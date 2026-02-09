#!/usr/bin/env python3
"""
Knowledge Base Embeddings Manager
Uses MLX (Apple Silicon) or Sentence Transformers + ChromaDB for semantic search
Auto-detects best backend for current hardware
"""
import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import frontmatter
import hashlib
import platform
import sys
from datetime import datetime


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/M4)."""
    return platform.machine() == "arm64" and platform.system() == "Darwin"


class MLXEmbeddingFunction(chromadb.EmbeddingFunction):
    """ChromaDB-compatible embedding function using MLX on Apple Silicon."""

    def __init__(self, model_name: str = "mlx-community/multilingual-e5-small-mlx"):
        from mlx_embeddings.utils import load
        self._model_name = model_name
        self.model, self.tokenizer = load(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        from mlx_embeddings.utils import generate
        embeddings = []
        for text in input:
            result = generate(self.model, self.tokenizer, text)
            embeddings.append(result.text_embeds[0].tolist())
        return embeddings

    def name(self) -> str:
        return "mlx_embedding_function"


class KnowledgeEmbeddings:
    def __init__(self, kb_path, chroma_path, backend: str | None = None):
        self.kb_path = Path(kb_path)
        self.chroma_path = chroma_path

        # Determine embedding backend
        self.embedding_function = self._init_embeddings(backend)

        # ChromaDB persistent client (SQLite backend, no Docker needed)
        self.client = chromadb.PersistentClient(path=chroma_path)

        # Collection name from env or default
        collection_name = os.environ.get("KB_COLLECTION", "knowledge_base")

        # Get or create collection (handle backend switch)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Knowledge base semantic search"}
            )
        except ValueError:
            # Embedding function changed — must reindex
            print("⚠️  Embedding backend changed, rebuilding collection...")
            self.client.delete_collection(collection_name)
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Knowledge base semantic search"}
            )

        print(f"✓ ChromaDB collection ready: {self.collection.count()} documents")

    def _init_embeddings(self, backend: str | None):
        """Initialize embedding function with auto-detection and fallback."""
        use_mlx = False

        if backend == "mlx":
            use_mlx = True
        elif backend == "st":
            use_mlx = False
        elif backend is None:
            # Auto-detect: prefer MLX on Apple Silicon
            use_mlx = _is_apple_silicon()

        if use_mlx:
            try:
                print("📚 Loading MLX embeddings (Apple Silicon native)...")
                ef = MLXEmbeddingFunction()
                print("✓ Using MLX embeddings (Apple Silicon)")
                return ef
            except Exception as e:
                print(f"⚠️  MLX failed ({e}), falling back to sentence-transformers...")

        print("📚 Loading embedding model (all-MiniLM-L6-v2)...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("✓ Using sentence-transformers embeddings")
        return ef

    def index_all_markdown(self, force=False):
        """Index all markdown files in knowledge base"""
        print(f"\n🔍 Scanning markdown files from {self.kb_path}...")

        indexed = 0
        skipped = 0
        errors = 0

        # Patterns to skip
        skip_patterns = [
            "README.md",
            "INDEX.md",
            ".archive_old",
            "archive/",
            ".venv/",
            "node_modules/",
            ".git/"
        ]

        for md_file in self.kb_path.rglob("*.md"):
            # Skip certain files
            if any(pattern in str(md_file) for pattern in skip_patterns):
                continue

            try:
                # Parse frontmatter + content
                with open(md_file, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)

                # Create unique ID from file path
                rel_path = str(md_file.relative_to(self.kb_path))
                file_id = hashlib.md5(rel_path.encode()).hexdigest()

                # Check if already indexed (unless force)
                if not force:
                    try:
                        existing = self.collection.get(ids=[file_id])
                        if existing['ids']:
                            skipped += 1
                            print(f"  ⏭️  Skipped (already indexed): {md_file.name}")
                            continue
                    except:
                        pass

                # Combine title + content for embedding
                title = post.metadata.get('title', md_file.stem)

                # Take first 3000 chars (enough for semantic understanding)
                content_preview = post.content[:3000] if post.content else ""
                text = f"{title}\n\n{content_preview}"

                # Prepare metadata (convert all values to strings for ChromaDB)
                created = post.metadata.get('created', '')
                updated = post.metadata.get('updated', str(datetime.now().date()))

                metadata = {
                    "file": rel_path,
                    "title": title,
                    "type": post.metadata.get('type', 'unknown'),
                    "status": post.metadata.get('status', 'active'),
                    "tags": ",".join(post.metadata.get('tags', [])),
                    "created": str(created) if created else '',
                    "updated": str(updated) if updated else '',
                }

                # Add opportunity score if present
                if 'opportunity_score' in post.metadata:
                    metadata['opportunity_score'] = str(post.metadata['opportunity_score'])

                # Add to ChromaDB (will auto-generate embedding)
                self.collection.upsert(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[file_id]
                )

                indexed += 1
                print(f"  ✓ Indexed: {md_file.name} ({len(content_preview)} chars)")

            except Exception as e:
                errors += 1
                print(f"  ✗ Error indexing {md_file.name}: {e}")

        print(f"\n📊 Indexing complete:")
        print(f"   • Indexed: {indexed}")
        print(f"   • Skipped: {skipped}")
        print(f"   • Errors: {errors}")
        print(f"   • Total in DB: {self.collection.count()}")

        return indexed

    def search(self, query, n_results=5, filter_dict=None, expand_graph=False):
        """
        Semantic search across knowledge base

        Args:
            query: Search query string (Russian or English)
            n_results: Number of results to return
            filter_dict: Optional metadata filter (e.g., {"type": "opportunity"})
            expand_graph: If True, expand top results with knowledge graph neighbors

        Returns:
            dict with 'documents', 'metadatas', 'distances'
        """
        print(f"\n🔍 Searching: '{query}'")

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )

        output = {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }

        if expand_graph and output['metadatas']:
            output = self._expand_with_graph(output)

        return output

    def _expand_with_graph(self, results):
        """Expand search results with knowledge graph neighbors (1-hop)."""
        import json as _json

        graph_path = Path(self.kb_path) / ".metadata" / "knowledge_graph.json"
        if not graph_path.exists():
            print("⚠️  Knowledge graph not found, skipping expansion. Run `make graph` first.")
            return results

        with open(graph_path, encoding="utf-8") as f:
            graph = _json.load(f)

        # Build adjacency for explicit + shared_tags edges only
        from collections import defaultdict
        adj = defaultdict(list)
        for e in graph["edges"]:
            if e["type"] in ("explicit", "shared_tags"):
                adj[e["source"]].append(e)
                adj[e["target"]].append({**e, "source": e["target"], "target": e["source"]})

        result_files = {m.get("file") for m in results["metadatas"]}

        # Expand top-3 results
        expanded_docs = []
        expanded_meta = []
        expanded_dist = []
        for meta in results["metadatas"][:3]:
            file_path = meta.get("file")
            if not file_path:
                continue
            for edge in adj.get(file_path, []):
                neighbor = edge["target"]
                if neighbor in result_files:
                    continue
                result_files.add(neighbor)
                node = graph["nodes"].get(neighbor, {})
                expanded_meta.append({
                    "file": neighbor,
                    "title": node.get("title", neighbor),
                    "type": node.get("type", "unknown"),
                    "tags": ",".join(node.get("tags", [])),
                    "graph_source": f"via {edge['type']}: {meta.get('title', file_path)}",
                })
                expanded_docs.append(f"[graph-expanded from {meta.get('title', '')}]")
                expanded_dist.append(1.0)  # placeholder distance for graph results

        if expanded_docs:
            print(f"  📊 Graph expansion: +{len(expanded_docs)} related documents")
            results["documents"].extend(expanded_docs)
            results["metadatas"].extend(expanded_meta)
            results["distances"].extend(expanded_dist)

        return results

    def search_by_tag(self, tag, n_results=10):
        """Search documents by tag"""
        results = self.collection.query(
            query_texts=[tag],
            n_results=n_results,
            where={"tags": {"$contains": tag}}
        )
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
        }

    def get_stats(self):
        """Get collection statistics"""
        count = self.collection.count()

        # Get all metadata to analyze
        all_data = self.collection.get()

        types = {}
        tags = set()

        for meta in all_data['metadatas']:
            doc_type = meta.get('type', 'unknown')
            types[doc_type] = types.get(doc_type, 0) + 1

            if meta.get('tags'):
                for tag in meta['tags'].split(','):
                    if tag.strip():
                        tags.add(tag.strip())

        return {
            'total_documents': count,
            'by_type': types,
            'unique_tags': len(tags),
            'model': 'all-MiniLM-L6-v2 (sentence-transformers)',
            'storage': self.chroma_path
        }

    def list_all(self):
        """List all indexed documents"""
        all_data = self.collection.get()

        print(f"\n📚 All indexed documents ({len(all_data['ids'])}):\n")

        for i, (doc_id, meta) in enumerate(zip(all_data['ids'], all_data['metadatas']), 1):
            title = meta.get('title', 'Untitled')
            file_path = meta.get('file', '')
            doc_type = meta.get('type', 'unknown')

            print(f"{i}. {title}")
            print(f"   Type: {doc_type} | File: {file_path}")
            print()


def main():
    """CLI interface for knowledge base search"""
    import argparse

    parser = argparse.ArgumentParser(description='Knowledge Base Semantic Search')
    parser.add_argument('command', choices=['index', 'search', 'stats', 'list'],
                       help='Command to execute')
    parser.add_argument('query', nargs='*', help='Search query (for search command)')
    parser.add_argument('--force', action='store_true', help='Force re-indexing')
    parser.add_argument('--type', help='Filter by type (principle, methodology, agent, opportunity)')
    parser.add_argument('--limit', type=int, default=5, help='Number of results')
    parser.add_argument('--backend', choices=['mlx', 'st'], default=None,
                       help='Embedding backend: mlx (Apple Silicon) or st (sentence-transformers)')
    parser.add_argument('--graph', action='store_true',
                       help='Expand results with knowledge graph neighbors')

    args = parser.parse_args()

    # Initialize
    project_root = Path(__file__).resolve().parent.parent
    kb = KnowledgeEmbeddings(
        kb_path=str(project_root),
        chroma_path=str(project_root / ".embeddings" / "chroma_db"),
        backend=args.backend
    )

    if args.command == 'index':
        kb.index_all_markdown(force=args.force)

    elif args.command == 'search':
        if not args.query:
            print("❌ Error: Please provide a search query")
            sys.exit(1)

        query = " ".join(args.query)
        filter_dict = {"type": args.type} if args.type else None

        results = kb.search(query, n_results=args.limit, filter_dict=filter_dict,
                           expand_graph=args.graph)

        if not results['documents']:
            print("❌ No results found")
            return

        print(f"\n📖 Found {len(results['documents'])} results:\n")

        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ), 1):
            graph_source = meta.get('graph_source')
            if graph_source:
                print(f"{i}. {meta['title']} [📊 {graph_source}]")
                print(f"   📄 {meta['file']}")
                print(f"   🏷️  Type: {meta['type']} | Tags: {meta.get('tags', 'none')}\n")
            else:
                score = 1 - dist  # Convert distance to similarity score
                print(f"{i}. {meta['title']} (relevance: {score:.2%})")
                print(f"   📄 {meta['file']}")
                print(f"   🏷️  Type: {meta['type']} | Tags: {meta.get('tags', 'none')}")
                preview = doc[:200].replace('\n', ' ').strip()
                print(f"   📝 {preview}...\n")

    elif args.command == 'stats':
        stats = kb.get_stats()
        print(f"\n📊 Knowledge Base Statistics:\n")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Embedding model: {stats['model']}")
        print(f"Storage: {stats['storage']}")
        print(f"Unique tags: {stats['unique_tags']}")
        print(f"\nBy type:")
        for doc_type, count in stats['by_type'].items():
            print(f"  • {doc_type}: {count}")

    elif args.command == 'list':
        kb.list_all()


if __name__ == "__main__":
    main()
