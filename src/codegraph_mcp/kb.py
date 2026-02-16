"""Knowledge Base Embeddings Manager — FalkorDB vector backend.

Indexes markdown files with YAML frontmatter into FalkorDB graph with vector embeddings.
Supports MLX (Apple Silicon, multilingual-e5-small) and sentence-transformers (all-MiniLM-L6-v2) fallback.

Features:
  - SHA-256 content dedup: skip re-embedding unchanged documents
  - Hybrid search: vector cosine + full-text TF-IDF with Reciprocal Rank Fusion
"""

import hashlib
import json as _json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import frontmatter

from .vectors.common import EMBEDDING_DIM, init_embedding_function

try:
    from redislite.falkordb_client import FalkorDB
except ImportError:
    FalkorDB = None


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion — merge multiple ranked ID lists into one.

    Each list is ordered by relevance (best first). RRF score for doc d:
        score(d) = sum(1 / (k + rank_i)) for each list where d appears.

    Returns [(doc_id, rrf_score)] sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class KnowledgeEmbeddings:
    def __init__(self, kb_path, db_path=None, backend: str | None = None):
        self.kb_path = Path(kb_path).expanduser()
        self._db_path = Path(db_path).expanduser() if db_path else self.kb_path / ".solo" / "kb"
        self._db_path.mkdir(parents=True, exist_ok=True)

        self._ef = init_embedding_function(backend)

        if FalkorDB is None:
            raise ImportError("falkordblite is required: uv add falkordblite")

        self._fdb = FalkorDB(str(self._db_path / "graph.db"))
        self._graph = self._fdb.select_graph("kb")
        self._init_schema()

        count = self._count()
        print(f"KB ready: {count} documents ({self._db_path})")

    def _init_schema(self):
        indexes = [
            f"CREATE VECTOR INDEX FOR (d:KBDoc) ON (d.embedding) "
            f"OPTIONS {{dimension: {EMBEDDING_DIM}, similarityFunction: 'cosine'}}",
            "CREATE INDEX FOR (d:KBDoc) ON (d.doc_id)",
            "CREATE INDEX FOR (d:KBDoc) ON (d.doc_type)",
            "CREATE INDEX FOR (d:KBDoc) ON (d.content_hash)",
        ]
        for idx in indexes:
            try:
                self._graph.query(idx)
            except Exception:
                pass

        # Full-text index for hybrid search (TF-IDF on title + content)
        try:
            self._graph.query("CALL db.idx.fulltext.createNodeIndex('KBDoc', 'title', 'content')")
        except Exception:
            pass  # already exists

    def _count(self) -> int:
        result = self._graph.query("MATCH (d:KBDoc) RETURN count(d)")
        return result.result_set[0][0] if result.result_set else 0

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        raw = self._ef(texts)
        return [[float(x) for x in emb] for emb in raw]

    def index_all_markdown(self, force=False):
        """Index all markdown files in knowledge base.

        Uses SHA-256 content hashing to skip re-embedding unchanged documents.
        Only new or modified content triggers embedding computation.
        """
        print(f"\nScanning markdown files from {self.kb_path}...")

        indexed = 0
        skipped = 0
        updated = 0
        errors = 0

        skip_patterns = [
            "README.md",
            "INDEX.md",
            ".archive_old",
            "archive/",
            ".venv/",
            "node_modules/",
            ".git/",
        ]

        for md_file in self.kb_path.rglob("*.md"):
            if any(pattern in str(md_file) for pattern in skip_patterns):
                continue

            try:
                with open(md_file, encoding="utf-8") as f:
                    post = frontmatter.load(f)

                rel_path = str(md_file.relative_to(self.kb_path))
                file_id = hashlib.md5(rel_path.encode()).hexdigest()

                title = post.metadata.get("title", md_file.stem)
                content_preview = post.content[:3000] if post.content else ""
                text = f"{title}\n\n{content_preview}"

                # SHA-256 content hash for dedup
                content_hash = hashlib.sha256(text.encode()).hexdigest()

                doc_type = post.metadata.get("type", "unknown")
                status = post.metadata.get("status", "active")
                tags = ",".join(post.metadata.get("tags", []))
                created = str(post.metadata.get("created", ""))
                updated_date = str(post.metadata.get("updated", str(datetime.now().date())))

                # Check existing document
                if not force:
                    existing = self._graph.query(
                        "MATCH (d:KBDoc {doc_id: $did}) RETURN d.content_hash",
                        {"did": file_id},
                    )
                    if existing.result_set:
                        old_hash = existing.result_set[0][0]
                        if old_hash == content_hash:
                            # Content unchanged — skip entirely
                            skipped += 1
                            continue
                        # Content changed — update metadata + re-embed
                        updated += 1
                    # else: new document

                embedding = self._embed([text])[0]

                self._graph.query(
                    "MERGE (d:KBDoc {doc_id: $did}) "
                    "SET d.file = $file, d.title = $title, "
                    "d.doc_type = $dtype, d.status = $status, "
                    "d.tags = $tags, d.created = $created, d.updated = $updated, "
                    "d.content = $content, d.content_hash = $hash, "
                    "d.embedding = vecf32($emb)",
                    {
                        "did": file_id,
                        "file": rel_path,
                        "title": title,
                        "dtype": doc_type,
                        "status": status,
                        "tags": tags,
                        "created": created,
                        "updated": updated_date,
                        "content": text[:500],
                        "hash": content_hash,
                        "emb": embedding,
                    },
                )

                # Store opportunity score if present
                if "opportunity_score" in post.metadata:
                    self._graph.query(
                        "MATCH (d:KBDoc {doc_id: $did}) SET d.opportunity_score = $score",
                        {
                            "did": file_id,
                            "score": float(post.metadata["opportunity_score"]),
                        },
                    )

                indexed += 1
                tag = "updated" if updated else "new"
                print(f"  Indexed ({tag}): {md_file.name} ({len(content_preview)} chars)")

            except Exception as e:
                errors += 1
                print(f"  Error: {md_file.name}: {e}")

        print(f"\nIndexing complete: {indexed} indexed, {skipped} unchanged, {errors} errors")
        print(f"Total in DB: {self._count()}")
        return indexed

    def search(self, query, n_results=5, filter_dict=None, expand_graph=False):
        """Hybrid search: vector cosine + full-text TF-IDF with RRF fusion.

        Args:
            query: Search query (Russian or English)
            n_results: Number of results
            filter_dict: Optional metadata filter (e.g. {"type": "opportunity"})
            expand_graph: Expand top results with knowledge graph neighbors
        """
        count = self._count()
        if count == 0:
            return {"documents": [], "metadatas": [], "distances": []}

        fetch_n = min(n_results * 3, count)
        dtype_filter = filter_dict.get("type") if filter_dict else None

        # 1. Vector search
        vec_ids = self._search_vector(query, fetch_n, dtype_filter)

        # 2. Full-text search
        ft_ids = self._search_fulltext(query, fetch_n, dtype_filter)

        # 3. RRF fusion
        if ft_ids:
            fused = _rrf([vec_ids, ft_ids])
            result_ids = [doc_id for doc_id, _ in fused[:n_results]]
        else:
            # Fallback to pure vector if full-text returns nothing
            result_ids = vec_ids[:n_results]

        if not result_ids:
            return {"documents": [], "metadatas": [], "distances": []}

        # 4. Fetch full documents by IDs (preserve RRF order)
        output = self._fetch_docs(result_ids)

        if expand_graph and output["metadatas"]:
            output = self._expand_with_graph(output)

        return output

    def _search_vector(self, query: str, n: int, dtype: str | None = None) -> list[str]:
        """Pure vector search. Returns ordered list of doc_ids."""
        query_emb = self._embed([query])[0]

        cypher = f"CALL db.idx.vector.queryNodes('KBDoc', 'embedding', {n}, vecf32($q)) YIELD node, score "
        params: dict = {"q": query_emb}
        if dtype:
            cypher += "WHERE node.doc_type = $dtype "
            params["dtype"] = dtype
        cypher += "RETURN node.doc_id, score ORDER BY score ASC"

        try:
            result = self._graph.query(cypher, params=params)
            return [row[0] for row in result.result_set]
        except Exception:
            return []

    def _search_fulltext(self, query: str, n: int, dtype: str | None = None) -> list[str]:
        """Full-text TF-IDF search. Returns ordered list of doc_ids."""
        # Sanitize query for full-text: escape special chars, keep words
        safe_query = " ".join(word for word in query.split() if not any(c in word for c in '(){}[]+-~*"\\'))
        if not safe_query.strip():
            return []

        cypher = "CALL db.idx.fulltext.queryNodes('KBDoc', $q) YIELD node, score "
        params: dict = {"q": safe_query}
        if dtype:
            cypher += "WHERE node.doc_type = $dtype "
            params["dtype"] = dtype
        cypher += f"RETURN node.doc_id, score ORDER BY score DESC LIMIT {n}"

        try:
            result = self._graph.query(cypher, params=params)
            return [row[0] for row in result.result_set]
        except Exception:
            return []

    def _fetch_docs(self, doc_ids: list[str]) -> dict:
        """Fetch full document data for a list of doc_ids, preserving order."""
        if not doc_ids:
            return {"documents": [], "metadatas": [], "distances": []}

        result = self._graph.query(
            "UNWIND $ids AS did "
            "MATCH (d:KBDoc {doc_id: did}) "
            "RETURN d.doc_id, d.file, d.title, d.doc_type, d.tags, d.content",
            {"ids": doc_ids},
        )

        # Build lookup for order preservation
        lookup = {}
        for row in result.result_set:
            doc_id, file, title, dtype, tags, content = row
            lookup[doc_id] = {
                "content": content or "",
                "meta": {
                    "file": file or "",
                    "title": title or "",
                    "type": dtype or "",
                    "tags": tags or "",
                },
            }

        documents = []
        metadatas = []
        distances = []
        for i, doc_id in enumerate(doc_ids):
            if doc_id in lookup:
                documents.append(lookup[doc_id]["content"])
                metadatas.append(lookup[doc_id]["meta"])
                # Synthetic distance: RRF rank as proxy (lower = better)
                distances.append(i * 0.1)

        return {"documents": documents, "metadatas": metadatas, "distances": distances}

    def _expand_with_graph(self, results):
        """Expand search results with knowledge graph neighbors (1-hop)."""
        graph_path = Path(self.kb_path) / ".metadata" / "knowledge_graph.json"
        if not graph_path.exists():
            return results

        with open(graph_path, encoding="utf-8") as f:
            graph = _json.load(f)

        adj = defaultdict(list)
        for e in graph["edges"]:
            if e["type"] in ("explicit", "shared_tags"):
                adj[e["source"]].append(e)
                adj[e["target"]].append({**e, "source": e["target"], "target": e["source"]})

        result_files = {m.get("file") for m in results["metadatas"]}

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
                expanded_meta.append(
                    {
                        "file": neighbor,
                        "title": node.get("title", neighbor),
                        "type": node.get("type", "unknown"),
                        "tags": ",".join(node.get("tags", [])),
                        "graph_source": f"via {edge['type']}: {meta.get('title', file_path)}",
                    }
                )
                expanded_docs.append(f"[graph-expanded from {meta.get('title', '')}]")
                expanded_dist.append(1.0)

        if expanded_docs:
            results["documents"].extend(expanded_docs)
            results["metadatas"].extend(expanded_meta)
            results["distances"].extend(expanded_dist)

        return results

    def get_stats(self):
        """Get collection statistics."""
        count = self._count()

        type_result = self._graph.query("MATCH (d:KBDoc) RETURN d.doc_type, count(d) ORDER BY count(d) DESC")
        types = {}
        for row in type_result.result_set:
            types[row[0] or "unknown"] = row[1]

        tag_result = self._graph.query("MATCH (d:KBDoc) WHERE d.tags <> '' RETURN d.tags")
        tags = set()
        for row in tag_result.result_set:
            for tag in (row[0] or "").split(","):
                if tag.strip():
                    tags.add(tag.strip())

        return {
            "total_documents": count,
            "by_type": types,
            "unique_tags": len(tags),
            "model": "MLX multilingual-e5-small / ST all-MiniLM-L6-v2 (384 dim)",
            "search": "hybrid (vector cosine + full-text TF-IDF + RRF)",
            "storage": str(self._db_path),
        }


def main():
    """CLI interface for knowledge base search."""
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Semantic Search")
    parser.add_argument("command", choices=["index", "search", "stats"], help="Command to execute")
    parser.add_argument("query", nargs="*", help="Search query (for search command)")
    parser.add_argument("--force", action="store_true", help="Force re-indexing")
    parser.add_argument("--type", help="Filter by type")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--backend", choices=["mlx", "st"], default=None, help="Embedding backend")
    parser.add_argument("--graph", action="store_true", help="Expand with knowledge graph neighbors")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    kb = KnowledgeEmbeddings(kb_path=str(project_root), backend=args.backend)

    if args.command == "index":
        kb.index_all_markdown(force=args.force)
    elif args.command == "search":
        if not args.query:
            print("Error: provide a search query")
            sys.exit(1)
        query = " ".join(args.query)
        filter_dict = {"type": args.type} if args.type else None
        results = kb.search(
            query,
            n_results=args.limit,
            filter_dict=filter_dict,
            expand_graph=args.graph,
        )
        if not results["documents"]:
            print("No results found")
            return
        print(f"\nFound {len(results['documents'])} results:\n")
        for i, (_doc, meta, _dist) in enumerate(
            zip(results["documents"], results["metadatas"], results["distances"]), 1
        ):
            print(f"{i}. {meta['title']}")
            print(f"   File: {meta['file']} | Type: {meta['type']} | Tags: {meta.get('tags', 'none')}")
            print()
    elif args.command == "stats":
        stats = kb.get_stats()
        print("\nKB Statistics:\n")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Embedding model: {stats['model']}")
        print(f"Search: {stats['search']}")
        print(f"Storage: {stats['storage']}")
        print(f"Unique tags: {stats['unique_tags']}")
        print("\nBy type:")
        for doc_type, count in stats["by_type"].items():
            print(f"  {doc_type}: {count}")


if __name__ == "__main__":
    main()
