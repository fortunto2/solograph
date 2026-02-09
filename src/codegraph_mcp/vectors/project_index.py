"""Per-project ChromaDB vector databases for code and documentation.

Each project gets its own PersistentClient at ~/.codegraph/vectors/{project}/chroma_db/.
Uses semantic-text-splitter (Rust core) with tree-sitter for AST-aware chunking.
"""

import shutil
from pathlib import Path

import chromadb

from ..scanner.code import LANG_MAP, SKIP_DIRS, SKIP_FILES

VECTORS_ROOT = Path.home() / ".codegraph" / "vectors"

# Extensions for markdown docs
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}

# All scannable extensions (code + docs)
ALL_EXTENSIONS = set(LANG_MAP.keys()) | DOC_EXTENSIONS

# tree-sitter grammar module mapping
_TS_GRAMMAR_MAP = {
    "python": "tree_sitter_python",
    "swift": "tree_sitter_swift",
    "typescript": "tree_sitter_typescript",
    "kotlin": "tree_sitter_kotlin",
}

# Chunk capacity range (min, max) in characters
CHUNK_CAPACITY = (200, 1500)


def _init_embedding_function(backend: str | None = None):
    """Initialize embedding function (reuses logic from session_index.py)."""
    import platform
    import sys

    use_mlx = False
    if backend == "mlx":
        use_mlx = True
    elif backend == "st":
        use_mlx = False
    elif backend is None:
        use_mlx = platform.machine() == "arm64" and platform.system() == "Darwin"

    if use_mlx:
        try:
            from ..kb import MLXEmbeddingFunction

            return MLXEmbeddingFunction()
        except Exception:
            pass

    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def _get_code_splitter(lang: str):
    """Create a CodeSplitter for the given language."""
    import importlib

    from semantic_text_splitter import CodeSplitter

    grammar_module = _TS_GRAMMAR_MAP.get(lang)
    if not grammar_module:
        return None
    try:
        mod = importlib.import_module(grammar_module)
        return CodeSplitter(mod.language(), CHUNK_CAPACITY)
    except Exception:
        return None


def _get_markdown_splitter():
    """Create a MarkdownSplitter."""
    from semantic_text_splitter import MarkdownSplitter

    return MarkdownSplitter(CHUNK_CAPACITY)


def _scan_project_files(project_path: Path) -> list[tuple[Path, str]]:
    """Scan project directory for indexable files. Returns (abs_path, language)."""
    extended_lang_map = dict(LANG_MAP)
    for ext in DOC_EXTENSIONS:
        extended_lang_map[ext] = "markdown"

    files = []
    for ext, lang in extended_lang_map.items():
        for fp in project_path.rglob(f"*{ext}"):
            if any(part in SKIP_DIRS for part in fp.parts):
                continue
            if fp.name in SKIP_FILES:
                continue
            files.append((fp, lang))
    return files


class ProjectIndex:
    """Per-project ChromaDB vector index for source code and documentation."""

    def __init__(self, backend: str | None = None):
        self._ef = _init_embedding_function(backend)
        self._clients: dict[str, chromadb.PersistentClient] = {}
        self._md_splitter = None

    def _get_collection(self, project_name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection for a project (lazy)."""
        if project_name not in self._clients:
            db_path = VECTORS_ROOT / project_name / "chroma_db"
            db_path.mkdir(parents=True, exist_ok=True)
            self._clients[project_name] = chromadb.PersistentClient(path=str(db_path))

        client = self._clients[project_name]
        return client.get_or_create_collection(
            name="content",
            embedding_function=self._ef,
            metadata={"description": f"Code and docs for {project_name}"},
        )

    def _chunk_file(self, file_path: Path, lang: str, rel_path: str) -> list[dict]:
        """Chunk a single file into documents with metadata."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        if not content.strip():
            return []

        chunks = []
        chunk_type = "doc" if lang == "markdown" else "code"

        if lang == "markdown":
            if self._md_splitter is None:
                self._md_splitter = _get_markdown_splitter()
            raw_chunks = self._md_splitter.chunks(content)
        elif lang in _TS_GRAMMAR_MAP:
            splitter = _get_code_splitter(lang)
            if splitter:
                try:
                    raw_chunks = splitter.chunks(content)
                except Exception:
                    # Fallback: whole file if tree-sitter fails
                    raw_chunks = [content[:CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]
            else:
                raw_chunks = [content[:CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]
        else:
            # Unknown language — whole file as one chunk
            raw_chunks = [content[:CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]

        total = len(raw_chunks)
        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            chunks.append({
                "id": f"{rel_path}::chunk_{i}",
                "document": chunk_text,
                "metadata": {
                    "file": rel_path,
                    "language": lang,
                    "chunk_type": chunk_type,
                    "chunk_index": i,
                    "total_chunks": total,
                },
            })

        return chunks

    def index_project(self, project_path: Path, project_name: str) -> dict:
        """Index all code and doc files in a project.

        Returns stats: {chunks, files, code_chunks, doc_chunks}.
        """
        collection = self._get_collection(project_name)

        # Clear existing data for clean reindex
        existing = collection.count()
        if existing > 0:
            # Delete all and recreate
            client = self._clients[project_name]
            client.delete_collection("content")
            collection = client.get_or_create_collection(
                name="content",
                embedding_function=self._ef,
                metadata={"description": f"Code and docs for {project_name}"},
            )

        import gc

        files = _scan_project_files(project_path)
        file_count = 0
        total_chunks = 0
        code_chunks = 0
        doc_chunks = 0
        batch: list[dict] = []
        batch_size = 32  # small batches — embeddings computed during upsert

        for abs_path, lang in files:
            rel = str(abs_path.relative_to(project_path))
            file_chunks = self._chunk_file(abs_path, lang, rel)
            if not file_chunks:
                continue
            file_count += 1
            for c in file_chunks:
                batch.append(c)
                if c["metadata"]["chunk_type"] == "code":
                    code_chunks += 1
                else:
                    doc_chunks += 1

                if len(batch) >= batch_size:
                    collection.upsert(
                        ids=[x["id"] for x in batch],
                        documents=[x["document"] for x in batch],
                        metadatas=[x["metadata"] for x in batch],
                    )
                    total_chunks += len(batch)
                    batch.clear()
                    gc.collect()

        # Flush remaining
        if batch:
            collection.upsert(
                ids=[x["id"] for x in batch],
                documents=[x["document"] for x in batch],
                metadatas=[x["metadata"] for x in batch],
            )
            total_chunks += len(batch)
            gc.collect()

        return {
            "chunks": total_chunks,
            "files": file_count,
            "code_chunks": code_chunks,
            "doc_chunks": doc_chunks,
        }

    def search(
        self,
        query: str,
        project: str | None = None,
        n_results: int = 5,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """Semantic search over project code/docs.

        Args:
            query: Search query
            project: Search one project. None = search all indexed projects.
            n_results: Number of results
            chunk_type: Filter by "code" or "doc"
        """
        where = {"chunk_type": chunk_type} if chunk_type else None

        if project:
            return self._search_one(query, project, n_results, where)

        # Search all projects, merge by distance
        all_results: list[dict] = []
        for proj_name in self._discover_projects():
            results = self._search_one(query, proj_name, n_results, where)
            for r in results:
                r["project"] = proj_name
            all_results.extend(results)

        # Sort by relevance (highest first) and take top N
        all_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return all_results[:n_results]

    def _search_one(
        self, query: str, project_name: str, n_results: int, where: dict | None
    ) -> list[dict]:
        """Search a single project's collection."""
        try:
            collection = self._get_collection(project_name)
        except Exception:
            return []

        if collection.count() == 0:
            return []

        actual_n = min(n_results, collection.count())

        try:
            results = collection.query(
                query_texts=[query],
                n_results=actual_n,
                where=where,
            )
        except Exception:
            return []

        output = []
        if not results["ids"] or not results["ids"][0]:
            return output

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 1.0
            doc = results["documents"][0][i] if results["documents"] else ""
            output.append({
                "id": doc_id,
                "file": meta.get("file", ""),
                "language": meta.get("language", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "relevance": round(1 - dist, 4),
                "snippet": doc[:500],
                "project": project_name,
            })

        return output

    def _discover_projects(self) -> list[str]:
        """Find all indexed project directories."""
        if not VECTORS_ROOT.exists():
            return []
        return [
            d.name
            for d in VECTORS_ROOT.iterdir()
            if d.is_dir() and (d / "chroma_db").exists()
        ]

    def list_projects(self) -> list[dict]:
        """List all indexed projects with stats."""
        projects = []
        for name in self._discover_projects():
            db_path = VECTORS_ROOT / name / "chroma_db"
            try:
                collection = self._get_collection(name)
                chunks = collection.count()
            except Exception:
                chunks = 0

            # Disk size
            size_bytes = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
            size_mb = round(size_bytes / 1024 / 1024, 2)

            projects.append({
                "name": name,
                "chunks": chunks,
                "size_mb": size_mb,
            })

        return sorted(projects, key=lambda x: x["name"])

    def delete_project(self, project_name: str) -> bool:
        """Delete a project's vector database."""
        proj_path = VECTORS_ROOT / project_name
        if proj_path.exists():
            # Close client if open
            self._clients.pop(project_name, None)
            shutil.rmtree(proj_path)
            return True
        return False

    def stats(self) -> dict:
        """Overall statistics across all projects."""
        projects = self.list_projects()
        return {
            "projects": len(projects),
            "total_chunks": sum(p["chunks"] for p in projects),
            "total_size_mb": round(sum(p["size_mb"] for p in projects), 2),
            "per_project": projects,
        }
