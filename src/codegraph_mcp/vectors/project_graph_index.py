"""Per-project FalkorDBLite vector databases for code and documentation.

Each project gets its own FalkorDBLite instance at {project_path}/.solo/vectors/graph.db.
Stores chunks as graph nodes with vector embeddings — enables hybrid graph+vector queries.
Uses semantic-text-splitter (Rust core) with tree-sitter for AST-aware chunking.

Vectors live on graph nodes, enabling hybrid queries:
  e.g. "find similar code -> show its imports -> find other files using same packages"
"""

import shutil
from pathlib import Path

from redislite.falkordb_client import FalkorDB

from .common import (
    VECTORS_ROOT,
    get_code_splitter,
    get_markdown_splitter,
    init_embedding_function,
    scan_project_files,
    TS_GRAMMAR_MAP,
    CHUNK_CAPACITY,
    EMBEDDING_DIM,
)

# Registry path from env or ~/.solo/
import os
_REGISTRY_ENV = os.environ.get("CODEGRAPH_REGISTRY", "")
_REGISTRY_PATH = Path(_REGISTRY_ENV).expanduser() if _REGISTRY_ENV else Path.home() / ".solo" / "registry.yaml"


class ProjectGraphIndex:
    """Per-project FalkorDBLite vector index for source code and documentation."""

    def __init__(self, backend: str | None = None):
        self._ef = init_embedding_function(backend)
        self._dbs: dict[str, FalkorDB] = {}
        self._paths: dict[str, Path] = {}  # name → project_path
        self._md_splitter = None
        self._registry_loaded = False

    def _ensure_registry(self):
        """Lazy-load project paths from registry.yaml."""
        if self._registry_loaded:
            return
        self._registry_loaded = True
        if not _REGISTRY_PATH.exists():
            return
        import yaml
        with open(_REGISTRY_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for p in data.get("projects", []):
            name = p["name"]
            path = Path(p["path"]).expanduser()
            if path.exists() and name not in self._paths:
                self._paths[name] = path

    def _db_dir(self, project_name: str) -> Path:
        """DB directory: {project_path}/.solo/vectors/ or legacy fallback."""
        self._ensure_registry()
        if project_name in self._paths:
            return self._paths[project_name] / ".solo" / "vectors"
        # Legacy fallback for projects not in registry
        return VECTORS_ROOT / project_name

    def _get_graph(self, project_name: str):
        """Get or create a FalkorDBLite graph for a project (lazy)."""
        if project_name not in self._dbs:
            db_path = self._db_dir(project_name)
            db_path.mkdir(parents=True, exist_ok=True)
            fdb = FalkorDB(str(db_path / "graph.db"))
            self._dbs[project_name] = fdb
            graph = fdb.select_graph("content")
            self._init_schema(graph)
            return graph
        return self._dbs[project_name].select_graph("content")

    def _init_schema(self, graph):
        """Create vector index and standard indexes."""
        indexes = [
            f"CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{dimension: {EMBEDDING_DIM}, similarityFunction: 'cosine'}}",
            "CREATE INDEX FOR (c:Chunk) ON (c.chunk_id)",
            "CREATE INDEX FOR (f:File) ON (f.path)",
        ]
        for idx in indexes:
            try:
                graph.query(idx)
            except Exception:
                pass  # already exists

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of texts. Returns plain Python floats."""
        if not texts:
            return []
        raw = self._ef(texts)
        # Convert numpy arrays to plain lists (FalkorDB needs native Python types)
        return [[float(x) for x in emb] for emb in raw]

    def _chunk_file(self, file_path: Path, lang: str, rel_path: str) -> list[dict]:
        """Chunk a single file into documents with metadata."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        if not content.strip():
            return []

        # Skip binary files masquerading as code (e.g. .ts files with binary data)
        if '\x00' in content or sum(1 for c in content[:2000] if ord(c) < 32 and c not in '\n\r\t') > 20:
            return []

        chunk_type = "doc" if lang == "markdown" else "code"

        if lang == "markdown":
            if self._md_splitter is None:
                self._md_splitter = get_markdown_splitter()
            raw_chunks = self._md_splitter.chunks(content)
        elif lang in TS_GRAMMAR_MAP:
            splitter = get_code_splitter(lang)
            if splitter:
                try:
                    raw_chunks = splitter.chunks(content)
                except Exception:
                    raw_chunks = [content[:CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]
            else:
                raw_chunks = [content[:CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]
        else:
            raw_chunks = [content[:CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]

        chunks = []
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

        Creates File and Chunk nodes with embeddings, linked by HAS_CHUNK edges.
        Returns stats: {chunks, files, code_chunks, doc_chunks}.
        """
        import gc

        # Register path so _db_dir resolves to {project_path}/.solo/
        self._paths[project_name] = project_path

        graph = self._get_graph(project_name)

        # Clear old data
        try:
            graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

        files = scan_project_files(project_path)
        file_count = 0
        total_chunks = 0
        code_chunks = 0
        doc_chunks = 0

        # Process file by file, embed + insert in small batches
        batch: list[dict] = []
        batch_file: str | None = None
        batch_size = 16

        for abs_path, lang in files:
            rel = str(abs_path.relative_to(project_path))
            file_chunks = self._chunk_file(abs_path, lang, rel)
            if not file_chunks:
                continue

            file_count += 1
            # Create File node
            graph.query(
                "MERGE (f:File {path: $path}) SET f.language = $lang",
                params={"path": rel, "lang": lang},
            )

            for c in file_chunks:
                batch.append(c)
                if c["metadata"]["chunk_type"] == "code":
                    code_chunks += 1
                else:
                    doc_chunks += 1

                if len(batch) >= batch_size:
                    self._flush_batch(graph, batch)
                    total_chunks += len(batch)
                    batch.clear()
                    gc.collect()

        # Flush remaining
        if batch:
            self._flush_batch(graph, batch)
            total_chunks += len(batch)
            gc.collect()

        return {
            "chunks": total_chunks,
            "files": file_count,
            "code_chunks": code_chunks,
            "doc_chunks": doc_chunks,
        }

    def _flush_batch(self, graph, batch: list[dict]):
        """Embed a batch of chunks and insert into graph using UNWIND (single query)."""
        texts = [c["document"] for c in batch]
        embeddings = self._embed(texts)

        # Build items list for UNWIND
        items = []
        for chunk, emb in zip(batch, embeddings):
            meta = chunk["metadata"]
            items.append({
                "cid": chunk["id"],
                "text": chunk["document"],
                "ct": meta["chunk_type"],
                "ci": meta["chunk_index"],
                "tc": meta["total_chunks"],
                "lang": meta["language"],
                "fp": meta["file"],
                "emb": emb,
            })

        # Single UNWIND query: create all Chunk nodes + link to File nodes
        graph.query(
            "UNWIND $items AS item "
            "CREATE (c:Chunk {"
            "  chunk_id: item.cid, text: item.text, chunk_type: item.ct,"
            "  chunk_index: item.ci, total_chunks: item.tc,"
            "  language: item.lang, file_path: item.fp,"
            "  embedding: vecf32(item.emb)"
            "}) "
            "WITH c, item "
            "MATCH (f:File {path: item.fp}) "
            "CREATE (f)-[:HAS_CHUNK]->(c)",
            params={"items": items},
        )

    def search(
        self,
        query: str,
        project: str | None = None,
        n_results: int = 5,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """Semantic search over project code/docs via FalkorDB vector index."""
        query_emb = self._embed([query])[0]

        if project:
            return self._search_one(query_emb, project, n_results, chunk_type)

        # Search all projects, merge by score
        all_results: list[dict] = []
        for proj_name in self._discover_projects():
            results = self._search_one(query_emb, proj_name, n_results, chunk_type)
            all_results.extend(results)

        # Sort by relevance (highest = best) and take top N
        all_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return all_results[:n_results]

    def _search_one(
        self, query_emb: list[float], project_name: str, n_results: int,
        chunk_type: str | None,
    ) -> list[dict]:
        """Search a single project's FalkorDB graph."""
        try:
            graph = self._get_graph(project_name)
        except Exception:
            return []

        # Check if there are any chunks
        count_result = graph.query("MATCH (c:Chunk) RETURN count(c)")
        count = count_result.result_set[0][0] if count_result.result_set else 0
        if count == 0:
            return []

        actual_n = min(n_results, count)

        # Vector search — k must be inlined (FalkorDB procedure limitation)
        if chunk_type:
            cypher = (
                f"CALL db.idx.vector.queryNodes('Chunk', 'embedding', {actual_n * 2}, vecf32($q)) "
                "YIELD node, score "
                "WHERE node.chunk_type = $ct "
                "RETURN node.chunk_id, node.file_path, node.language, node.chunk_type, "
                "node.chunk_index, node.text, score "
                f"LIMIT {actual_n}"
            )
            params = {"q": query_emb, "ct": chunk_type}
        else:
            cypher = (
                f"CALL db.idx.vector.queryNodes('Chunk', 'embedding', {actual_n}, vecf32($q)) "
                "YIELD node, score "
                "RETURN node.chunk_id, node.file_path, node.language, node.chunk_type, "
                "node.chunk_index, node.text, score"
            )
            params = {"q": query_emb}

        try:
            result = graph.query(cypher, params=params)
        except Exception:
            return []

        output = []
        for row in result.result_set[:n_results]:
            doc_id, file_path, lang, ct, ci, text, score = row
            output.append({
                "id": doc_id,
                "file": file_path or "",
                "language": lang or "",
                "chunk_type": ct or "",
                "chunk_index": ci or 0,
                "relevance": round(1 - score, 4),  # cosine distance → similarity
                "snippet": (text or "")[:500],
                "project": project_name,
            })

        return output

    def search_hybrid(
        self,
        query: str,
        project: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Hybrid search: vector similarity + graph traversal.

        Finds similar chunks, then returns neighboring chunks from the same files.
        Structural context via graph traversal — finds neighboring chunks from same files.
        """
        query_emb = self._embed([query])[0]

        try:
            graph = self._get_graph(project)
        except Exception:
            return []

        # Find top chunks by vector similarity, then get sibling chunks from same files
        cypher = (
            f"CALL db.idx.vector.queryNodes('Chunk', 'embedding', {n_results}, vecf32($q)) "
            "YIELD node, score "
            "MATCH (f:File)-[:HAS_CHUNK]->(node) "
            "OPTIONAL MATCH (f)-[:HAS_CHUNK]->(sibling:Chunk) "
            "WHERE sibling <> node "
            "RETURN node.chunk_id, f.path, f.language, node.chunk_type, "
            "node.text, score, collect(DISTINCT sibling.chunk_index) AS sibling_chunks "
            "ORDER BY score ASC"
        )

        try:
            result = graph.query(cypher, params={"q": query_emb})
        except Exception:
            return []

        output = []
        for row in result.result_set:
            doc_id, file_path, lang, ct, text, score, siblings = row
            output.append({
                "id": doc_id,
                "file": file_path or "",
                "language": lang or "",
                "chunk_type": ct or "",
                "relevance": round(1 - score, 4),
                "snippet": (text or "")[:500],
                "project": project,
                "sibling_chunks": siblings or [],
            })

        return output

    def _discover_projects(self) -> list[str]:
        """Find all indexed projects (with .solo/vectors/ in project dir)."""
        self._ensure_registry()
        found = []
        # Check in-project .solo/ dirs (new location)
        for name, path in self._paths.items():
            if (path / ".solo" / "vectors").exists():
                found.append(name)
        # Legacy: check ~/.solo/vectors/
        if VECTORS_ROOT.exists():
            for d in VECTORS_ROOT.iterdir():
                if d.is_dir() and (d / "graph.db").exists() and d.name not in found:
                    found.append(d.name)
        return found

    def list_projects(self) -> list[dict]:
        """List all FalkorDB-indexed projects with stats."""
        projects = []
        for name in self._discover_projects():
            db_path = self._db_dir(name)
            try:
                graph = self._get_graph(name)
                result = graph.query("MATCH (c:Chunk) RETURN count(c)")
                chunks = result.result_set[0][0] if result.result_set else 0
            except Exception:
                chunks = 0

            size_bytes = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
            size_mb = round(size_bytes / 1024 / 1024, 2)

            projects.append({
                "name": name,
                "chunks": chunks,
                "size_mb": size_mb,
            })

        return sorted(projects, key=lambda x: x["name"])

    def delete_project(self, project_name: str) -> bool:
        """Delete a project's FalkorDB vector database."""
        db_path = self._db_dir(project_name)
        if db_path.exists():
            self._dbs.pop(project_name, None)
            shutil.rmtree(db_path)
            return True
        return False

    def stats(self) -> dict:
        """Overall statistics across all FalkorDB-indexed projects."""
        projects = self.list_projects()
        return {
            "projects": len(projects),
            "total_chunks": sum(p["chunks"] for p in projects),
            "total_size_mb": round(sum(p["size_mb"] for p in projects), 2),
            "per_project": projects,
        }
