"""Per-project FalkorDBLite vector databases for code and documentation.

Each project gets its own FalkorDBLite instance at {project_path}/.solo/vectors/graph.db.
Stores chunks as graph nodes with vector embeddings — enables hybrid graph+vector queries.
Uses semantic-text-splitter (Rust core) with tree-sitter for AST-aware chunking.

Vectors live on graph nodes, enabling hybrid queries:
  e.g. "find similar code -> show its imports -> find other files using same packages"
"""

# Registry path from env or ~/.solo/
import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from redislite.falkordb_client import FalkorDB

from .common import (
    CHUNK_CAPACITY,
    EMBEDDING_DIM,
    MIN_CHUNK_CHARS,
    TS_GRAMMAR_MAP,
    VECTORS_ROOT,
    get_code_splitter,
    get_markdown_splitter,
    get_text_splitter,
    init_embedding_function,
    scan_project_files,
)

_REGISTRY_ENV = os.environ.get("CODEGRAPH_REGISTRY", "")
_REGISTRY_PATH = Path(_REGISTRY_ENV).expanduser() if _REGISTRY_ENV else Path.home() / ".solo" / "registry.yaml"


class VectorIndexBusy(RuntimeError):
    """Another process is already indexing this project."""


@contextmanager
def _exclusive(db_dir: Path, project_name: str):
    """Hold an exclusive lock on one project's vector store for the duration of a write.

    index_project opens with `MATCH (n) DETACH DELETE n`, so two runs on the same
    project do not merely race — the later one wipes what the earlier had written and
    both then append into the hole. Observed 22 Aug 2026 on epiphan/sgr-chat-agent: a
    full sweep and a single-project reindex overlapped and left 45,454 chunks in a store
    that reported indexing 26,654, including 1,883 chunks from a deleted agent worktree
    the current scanner excludes. Search returned them at 84% relevance, which is the
    failure this whole subsystem exists to prevent: a stale index answers confidently.

    flock, so the lock dies with the process — a killed indexer must not leave a store
    that nothing can ever write to again. Non-blocking: the second run is told to wait
    rather than queued, because these runs take minutes and a silent queue looks like a
    hang.
    """
    db_dir.mkdir(parents=True, exist_ok=True)
    lock_path = db_dir / ".index.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = ""
            try:
                holder = os.read(fd, 32).decode().strip()
            except Exception:
                pass
            raise VectorIndexBusy(
                f"{project_name} is being indexed by another process"
                + (f" (pid {holder})" if holder else "")
                + f" — lock: {lock_path}"
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class ProjectGraphIndex:
    """Per-project FalkorDBLite vector index for source code and documentation."""

    def __init__(self, backend: str | None = None):
        self._ef = init_embedding_function(backend)
        self._dbs: dict[str, FalkorDB] = {}
        self._paths: dict[str, Path] = {}  # name → project_path
        self._md_splitter = None
        self._text_splitter = None
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
        if "\x00" in content or sum(1 for c in content[:2000] if ord(c) < 32 and c not in "\n\r\t") > 20:
            return []

        chunk_type = "doc" if lang in ("markdown", "text") else "code"

        if lang == "text":
            if self._text_splitter is None:
                self._text_splitter = get_text_splitter()
            raw_chunks = self._text_splitter.chunks(content)
        elif lang == "markdown":
            if self._md_splitter is None:
                self._md_splitter = get_markdown_splitter()
            raw_chunks = self._md_splitter.chunks(content)
        elif lang in TS_GRAMMAR_MAP:
            splitter = get_code_splitter(lang)
            if splitter:
                try:
                    raw_chunks = splitter.chunks(content)
                except Exception:
                    raw_chunks = [content[: CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]
            else:
                raw_chunks = [content[: CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]
        else:
            raw_chunks = [content[: CHUNK_CAPACITY[1]]] if len(content) > CHUNK_CAPACITY[1] else [content]

        # Drop fragments. CHUNK_CAPACITY's lower bound is a target the splitter aims
        # for, not a floor it honours: an AST boundary can end a chunk anywhere, and
        # what comes out includes `;`, `() =>` and `describe`. Measured 22 Aug 2026 on
        # a 43,156-chunk store, 4,472 of them — 10.4% — were under 20 characters, and
        # a two-character chunk still gets a 384-dimension embedding that can outrank
        # real code on a short query.
        #
        # Only when the file produced more than one chunk: a genuinely tiny file is
        # still a file, and dropping it would make it unsearchable rather than tidy.
        if len(raw_chunks) > 1:
            raw_chunks = [c for c in raw_chunks if len(c.strip()) >= MIN_CHUNK_CHARS]

        chunks = []
        total = len(raw_chunks)
        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            chunks.append(
                {
                    "id": f"{rel_path}::chunk_{i}",
                    "document": chunk_text,
                    "metadata": {
                        "file": rel_path,
                        "language": lang,
                        "chunk_type": chunk_type,
                        "chunk_index": i,
                        "total_chunks": total,
                    },
                }
            )
        return chunks

    def index_project(self, project_path: Path, project_name: str) -> dict:
        """Index all code and doc files in a project.

        Creates File and Chunk nodes with embeddings, linked by HAS_CHUNK edges.
        Returns stats: {chunks, files, code_chunks, doc_chunks}.

        Raises VectorIndexBusy if another process holds this project's lock.
        """
        # Register path so _db_dir resolves to {project_path}/.solo/
        self._paths[project_name] = project_path

        with _exclusive(self._db_dir(project_name), project_name):
            return self._index_locked(project_path, project_name)

    def _index_locked(self, project_path: Path, project_name: str) -> dict:
        """The write itself. Always called with the project's lock held."""
        import gc

        graph = self._get_graph(project_name)
        file_count = 0
        total_chunks = 0
        code_chunks = 0
        doc_chunks = 0

        # Clear old data
        try:
            graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

        files = scan_project_files(project_path)

        # Process file by file, embed + insert in batches.
        #
        # Two round-trips per batch, not one per file plus one per 16 chunks. Measured
        # 22 Aug 2026 on epiphan/epiphan-students, 45,465 chunks, the two shapes back to
        # back in one process: 63.5 chunks/sec at batch 16 with a MERGE per file, 121.7
        # at batch 128 with the merge folded into the flush.
        batch: list[dict] = []
        batch_size = 128

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
                    self._flush_batch(graph, batch)
                    total_chunks += len(batch)
                    batch.clear()

        # Flush remaining
        if batch:
            self._flush_batch(graph, batch)
            total_chunks += len(batch)

        # One collection at the end instead of one per batch. The per-batch call cost a
        # measured 25% of indexing time (13.5 -> 17.9 chunks/sec with it removed) to
        # reclaim memory that the next batch would have reclaimed anyway.
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
            items.append(
                {
                    "cid": chunk["id"],
                    "text": chunk["document"],
                    "ct": meta["chunk_type"],
                    "ci": meta["chunk_index"],
                    "tc": meta["total_chunks"],
                    "lang": meta["language"],
                    "fp": meta["file"],
                    "emb": emb,
                }
            )

        # The File nodes this batch references, merged in one query. Chunks below MATCH
        # on them, so they have to exist first — and a batch usually spans a handful of
        # files, which is why this is cheap where a per-file MERGE was not.
        seen: dict[str, str] = {}
        for item in items:
            seen.setdefault(item["fp"], item["lang"])
        graph.query(
            "UNWIND $files AS f MERGE (n:File {path: f.path}) SET n.language = f.lang",
            params={"files": [{"path": k, "lang": v} for k, v in seen.items()]},
        )

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
        self,
        query_emb: list[float],
        project_name: str,
        n_results: int,
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
            output.append(
                {
                    "id": doc_id,
                    "file": file_path or "",
                    "language": lang or "",
                    "chunk_type": ct or "",
                    "chunk_index": ci or 0,
                    "relevance": round(1 - score, 4),  # cosine distance → similarity
                    "snippet": (text or "")[:500],
                    "project": project_name,
                }
            )

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
            output.append(
                {
                    "id": doc_id,
                    "file": file_path or "",
                    "language": lang or "",
                    "chunk_type": ct or "",
                    "relevance": round(1 - score, 4),
                    "snippet": (text or "")[:500],
                    "project": project,
                    "sibling_chunks": siblings or [],
                }
            )

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

            projects.append(
                {
                    "name": name,
                    "chunks": chunks,
                    "size_mb": size_mb,
                }
            )

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
