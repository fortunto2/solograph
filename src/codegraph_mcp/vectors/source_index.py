"""FalkorDB vector index for external sources (Telegram, YouTube, etc.).

Each source gets its own FalkorDB graph at ~/.solo/sources/{name}/graph.db.
Keeps external content out of the main KB graph.
"""

import os
from pathlib import Path

from redislite.falkordb_client import FalkorDB

from ..models import SourceDoc
from .common import EMBEDDING_DIM, init_embedding_function

_DEFAULT_ROOT = str(Path.home() / ".solo" / "sources")


class SourceIndex:
    """Per-source FalkorDB vector indexes for external content."""

    def __init__(self, backend: str | None = None, sources_root: str | None = None):
        self._ef = init_embedding_function(backend)
        self._root = Path(
            sources_root or os.environ.get("SOURCES_ROOT", _DEFAULT_ROOT)
        ).expanduser()
        self._dbs: dict[str, tuple[FalkorDB, object]] = {}

    def _get_graph(self, source_name: str):
        """Lazy-open a per-source FalkorDB graph."""
        if source_name not in self._dbs:
            path = self._root / source_name
            path.mkdir(parents=True, exist_ok=True)
            fdb = FalkorDB(str(path / "graph.db"))
            graph = fdb.select_graph("source")
            self._init_schema(graph)
            self._dbs[source_name] = (fdb, graph)
        return self._dbs[source_name][1]

    def _init_schema(self, graph):
        indexes = [
            f"CREATE VECTOR INDEX FOR (d:SourceDoc) ON (d.embedding) "
            f"OPTIONS {{dimension: {EMBEDDING_DIM}, similarityFunction: 'cosine'}}",
            "CREATE INDEX FOR (d:SourceDoc) ON (d.doc_id)",
            "CREATE INDEX FOR (d:SourceDoc) ON (d.source_name)",
        ]
        for idx in indexes:
            try:
                graph.query(idx)
            except Exception:
                pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings. Public — scripts use this for centroid computation."""
        if not texts:
            return []
        raw = self._ef(texts)
        return [[float(x) for x in emb] for emb in raw]

    def exists(self, source_name: str, doc_id: str) -> bool:
        """Check if a document already exists in a source graph."""
        graph = self._get_graph(source_name)
        result = graph.query(
            "MATCH (d:SourceDoc {doc_id: $did}) RETURN d.doc_id",
            {"did": doc_id},
        )
        return bool(result.result_set)

    def upsert_one(
        self,
        source_name: str,
        doc: SourceDoc,
        embedding: list[float] | None = None,
    ) -> bool:
        """Upsert a single document. Returns True if new (inserted)."""
        graph = self._get_graph(source_name)

        if embedding is None:
            text = doc.embed_text or doc.content or doc.title
            embedding = self.embed([text[:3000]])[0]

        was_new = not self.exists(source_name, doc.doc_id)

        graph.query(
            "MERGE (d:SourceDoc {doc_id: $did}) "
            "SET d.source_type = $stype, d.source_name = $sname, "
            "d.title = $title, d.url = $url, "
            "d.tags = $tags, d.created = $created, "
            "d.content = $content, d.embedding = vecf32($emb)",
            {
                "did": doc.doc_id,
                "stype": doc.source_type,
                "sname": doc.source_name,
                "title": doc.title,
                "url": doc.url,
                "tags": doc.tags,
                "created": doc.created,
                "content": doc.content[:500],
                "emb": embedding,
            },
        )
        return was_new

    def search(
        self,
        query: str,
        source: str | None = None,
        n_results: int = 5,
    ) -> list[dict]:
        """Semantic search. If source given, search that graph only.
        Otherwise, search all discovered sources and merge by score.
        """
        query_emb = self.embed([query])[0]

        if source:
            return self._search_one(source, query_emb, n_results)

        # Cross-source: search all, merge by relevance
        sources = self._discover_sources()
        if not sources:
            return []

        all_results = []
        for src in sources:
            all_results.extend(self._search_one(src, query_emb, n_results))

        all_results.sort(key=lambda r: r.get("relevance", 0), reverse=True)
        return all_results[:n_results]

    def _search_one(
        self, source_name: str, query_emb: list[float], n_results: int
    ) -> list[dict]:
        """Search a single source graph."""
        graph = self._get_graph(source_name)

        count_result = graph.query("MATCH (d:SourceDoc) RETURN count(d)")
        count = count_result.result_set[0][0] if count_result.result_set else 0
        if count == 0:
            return []

        fetch_n = min(n_results, count)

        cypher = (
            f"CALL db.idx.vector.queryNodes('SourceDoc', 'embedding', {fetch_n}, vecf32($q)) "
            "YIELD node, score "
            "RETURN node.doc_id, node.source_type, node.source_name, "
            "node.title, node.url, node.content, node.created, node.tags, score "
            f"LIMIT {n_results}"
        )

        try:
            result = graph.query(cypher, params={"q": query_emb})
        except Exception:
            return []

        output = []
        for row in result.result_set:
            doc_id, stype, sname, title, url, content, created, tags, score = row
            output.append({
                "doc_id": doc_id or "",
                "source_type": stype or "",
                "source_name": sname or "",
                "title": (title or "")[:100],
                "url": url or "",
                "content": (content or "")[:300],
                "created": created or "",
                "tags": tags or "",
                "relevance": round(1 - score, 4),
            })
        return output

    def _discover_sources(self) -> list[str]:
        """Scan ~/.solo/sources/ for subdirectories with graph.db."""
        if not self._root.exists():
            return []
        sources = []
        for child in sorted(self._root.iterdir()):
            if child.is_dir() and (child / "graph.db").exists():
                sources.append(child.name)
        return sources

    def list_sources(self) -> list[dict]:
        """List all indexed sources with document counts."""
        sources = self._discover_sources()
        result = []
        for src in sources:
            result.append({
                "source": src,
                "count": self.count(src),
                "path": str(self._root / src / "graph.db"),
            })
        return result

    def count(self, source_name: str | None = None) -> int:
        """Count documents in a source (or all sources)."""
        if source_name:
            graph = self._get_graph(source_name)
            result = graph.query("MATCH (d:SourceDoc) RETURN count(d)")
            return result.result_set[0][0] if result.result_set else 0

        return sum(self.count(src) for src in self._discover_sources())

    def delete_source(self, source_name: str) -> bool:
        """Delete a source graph entirely."""
        path = self._root / source_name
        if not path.exists():
            return False

        # Close connection if open
        if source_name in self._dbs:
            del self._dbs[source_name]

        import shutil
        shutil.rmtree(path)
        return True
