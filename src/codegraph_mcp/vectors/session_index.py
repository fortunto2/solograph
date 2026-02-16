"""FalkorDB vector index for session summaries.

Stores SessionDoc nodes with text embeddings in a dedicated FalkorDB graph.
"""

import os
from pathlib import Path

from redislite.falkordb_client import FalkorDB

from ..models import SessionSummary
from .common import EMBEDDING_DIM, init_embedding_function

_DB_PATH = os.environ.get("CODEGRAPH_SESSIONS_DB", str(Path.home() / ".solo" / "sessions"))


class SessionIndex:
    """FalkorDB vector index for session summaries."""

    def __init__(self, backend: str | None = None, db_path: str | None = None):
        self._ef = init_embedding_function(backend)
        path = Path(db_path or _DB_PATH).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self._fdb = FalkorDB(str(path / "graph.db"))
        self._graph = self._fdb.select_graph("sessions")
        self._init_schema()

    def _init_schema(self):
        indexes = [
            f"CREATE VECTOR INDEX FOR (s:SessionDoc) ON (s.embedding) "
            f"OPTIONS {{dimension: {EMBEDDING_DIM}, similarityFunction: 'cosine'}}",
            "CREATE INDEX FOR (s:SessionDoc) ON (s.session_id)",
            "CREATE INDEX FOR (s:SessionDoc) ON (s.project_name)",
        ]
        for idx in indexes:
            try:
                self._graph.query(idx)
            except Exception:
                pass

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        raw = self._ef(texts)
        return [[float(x) for x in emb] for emb in raw]

    def upsert(self, summaries: list[SessionSummary]) -> int:
        """Upsert session summaries. Idempotent by session_id."""
        if not summaries:
            return 0

        # Batch embed
        texts = [s.summary_text for s in summaries]
        embeddings = self._embed(texts)

        for s, emb in zip(summaries, embeddings):
            self._graph.query(
                "MERGE (d:SessionDoc {session_id: $sid}) "
                "SET d.project_name = $pname, d.started_at = $start, "
                "d.tags = $tags, d.summary = $summary, "
                "d.embedding = vecf32($emb)",
                {
                    "sid": s.session_id,
                    "pname": s.project_name,
                    "start": s.started_at,
                    "tags": ",".join(s.tags),
                    "summary": s.summary_text,
                    "emb": emb,
                },
            )

        return len(summaries)

    def search(
        self,
        query: str,
        n_results: int = 5,
        project: str | None = None,
    ) -> list[dict]:
        """Semantic search over session summaries."""
        query_emb = self._embed([query])[0]

        count_result = self._graph.query("MATCH (s:SessionDoc) RETURN count(s)")
        count = count_result.result_set[0][0] if count_result.result_set else 0
        if count == 0:
            return []

        fetch_n = min(n_results * 2 if project else n_results, count)

        cypher = f"CALL db.idx.vector.queryNodes('SessionDoc', 'embedding', {fetch_n}, vecf32($q)) YIELD node, score "
        if project:
            cypher += "WHERE node.project_name = $proj "
        cypher += f"RETURN node.session_id, node.project_name, node.started_at, node.summary, score LIMIT {n_results}"

        params: dict = {"q": query_emb}
        if project:
            params["proj"] = project

        try:
            result = self._graph.query(cypher, params=params)
        except Exception:
            return []

        output = []
        for row in result.result_set:
            sid, pname, start, summary, score = row
            output.append(
                {
                    "session_id": sid or "",
                    "project_name": pname or "",
                    "started_at": start or "",
                    "relevance": round(1 - score, 4),
                    "summary": (summary or "")[:300],
                }
            )

        return output

    def count(self) -> int:
        result = self._graph.query("MATCH (s:SessionDoc) RETURN count(s)")
        return result.result_set[0][0] if result.result_set else 0
