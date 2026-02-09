"""ChromaDB index for session summaries.

Reuses MLX/sentence-transformers infrastructure from knowledge_embeddings.py.
Collection: codegraph_sessions
"""

import os
import sys
from pathlib import Path

import chromadb

from ..models import SessionSummary

# Embeddings storage next to KB embeddings
_CHROMA_PATH = os.environ.get("CODEGRAPH_SESSIONS_CHROMA", str(Path.home() / ".codegraph" / "sessions_chroma"))


def _init_embedding_function(backend: str | None = None):
    """Initialize embedding function (same logic as knowledge_embeddings.py)."""
    import platform

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


class SessionIndex:
    """ChromaDB index for session summaries."""

    def __init__(self, backend: str | None = None, chroma_path: str | None = None):
        self._ef = _init_embedding_function(backend)
        self._client = chromadb.PersistentClient(path=chroma_path or _CHROMA_PATH)
        self.collection = self._client.get_or_create_collection(
            name="codegraph_sessions",
            embedding_function=self._ef,
            metadata={"description": "Claude Code session summaries"},
        )

    def upsert(self, summaries: list[SessionSummary]) -> int:
        """Upsert session summaries. Idempotent by session_id."""
        if not summaries:
            return 0

        ids = [s.session_id for s in summaries]
        documents = [s.summary_text for s in summaries]
        metadatas = [
            {
                "project_name": s.project_name,
                "started_at": s.started_at,
                "tags": ",".join(s.tags),
            }
            for s in summaries
        ]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def search(
        self,
        query: str,
        n_results: int = 5,
        project: str | None = None,
    ) -> list[dict]:
        """Semantic search over session summaries."""
        where = {"project_name": project} if project else None

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        output = []
        if not results["ids"] or not results["ids"][0]:
            return output

        for i, sid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 1.0
            doc = results["documents"][0][i] if results["documents"] else ""
            output.append(
                {
                    "session_id": sid,
                    "project_name": meta.get("project_name", ""),
                    "started_at": meta.get("started_at", ""),
                    "relevance": round(1 - dist, 4),
                    "summary": doc[:300],
                }
            )

        return output

    def count(self) -> int:
        return self.collection.count()
