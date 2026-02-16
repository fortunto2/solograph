"""FalkorDB vector index for external sources (Telegram, YouTube, etc.).

Each source gets its own FalkorDB graph at ~/.solo/sources/{name}/graph.db.
Keeps external content out of the main KB graph.

Graph structure:
  - SourceDoc nodes: Telegram posts and flat content (existing)
  - Video → HAS_CHUNK → VideoChunk: YouTube videos with chunked transcripts
  Both coexist in the same graph and are searched together.
"""

import os
from pathlib import Path

from redislite.falkordb_client import FalkorDB

from ..models import SourceDoc, VideoDoc
from .common import (
    DEFAULT_TOPICS,
    EMBEDDING_DIM,
    cosine_similarity,
    init_embedding_function,
)

_DEFAULT_ROOT = str(Path.home() / ".solo" / "sources")


class SourceIndex:
    """Per-source FalkorDB vector indexes for external content."""

    def __init__(self, backend: str | None = None, sources_root: str | None = None):
        self._ef = init_embedding_function(backend)
        self._root = Path(sources_root or os.environ.get("SOURCES_ROOT", _DEFAULT_ROOT)).expanduser()
        self._dbs: dict[str, tuple[FalkorDB, object]] = {}
        self._topic_embeddings: list[tuple[str, list[float]]] | None = None

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
            # SourceDoc indexes (Telegram, flat content)
            f"CREATE VECTOR INDEX FOR (d:SourceDoc) ON (d.embedding) "
            f"OPTIONS {{dimension: {EMBEDDING_DIM}, similarityFunction: 'cosine'}}",
            "CREATE INDEX FOR (d:SourceDoc) ON (d.doc_id)",
            "CREATE INDEX FOR (d:SourceDoc) ON (d.source_name)",
            # Video indexes (YouTube chunked)
            "CREATE INDEX FOR (v:Video) ON (v.video_id)",
            "CREATE INDEX FOR (v:Video) ON (v.source_name)",
            # VideoChunk vector index
            f"CREATE VECTOR INDEX FOR (c:VideoChunk) ON (c.embedding) "
            f"OPTIONS {{dimension: {EMBEDDING_DIM}, similarityFunction: 'cosine'}}",
            "CREATE INDEX FOR (c:VideoChunk) ON (c.chunk_id)",
            # Channel indexes
            "CREATE INDEX FOR (ch:Channel) ON (ch.handle)",
            # Tag indexes
            "CREATE INDEX FOR (t:Tag) ON (t.name)",
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
        """Check if a SourceDoc already exists in a source graph."""
        graph = self._get_graph(source_name)
        result = graph.query(
            "MATCH (d:SourceDoc {doc_id: $did}) RETURN d.doc_id",
            {"did": doc_id},
        )
        return bool(result.result_set)

    def video_exists(self, source_name: str, video_id: str) -> bool:
        """Check if a Video node already exists."""
        graph = self._get_graph(source_name)
        result = graph.query(
            "MATCH (v:Video {video_id: $vid}) RETURN v.video_id",
            {"vid": video_id},
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

    def auto_tag_video(self, title: str, description: str, threshold: float = 0.3) -> list[tuple[str, float]]:
        """Zero-shot topic tagging via embedding similarity.

        Embeds title+description, compares with pre-embedded topic list.
        Returns list of (topic_name, confidence_score) with similarity > threshold.
        Confidence score is cosine similarity (0-1), stored as edge weight.
        """
        if self._topic_embeddings is None:
            embs = self._ef(DEFAULT_TOPICS)
            self._topic_embeddings = list(zip(DEFAULT_TOPICS, embs))

        text = f"{title}\n{description[:500]}"
        text_emb = self._ef([text])[0]

        tags = []
        for topic, topic_emb in self._topic_embeddings:
            sim = cosine_similarity(text_emb, topic_emb)
            if sim >= threshold:
                tags.append((topic, round(sim, 4)))
        return sorted(tags, key=lambda x: x[1], reverse=True)

    def upsert_video(
        self,
        source_name: str,
        video: VideoDoc,
        chunks: list[dict],
        channel_name: str | None = None,
    ) -> int:
        """Insert/update a Video node with chunked VideoChunk nodes.

        Args:
            source_name: Source graph name (e.g. "youtube")
            video: VideoDoc with metadata
            chunks: list of {text, chapter, start_time, chunk_index, chunk_type?}
            channel_name: Human-readable channel name (for Channel node)

        Returns: number of chunks inserted
        """
        graph = self._get_graph(source_name)

        # Auto-tag via zero-shot embedding similarity
        auto_tags = self.auto_tag_video(video.title, video.description)

        # Merge Video node
        graph.query(
            "MERGE (v:Video {video_id: $vid}) "
            "SET v.title = $title, v.url = $url, v.source_name = $sname, "
            "v.description = $desc, v.created = $created, v.tags = $tags, "
            "v.duration_seconds = $dur, "
            "v.chapters = $chapters",
            {
                "vid": video.video_id,
                "title": video.title,
                "url": video.url,
                "sname": video.source_name,
                "desc": video.description[:5000],
                "created": video.created,
                "tags": video.tags,
                "dur": video.duration_seconds,
                "chapters": ", ".join(f"{ch.start_time} {ch.title}" for ch in video.chapters) if video.chapters else "",
            },
        )

        # Channel node + edge
        handle = video.source_name
        ch_name = channel_name or handle
        graph.query(
            "MERGE (ch:Channel {handle: $handle}) "
            "SET ch.name = $ch_name "
            "WITH ch "
            "MATCH (v:Video {video_id: $vid}) "
            "MERGE (ch)-[:HAS_VIDEO]->(v)",
            {"handle": handle, "ch_name": ch_name, "vid": video.video_id},
        )

        # Tag nodes with confidence weights on HAS_TAG edges
        if auto_tags:
            tag_items = [{"name": name, "weight": weight} for name, weight in auto_tags]
            graph.query(
                "UNWIND $tags AS item "
                "MERGE (t:Tag {name: item.name}) "
                "WITH t, item "
                "MATCH (v:Video {video_id: $vid}) "
                "MERGE (v)-[r:HAS_TAG]->(t) "
                "SET r.weight = item.weight",
                {"tags": tag_items, "vid": video.video_id},
            )

        # Delete old chunks for this video
        graph.query(
            "MATCH (v:Video {video_id: $vid})-[:HAS_CHUNK]->(c:VideoChunk) DETACH DELETE c",
            {"vid": video.video_id},
        )

        if not chunks:
            return 0

        # Embed all chunk texts in batch
        texts = [c["text"] for c in chunks]
        embeddings = self.embed(texts)

        # Build items for UNWIND
        items = []
        for chunk, emb in zip(chunks, embeddings):
            items.append(
                {
                    "cid": f"{video.video_id}:{chunk['chunk_index']}",
                    "text": chunk["text"],
                    "ci": chunk["chunk_index"],
                    "chapter": chunk.get("chapter", ""),
                    "start_time": chunk.get("start_time", ""),
                    "start_seconds": chunk.get("start_seconds", 0.0),
                    "chunk_type": chunk.get("chunk_type", "transcript"),
                    "emb": emb,
                }
            )

        # Batch insert with UNWIND
        graph.query(
            "UNWIND $items AS item "
            "CREATE (c:VideoChunk {"
            "  chunk_id: item.cid, text: item.text,"
            "  chunk_index: item.ci, chapter: item.chapter,"
            "  start_time: item.start_time, start_seconds: item.start_seconds,"
            "  chunk_type: item.chunk_type,"
            "  embedding: vecf32(item.emb)"
            "}) "
            "WITH c, item "
            "MATCH (v:Video {video_id: $vid}) "
            "CREATE (v)-[:HAS_CHUNK]->(c)",
            params={"items": items, "vid": video.video_id},
        )

        return len(items)

    def video_count(self, source_name: str) -> dict:
        """Count Video nodes and VideoChunk nodes in a source graph."""
        graph = self._get_graph(source_name)
        try:
            vr = graph.query("MATCH (v:Video) RETURN count(v)")
            videos = vr.result_set[0][0] if vr.result_set else 0
        except Exception:
            videos = 0
        try:
            cr = graph.query("MATCH (c:VideoChunk) RETURN count(c)")
            chunks = cr.result_set[0][0] if cr.result_set else 0
        except Exception:
            chunks = 0
        return {"videos": videos, "chunks": chunks}

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

    def _search_one(self, source_name: str, query_emb: list[float], n_results: int) -> list[dict]:
        """Search a single source graph — SourceDoc + VideoChunk, merged by score."""
        graph = self._get_graph(source_name)
        output = []

        # 1. Search SourceDoc nodes (Telegram, flat YouTube imports)
        sd_count_result = graph.query("MATCH (d:SourceDoc) RETURN count(d)")
        sd_count = sd_count_result.result_set[0][0] if sd_count_result.result_set else 0

        if sd_count > 0:
            fetch_n = min(n_results, sd_count)
            cypher = (
                f"CALL db.idx.vector.queryNodes('SourceDoc', 'embedding', {fetch_n}, vecf32($q)) "
                "YIELD node, score "
                "RETURN node.doc_id, node.source_type, node.source_name, "
                "node.title, node.url, node.content, node.created, node.tags, score "
                f"LIMIT {n_results}"
            )
            try:
                result = graph.query(cypher, params={"q": query_emb})
                for row in result.result_set:
                    doc_id, stype, sname, title, url, content, created, tags, score = row
                    output.append(
                        {
                            "doc_id": doc_id or "",
                            "source_type": stype or "",
                            "source_name": sname or "",
                            "title": (title or "")[:100],
                            "url": url or "",
                            "content": (content or "")[:300],
                            "created": created or "",
                            "tags": tags or "",
                            "relevance": round(1 - score, 4),
                        }
                    )
            except Exception:
                pass

        # 2. Search VideoChunk nodes (chunked YouTube) with ±1 sibling expansion
        vc_count_result = graph.query("MATCH (c:VideoChunk) RETURN count(c)")
        vc_count = vc_count_result.result_set[0][0] if vc_count_result.result_set else 0

        if vc_count > 0:
            fetch_n = min(n_results, vc_count)
            cypher = (
                f"CALL db.idx.vector.queryNodes('VideoChunk', 'embedding', {fetch_n}, vecf32($q)) "
                "YIELD node, score "
                "MATCH (v:Video)-[:HAS_CHUNK]->(node) "
                "OPTIONAL MATCH (v)-[:HAS_CHUNK]->(prev:VideoChunk) "
                "  WHERE prev.chunk_index = node.chunk_index - 1 "
                "OPTIONAL MATCH (v)-[:HAS_CHUNK]->(next:VideoChunk) "
                "  WHERE next.chunk_index = node.chunk_index + 1 "
                "RETURN node.chunk_id, v.title, v.url, v.source_name, "
                "node.text, node.chapter, node.start_time, node.start_seconds, node.chunk_type, "
                "prev.text, next.text, "
                "v.created, v.tags, score "
                f"LIMIT {n_results}"
            )
            try:
                result = graph.query(cypher, params={"q": query_emb})
                for row in result.result_set:
                    (
                        chunk_id,
                        title,
                        url,
                        sname,
                        text,
                        chapter,
                        start_time,
                        start_seconds,
                        chunk_type,
                        prev_text,
                        next_text,
                        created,
                        tags,
                        score,
                    ) = row
                    # Build context from siblings
                    parts = [p for p in [prev_text, text, next_text] if p]
                    context = "\n".join(parts) if len(parts) > 1 else ""
                    entry = {
                        "doc_id": chunk_id or "",
                        "source_type": f"youtube-{chunk_type or 'transcript'}",
                        "source_name": sname or "",
                        "title": (title or "")[:100],
                        "url": url or "",
                        "content": (text or "")[:300],
                        "created": created or "",
                        "tags": tags or "",
                        "chapter": chapter or "",
                        "start_time": start_time or "",
                        "start_seconds": start_seconds or 0.0,
                        "relevance": round(1 - score, 4),
                    }
                    if context:
                        entry["context"] = context[:1000]
                    output.append(entry)
            except Exception:
                pass

        # Merge by relevance
        output.sort(key=lambda r: r.get("relevance", 0), reverse=True)
        return output[:n_results]

    def search_by_tag(self, source_name: str, tag: str, n_results: int = 10) -> list[dict]:
        """Find all videos with a specific tag."""
        graph = self._get_graph(source_name)
        result = graph.query(
            "MATCH (t:Tag {name: $tag})<-[:HAS_TAG]-(v:Video) "
            "RETURN v.video_id, v.title, v.url, v.source_name, v.created, v.tags "
            "LIMIT $limit",
            {"tag": tag, "limit": n_results},
        )
        return [
            {
                "video_id": row[0] or "",
                "title": row[1] or "",
                "url": row[2] or "",
                "source_name": row[3] or "",
                "created": row[4] or "",
                "tags": row[5] or "",
            }
            for row in result.result_set
        ]

    def list_tags(self, source_name: str) -> list[dict]:
        """List all tags with video counts and average confidence."""
        graph = self._get_graph(source_name)
        result = graph.query(
            "MATCH (t:Tag)<-[r:HAS_TAG]-(v:Video) "
            "RETURN t.name, count(v) AS cnt, avg(r.weight) AS avg_weight "
            "ORDER BY cnt DESC"
        )
        return [
            {
                "name": row[0],
                "count": row[1],
                "avg_confidence": round(row[2], 3) if row[2] else None,
            }
            for row in result.result_set
        ]

    def related_videos(self, source_name: str, video_id: str) -> list[dict]:
        """Find videos sharing tags with the given video.

        Uses sum of tag weights as relevance score — videos sharing
        high-confidence tags rank higher than those sharing weak tags.
        """
        graph = self._get_graph(source_name)
        result = graph.query(
            "MATCH (v1:Video {video_id: $vid})-[r1:HAS_TAG]->(t:Tag)<-[r2:HAS_TAG]-(v2:Video) "
            "WHERE v1 <> v2 "
            "RETURN v2.video_id, v2.title, v2.url, "
            "collect(t.name) AS shared_tags, count(t) AS overlap, "
            "sum(r1.weight + r2.weight) / 2 AS relevance "
            "ORDER BY relevance DESC",
            {"vid": video_id},
        )
        return [
            {
                "video_id": row[0] or "",
                "title": row[1] or "",
                "url": row[2] or "",
                "shared_tags": row[3] or [],
                "overlap": row[4] or 0,
                "relevance": round(row[5], 3) if row[5] else 0,
            }
            for row in result.result_set
        ]

    def channel_videos(self, source_name: str, handle: str) -> list[dict]:
        """List all videos for a channel."""
        graph = self._get_graph(source_name)
        result = graph.query(
            "MATCH (ch:Channel {handle: $handle})-[:HAS_VIDEO]->(v:Video) "
            "RETURN v.video_id, v.title, v.url, v.created, v.tags "
            "ORDER BY v.created DESC",
            {"handle": handle},
        )
        return [
            {
                "video_id": row[0] or "",
                "title": row[1] or "",
                "url": row[2] or "",
                "created": row[3] or "",
                "tags": row[4] or "",
            }
            for row in result.result_set
        ]

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
            entry = {
                "source": src,
                "count": self.count(src),
                "path": str(self._root / src / "graph.db"),
            }
            # Enrich YouTube with video/chunk breakdown
            vc = self.video_count(src)
            if vc["videos"] > 0:
                entry["videos"] = vc["videos"]
                entry["video_chunks"] = vc["chunks"]
            result.append(entry)
        return result

    def count(self, source_name: str | None = None) -> int:
        """Count total searchable items (SourceDoc + Video nodes)."""
        if source_name:
            graph = self._get_graph(source_name)
            sd = graph.query("MATCH (d:SourceDoc) RETURN count(d)")
            sd_count = sd.result_set[0][0] if sd.result_set else 0
            try:
                vr = graph.query("MATCH (v:Video) RETURN count(v)")
                v_count = vr.result_set[0][0] if vr.result_set else 0
            except Exception:
                v_count = 0
            return sd_count + v_count

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
