"""Copy the FalkorDB source graphs into the full-text sources index. One-off.

Reads ~/.solo/sources/<source>/graph.db (graph "source": SourceDoc, Video,
VideoChunk, Channel, Maker) and writes ~/.solo/sources/search.db, so nothing
is re-scraped and no transcript is re-downloaded. Video tags are recomputed by
the keyword tagger rather than copied: the embedding tagger had put all fifteen
topics on 740 of 742 videos. The graph.db files are left where they are.

    uv run python scripts/migrate-sources-to-fts.py [SOURCES_ROOT]
"""

import os
import sys
import time
from pathlib import Path

from redislite.falkordb_client import FalkorDB

from codegraph_mcp.models import MakerProfile, SourceDoc
from codegraph_mcp.sources import SourceIndex

root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SOURCES_ROOT", "~/.solo/sources")).expanduser()
idx = SourceIndex(sources_root=str(root))


def rows(graph, cypher, params=None):
    return graph.query(cypher, params or {}).result_set


for src_dir in sorted(p for p in root.iterdir() if (p / "graph.db").exists()):
    source = src_dir.name
    t = time.time()
    graph = FalkorDB(str(src_dir / "graph.db")).select_graph("source")
    before = dict(rows(graph, "MATCH (n) RETURN labels(n)[0], count(n)"))

    for r in rows(
        graph,
        "MATCH (d:SourceDoc) RETURN d.doc_id, d.source_type, d.source_name, d.title, d.url, "
        "d.tags, d.created, d.content, d.popularity",
    ):
        doc_id, stype, sname, title, url, tags, created, content, pop = r
        idx.upsert_one(
            source,
            SourceDoc(
                doc_id=doc_id,
                source_type=stype or "",
                source_name=sname or source,
                title=title or "",
                url=url or "",
                tags=tags or "",
                created=str(created or ""),
                content=content or "",
                popularity=int(pop or 0),
            ),
        )

    for r in rows(
        graph,
        "MATCH (m:Maker) RETURN m.username, m.name, m.headline, m.bio, m.twitter, m.linkedin, m.website, "
        "m.points, m.streak_days, m.followers, m.is_maker",
    ):
        username, name, headline, bio, tw, li, web, points, streak, followers, is_maker = r
        idx.upsert_maker(
            source,
            MakerProfile(
                username=username,
                name=name or "",
                headline=headline or "",
                bio=bio or "",
                twitter_username=tw or "",
                linkedin_url=li or "",
                website_url=web or "",
                points=int(points or 0),
                streak_days=int(streak or 0),
                followers_count=int(followers or 0),
                is_maker=bool(is_maker),
            ),
        )

    videos = rows(
        graph,
        "MATCH (v:Video) OPTIONAL MATCH (ch:Channel)-[:HAS_VIDEO]->(v) "
        "RETURN v.video_id, v.title, v.url, v.source_name, ch.name, v.created, v.tags, v.description, "
        "v.duration_seconds, v.chapters",
    )
    for r in videos:
        vid, title, url, handle, ch_name, created, tags, desc, dur, chapters = r
        chunks = [
            {
                "chunk_index": ci,
                "text": text or "",
                "chapter": chapter or "",
                "start_time": start or "",
                "start_seconds": secs or 0.0,
                "chunk_type": ctype or "transcript",
            }
            for ci, text, chapter, start, secs, ctype in rows(
                graph,
                "MATCH (v:Video {video_id: $v})-[:HAS_CHUNK]->(c:VideoChunk) "
                "RETURN c.chunk_index, c.text, c.chapter, c.start_time, c.start_seconds, c.chunk_type",
                {"v": vid},
            )
        ]
        idx.write_video(
            source,
            {
                "video_id": vid,
                "title": title or "",
                "url": url or "",
                "channel": handle or "",
                "channel_name": ch_name or handle or "",
                "created": str(created or ""),
                "tags": tags or "",
                "description": desc or "",
                "duration_seconds": int(dur or 0),
                "chapters": chapters or "",
            },
            idx.auto_tag_video(title or "", desc or ""),
            chunks,
        )

    after = {
        "docs+videos+makers": idx.count(source),
        **idx.video_count(source),
        "makers": idx.maker_count(source),
    }
    print(f"{source}: graph {before} -> fts {after} in {time.time() - t:.1f}s")
