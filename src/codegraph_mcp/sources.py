"""External sources — YouTube, ProductHunt, TrustMRR, Telegram — written to SQLite FTS5.

The write side only. The scrapers and indexers stay in Python because yt-dlp,
Scrapy and Playwright live here; everything that reads <sources_root>/search.db
(search, tags, related videos, the HTTP API) is the Rust binary since 0.9
(rust/solograph/src/sources.rs). The tables below are the contract between them.

Documents are keyed "<source>/<id>" so two sources cannot collide. A YouTube
transcript is written chunk by chunk, so a hit carries a timestamped URL.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .fts import FtsStore, _fold
from .models import MakerProfile, SourceDoc, VideoDoc

_DEFAULT_ROOT = str(Path.home() / ".solo" / "sources")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos(
    video_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT, url TEXT, channel TEXT, channel_name TEXT,
    created TEXT, tags TEXT, description TEXT,
    duration_seconds INTEGER, chapters TEXT
);
CREATE INDEX IF NOT EXISTS videos_source ON videos(source, channel);
CREATE TABLE IF NOT EXISTS video_tags(
    video_id TEXT NOT NULL, tag TEXT NOT NULL, weight REAL,
    PRIMARY KEY (video_id, tag)
);
"""

# Topic -> keyword prefixes. Was zero-shot cosine similarity against these same
# fifteen topic names. A keyword of three letters or fewer must match a whole
# word ("ai", "seo"); a longer one matches any word it starts ("automat").
# Phrases (with a space or hyphen) match as substrings.
TOPICS: dict[str, tuple[str, ...]] = {
    "AI agents and automation": (
        "ai",
        "agent",
        "agentic",
        "automat",
        "llm",
        "gpt",
        "chatgpt",
        "claude",
        "openai",
        "copilot",
    ),
    "revenue model and pricing strategy": (
        "pricing",
        "price",
        "revenue",
        "mrr",
        "arr",
        "monetiz",
        "subscription",
        "paywall",
        "freemium",
        "churn",
    ),
    "community building and audience growth": (
        "community",
        "audience",
        "newsletter",
        "discord",
        "engagement",
        "fanbase",
    ),
    "security and privacy": ("security", "secure", "privacy", "encrypt", "vulnerab", "gdpr", "hacker attack"),
    "hardware setup and infrastructure": (
        "hardware",
        "server",
        "infrastructure",
        "homelab",
        "gpu",
        "mac mini",
        "raspberry",
        "kubernetes",
        "vps",
    ),
    "productivity and workflows": ("productiv", "workflow", "habit", "routine", "focus", "notion", "obsidian"),
    "marketing and growth hacking": (
        "marketing",
        "growth",
        "seo",
        "ads",
        "advertis",
        "viral",
        "funnel",
        "acquisition",
        "outreach",
        "cold email",
    ),
    "product development and MVP": ("mvp", "prototype", "product development", "feature", "ship", "build in public"),
    "fundraising and investment": ("fundrais", "investor", "venture", "vc", "angel", "seed round", "valuation"),
    "solo founder and bootstrapping": (
        "solo",
        "bootstrap",
        "indie",
        "indiehacker",
        "solopreneur",
        "one-person",
        "side project",
    ),
    "open source and self-hosting": ("open source", "open-source", "opensource", "self-host", "selfhost", "self host"),
    "e-commerce and conversion optimization": (
        "ecommerce",
        "e-commerce",
        "shopify",
        "conversion",
        "checkout",
        "dropship",
        "amazon fba",
    ),
    "content creation and YouTube": ("youtube", "creator", "podcast", "thumbnail", "tiktok", "content creation"),
    "developer tools and coding": (
        "coding",
        "developer",
        "programming",
        "api",
        "cursor",
        "ide",
        "debug",
        "python",
        "javascript",
        "typescript",
    ),
    "market research and validation": (
        "market research",
        "validat",
        "niche",
        "competitor",
        "customer interview",
        "demand",
    ),
}
DEFAULT_TOPICS = list(TOPICS)


def _topic_hits(text: str, keywords: tuple[str, ...]) -> int:
    folded = _fold(text.lower())
    words = set(re.findall(r"\w+", folded))
    hits = 0
    for kw in keywords:
        if " " in kw or "-" in kw:
            hits += kw in folded
        elif len(kw) <= 3:
            hits += kw in words
        else:
            hits += any(w.startswith(kw) for w in words)
    return hits


def _seconds(chunk: dict) -> float:
    """start_seconds, or parsed from "H:MM:SS"/"M:SS" — the old indexer mostly filled only the string."""
    if chunk.get("start_seconds"):
        return float(chunk["start_seconds"])
    parts = str(chunk.get("start_time") or "").split(":")
    if not all(p.isdigit() for p in parts):
        return 0.0
    return float(sum(int(p) * 60**i for i, p in enumerate(reversed(parts))))


def _timestamped(url: str, seconds) -> str:
    if not url or not seconds:
        return url
    return f"{url}{'&' if '?' in url else '?'}t={int(seconds)}s"


class SourceIndex:
    """Full-text index over external content, one store for every source."""

    def __init__(self, sources_root: str | None = None):
        self._root = Path(sources_root or os.environ.get("SOURCES_ROOT", _DEFAULT_ROOT)).expanduser()
        self._store = FtsStore(self._root / "search.db")
        with self._store.transaction() as db:
            db.executescript(_SCHEMA)

    @staticmethod
    def _key(source: str, doc_id: str) -> str:
        return f"{source}/{doc_id}"

    # ── writes ────────────────────────────────────────────────────

    def exists(self, source_name: str, doc_id: str) -> bool:
        with self._store.transaction() as db:
            row = db.execute("SELECT 1 FROM docs WHERE doc_id = ?", (self._key(source_name, doc_id),)).fetchone()
        return row is not None

    def upsert_one(self, source_name: str, doc: SourceDoc, embedding=None) -> bool:
        """Insert or update one document. Returns True if it was new.

        `embedding` is accepted and ignored, so indexers written for the vector
        store keep working.
        """
        was_new = not self.exists(source_name, doc.doc_id)
        body = "\n".join(p for p in (doc.tags, doc.embed_text or doc.content) if p)
        with self._store.transaction():
            self._store.put(
                self._key(source_name, doc.doc_id),
                {
                    "title": doc.title,
                    "body": body,
                    "kind": source_name,
                    "meta": {
                        "type": "doc",
                        "source_type": doc.source_type,
                        "source_name": doc.source_name,
                        "url": doc.url,
                        "tags": doc.tags,
                        "created": doc.created,
                        "popularity": doc.popularity,
                        "content": doc.content[:300],
                    },
                },
            )
        return was_new

    def upsert_maker(self, source_name: str, maker: MakerProfile, embedding=None) -> bool:
        was_new = not self.exists(source_name, f"maker:{maker.username}")
        with self._store.transaction():
            self._store.put(
                self._key(source_name, f"maker:{maker.username}"),
                {
                    "title": f"{maker.name or maker.username} (@{maker.username})",
                    "body": "\n".join(p for p in (maker.headline, maker.bio) if p),
                    "kind": source_name,
                    "meta": {
                        "type": "maker",
                        "username": maker.username,
                        "headline": maker.headline,
                        "points": maker.points,
                        "streak_days": maker.streak_days,
                        "followers": maker.followers_count,
                        "twitter": maker.twitter_username,
                        "linkedin": maker.linkedin_url,
                        "website": maker.website_url,
                        "is_maker": maker.is_maker,
                    },
                },
            )
        return was_new

    def auto_tag_video(self, title: str, description: str, threshold: float = 0.25) -> list[tuple[str, float]]:
        """Topic tags from keywords: a title hit counts 2, a description hit 1.

        weight = score / (score + 3); the default threshold takes any one hit.
        Only the first 500 characters of the description count — the rest is
        links, sponsors and "subscribe". The embedding tagger this replaces put
        all fifteen topics on 740 of 742 videos (e5 cosine is above 0.3 for
        almost anything), so every video was "related" to every other. On the
        same videos this gives 2.0 tags per video, 74 with none.
        """
        tags = []
        for topic, keywords in TOPICS.items():
            score = 2 * _topic_hits(title, keywords) + _topic_hits(description[:500], keywords)
            weight = score / (score + 3) if score else 0.0
            if weight >= threshold:
                tags.append((topic, round(weight, 3)))
        return sorted(tags, key=lambda t: t[1], reverse=True)

    def upsert_video(self, source_name: str, video: VideoDoc, chunks: list[dict], channel_name: str | None = None):
        """Insert or replace a video and its transcript chunks. Returns the chunk count."""
        tags = self.auto_tag_video(video.title, video.description)
        chapters = ", ".join(f"{ch.start_time} {ch.title}" for ch in video.chapters) if video.chapters else ""
        self.write_video(
            source_name,
            {
                "video_id": video.video_id,
                "title": video.title,
                "url": video.url,
                "channel": video.source_name,
                "channel_name": channel_name or video.source_name,
                "created": video.created,
                "tags": video.tags,
                "description": video.description[:5000],
                "duration_seconds": video.duration_seconds,
                "chapters": chapters,
            },
            tags,
            chunks,
        )
        return len(chunks)

    def write_video(self, source_name: str, video: dict, tags: list[tuple[str, float]], chunks: list[dict]) -> None:
        """Store a video row, its tags and its chunks, replacing whatever was there."""
        vid = video["video_id"]
        prefix = self._key(source_name, f"{vid}:")
        with self._store.transaction() as db:
            old = {r[0] for r in db.execute("SELECT doc_id FROM docs WHERE doc_id LIKE ?", (prefix + "%",))}
            self._store.forget(old)
            cols = ("video_id", "title", "url", "channel", "channel_name", "created", "tags", "description")
            cols += ("duration_seconds", "chapters")
            db.execute(
                f"INSERT OR REPLACE INTO videos(source, {', '.join(cols)}) VALUES (?{', ?' * len(cols)})",
                (source_name, *(video.get(c) for c in cols)),
            )
            db.execute("DELETE FROM video_tags WHERE video_id = ?", (vid,))
            db.executemany("INSERT INTO video_tags VALUES (?, ?, ?)", [(vid, t, w) for t, w in tags])
            for c in chunks:
                chapter = c.get("chapter", "")
                self._store.put(
                    f"{prefix}{c['chunk_index']}",
                    {
                        "title": f"{video['title']} — {chapter}" if chapter else video["title"],
                        "body": c["text"],
                        "kind": source_name,
                        "meta": {
                            "type": "video_chunk",
                            "video_id": vid,
                            "source_name": video.get("channel", ""),
                            "url": _timestamped(video["url"], _seconds(c)),
                            "created": video.get("created", ""),
                            "tags": video.get("tags", ""),
                            "chapter": chapter,
                            "start_time": c.get("start_time", ""),
                            "start_seconds": _seconds(c),
                            "chunk_type": c.get("chunk_type", "transcript"),
                        },
                    },
                )

    def delete_source(self, source_name: str) -> bool:
        with self._store.transaction() as db:
            keys = {r[0] for r in db.execute("SELECT doc_id FROM docs WHERE kind = ?", (source_name,))}
            vids = [r[0] for r in db.execute("SELECT video_id FROM videos WHERE source = ?", (source_name,))]
            self._store.forget(keys)
            db.executemany("DELETE FROM video_tags WHERE video_id = ?", [(v,) for v in vids])
            db.execute("DELETE FROM videos WHERE source = ?", (source_name,))
        return bool(keys or vids)

    # ── what the writers ask before writing ───────────────────────

    def get_all_urls(self, source_name: str) -> set[str]:
        with self._store.transaction() as db:
            rows = db.execute(
                "SELECT json_extract(meta, '$.url') FROM docs WHERE kind = ? AND json_extract(meta, '$.type') = 'doc'",
                (source_name,),
            ).fetchall()
        return {r[0] for r in rows if r[0]}

    def video_exists(self, source_name: str, video_id: str) -> bool:
        with self._store.transaction() as db:
            row = db.execute(
                "SELECT 1 FROM videos WHERE source = ? AND video_id = ?", (source_name, video_id)
            ).fetchone()
        return row is not None

    def _count_type(self, source_name: str, doc_type: str) -> int:
        with self._store.transaction() as db:
            return db.execute(
                "SELECT count(*) FROM docs WHERE kind = ? AND json_extract(meta, '$.type') = ?",
                (source_name, doc_type),
            ).fetchone()[0]

    def maker_count(self, source_name: str) -> int:
        return self._count_type(source_name, "maker")
