"""Pydantic schemas for Solograph entities.

SGR-first: these schemas are the source of truth.
All scanners produce data conforming to these models.
"""

from pydantic import BaseModel, Field


class SymbolNode(BaseModel):
    """A code symbol (function, class, protocol)."""

    name: str
    kind: str = Field(..., description="class, function, method, protocol, interface")
    file: str = Field(..., description="Relative file path")
    project: str
    line: int = 0


class PackageNode(BaseModel):
    """A dependency package."""

    name: str
    version: str | None = None
    source: str = Field("", description="pip, npm, spm, gradle, system")


class SourceDoc(BaseModel):
    """A document from an external source (Telegram, YouTube, etc.)."""

    doc_id: str = Field(..., description="md5 hash (e.g. tg:channel:post_id)")
    source_type: str = Field(..., description="telegram-post, youtube-transcript")
    source_name: str = Field(..., description="Channel or source name")
    title: str = ""
    content: str = Field("", description="Preview text (up to 500 chars)")
    url: str = ""
    created: str = ""
    tags: str = Field("", description="Comma-separated tags")
    embed_text: str = Field("", description="Full text for embedding (up to 3000 chars)")
    popularity: int = Field(0, description="Popularity score (e.g. upvotes) for search ranking boost")


class VideoChapter(BaseModel):
    """A chapter marker from a YouTube video description."""

    title: str
    start_time: str = ""  # "5:30" or "1:05:20"
    start_seconds: int = 0


class VideoDoc(BaseModel):
    """A YouTube video with metadata for chunked indexing."""

    video_id: str  # YouTube 11-char ID
    doc_id: str  # md5("yt:{video_id}")
    source_name: str  # channel handle
    title: str = ""
    description: str = ""  # full YouTube description (up to 5000 chars)
    url: str = ""
    created: str = ""
    tags: str = ""
    duration_seconds: int = 0
    chapters: list[VideoChapter] = []
    transcript: str = ""  # transient — used during indexing, not stored on node


class MakerProfile(BaseModel):
    """A ProductHunt maker profile (scraped via Playwright)."""

    username: str  # PH handle
    name: str = ""
    headline: str = ""
    bio: str = ""
    twitter_username: str = ""
    linkedin_url: str = ""
    website_url: str = ""
    points: int = 0  # karmaBadge.score
    streak_days: int = 0  # visitStreak.duration
    followers_count: int = 0
    following_count: int = 0
    products_count: int = 0  # products made
    hunted_count: int = 0  # submitted posts
    is_maker: bool = False
    avatar_url: str = ""
    created_at: str = ""
    ph_user_id: str = ""
