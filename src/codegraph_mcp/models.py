"""Pydantic schemas for Solograph entities.

SGR-first: these schemas are the source of truth.
All scanners produce data conforming to these models.
"""

from pydantic import BaseModel, Field


class ProjectNode(BaseModel):
    """A project from the registry."""

    name: str
    stack: str = ""
    path: str
    status: str = "active"
    has_claude_md: bool = False
    description: str = ""


class FileNode(BaseModel):
    """A source code file."""

    path: str = Field(..., description="Path relative to project root")
    project: str
    lang: str = ""
    lines: int = 0
    last_modified: str = ""


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


class ModifiedEdge(BaseModel):
    """Edge: something modified a file."""

    file_path: str
    project: str
    author: str = ""
    date: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    commit_hash: str = ""


class SessionNode(BaseModel):
    """A Claude Code chat session."""

    session_id: str
    project_path: str
    project_name: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_ms: int = 0
    user_prompts: int = 0
    tool_uses: int = 0
    model: str = ""
    git_branch: str = ""
    slug: str = ""


class SessionFileEdge(BaseModel):
    """Edge: session touched/edited/created a file."""

    session_id: str
    file_path: str = Field(..., description="Path relative to project root")
    project_name: str
    action: str = Field(..., description="TOUCHED | EDITED | CREATED")
    count: int = 1


class SessionSummary(BaseModel):
    """Summary text for vector indexing."""

    session_id: str
    project_name: str
    summary_text: str
    started_at: str = ""
    tags: list[str] = []


class ImportEdge(BaseModel):
    """Edge: file imports a module/package."""

    source_file: str = Field(..., description="Relative path of importing file")
    project: str
    module: str = Field(..., description="Module name (e.g. 'os.path', 'react', './utils')")
    kind: str = Field(..., description="'internal' or 'external'")


class CallEdge(BaseModel):
    """Edge: file calls a symbol."""

    source_file: str = Field(..., description="Relative path of calling file")
    project: str
    callee_name: str = Field(..., description="Name of called function/method")


class InheritsEdge(BaseModel):
    """Edge: class inherits from another class."""

    child_name: str
    parent_name: str
    child_file: str = Field(..., description="Relative path where child is defined")
    project: str


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


class ScanResult(BaseModel):
    """Aggregated result from scanning a project."""

    project: ProjectNode
    files: list[FileNode] = []
    symbols: list[SymbolNode] = []
    packages: list[PackageNode] = []
    modifications: list[ModifiedEdge] = []
