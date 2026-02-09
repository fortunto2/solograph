"""Pydantic schemas for CodeGraph entities.

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
    """Summary text for ChromaDB indexing."""

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


class ScanResult(BaseModel):
    """Aggregated result from scanning a project."""

    project: ProjectNode
    files: list[FileNode] = []
    symbols: list[SymbolNode] = []
    packages: list[PackageNode] = []
    modifications: list[ModifiedEdge] = []
