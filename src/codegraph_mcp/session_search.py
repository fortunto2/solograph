"""Search over Claude Code transcripts, including the ones the disk has deleted.

Claude Code removes transcripts after 30 days (`cleanupPeriodDays`). Measured
on 26 Oct 2026: the old vector index held 640 sessions from Dec–Aug, the disk
633 from Sep 28–Oct 26, overlap zero. So the index is not a cache of the disk
but the archive: every search first syncs new and changed transcripts, and a
transcript that disappears keeps its text here, marked `gone`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .fts import FtsStore
from .scanner.sessions import _derive_project_name


def claude_dir() -> Path:
    env = os.environ.get("CLAUDE_SESSIONS_DIR", "")
    return Path(env).expanduser() if env else Path.home() / ".claude"


def default_db_path() -> Path:
    base = os.environ.get("CODEGRAPH_SESSIONS_DB", str(Path.home() / ".solo" / "sessions"))
    return Path(base).expanduser() / "search.db"


# Sessions started in a temp directory are batch runs (`claude -p` from /tmp, an
# agent's scratchpad), not conversations. On 26 Oct they were 322 of 633, and the
# labeling jobs among them embed other sessions' text in their prompt — they came
# first for nearly every real query.
_BATCH_DIRS = ("-private-tmp", "-tmp-", "-private-var-folders")


def transcript_files(projects_dir: Path) -> dict[str, Path] | None:
    """{session_id: path} for conversation transcripts, or None if the directory cannot be read."""
    try:
        if not projects_dir.is_dir():
            return None
        return {
            p.stem: p
            for p in projects_dir.glob("*/*.jsonl")
            if not p.parent.name.startswith(_BATCH_DIRS) and p.parent.name != "-tmp"
        }
    except OSError:
        return None


# Byte markers checked before json.loads: tool results are most of a transcript's
# bytes and none of its text, so they are never parsed.
_USER = b'"type":"user"'
_ASSISTANT = b'"type":"assistant"'
_TITLE = b'"type":"ai-title"'
_SKIP_USER = (b'"toolUseResult"', b'"isMeta":true')
_FILE_TOOLS = ("Edit", "Write")


def read_transcript(path: Path, offset: int = 0) -> dict:
    """The searchable text of a transcript from `offset` on, and where reading stopped.

    Prompts, answers and edited files, in order. Skill expansions arrive as user
    messages marked isMeta and are not what anyone said, so they are skipped. A
    last line without its newline is still being written: reading stops before
    it, and `end` says where to resume.
    """
    parts: list[str] = []
    title = ""
    first_prompt = ""
    meta: dict = {}
    end = offset
    with open(path, "rb") as f:
        f.seek(offset)
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            end += len(raw)
            if _USER in raw:
                if any(m in raw for m in _SKIP_USER):
                    continue
            elif _ASSISTANT in raw:
                if b'"type":"text"' not in raw and not any(f'"name":"{t}"'.encode() in raw for t in _FILE_TOOLS):
                    continue
            elif _TITLE not in raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if obj.get("type") == "ai-title":
                title = obj.get("aiTitle") or title
                continue
            if obj.get("cwd") and "project_path" not in meta:
                meta["project_path"] = obj["cwd"]
                meta["branch"] = obj.get("gitBranch", "")
            if obj.get("timestamp"):
                meta.setdefault("started_at", obj["timestamp"])
                meta["ended_at"] = obj["timestamp"]
            content = (obj.get("message") or {}).get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if len(text) > 5 and not text.startswith("<"):
                        parts.append(text)
                        if obj["type"] == "user" and not first_prompt:
                            first_prompt = text
                elif block.get("type") == "tool_use" and block.get("name") in _FILE_TOOLS:
                    inp = block.get("input")
                    fp = inp.get("file_path", "") if isinstance(inp, dict) else ""
                    if fp:
                        parts.append(f"Edited: {fp}")
    if offset == 0:
        title = title or first_prompt.replace("\n", " ").strip()[:200]
    else:
        meta.pop("started_at", None)  # the head already has the real start
        meta.pop("project_path", None)
        meta.pop("branch", None)
    project = _derive_project_name(path.parent.name)
    return {
        "title": title,
        "body": "\n\n".join(parts),
        "kind": project,
        "meta": {"session_id": path.stem, "project": project, **meta},
        "end": end,
    }


# A session whose working directory is inside a folder holding one of these is
# never indexed: its text would otherwise surface in snippets from any project.
# `.metadata_never_index` is Spotlight's marker, so a folder already private to
# macOS search is private here too.
PRIVATE_MARKERS = (".solograph-private", ".metadata_never_index")


def _session_cwd(path: Path) -> str:
    with open(path, "rb") as f:
        for i, raw in enumerate(f):
            if b'"cwd"' in raw:
                try:
                    return json.loads(raw).get("cwd") or ""
                except ValueError:
                    pass
            if i > 200:
                break
    return ""


def _is_private(cwd: str) -> bool:
    if not cwd:
        return False
    d = Path(cwd)
    return any((p / m).exists() for p in (d, *d.parents) for m in PRIVATE_MARKERS)


class SessionSearch:
    def __init__(self, db_path: str | Path | None = None, sessions_root: Path | None = None):
        self.store = FtsStore(db_path or default_db_path())
        self.projects_dir = (sessions_root or claude_dir()) / "projects"
        self._private_dirs: dict[str, bool] = {}

    def _private(self, path: Path) -> bool:
        """Per project directory: all its sessions share one working directory."""
        key = path.parent.name
        if key not in self._private_dirs:
            self._private_dirs[key] = _is_private(_session_cwd(path))
        return self._private_dirs[key]

    def sync(self) -> dict:
        files = transcript_files(self.projects_dir)
        if files is None:
            return {"unreadable": True, "indexed": self.store.count()}
        private = {doc_id for doc_id, path in files.items() if self._private(path)}
        self.store.forget(private)
        files = {k: v for k, v in files.items() if k not in private}
        report = self.store.sync(files, read_transcript, keep_gone=True, append=read_transcript)
        report["on_disk"] = len(files)
        report["private"] = len(private)
        return report

    def search(self, query: str, n_results: int = 5, project: str | None = None) -> dict:
        report = self.sync()
        results = []
        for hit in self.store.search(query, n_results=n_results, kind=project):
            result = {
                "session_id": hit["doc_id"],
                "project": hit.get("project", hit["kind"]),
                "started_at": hit.get("started_at", ""),
                "title": hit["title"],
                "snippet": hit["snippet"],
                "score": hit["score"],
            }
            if hit["gone"]:
                result["transcript"] = "deleted from disk; this text is the only copy"
            elif hit.get("project_path"):
                result["resume"] = f"cd {hit['project_path']} && claude --resume {hit['doc_id']}"
            results.append(result)
        return {"results": results, "coverage": coverage(report)}


def coverage(report: dict) -> dict:
    """What this answer could see. Never "complete" when we could not look."""
    if report.get("unreadable"):
        return {
            "status": "unknown",
            "indexed": report["indexed"],
            "why": "the transcript directory is missing or unreadable; only the archive was searched",
        }
    cov = {
        "on_disk": report["on_disk"],
        "indexed": report["indexed"],
        "archived": report["gone"],
        "read_now": report["read"],
        "private_skipped": report.get("private", 0),
    }
    if report["failed"]:
        cov["status"] = "partial"
        cov["failed"] = report["failed"][:10]
        cov["why"] = f"{len(report['failed'])} transcript(s) could not be read, so this answer is not about them"
    else:
        cov["status"] = "complete"
    return cov
