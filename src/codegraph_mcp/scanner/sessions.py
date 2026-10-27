"""Streaming parser for Claude Code chat sessions.

Reads ~/.claude/projects/*/uuid.jsonl files, extracts:
- Session metadata (project, dates, model, branch)
- File interactions (Read→TOUCHED, Edit→EDITED, Write→CREATED)

The searchable text is not built here: session_search.read_transcript streams
it from a byte offset, so a 400 MB live transcript is not re-read per search.
"""

import json
import os
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..models import SessionFileEdge, SessionNode

# Tools → action mapping
_TOOL_FILE_KEY = {
    "Read": ("file_path", "TOUCHED"),
    "Edit": ("file_path", "EDITED"),
    "Write": ("file_path", "CREATED"),
    "Grep": ("path", "TOUCHED"),
    "Glob": ("path", "TOUCHED"),
}

# Lines to skip
_SKIP_TYPES = {"progress", "file-history-snapshot"}


def _derive_project_name(project_dir_name: str) -> str:
    """Derive human-readable project name from Claude project dir name.

    '-Users-john-projects-MyApp-ios' → 'MyApp/ios'
    '-Users-john-projects-backend' → 'backend'
    '-Users-john-projects' → 'projects'
    '-Users-john' → 'john'
    """
    # Replace leading dashes with slashes to reconstruct path
    parts = project_dir_name.lstrip("-").split("-")
    # Find 'projects' index to strip prefix
    try:
        idx = parts.index("projects")
        remainder = parts[idx + 1 :]
        if remainder:
            return "/".join(remainder)
        return "projects"
    except ValueError:
        # No 'projects' in path — return last segment
        return parts[-1] if parts else project_dir_name


def _extract_file_path(tool_name: str, tool_input: dict) -> str | None:
    """Extract file path from tool_use input."""
    if tool_name not in _TOOL_FILE_KEY:
        return None
    key, _ = _TOOL_FILE_KEY[tool_name]
    return tool_input.get(key)


def _make_relative(file_path: str, project_path: str) -> str:
    """Make absolute path relative to project root."""
    if not file_path or not project_path:
        return file_path or ""
    try:
        return str(Path(file_path).relative_to(project_path))
    except ValueError:
        # File outside project — keep as-is
        return file_path


def scan_session(session_path: Path, project_dir_name: str) -> tuple[SessionNode, list[SessionFileEdge]] | None:
    """Parse one JSONL session file. Streams line-by-line.

    Returns None for empty/broken sessions.
    """
    session_id = session_path.stem
    project_name = _derive_project_name(project_dir_name)

    # Accumulators
    project_path = ""
    started_at = ""
    ended_at = ""
    duration_ms = 0
    model = ""
    git_branch = ""
    slug = ""
    user_prompt_count = 0
    tool_use_count = 0

    # For file edges
    file_actions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    try:
        with open(session_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = obj.get("type")
                if msg_type in _SKIP_TYPES:
                    continue

                # Extract common metadata
                if not project_path and obj.get("cwd"):
                    project_path = obj["cwd"]
                if not started_at and obj.get("timestamp"):
                    started_at = obj["timestamp"]
                if obj.get("timestamp"):
                    ended_at = obj["timestamp"]
                if not git_branch and obj.get("gitBranch"):
                    git_branch = obj["gitBranch"]
                if not slug and obj.get("slug"):
                    slug = obj["slug"]

                if msg_type == "user":
                    msg = obj.get("message", {})
                    content = msg.get("content")
                    if isinstance(content, str):
                        # Skip tool_results and system messages
                        if not content.startswith("<") and len(content) > 5:
                            user_prompt_count += 1
                    elif isinstance(content, list):
                        # Array of content blocks
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text = block["text"]
                                    if not text.startswith("<") and len(text) > 5:
                                        user_prompt_count += 1

                elif msg_type == "assistant":
                    msg = obj.get("message", {})
                    # Extract model
                    if not model and msg.get("model"):
                        model = msg["model"]

                    content = msg.get("content", [])
                    if not isinstance(content, list):
                        continue

                    for block in content:
                        if not isinstance(block, dict):
                            continue

                        if block.get("type") == "tool_use":
                            tool_use_count += 1
                            tool_name = block.get("name", "")
                            tool_input = block.get("input", {})

                            fp = _extract_file_path(tool_name, tool_input)
                            if fp:
                                _, action = _TOOL_FILE_KEY[tool_name]
                                rel = _make_relative(fp, project_path)
                                file_actions[rel][action] += 1

                elif msg_type == "system":
                    if obj.get("subtype") == "turn_duration":
                        duration_ms += obj.get("durationMs", 0)

    except (OSError, UnicodeDecodeError):
        return None

    # Skip empty sessions
    if user_prompt_count == 0 and tool_use_count == 0:
        return None

    # Build session node
    session = SessionNode(
        session_id=session_id,
        project_path=project_path,
        project_name=project_name,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        user_prompts=user_prompt_count,
        tool_uses=tool_use_count,
        model=model,
        git_branch=git_branch,
        slug=slug,
    )

    # Build file edges (pick highest-priority action per file)
    edges: list[SessionFileEdge] = []
    action_priority = {"CREATED": 3, "EDITED": 2, "TOUCHED": 1}
    for fpath, actions in file_actions.items():
        # Use highest-priority action
        best_action = max(actions.keys(), key=lambda a: action_priority.get(a, 0))
        total_count = sum(actions.values())
        edges.append(
            SessionFileEdge(
                session_id=session_id,
                file_path=fpath,
                project_name=project_name,
                action=best_action,
                count=total_count,
            )
        )

    return session, edges


def count_session_files(claude_dir: Path | None = None) -> int | None:
    """How many session transcripts exist on disk, or None if we cannot tell.

    None rather than 0 when the directory is missing or unreadable. Zero is a
    measurement — "there are no sessions" — and an unreadable directory is not one.
    Reporting the second as the first turns "we cannot see your history" into
    "you have no history", which is the same shape as a lint tool that is absent
    being read as a lint tool that found nothing.
    """
    if claude_dir is None:
        env = os.environ.get("CLAUDE_SESSIONS_DIR", "")
        claude_dir = Path(env).expanduser() if env else Path.home() / ".claude"

    projects_dir = claude_dir / "projects"
    try:
        if not projects_dir.is_dir():
            return None
        return sum(1 for _ in projects_dir.rglob("*.jsonl"))
    except OSError:
        return None


def scan_all_sessions(
    claude_dir: Path | None = None,
    project_filter: str | None = None,
) -> Iterator[tuple[SessionNode, list[SessionFileEdge]]]:
    """Iterate over all session files in ~/.claude/projects/."""
    if claude_dir is None:
        # Read here rather than threaded in from one caller: the MCP server was the
        # only path that passed it, so `solograph sessions scan` and the post-commit
        # hook kept ingesting the personal ~/.claude into a client graph. Its
        # sibling CODEGRAPH_SESSIONS_DB is read at the store for the same reason.
        env = os.environ.get("CLAUDE_SESSIONS_DIR", "")
        claude_dir = Path(env).expanduser() if env else Path.home() / ".claude"

    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        return

    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir():
            continue

        project_dir_name = proj_dir.name
        derived_name = _derive_project_name(project_dir_name)

        if project_filter and project_filter.lower() not in derived_name.lower():
            continue

        for session_file in sorted(proj_dir.glob("*.jsonl")):
            result = scan_session(session_file, project_dir_name)
            if result is not None:
                yield result


def ingest_sessions(graph, sessions: list[SessionNode]) -> int:
    """MERGE Session nodes into FalkorDB."""
    count = 0
    for s in sessions:
        graph.query(
            "MERGE (sess:Session {session_id: $sid}) "
            "SET sess.project_name = $pname, "
            "    sess.project_path = $ppath, "
            "    sess.started_at = $start, "
            "    sess.ended_at = $end, "
            "    sess.duration_ms = $dur, "
            "    sess.user_prompts = $up, "
            "    sess.tool_uses = $tu, "
            "    sess.model = $model, "
            "    sess.git_branch = $branch, "
            "    sess.slug = $slug",
            {
                "sid": s.session_id,
                "pname": s.project_name,
                "ppath": s.project_path,
                "start": s.started_at,
                "end": s.ended_at,
                "dur": s.duration_ms,
                "up": s.user_prompts,
                "tu": s.tool_uses,
                "model": s.model,
                "branch": s.git_branch,
                "slug": s.slug,
            },
        )
        count += 1

    return count


def ingest_session_files(graph, edges: list[SessionFileEdge]) -> int:
    """MERGE session→file edges. Creates File nodes if they don't exist."""
    count = 0
    for e in edges:
        rel_type = e.action  # TOUCHED, EDITED, or CREATED
        graph.query(
            f"MATCH (sess:Session {{session_id: $sid}}) "
            f"MERGE (f:File {{path: $fpath, project: $proj}}) "
            f"MERGE (sess)-[r:{rel_type}]->(f) "
            f"SET r.count = $cnt",
            {
                "sid": e.session_id,
                "fpath": e.file_path,
                "proj": e.project_name,
                "cnt": e.count,
            },
        )
        count += 1

    return count


def link_sessions_to_projects(graph) -> int:
    """Create IN_PROJECT edges from Session to Project nodes."""
    # Join on path, not on name. The name is derived by splitting the session
    # directory on every dash (_derive_project_name), so `video-analyzer` becomes
    # `analyzer` and `sgr-chat-agent` becomes `epiphan/sgr/chat/agent` — and it is
    # not recoverable, because the directory encoding flattens underscores to
    # dashes too (`legacy_crm_php` arrives as `legacy-crm-php`). project_path is
    # the session's real cwd and matches Project.path exactly. The name match is
    # kept as a fallback for sessions recorded before project_path existed.
    result = graph.query(
        "MATCH (sess:Session), (p:Project) "
        "WHERE sess.project_path = p.path "
        "   OR (sess.project_path IS NULL AND sess.project_name = p.name) "
        "MERGE (sess)-[:IN_PROJECT]->(p) "
        "RETURN COUNT(*) AS cnt"
    )
    return result.result_set[0][0] if result.result_set else 0
