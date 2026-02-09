"""Streaming parser for Claude Code chat sessions.

Reads ~/.claude/projects/*/uuid.jsonl files, extracts:
- Session metadata (project, dates, model, branch)
- File interactions (Read→TOUCHED, Edit→EDITED, Write→CREATED)
- Summary text for vector indexing
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from ..models import SessionFileEdge, SessionNode, SessionSummary

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

# Max chars per assistant text block in summary
_TEXT_LIMIT = 200
_THINKING_LIMIT = 100
_SUMMARY_LIMIT = 2000


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


def scan_session(
    session_path: Path, project_dir_name: str
) -> tuple[SessionNode, list[SessionFileEdge], SessionSummary] | None:
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

    # For summary
    user_prompts: list[str] = []
    assistant_texts: list[str] = []
    thinking_texts: list[str] = []

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
                            user_prompts.append(content[:300])
                    elif isinstance(content, list):
                        # Array of content blocks
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text = block["text"]
                                    if not text.startswith("<") and len(text) > 5:
                                        user_prompt_count += 1
                                        user_prompts.append(text[:300])

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

                        elif block.get("type") == "text":
                            text = block.get("text", "")
                            if text and len(text) > 10:
                                assistant_texts.append(text[:_TEXT_LIMIT])

                        elif block.get("type") == "thinking":
                            text = block.get("thinking", "")
                            if text and len(text) > 10:
                                thinking_texts.append(text[:_THINKING_LIMIT])

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

    # Build summary
    summary_parts = []
    if project_name:
        header = f"[Project: {project_name}"
        if git_branch:
            header += f" | Branch: {git_branch}"
        if started_at:
            header += f" | {started_at[:10]}"
        header += "]"
        summary_parts.append(header)

    if user_prompts:
        summary_parts.append("\nPrompts:")
        for i, p in enumerate(user_prompts[:5], 1):
            clean = p.replace("\n", " ").strip()
            summary_parts.append(f"{i}. {clean}")

    if file_actions:
        edited = [f for f, a in file_actions.items() if "EDITED" in a or "CREATED" in a]
        touched = [f for f, a in file_actions.items() if f not in edited]
        if edited:
            summary_parts.append(f"\nEdited: {', '.join(edited[:10])}")
        if touched:
            summary_parts.append(f"Read: {', '.join(touched[:10])}")

    if assistant_texts:
        summary_parts.append("\nDecisions:")
        for t in assistant_texts[:3]:
            clean = t.replace("\n", " ").strip()
            summary_parts.append(f"- {clean}")

    summary_text = "\n".join(summary_parts)
    if len(summary_text) > _SUMMARY_LIMIT:
        summary_text = summary_text[:_SUMMARY_LIMIT] + "..."

    summary = SessionSummary(
        session_id=session_id,
        project_name=project_name,
        summary_text=summary_text,
        started_at=started_at,
        tags=[project_name] if project_name else [],
    )

    return session, edges, summary


def scan_all_sessions(
    claude_dir: Path | None = None,
    project_filter: str | None = None,
) -> Iterator[tuple[SessionNode, list[SessionFileEdge], SessionSummary]]:
    """Iterate over all session files in ~/.claude/projects/."""
    if claude_dir is None:
        claude_dir = Path.home() / ".claude"

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
    result = graph.query(
        "MATCH (sess:Session), (p:Project) "
        "WHERE sess.project_name = p.name "
        "MERGE (sess)-[:IN_PROJECT]->(p) "
        "RETURN COUNT(*) AS cnt"
    )
    return result.result_set[0][0] if result.result_set else 0
