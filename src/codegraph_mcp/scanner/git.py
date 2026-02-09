"""Scan git log → MODIFIED edges on File nodes."""

from pathlib import Path

from ..models import ModifiedEdge


def scan_git_log(project_path: Path, project_name: str, max_commits: int = 100) -> list[ModifiedEdge]:
    """Extract recent file modifications from git log."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "git", "log", f"--max-count={max_commits}",
                "--numstat", "--format=%H|%an|%aI",
            ],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []
    except Exception:
        return []

    modifications = []
    current_commit = ""
    current_author = ""
    current_date = ""

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line and line.count("|") == 2:
            parts = line.split("|")
            current_commit = parts[0]
            current_author = parts[1]
            current_date = parts[2][:10]  # YYYY-MM-DD
        elif "\t" in line:
            parts = line.split("\t")
            if len(parts) == 3:
                added, removed, file_path = parts
                try:
                    lines_added = int(added) if added != "-" else 0
                    lines_removed = int(removed) if removed != "-" else 0
                except ValueError:
                    continue
                modifications.append(
                    ModifiedEdge(
                        file_path=file_path,
                        project=project_name,
                        author=current_author,
                        date=current_date,
                        lines_added=lines_added,
                        lines_removed=lines_removed,
                        commit_hash=current_commit[:8],
                    )
                )

    return modifications


def ingest_modifications(graph, mods: list[ModifiedEdge]) -> int:
    """Create MODIFIED edges between commits and files."""
    count = 0
    for m in mods:
        path_escaped = m.file_path.replace("'", "\\'")
        author_escaped = m.author.replace("'", "\\'")
        # Only link to files that exist in the graph
        graph.query(
            f"MATCH (f:File {{path: '{path_escaped}', project: '{m.project}'}}) "
            f"CREATE (f)<-[:MODIFIED {{author: '{author_escaped}', date: '{m.date}', "
            f"lines_added: {m.lines_added}, lines_removed: {m.lines_removed}, "
            f"commit: '{m.commit_hash}'}}]-(f)"
        )
        count += 1
    return count
