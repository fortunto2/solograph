"""Scan git log → MODIFIED edges on File nodes."""

from pathlib import Path

from ..models import ModifiedEdge


def scan_git_log(project_path: Path, project_name: str, max_commits: int = 100) -> list[ModifiedEdge]:
    """Extract recent file modifications from git log."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "git",
                "log",
                f"--max-count={max_commits}",
                "--numstat",
                "--format=%H|%an|%aI",
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
    """Create MODIFIED edges between commits and files, a batch at a time.

    MERGE, not CREATE. Every scan re-reads the last 100 commits, so CREATE
    wrote the same history again on each pass: the edge count grew with the
    number of scans rather than with the number of commits, and nothing in
    the graph said which copy was which. Keyed on (file, commit), because
    that is what makes a modification the same modification.
    """
    count = 0
    for i in range(0, len(mods), 500):
        chunk = mods[i : i + 500]
        rows = [
            {
                "path": m.file_path,
                "project": m.project,
                "author": m.author,
                "date": m.date,
                "added": m.lines_added,
                "removed": m.lines_removed,
                "commit": m.commit_hash,
            }
            for m in chunk
        ]
        graph.query(
            "UNWIND $rows AS r "
            "MATCH (f:File {path: r.path, project: r.project}) "
            "MERGE (f)<-[e:MODIFIED {commit: r.commit}]-(f) "
            "SET e.author = r.author, e.date = r.date, "
            "e.lines_added = r.added, e.lines_removed = r.removed",
            {"rows": rows},
        )
        count += len(chunk)
    return count
