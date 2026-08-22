"""Rescan only what git says changed since the last pass.

A full deep scan of a mid-sized project is about twenty seconds, and that is
the *floor*: it re-parses every file and rewrites every node, whether or not
anything moved. Between two commits, three files change. Doing the same
twenty seconds of work for three files is what keeps an index from being
refreshed often, and an index that is refreshed rarely is the one failure
this whole subsystem is written against — a stale graph answers confidently
and wrongly.

So: remember the commit each project was scanned at, ask git what changed,
and touch only those files. A no-op costs one `git rev-parse`.

What this deliberately does NOT do
----------------------------------
It never *widens* the graph beyond the files it looked at. A CALLS edge from
an unchanged file to a symbol that has just been renamed will be stale until
the next full pass, because finding it would mean re-reading every file that
might mention the old name — the very work being avoided. `--deep` on a full
scan is the correction, and the nightly refresh does one when the drift
counter says so (see `should_full_rescan`).

The honest framing: incremental keeps the graph *current about what moved*.
Full keeps it *consistent*. Both are needed, and confusing them is how an
index quietly starts lying.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# After this many incremental passes a project gets a full rescan, so
# cross-file drift (a renamed symbol other files still call) cannot
# accumulate forever. Twenty passes is roughly a fortnight of ordinary
# commits; the full pass costs twenty seconds.
FULL_RESCAN_EVERY = 20


@dataclass
class Delta:
    """What changed between the scanned commit and HEAD."""

    head: str
    changed: list[str]  # paths to (re)scan, relative to the project
    deleted: list[str]  # paths whose nodes must go
    from_commit: str | None  # None when the project was never scanned

    @property
    def is_noop(self) -> bool:
        return bool(self.from_commit) and not self.changed and not self.deleted


def _git(path: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def scanned_commit(graph, project: str) -> str | None:
    """The commit this project was last scanned at, or None."""
    rows = graph.query("MATCH (p:Project {name: $n}) RETURN p.scanned_commit", {"n": project}).result_set
    return rows[0][0] if rows and rows[0][0] else None


def scan_count(graph, project: str) -> int:
    rows = graph.query("MATCH (p:Project {name: $n}) RETURN p.incremental_passes", {"n": project}).result_set
    return int(rows[0][0]) if rows and rows[0][0] else 0


def record_scan(graph, project: str, head: str, incremental: bool) -> None:
    """Stamp the commit, and count incremental passes since the last full one."""
    graph.query(
        "MATCH (p:Project {name: $n}) "
        "SET p.scanned_commit = $c, "
        "p.incremental_passes = CASE WHEN $inc THEN coalesce(p.incremental_passes, 0) + 1 ELSE 0 END",
        {"n": project, "c": head, "inc": incremental},
    )


def should_full_rescan(graph, project: str) -> bool:
    return scan_count(graph, project) >= FULL_RESCAN_EVERY


def delta_since_last_scan(graph, project: str, path: Path) -> Delta:
    """Ask git what moved. An unscanned or non-git project asks for everything."""
    head = _git(path, "rev-parse", "HEAD")
    previous = scanned_commit(graph, project)

    if not head or not previous:
        return Delta(head=head, changed=[], deleted=[], from_commit=None)

    # Does the recorded commit still exist? A rebase or a fresh clone can
    # leave a hash nothing points at, and `git diff` against it fails —
    # silently returning "nothing changed" would freeze the graph forever.
    if not _git(path, "cat-file", "-e", f"{previous}^{{commit}}") and _git(
        path, "rev-parse", "--verify", "--quiet", previous
    ) in ("", None):
        return Delta(head=head, changed=[], deleted=[], from_commit=None)

    raw = _git(path, "diff", "--name-status", f"{previous}..{head}")
    changed: list[str] = []
    deleted: list[str] = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, rel = parts[0], parts[-1]
        if status.startswith("D"):
            deleted.append(rel)
        else:
            changed.append(rel)
            if status.startswith("R") and len(parts) == 3:
                deleted.append(parts[1])  # the old name of a rename

    # Uncommitted work counts too: an agent asking about code it just wrote
    # gets an answer about the last commit otherwise. Both halves are needed
    # and the second is easy to forget — `git diff` never mentions a file git
    # has not been told about, so a brand-new file was invisible until it was
    # committed. Measured: a fresh Swift file, scanned incrementally, did not
    # reach the graph at all.
    for rel in _git(path, "diff", "--name-only", "HEAD").splitlines():
        if rel and rel not in changed:
            changed.append(rel)
    for rel in _git(path, "ls-files", "--others", "--exclude-standard").splitlines():
        if rel and rel not in changed:
            changed.append(rel)

    # Files the graph holds that are no longer on disk. git cannot tell us
    # about these when they were never tracked: an untracked file that is
    # created and then deleted leaves a symbol behind forever, because
    # neither `diff` nor `ls-files --others` mentions something that is gone.
    # Measured — a probe file removed this way stayed in the graph and the
    # pass reported "unchanged". One existence check per known file is
    # microseconds and runs only on a partial pass.
    for row in graph.query("MATCH (f:File {project: $p}) RETURN f.path", {"p": project}).result_set:
        rel = row[0]
        if rel and not (path / rel).exists() and rel not in deleted:
            deleted.append(rel)

    return Delta(head=head, changed=changed, deleted=deleted, from_commit=previous)


def forget_files(graph, project: str, paths: list[str]) -> None:
    """Drop these files and everything defined in them, in one query each."""
    if not paths:
        return
    graph.query(
        "UNWIND $paths AS p MATCH (s:Symbol {project: $proj, file: p}) DETACH DELETE s",
        {"paths": paths, "proj": project},
    )
    graph.query(
        "UNWIND $paths AS p MATCH (f:File {project: $proj, path: p}) DETACH DELETE f",
        {"paths": paths, "proj": project},
    )
