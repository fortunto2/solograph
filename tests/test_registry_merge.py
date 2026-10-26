"""Refreshing the registry must add what is new without losing what was put there.

The registry was last generated on 27 Feb from ~/startups/active and
~/projects/epiphan; every project after that was appended by hand or not at all.
By 26 Oct, 40 directories in active/ were missing from it, so nothing ever
scanned them. Regenerating it the old way was not an option either: `init`
takes one directory and writes a fresh file, which would have dropped the
nested video-analyzer, shared/superduperai-analytics and every epiphan project.

So a refresh merges: entries already there stay, entries whose directory is
gone leave and are named, new checkouts are appended.
"""

from __future__ import annotations

import yaml

from codegraph_mcp.registry import (
    ProjectInfo,
    ProjectRegistry,
    detect_stacks,
    refresh_registry,
)


def _repo(path, files=()):
    (path / ".git").mkdir(parents=True)
    for rel, text in files:
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return path


def _write(registry_path, projects, scan_path=""):
    reg = ProjectRegistry(projects=projects, scan_path=scan_path)
    registry_path.write_text(yaml.dump(reg.model_dump(), sort_keys=False))


def _names(registry_path):
    return [p["name"] for p in yaml.safe_load(registry_path.read_text())["projects"]]


def test_new_checkouts_are_appended_and_outside_entries_kept(tmp_path):
    active = tmp_path / "active"
    _repo(active / "old-one")
    _repo(active / "new-one")
    nested = _repo(tmp_path / "elsewhere" / "nested")
    reg = tmp_path / "registry.yaml"
    _write(
        reg,
        [
            ProjectInfo(name="old-one", path=str(active / "old-one")),
            ProjectInfo(name="nested", path=str(nested)),
        ],
    )

    report = refresh_registry([active], reg)

    assert _names(reg) == ["old-one", "nested", "new-one"]
    assert report.added == ["new-one"]
    assert report.removed == []


def test_an_entry_whose_directory_is_gone_is_dropped_and_named(tmp_path):
    active = tmp_path / "active"
    _repo(active / "here")
    reg = tmp_path / "registry.yaml"
    _write(
        reg,
        [
            ProjectInfo(name="here", path=str(active / "here")),
            ProjectInfo(name="deleted", path=str(tmp_path / "nowhere")),
        ],
    )

    report = refresh_registry([active], reg)

    assert _names(reg) == ["here"]
    assert report.removed == ["deleted"]


def test_only_real_checkouts_count_as_new_projects(tmp_path):
    active = tmp_path / "active"
    _repo(active / "repo")
    (active / "no-git").mkdir(parents=True)
    (active / "no-git" / "main.py").write_text("x = 1\n")
    (active / "worktree").mkdir()
    (active / "worktree" / ".git").write_text("gitdir: /somewhere/.git/worktrees/w\n")
    (active / "demo.zip").write_text("PK")
    (active / "CLAUDE.md").write_text("# notes\n")
    (active / ".hidden").mkdir()
    reg = tmp_path / "registry.yaml"

    refresh_registry([active], reg)

    assert _names(reg) == ["repo"]


def test_a_new_directory_never_takes_an_existing_name(tmp_path):
    """The graph keys projects by name: a second `api` would overwrite the first."""
    active = tmp_path / "active"
    _repo(active / "api")
    first = _repo(tmp_path / "other" / "api")
    reg = tmp_path / "registry.yaml"
    _write(reg, [ProjectInfo(name="api", path=str(first))])

    report = refresh_registry([active], reg)

    data = yaml.safe_load(reg.read_text())["projects"]
    assert [(p["name"], p["path"]) for p in data] == [("api", str(first))]
    assert report.collisions == [str(active / "api")]


def test_the_previous_file_is_backed_up_before_writing(tmp_path):
    active = tmp_path / "active"
    _repo(active / "a")
    reg = tmp_path / "registry.yaml"
    _write(reg, [ProjectInfo(name="kept", path=str(_repo(tmp_path / "kept")))])
    before = reg.read_text()

    report = refresh_registry([active], reg)

    assert report.backup is not None
    assert report.backup.read_text() == before


def test_backups_do_not_pile_up(tmp_path):
    """The nightly job refreshes every night: a backup each time is 365 files a year."""
    active = tmp_path / "active"
    _repo(active / "a")
    reg = tmp_path / "registry.yaml"
    _write(reg, [])

    for _ in range(8):
        refresh_registry([active], reg)

    assert len(list(tmp_path.glob("registry.yaml.*.bak"))) == 5


def test_scan_paths_default_to_the_ones_the_registry_recorded(tmp_path):
    """The nightly job runs with no CODEGRAPH_SCAN_PATH; the file knows where it scans."""
    active, epiphan = tmp_path / "active", tmp_path / "epiphan"
    _repo(active / "a")
    _repo(epiphan / "b")
    reg = tmp_path / "registry.yaml"
    _write(reg, [], scan_path=f"{active},{epiphan}")

    refresh_registry(None, reg)

    assert _names(reg) == ["a", "b"]


def test_stack_detection_sees_shallow_markers_and_ignores_node_modules(tmp_path):
    proj = _repo(
        tmp_path / "app",
        files=[
            ("web/package.json", '{"dependencies": {"next": "15", "ai": "4"}}'),
            ("node_modules/x/pyproject.toml", "[project]\nname='x'\n"),
            ("ios/App/App.xcodeproj/project.pbxproj", ""),
        ],
    )

    assert detect_stacks(proj) == ["ios-swift", "nextjs-ai-agents", "nextjs-supabase"]
