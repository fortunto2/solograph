"""The portfolio store: projects, packages, files and symbol definitions in SQLite.

It replaced a FalkorDB code graph whose structural layer — 781k CALLS edges,
imports, inheritance, session and git edges — got zero structural questions in
four months (5 calls in total, all of them "list the projects" or stats). What
stays is what no per-repo tool does: one index across every project, and
symbol names searchable by their parts.
"""

from __future__ import annotations

import os

from codegraph_mcp.portfolio import Portfolio, split_identifier


def test_identifiers_split_into_their_words():
    assert split_identifier("resolve_montage_profile") == "resolve montage profile"
    assert split_identifier("MontageProfile") == "montage profile"
    assert split_identifier("HTTPServer") == "http server"
    assert split_identifier("parseJSON2") == "parse json 2"


def _project(tmp_path, name="video", files=None):
    root = tmp_path / name
    root.mkdir()
    for rel, text in (files or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return {"name": name, "path": str(root), "stacks": ["python"], "status": "active"}


VIDEO = {
    "src/montage.py": "class MontageProfile:\n    pass\n\n\ndef resolve_montage_profile(name):\n    return name\n",
    "src/render.py": "def render_frame(x):\n    return x\n",
    "pyproject.toml": '[project]\nname = "video"\ndependencies = ["requests>=2", "numpy"]\n',
}


def _portfolio(tmp_path):
    return Portfolio(tmp_path / "portfolio.db")


def test_a_query_by_words_finds_both_spellings(tmp_path):
    pf = _portfolio(tmp_path)
    proj = _project(tmp_path, files=VIDEO)
    pf.sync_projects([proj])
    pf.scan_project(proj["name"], proj["path"])

    names = [h["name"] for h in pf.search_symbols("montage profile")]
    assert set(names[:2]) == {"MontageProfile", "resolve_montage_profile"}
    hit = next(h for h in pf.search_symbols("montage profile") if h["name"] == "resolve_montage_profile")
    assert hit["project"] == "video" and hit["file"] == "src/montage.py" and hit["line"] == 5
    assert hit["kind"] == "function"


def test_an_exact_name_comes_first(tmp_path):
    pf = _portfolio(tmp_path)
    proj = _project(tmp_path, files=VIDEO)
    pf.sync_projects([proj])
    pf.scan_project(proj["name"], proj["path"])
    assert pf.search_symbols("MontageProfile")[0]["name"] == "MontageProfile"
    assert pf.search_symbols("resolve_montage_profile")[0]["name"] == "resolve_montage_profile"


def test_project_and_kind_filters(tmp_path):
    pf = _portfolio(tmp_path)
    a = _project(tmp_path, "a", VIDEO)
    b = _project(tmp_path, "b", {"x.py": "def montage_profile():\n    pass\n"})
    pf.sync_projects([a, b])
    pf.scan_project("a", a["path"])
    pf.scan_project("b", b["path"])
    assert {h["project"] for h in pf.search_symbols("montage profile", project="b")} == {"b"}
    assert {h["kind"] for h in pf.search_symbols("montage profile", kind="class")} == {"class"}


def test_a_second_scan_reads_nothing_and_an_edit_reads_one_file(tmp_path):
    pf = _portfolio(tmp_path)
    proj = _project(tmp_path, files=VIDEO)
    pf.sync_projects([proj])
    assert pf.scan_project("video", proj["path"])["read"] == 3
    assert pf.scan_project("video", proj["path"])["read"] == 0

    f = tmp_path / "video" / "src" / "render.py"
    f.write_text("def render_thumbnail(x):\n    return x\n")
    os.utime(f, (1, 1))  # a different mtime even within the same second
    report = pf.scan_project("video", proj["path"])
    assert report["read"] == 1
    assert "render_frame" not in [h["name"] for h in pf.search_symbols("render frame")]
    assert pf.search_symbols("render thumbnail")[0]["name"] == "render_thumbnail"


def test_a_deleted_file_takes_its_symbols_with_it(tmp_path):
    pf = _portfolio(tmp_path)
    proj = _project(tmp_path, files=VIDEO)
    pf.sync_projects([proj])
    pf.scan_project("video", proj["path"])
    (tmp_path / "video" / "src" / "render.py").unlink()
    assert pf.scan_project("video", proj["path"])["removed"] == 1
    assert pf.search_symbols("render") == []


def test_packages_and_who_shares_them(tmp_path):
    pf = _portfolio(tmp_path)
    a = _project(tmp_path, "a", VIDEO)
    b = _project(tmp_path, "b", {"pyproject.toml": '[project]\nname = "b"\ndependencies = ["requests"]\n'})
    pf.sync_projects([a, b])
    pf.scan_project("a", a["path"])
    pf.scan_project("b", b["path"])

    shared = pf.shared_packages()
    assert shared == [{"package": "requests", "source": "pip", "projects": ["a", "b"]}]
    assert pf.shared_packages(package="numpy") == [{"package": "numpy", "source": "pip", "projects": ["a"]}]


def test_a_project_dropped_from_the_registry_leaves_the_store(tmp_path):
    pf = _portfolio(tmp_path)
    a = _project(tmp_path, "a", VIDEO)
    b = _project(tmp_path, "b", {"x.py": "def only_in_b():\n    pass\n"})
    pf.sync_projects([a, b])
    pf.scan_project("b", b["path"])
    pf.sync_projects([a])
    assert pf.search_symbols("only in b") == []
    assert pf.stats()["projects"] == 1


def test_a_name_with_a_quote_is_just_a_name(tmp_path):
    pf = _portfolio(tmp_path)
    proj = _project(tmp_path, "Rustam's app", VIDEO)
    pf.sync_projects([proj])
    pf.scan_project("Rustam's app", proj["path"])
    assert pf.explain("Rustam's app")["counts"]["files"] == 3
    assert pf.search_symbols("montage", project="Rustam's app")


def test_explain_says_what_a_project_is_made_of(tmp_path):
    pf = _portfolio(tmp_path)
    proj = _project(tmp_path, files=VIDEO)
    pf.sync_projects([proj])
    pf.scan_project("video", proj["path"])
    data = pf.explain("video")
    assert data["stack"] == "python"
    assert data["counts"] == {"files": 3, "symbols": 3, "packages": 2}
    assert data["languages"][0][0] == "python"
    assert ("requests", "pip") in data["dependencies"]
    assert pf.explain("nope") == "Project 'nope' not found."


def test_all_words_rank_above_some_words(tmp_path):
    pf = _portfolio(tmp_path)
    proj = _project(tmp_path, files={"a.py": "def render_frame():\n    pass\n\n\ndef render_page():\n    pass\n"})
    pf.sync_projects([proj])
    pf.scan_project("video", proj["path"])
    names = [h["name"] for h in pf.search_symbols("render frame")]
    assert names == ["render_frame", "render_page"]
