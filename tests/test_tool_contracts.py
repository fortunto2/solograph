"""Every tool's return value must pass its own declared output schema.

The MCP server builds an output schema from each tool's return annotation and
validates the result against it before replying. A test that calls the Python
function directly never runs that check. That is how session_search shipped on
6 Sep returning {results, coverage} under a `-> list[dict]` annotation: its nine
tests called the helper, all passed, and every real call through MCP failed with
"Input should be a valid list" for seven weeks.

So these go through `mcp.call_tool`, the same path a client takes, against real
stores in a temporary directory.
"""

from __future__ import annotations

import asyncio
import json

import codegraph_mcp.server as server


def _call(name, args):
    result = asyncio.run(server.mcp.call_tool(name, args))
    assert not result.is_error, result.content[0].text
    return result


def test_session_search_passes_its_output_schema(monkeypatch, tmp_path):
    proj = tmp_path / "claude" / "projects" / "-w-app"
    proj.mkdir(parents=True)
    (proj / "s1.jsonl").write_text(
        json.dumps({"type": "user", "cwd": "/w/app", "message": {"content": "graph rebuild"}}, separators=(",", ":"))
        + "\n"
    )
    monkeypatch.setenv("CLAUDE_SESSIONS_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEGRAPH_SESSIONS_DB", str(tmp_path / "db"))
    monkeypatch.setattr(server, "_sessions", None)

    payload = json.loads(_call("session_search", {"query": "graph"}).content[0].text)

    assert payload["results"][0]["session_id"] == "s1"
    assert payload["coverage"]["status"] == "complete"


def test_kb_search_passes_its_output_schema(monkeypatch, tmp_path):
    (tmp_path / "note.md").write_text("---\ntitle: Graph notes\n---\ntree-sitter")
    monkeypatch.setattr(server, "KB_PATH", str(tmp_path))
    monkeypatch.setattr(server, "_kb", None)

    result = _call("kb_search", {"query": "graph"})

    assert result.structured_content["result"][0]["file"] == "note.md"


def _scanned_portfolio(tmp_path, monkeypatch):
    from codegraph_mcp.portfolio import Portfolio

    root = tmp_path / "app"
    root.mkdir()
    (root / "m.py").write_text("class MontageProfile:\n    pass\n\n\ndef resolve_montage_profile():\n    pass\n")
    (root / "pyproject.toml").write_text('[project]\nname = "app"\ndependencies = ["requests"]\n')
    other = tmp_path / "other"
    other.mkdir()
    (other / "pyproject.toml").write_text('[project]\nname = "other"\ndependencies = ["requests"]\n')
    pf = Portfolio(tmp_path / "portfolio.db")
    pf.sync_projects([{"name": "app", "path": str(root)}, {"name": "other", "path": str(other)}])
    pf.scan_project("app", root)
    pf.scan_project("other", other)
    monkeypatch.setattr(server, "_portfolio", pf)
    return pf


def test_symbol_search_passes_its_output_schema(monkeypatch, tmp_path):
    _scanned_portfolio(tmp_path, monkeypatch)
    hits = _call("symbol_search", {"query": "montage profile"}).structured_content["result"]
    assert {h["name"] for h in hits} == {"MontageProfile", "resolve_montage_profile"}
    assert hits[0]["file"] == "m.py" and hits[0]["project"] == "app"


def test_portfolio_tools_pass_their_output_schemas(monkeypatch, tmp_path):
    _scanned_portfolio(tmp_path, monkeypatch)
    stats = json.loads(_call("codegraph_stats", {}).content[0].text)
    assert stats["projects"] == 2 and stats["symbols"] == 2
    explain = json.loads(_call("codegraph_explain", {"project": "app"}).content[0].text)
    assert explain["counts"]["symbols"] == 2
    shared = _call("codegraph_shared", {}).structured_content["result"]
    assert shared == [{"package": "requests", "source": "pip", "projects": ["app", "other"]}]
    one = _call("codegraph_shared", {"package": "req"}).structured_content["result"]
    assert one[0]["projects"] == ["app", "other"]


def test_project_info_passes_its_output_schema(monkeypatch, tmp_path):
    reg = tmp_path / "registry.yaml"
    reg.write_text("projects:\n- name: app\n  path: /tmp/app\n  stacks: [python]\n")
    monkeypatch.setattr(server, "CODEGRAPH_REGISTRY", str(reg))
    listed = _call("project_info", {}).structured_content["result"]
    assert listed[0]["name"] == "app"
    one = json.loads(_call("project_info", {"name": "app"}).content[0].text)
    assert one["stacks"] == ["python"]
