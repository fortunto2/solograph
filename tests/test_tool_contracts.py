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
