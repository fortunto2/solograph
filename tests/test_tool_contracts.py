"""Every tool's return value must pass its own declared output schema.

The MCP server builds an output schema from each tool's return annotation and
validates the result against it before replying. A test that calls the Python
function directly never runs that check. That is how session_search shipped on
6 Sep returning {results, coverage} under a `-> list[dict]` annotation: its nine
tests called the helper, all passed, and every real call through MCP failed with
"Input should be a valid list" for seven weeks.

So these go through `mcp.call_tool`, the same path a client takes.
"""

from __future__ import annotations

import asyncio
import json

import codegraph_mcp.server as server


class FakeSessionIndex:
    def count(self):
        return 1

    def search(self, query, n_results=5, project=None):
        return [{"session_id": "s1", "project": "p", "snippet": query}]


def test_session_search_passes_its_output_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_get_session_index", lambda: FakeSessionIndex())
    monkeypatch.setattr(server, "_auto_scan_sessions_if_empty", lambda idx: None)
    monkeypatch.setenv("CLAUDE_SESSIONS_DIR", str(tmp_path))

    result = asyncio.run(server.mcp.call_tool("session_search", {"query": "graph"}))

    assert not result.is_error, result.content[0].text
    payload = json.loads(result.content[0].text)
    assert payload["results"][0]["session_id"] == "s1"
    assert "status" in payload["coverage"]
