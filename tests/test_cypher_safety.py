"""Names reach Cypher as string literals; they must not be able to end them.

Found in review on 26 Oct: project, file and symbol names were pasted into
queries as '{name}', a few with a hand-rolled .replace("'", "\\'") that missed
backslashes. A project called "Rustam's app" made codegraph_explain fail with a
Cypher syntax error (reproduced). A name could also close the literal early; the
payloads tried did not manage a write, but nothing stopped one. codegraph_query
itself ran any Cypher with write access, so one bad agent query could empty the
graph.
"""

from __future__ import annotations

import asyncio

import pytest
from redislite.falkordb_client import FalkorDB

import codegraph_mcp.server as server
from codegraph_mcp.db import cypher_str
from codegraph_mcp.output.explain import explain_project
from codegraph_mcp.query.search import callers_of, files_by_project


@pytest.fixture
def graph(tmp_path):
    g = FalkorDB(str(tmp_path / "g.db")).select_graph("codegraph")
    g.query("CREATE (:Project {name: $n, stack: 'python'})", {"n": "Rustam's app\\x"})
    g.query(
        "MATCH (p:Project {name: $n}) CREATE (p)-[:HAS_FILE]->(:File {project: $n, path: 'a.py', lang: 'python', lines: 3})",
        {"n": "Rustam's app\\x"},
    )
    return g


def test_quote_and_backslash_are_escaped():
    assert cypher_str("it's") == "'it\\'s'"
    assert cypher_str("a\\") == "'a\\\\'"


def test_a_name_with_a_quote_and_a_backslash_is_found(graph):
    data = explain_project(graph, "Rustam's app\\x")
    assert isinstance(data, dict), data
    assert data["counts"]["files"] == 1
    assert files_by_project(graph, "Rustam's app\\x")[0]["path"] == "a.py"


def test_a_name_cannot_close_the_literal_and_write(graph):
    for evil in ["x'}) DETACH DELETE p //", "x\\'}) DETACH DELETE p //"]:
        explain_project(graph, evil)
        callers_of(graph, evil, project=evil)
    assert graph.query("MATCH (p:Project) RETURN count(p)").result_set[0][0] == 1


def test_codegraph_query_refuses_writes(graph, monkeypatch):
    monkeypatch.setattr(server, "_graph", graph)
    with pytest.raises(Exception, match="read-only"):
        asyncio.run(server.mcp.call_tool("codegraph_query", {"query": "MATCH (p:Project) DETACH DELETE p"}))
    assert graph.query("MATCH (p:Project) RETURN count(p)").result_set[0][0] == 1
