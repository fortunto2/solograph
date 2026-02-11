#!/usr/bin/env python3
"""
Solograph MCP Server — code intelligence, knowledge base, sessions, sources, web search.

13 tools for Claude Code. Configure via environment variables:
  CODEGRAPH_DB_PATH     — FalkorDB path (default: ~/.solo/codegraph.db)
  CODEGRAPH_REGISTRY    — registry.yaml path (default: auto-detect)
  KB_PATH               — Knowledge base root (markdown files)
  TAVILY_API_URL        — Tavily-compatible search URL (default: http://localhost:8013)
  TAVILY_API_KEY        — API key for Tavily (ignored for SearXNG)

Run:
  solograph                         # via entry point
  uvx solograph                     # via uvx
  uv run solograph                  # via uv
"""

import os
import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("solograph")

# Redirect print() to stderr — MCP uses stdout for JSON-RPC
_real_stdout = sys.stdout

# ── Config from env vars ──────────────────────────────────────────

CODEGRAPH_DB_PATH = os.environ.get("CODEGRAPH_DB_PATH", str(Path.home() / ".solo" / "codegraph.db"))
CODEGRAPH_REGISTRY = os.environ.get("CODEGRAPH_REGISTRY", "")
KB_PATH = os.environ.get("KB_PATH", "")
TAVILY_API_URL = os.environ.get("TAVILY_API_URL", "http://localhost:8013")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# ── Lazy singletons ──────────────────────────────────────────────

_kb = None
_session_idx = None
_source_idx = None
_project_idx = None
_graph = None
_graph_db = None


def _get_kb():
    global _kb
    if _kb is None:
        if not KB_PATH:
            return None
        from codegraph_mcp.kb import KnowledgeEmbeddings
        sys.stdout = sys.stderr
        try:
            _kb = KnowledgeEmbeddings(KB_PATH)
        finally:
            sys.stdout = _real_stdout
    return _kb


def _get_session_index():
    global _session_idx
    if _session_idx is None:
        from codegraph_mcp.vectors.session_index import SessionIndex
        _session_idx = SessionIndex()
    return _session_idx


def _get_source_index():
    global _source_idx
    if _source_idx is None:
        from codegraph_mcp.vectors.source_index import SourceIndex
        _source_idx = SourceIndex()
    return _source_idx


def _get_project_index():
    global _project_idx
    if _project_idx is None:
        from codegraph_mcp.vectors.project_graph_index import ProjectGraphIndex
        sys.stdout = sys.stderr
        try:
            _project_idx = ProjectGraphIndex(backend="st")
        finally:
            sys.stdout = _real_stdout
    return _project_idx


def _get_graph():
    global _graph, _graph_db
    if _graph is None:
        from codegraph_mcp.db import get_db, get_graph
        _graph_db = get_db(Path(CODEGRAPH_DB_PATH).expanduser())
        _graph = get_graph(_graph_db)
        # Auto-refresh registry → project stacks always current
        _auto_refresh_registry()
    return _graph


def _auto_refresh_registry():
    """Ingest registry.yaml into graph so project stacks are always current."""
    registry_path = _get_registry_path()
    if not registry_path:
        return
    try:
        from codegraph_mcp.scanner.registry import ingest_projects, scan_registry
        projects = scan_registry(registry_path)
        ingest_projects(_graph, projects)
    except Exception as exc:
        print(f"Auto-refresh registry error: {exc}", file=sys.stderr)


def _get_registry_path() -> Path | None:
    """Find registry.yaml from env or auto-detect."""
    if CODEGRAPH_REGISTRY:
        p = Path(CODEGRAPH_REGISTRY).expanduser()
        return p if p.exists() else None
    # Auto-detect common locations
    candidates = [
        Path.home() / ".solo" / "registry.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _resolve_project_path(name: str) -> Path | None:
    """Look up project path by name (registry) or by direct path.

    Accepts:
      - Project name: "my-app" → looks up in registry.yaml
      - Absolute path: "/Users/x/projects/my-app" → uses directly
      - Home-relative: "~/projects/my-app" → expands and uses
    """
    # Direct path — bypass registry entirely
    if "/" in name or name.startswith("~"):
        p = Path(name).expanduser().resolve()
        return p if p.is_dir() else None

    # Registry lookup
    import yaml

    registry_path = _get_registry_path()
    if not registry_path:
        return None

    with open(registry_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for p in data.get("projects", []):
        if p["name"].lower() == name.lower():
            proj_path = Path(p["path"]).expanduser()
            return proj_path if proj_path.exists() else None
    return None


def _auto_index_if_needed(idx, project: str) -> bool:
    """Auto-index a project if it has no vector index."""
    # Resolve name for path-based projects
    proj_name = project
    if "/" in project or project.startswith("~"):
        p = Path(project).expanduser().resolve()
        if p.is_dir():
            proj_name = p.name
            idx._paths[proj_name] = p
        else:
            return False

    db_dir = idx._db_dir(proj_name)
    if db_dir.exists():
        try:
            graph = idx._get_graph(proj_name)
            result = graph.query("MATCH (c:Chunk) RETURN count(c)")
            if result.result_set and result.result_set[0][0] > 0:
                return False
        except Exception:
            pass

    proj_path = _resolve_project_path(project)
    if not proj_path:
        return False

    sys.stdout = sys.stderr
    try:
        print(f"Auto-indexing {proj_name}...", file=sys.stderr)
        idx.index_project(proj_path, proj_name)
    finally:
        sys.stdout = _real_stdout
    return True


# ── Tools ────────────────────────────────────────────────────────


@mcp.tool()
def kb_search(
    query: str,
    n_results: int = 5,
    doc_type: str | None = None,
    expand_graph: bool = False,
) -> list[dict]:
    """Semantic search over the knowledge base.

    Searches markdown documents with YAML frontmatter.
    Understands Russian and English. Use expand_graph=true to include
    knowledge graph neighbors (structurally related docs).

    Args:
        query: Search query (e.g. "privacy architecture", "API design patterns")
        n_results: Number of results (default 5)
        doc_type: Filter by type (depends on your KB schema)
        expand_graph: Expand results with knowledge graph neighbors (1-hop)
    """
    kb = _get_kb()
    if kb is None:
        return [{"error": "KB not configured. Set KB_PATH env var."}]

    filter_dict = {"type": doc_type} if doc_type else None
    sys.stdout = sys.stderr
    try:
        results = kb.search(query, n_results=n_results, filter_dict=filter_dict, expand_graph=expand_graph)
    finally:
        sys.stdout = _real_stdout

    output = []
    for i, meta in enumerate(results.get("metadatas", [])):
        entry = {
            "file": meta.get("file", ""),
            "title": meta.get("title", ""),
            "type": meta.get("type", ""),
            "tags": meta.get("tags", ""),
        }
        if i < len(results.get("distances", [])):
            entry["relevance"] = round(1 - results["distances"][i], 4)
        if i < len(results.get("documents", [])):
            entry["snippet"] = results["documents"][i][:300]
        if meta.get("graph_source"):
            entry["graph_source"] = meta["graph_source"]
        output.append(entry)

    return output


@mcp.tool()
def session_search(
    query: str,
    n_results: int = 5,
    project: str | None = None,
) -> list[dict]:
    """Semantic search over Claude Code chat session history.

    Finds past sessions by what was discussed or worked on.
    Useful for "how did I solve X?" or "when did I work on Y?" questions.

    Args:
        query: Search query (e.g. "knowledge graph implementation", "OCR receipt scanning")
        n_results: Number of results (default 5)
        project: Filter by project name (e.g. "my-app", "backend")
    """
    idx = _get_session_index()
    _auto_scan_sessions_if_empty(idx)
    return idx.search(query, n_results=n_results, project=project)


_sessions_auto_scanned = False


def _auto_scan_sessions_if_empty(idx):
    """Scan all Claude Code sessions into graph + vectors if index is empty."""
    global _sessions_auto_scanned
    if _sessions_auto_scanned:
        return
    _sessions_auto_scanned = True

    try:
        if idx.count() > 0:
            return
    except Exception:
        return

    try:
        graph = _get_graph()
        from codegraph_mcp.scanner.sessions import (
            ingest_session_files,
            ingest_sessions,
            link_sessions_to_projects,
            scan_all_sessions,
        )

        sessions, edges, summaries = [], [], []
        for s, e, sm in scan_all_sessions():
            sessions.append(s)
            edges.extend(e)
            summaries.append(sm)

        if sessions:
            sys.stdout = sys.stderr
            try:
                print(f"Auto-scanning {len(sessions)} sessions...", file=sys.stderr)
                ingest_sessions(graph, sessions)
                ingest_session_files(graph, edges)
                link_sessions_to_projects(graph)
                idx.upsert(summaries)
            finally:
                sys.stdout = _real_stdout
    except Exception as exc:
        print(f"Auto-scan sessions error: {exc}", file=sys.stderr)


@mcp.tool()
def codegraph_query(query: str) -> list[dict]:
    """Execute a raw Cypher query against the code intelligence graph.

    The graph contains: Project, File, Symbol, Package, Session nodes.
    Edges: HAS_FILE, DEFINES, DEPENDS_ON, MODIFIED, IN_PROJECT, TOUCHED, EDITED, CREATED,
           IMPORTS (File->File or File->Package), CALLS (File->Symbol), INHERITS (Symbol->Symbol).

    Example queries:
      - MATCH (p:Project) RETURN p.name, p.path LIMIT 10
      - MATCH (f:File {project: 'my-app'}) RETURN f.path, f.lang LIMIT 20
      - MATCH (p:Project)-[:DEPENDS_ON]->(pkg:Package) WHERE pkg.name = 'react' RETURN p.name
      - MATCH (s:Session)-[:EDITED]->(f:File) RETURN f.path, COUNT(s) ORDER BY COUNT(s) DESC LIMIT 10

    Args:
        query: Cypher query string
    """
    graph = _get_graph()
    result = graph.query(query)
    rows = []
    for row in result.result_set:
        if len(row) == 1:
            rows.append({"value": row[0]})
        else:
            rows.append({f"col_{i}": v for i, v in enumerate(row)})
    return rows


@mcp.tool()
def codegraph_stats() -> dict:
    """Get code intelligence graph statistics.

    Returns counts of projects, files, symbols, packages, sessions,
    and edge type breakdown.
    """
    graph = _get_graph()
    from codegraph_mcp.db import graph_stats as _graph_stats
    return _graph_stats(graph)


@mcp.tool()
def codegraph_explain(project: str) -> dict:
    """Architecture overview of a project from the code graph.

    Returns structured data: stack, languages, directory layers, key patterns
    (mixins, base classes, CRUD schemas), top dependencies, and hub files.

    Args:
        project: Project name (e.g. "my-app", "backend-api")
    """
    graph = _get_graph()
    from codegraph_mcp.output.explain import explain_project
    result = explain_project(graph, project)
    if isinstance(result, str):
        return {"error": result}
    return result


@mcp.tool()
def codegraph_shared() -> list[dict]:
    """Packages shared across multiple projects in the code graph.

    Returns list of packages with the projects that depend on them,
    sorted by number of projects (most shared first).
    Useful for finding common dependencies and potential shared infrastructure.
    """
    graph = _get_graph()
    from codegraph_mcp.query.search import shared_packages
    return shared_packages(graph)


@mcp.tool()
def project_info(name: str | None = None) -> list[dict] | dict:
    """Get project information from the registry.

    Without name: returns list of all active projects with stacks.
    With name: returns detailed info for one project.

    Args:
        name: Project name (e.g. "my-app"). Omit for full list.
    """
    import yaml

    registry_path = _get_registry_path()
    if not registry_path:
        return {"error": "Registry not found. Set CODEGRAPH_REGISTRY env var or run: solograph-cli scan"}

    with open(registry_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    projects = data.get("projects", [])

    if name is None:
        return [
            {
                "name": p["name"],
                "stacks": p.get("stacks", []),
                "status": p.get("status", "active"),
                "last_commit": p.get("last_commit", ""),
            }
            for p in projects
        ]

    for p in projects:
        if p["name"].lower() == name.lower():
            return p

    return {"error": f"Project '{name}' not found in registry"}


@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 10,
    engines: str | None = None,
    include_raw_content: bool = False,
) -> dict:
    """Search the web via SearXNG (Tavily-compatible API).

    Uses smart engine routing: auto-selects engines based on query keywords.
    Override with engines param for specific sources.

    Engine groups (auto-detected):
      - academic: arxiv, google scholar (keywords: research, paper, algorithm)
      - tech: github, stackoverflow (keywords: python, react, code, framework)
      - product: brave, reddit, app stores (keywords: app, competitor, pricing, vs)
      - news: google news, hacker news (keywords: news, latest, 2026, trend)
      - general: google, duckduckgo, brave, reddit (default fallback)

    Args:
        query: Search query
        max_results: Number of results (default 10)
        engines: Override engine selection (e.g. "reddit", "hacker news", "arxiv,google scholar")
        include_raw_content: Include full page content (up to 5000 chars per page)
    """
    payload = {
        "query": query,
        "max_results": max_results,
        "include_raw_content": include_raw_content,
    }
    if engines:
        payload["engines"] = engines

    headers = {}
    if TAVILY_API_KEY:
        headers["Authorization"] = f"Bearer {TAVILY_API_KEY}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{TAVILY_API_URL}/search", json=payload, headers=headers)
        if resp.status_code != 200:
            return {"error": f"Search returned {resp.status_code}", "detail": resp.text[:500]}
        return resp.json()


@mcp.tool()
def project_code_search(
    query: str,
    project: str | None = None,
    n_results: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:
    """Semantic search over project source code and documentation.

    Searches indexed project codebases (per-project FalkorDB vector DBs).
    Auto-indexes the project on first search if no index exists.
    Useful for finding code patterns, functions, classes, and docs across projects.

    Args:
        query: Search query (e.g. "authentication middleware", "API route handler")
        n_results: Number of results (default 5)
        project: Project name or path (e.g. "my-app", "~/projects/my-app"). Omit to search all.
        chunk_type: Filter by "code" or "doc"
    """
    idx = _get_project_index()

    # Resolve path-based project to name
    proj_name = project
    if project and ("/" in project or project.startswith("~")):
        p = Path(project).expanduser().resolve()
        if p.is_dir():
            proj_name = p.name
            idx._paths[proj_name] = p

    if proj_name:
        _auto_index_if_needed(idx, project)

    sys.stdout = sys.stderr
    try:
        return idx.search(query, project=proj_name, n_results=n_results, chunk_type=chunk_type)
    finally:
        sys.stdout = _real_stdout


@mcp.tool()
def project_code_reindex(
    project: str,
) -> dict:
    """Reindex a project's source code and docs into FalkorDB vectors.

    Call this after significant code changes to update the search index.
    Uses sentence-transformers backend (safe for memory).

    Args:
        project: Project name or path (e.g. "my-app", "~/projects/my-app")
    """
    proj_path = _resolve_project_path(project)
    if not proj_path:
        return {"error": f"Project '{project}' not found in registry. Pass an absolute path to index any project."}

    # Use directory name as project key when path is passed directly
    proj_name = proj_path.name if "/" in project or project.startswith("~") else project

    idx = _get_project_index()
    sys.stdout = sys.stderr
    try:
        stats = idx.index_project(proj_path, proj_name)
    finally:
        sys.stdout = _real_stdout
    stats["project"] = proj_name
    stats["path"] = str(proj_path)
    return stats


@mcp.tool()
def source_search(
    query: str,
    source: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """Search indexed external sources (Telegram, YouTube, etc.).

    Each source is stored in its own FalkorDB graph.
    YouTube videos are chunked by chapters — results include chapter name and timecode.
    Without source filter, searches all sources and merges by relevance.

    Args:
        query: Search query (e.g. "startup idea", "revenue growth")
        source: Filter by source name (e.g. "telegram", "youtube"). Omit to search all.
        n_results: Number of results (default 5)
    """
    idx = _get_source_index()
    return idx.search(query, source=source, n_results=n_results)


@mcp.tool()
def source_tags(source: str = "youtube") -> list[dict]:
    """List all auto-detected topics with video counts.

    Tags are assigned automatically via zero-shot embedding similarity
    during video indexing. Shared across videos — enables topic clustering.

    Args:
        source: Source name (default: "youtube")
    """
    idx = _get_source_index()
    return idx.list_tags(source)


@mcp.tool()
def source_related(video_url: str, source: str = "youtube") -> list[dict]:
    """Find related videos by shared tags.

    Given a video URL, finds other videos that share auto-detected topics.
    Returns videos sorted by number of shared tags (most related first).

    Args:
        video_url: YouTube video URL or video ID
        source: Source name (default: "youtube")
    """
    import re

    # Extract video ID from URL or use as-is
    video_id = video_url
    m = re.search(r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})", video_url)
    if m:
        video_id = m.group(1)

    idx = _get_source_index()
    return idx.related_videos(source, video_id)


@mcp.tool()
def source_list() -> list[dict]:
    """List indexed external sources with document counts.

    Shows all source graphs under ~/.solo/sources/ with their sizes.
    YouTube sources include video/chunk breakdown (videos, video_chunks fields).
    """
    idx = _get_source_index()
    return idx.list_sources()


# ── Entry point ──────────────────────────────────────────────────

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
