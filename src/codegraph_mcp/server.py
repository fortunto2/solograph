#!/usr/bin/env python3
"""
Solograph MCP Server — code intelligence, knowledge base, sessions, sources.

13 tools for Claude Code. Web search is NOT here: it lives on the standalone
searxng MCP server, so a broken import in this module cannot take search down
with it, and so search runs without FalkorDB or embeddings.

One codebase, several stores. Everything below is per-instance, so the same
server serves the personal graph and a client one from separate registrations:

  CODEGRAPH_DB_PATH     — FalkorDB path (default: ~/.solo/codegraph.db)
  CODEGRAPH_REGISTRY    — registry.yaml path (default: auto-detect)
  KB_PATH               — Knowledge base root (markdown files)
  CLAUDE_SESSIONS_DIR   — Claude config dir whose projects/ holds the session
                          logs (default: ~/.claude; e.g. ~/.claude-epiphan)
  CODEGRAPH_SESSIONS_DB — where the session index lives (default: ~/.solo/sessions)

Run:
  solograph                         # via entry point
  uvx solograph                     # via uvx
  uv run solograph                  # via uv
"""

import os
import sys
from pathlib import Path

import httpx
from mcp.server import MCPServer

mcp = MCPServer("solograph")

# Redirect print() to stderr — MCP uses stdout for JSON-RPC
_real_stdout = sys.stdout

# ── Config from env vars ──────────────────────────────────────────

CODEGRAPH_DB_PATH = os.environ.get("CODEGRAPH_DB_PATH", str(Path.home() / ".solo" / "codegraph.db"))
CODEGRAPH_REGISTRY = os.environ.get("CODEGRAPH_REGISTRY", "")
KB_PATH = os.environ.get("KB_PATH", "")


def _detect_kb_path() -> str:
    """Auto-detect KB path if not set via env var.

    Checks CWD for markdown files with YAML frontmatter (--- header).
    Falls back to empty string (KB disabled).
    """
    if KB_PATH:
        return KB_PATH
    cwd = Path.cwd()
    # Check if CWD has .solo/kb/ (indexed KB) or markdown files with frontmatter
    if (cwd / ".solo" / "kb").exists():
        return str(cwd)
    # Quick check: any .md file with frontmatter in top-level dirs
    for md in list(cwd.glob("*/*.md"))[:5]:
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")[:50]
            if text.startswith("---"):
                return str(cwd)
        except OSError:
            continue
    return ""


# ── Lazy singletons ──────────────────────────────────────────────

_kb = None
_sessions = None
_source_idx = None
_graph = None
_graph_db = None


def _get_kb():
    global _kb
    if _kb is None:
        kb_path = _detect_kb_path()
        if not kb_path:
            return None
        from codegraph_mcp.kb import KnowledgeBase

        _kb = KnowledgeBase(kb_path)
    return _kb


def _get_sessions():
    global _sessions
    if _sessions is None:
        from codegraph_mcp.session_search import SessionSearch

        _sessions = SessionSearch()
    return _sessions


def _get_source_index():
    global _source_idx
    if _source_idx is None:
        from codegraph_mcp.sources import SourceIndex

        _source_idx = SourceIndex()
    return _source_idx


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


# ── Tools ────────────────────────────────────────────────────────


@mcp.tool()
def kb_search(
    query: str,
    n_results: int = 5,
    doc_type: str | None = None,
    expand_graph: bool = False,
) -> list[dict]:
    """Full-text search over the knowledge base (markdown with YAML frontmatter).

    BM25 ranking with Russian and English stemming: "графами" finds "граф".
    Matching is lexical, so use the words the document would use, in its
    language. New and edited files are picked up on every call. `score` is
    BM25: higher is better, comparable only within one answer.

    Args:
        query: Search words (e.g. "privacy architecture", "агент память")
        n_results: Number of results (default 5)
        doc_type: Filter by frontmatter `type` (substring match)
        expand_graph: Add knowledge-graph neighbours of the top hits (1-hop)
    """
    kb = _get_kb()
    if kb is None:
        return [{"error": "KB not configured. Set KB_PATH env var."}]
    return kb.search(query, n_results=n_results, doc_type=doc_type, expand_graph=expand_graph)


@mcp.tool()
def session_search(
    query: str,
    n_results: int = 5,
    project: str | None = None,
) -> dict:
    """Full-text search over Claude Code session history — what was asked and answered.

    For "how did I solve X?", "when did I work on Y?", "what did we decide about Z?".
    Searches every prompt and every answer, BM25-ranked, Russian and English
    stemming. Matching is lexical: use the words the conversation would have
    used, and try the other language if the first returns nothing — sessions
    are mostly in Russian. Includes sessions Claude Code has already deleted
    from disk (it keeps transcripts 30 days); those come back marked
    `transcript: deleted`. Live ones carry a `resume` command.

    Args:
        query: Search words (e.g. "hubspot webhook", "транскрипция midi")
        n_results: Number of results (default 5)
        project: Filter by project name, substring (e.g. "life2film")
    """
    return _get_sessions().search(query, n_results=n_results, project=project)


@mcp.tool()
def codegraph_query(query: str) -> list[dict]:
    """Execute a read-only Cypher query against the code intelligence graph.

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
    # Read-only: one mistaken DETACH DELETE from an agent would empty the graph,
    # and nothing a question needs is a write.
    result = graph.ro_query(query)
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
def codegraph_repomap(project: str, limit: int = 20) -> dict:
    """Generate a YAML-formatted repository map showing the most important files and their symbols.

    Helps understand the global structure of a project without reading all files.

    Args:
        project: Project name (e.g. "my-app")
        limit: Maximum number of top files to include (default 20)
    """
    graph = _get_graph()
    from codegraph_mcp.output.repomap import generate_repomap

    result = generate_repomap(graph, project, max_files=limit)
    return {"repomap": result}


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


# ── Random User-Agent pool for web_fetch ────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
]


@mcp.tool()
async def web_fetch(
    url: str,
    max_length: int = 20000,
    extract_text: bool = True,
    timeout: int = 30,
) -> dict:
    """Fetch a URL with browser-like headers to bypass basic bot protection.

    Returns page content as plain text (HTML tags stripped) or raw HTML.
    Handles redirects, cookies, and common anti-bot headers automatically.

    Args:
        url: URL to fetch
        max_length: Max content length in chars (default 20000)
        extract_text: Strip HTML tags and return plain text (default true)
        timeout: Request timeout in seconds (default 30)
    """
    import random
    import re

    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=10,
        ) as client:
            resp = await client.get(url, headers=headers)

            if resp.status_code >= 400:
                return {
                    "url": str(resp.url),
                    "status": resp.status_code,
                    "error": f"HTTP {resp.status_code}",
                    "content": resp.text[:500],
                }

            content = resp.text
            if extract_text:
                # Strip script/style blocks, then all tags
                content = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<[^>]+>", " ", content)
                content = re.sub(r"\s+", " ", content).strip()

            return {
                "url": str(resp.url),
                "status": resp.status_code,
                "content_type": resp.headers.get("content-type", ""),
                "content": content[:max_length],
                "truncated": len(content) > max_length,
            }
    except httpx.TimeoutException:
        return {"url": url, "error": "Timeout", "status": 0}
    except httpx.ConnectError as e:
        return {"url": url, "error": f"Connection failed: {e}", "status": 0}
    except Exception as e:
        return {"url": url, "error": str(e), "status": 0}


@mcp.tool()
def source_search(
    query: str,
    source: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """Full-text search over indexed external sources (YouTube, Telegram, ProductHunt, TrustMRR).

    BM25 with Russian and English stemming; matching is lexical, so use the
    words the content would use. YouTube hits are the best transcript chunk per
    video, with chapter, timecode and a URL that opens at that moment. `score`
    is BM25 (ProductHunt upvotes add up to 15%): higher is better.

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

    Tags are assigned at indexing from topic keywords in the title and the
    start of the description. Shared across videos — enables topic clustering.

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

    One index at ~/.solo/sources/search.db. YouTube sources include the
    video/chunk breakdown (videos, video_chunks fields).
    """
    idx = _get_source_index()
    return idx.list_sources()


# ── Entry point ──────────────────────────────────────────────────


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
