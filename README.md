# solograph

Code intelligence MCP server for Claude Code. Multi-project code graph, semantic search, session history, knowledge base, web search.

> PyPI: `pip install solograph` / `uvx solograph`

All vector search powered by **FalkorDB** (embedded, no Docker). No ChromaDB dependency.

## Embeddings

Two backends, both produce **384-dimensional** vectors:

| Backend | Model | Platform | Languages |
|---------|-------|----------|-----------|
| **MLX** (primary) | `multilingual-e5-small-mlx` | Apple Silicon | RU + EN |
| **sentence-transformers** (fallback) | `all-MiniLM-L6-v2` | Any | EN |

Auto-detects Apple Silicon → uses MLX. Falls back to sentence-transformers on other platforms.

Install MLX support (optional):
```bash
uv add solograph[mlx]
```

## Install

```bash
uv add solograph
# or
pip install solograph
```

## Usage

### MCP Server (for Claude Code)

Add to `.mcp.json`:
```json
{
  "mcpServers": {
    "codegraph": {
      "command": "uvx",
      "args": ["solograph"]
    }
  }
}
```

### CLI

```bash
codegraph scan              # Build code graph
codegraph stats             # Graph statistics
codegraph explain my-app    # Architecture overview
codegraph query "MATCH ..." # Raw Cypher
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEGRAPH_DB_PATH` | `~/.codegraph/codegraph.db` | FalkorDB code graph path |
| `CODEGRAPH_REGISTRY` | auto-detect | registry.yaml path |
| `KB_PATH` | (none) | Knowledge base root (markdown files with YAML frontmatter) |
| `TAVILY_API_URL` | `http://localhost:8013` | Tavily-compatible search URL |
| `TAVILY_API_KEY` | (none) | API key for Tavily |

## 11 MCP Tools

- `codegraph_query` — Cypher queries against code graph
- `codegraph_stats` — graph statistics (projects, files, symbols, packages)
- `codegraph_explain` — architecture overview of a project
- `codegraph_shared` — packages shared across projects
- `project_code_search` — semantic code search (auto-indexes on first call)
- `project_code_reindex` — reindex project code into FalkorDB vectors
- `session_search` — Claude Code session history search
- `project_info` — project registry info
- `kb_search` — knowledge base semantic search
- `web_search` — web search (Tavily/SearXNG)

## Web Search

The `web_search` tool connects to any **Tavily-compatible API**. Works great with self-hosted [SearXNG + Tavily Adapter](https://github.com/fortunto2/searxng-docker-tavily-adapter) — private, no API keys, smart engine routing.

```bash
# Self-hosted (Docker, 1 minute setup)
git clone https://github.com/fortunto2/searxng-docker-tavily-adapter.git
cd searxng-docker-tavily-adapter
cp config.example.yaml config.yaml
docker compose up -d
# → http://localhost:8013/search (Tavily API)
# → http://localhost:8999 (SearXNG UI)
```

Or use [Tavily API](https://tavily.com) directly — set `TAVILY_API_URL=https://api.tavily.com` and `TAVILY_API_KEY`.

Smart engine routing auto-selects search engines by query type:
- **tech**: github, stackoverflow (keywords: python, react, code)
- **academic**: arxiv, google scholar (keywords: research, paper)
- **product**: brave, reddit, app stores (keywords: app, competitor, pricing)
- **news**: google news (keywords: news, latest, trend)
- **general**: google, duckduckgo, brave (default)

## Storage

- **Code graph:** `~/.codegraph/codegraph.db` (FalkorDB)
- **Session vectors:** `~/.codegraph/sessions_vectors/graph.db` (FalkorDB)
- **KB vectors:** `{KB_PATH}/.codegraph/kb/graph.db` (FalkorDB)
- **Project vectors:** `{project_path}/.codegraph/falkordb/graph.db` (per-project FalkorDB)

## License

MIT
