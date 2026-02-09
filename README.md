# codegraph-mcp

Code intelligence MCP server for Claude Code. Multi-project code graph, semantic search, session history, knowledge base, web search.

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
uv add codegraph-mcp[mlx]
```

## Install

```bash
uv add codegraph-mcp
# or
pip install codegraph-mcp
```

## Usage

### MCP Server (for Claude Code)

Add to `.mcp.json`:
```json
{
  "mcpServers": {
    "codegraph": {
      "command": "uvx",
      "args": ["codegraph-mcp"]
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

## Storage

- **Code graph:** `~/.codegraph/codegraph.db` (FalkorDB)
- **Session vectors:** `~/.codegraph/sessions_vectors/graph.db` (FalkorDB)
- **KB vectors:** `{KB_PATH}/.codegraph/kb/graph.db` (FalkorDB)
- **Project vectors:** `{project_path}/.codegraph/falkordb/graph.db` (per-project FalkorDB)

## License

MIT
