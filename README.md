# codegraph-mcp

Code intelligence MCP server for Claude Code. Multi-project code graph, semantic search, session history, knowledge base, web search.

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
codegraph explain FaceAlarm # Architecture overview
codegraph query "MATCH ..." # Raw Cypher
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEGRAPH_DB_PATH` | `~/.codegraph/codegraph.db` | FalkorDB path |
| `CODEGRAPH_REGISTRY` | auto-detect | registry.yaml path |
| `KB_PATH` | (none) | Knowledge base root |
| `KB_CHROMA_PATH` | `{KB_PATH}/.embeddings/chroma_db` | ChromaDB for KB |
| `TAVILY_API_URL` | `http://localhost:8013` | Tavily-compatible search URL |
| `TAVILY_API_KEY` | (none) | API key for Tavily |

## 11 MCP Tools

- `codegraph_query` — Cypher queries
- `codegraph_stats` — graph statistics
- `codegraph_explain` — architecture overview
- `codegraph_shared` — shared packages
- `project_code_search` — semantic code search
- `project_code_reindex` — reindex project
- `session_search` — Claude Code session history
- `project_info` — project registry
- `kb_search` — knowledge base search
- `web_search` — web search (Tavily/SearXNG)

## License

MIT
