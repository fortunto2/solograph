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

```bash
claude mcp add -s project solograph -- uvx solograph
```

Or add manually to `.mcp.json`:
```json
{
  "mcpServers": {
    "solograph": {
      "command": "uvx",
      "args": ["solograph"]
    }
  }
}
```

### CLI

```bash
solograph-cli init ~/my-projects       # First-time setup (scan + build graph)
solograph-cli init ~/my-projects --deep # + imports, calls, inheritance
solograph-cli scan                     # Re-scan projects into graph
solograph-cli scan --deep              # + imports, calls, inheritance (tree-sitter)
solograph-cli stats                    # Graph statistics
solograph-cli explain my-app           # Architecture overview
solograph-cli xray ~/my-projects       # Portfolio X-Ray (all projects at once)
solograph-cli diagram my-app           # Mermaid diagram
solograph-cli query "MATCH (n) RETURN n LIMIT 5"
solograph-cli web-search "query"       # Web search via SearXNG/Tavily
```

Install globally:
```bash
uv tool install solograph              # → solograph + solograph-cli in PATH
```

### Quick Start

```bash
# 1. Install
uv tool install solograph

# 2. Init — creates ~/.codegraph/, scans projects, builds graph
solograph-cli init ~/my-projects

# 3. Add MCP to Claude Code
claude mcp add -s project solograph -- uvx solograph

# 4. Done — MCP tools available in Claude Code
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEGRAPH_DB_PATH` | `~/.codegraph/codegraph.db` | FalkorDB code graph path |
| `CODEGRAPH_REGISTRY` | `~/.codegraph/registry.yaml` | Project registry path |
| `CODEGRAPH_SCAN_PATH` | `~/projects` | Where to scan for projects |
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

## Graph Schema

### Nodes

| Node | Key Properties | Source |
|------|---------------|--------|
| `Project` | name, stack, path | `registry.yaml` |
| `File` | path, lang, lines, project | tree-sitter scan |
| `Symbol` | name, kind (class/function/method), file, line | tree-sitter AST |
| `Package` | name, version, source (npm/pip/spm/gradle) | manifest files |
| `Session` | session_id, project_name, started_at, slug | `.claude/` history |

### Edges

| Edge | From → To | Description |
|------|-----------|-------------|
| `HAS_FILE` | Project → File | Project contains file |
| `DEFINES` | File → Symbol | File defines symbol |
| `IMPORTS` | File → File/Package | Import relationship |
| `CALLS` | File → Symbol | File calls symbol |
| `INHERITS` | Symbol → Symbol | Class inheritance |
| `DEPENDS_ON` | Project → Package | Package dependency |
| `MODIFIED` | Session → File | Git history (lines added/removed) |
| `TOUCHED` / `EDITED` / `CREATED` | Session → File | Session file operations |
| `IN_PROJECT` | Session → Project | Session belongs to project |

### Example Cypher Queries

```cypher
-- Hub files (most imported)
MATCH (f:File)<-[:IMPORTS]-(other:File)
RETURN f.path, COUNT(other) AS importers
ORDER BY importers DESC LIMIT 10

-- Shared packages across projects
MATCH (p1:Project)-[:DEPENDS_ON]->(pkg:Package)<-[:DEPENDS_ON]-(p2:Project)
WHERE p1.name <> p2.name
RETURN pkg.name, COLLECT(DISTINCT p1.name) AS projects

-- Impact analysis: what breaks if I change this file?
MATCH (f:File {path: 'lib/utils.ts'})<-[:IMPORTS*1..3]-(dep:File)
RETURN dep.path

-- Most edited files (from session history)
MATCH (s:Session)-[:EDITED]->(f:File)
RETURN f.path, COUNT(s) AS sessions
ORDER BY sessions DESC LIMIT 10

-- Files touched by sessions in a project
MATCH (s:Session {project_name: 'my-app'})-[r]->(f:File)
RETURN f.path, type(r) AS action, COUNT(s) AS times
ORDER BY times DESC
```

## Storage

- **Code graph:** `~/.codegraph/codegraph.db` (FalkorDB)
- **Session vectors:** `~/.codegraph/sessions_vectors/graph.db` (FalkorDB)
- **KB vectors:** `{KB_PATH}/.codegraph/kb/graph.db` (FalkorDB)
- **Project vectors:** `{project_path}/.codegraph/falkordb/graph.db` (per-project FalkorDB)

## Part of Solo Factory

Solograph is the MCP backend for [**Solo Factory**](https://github.com/fortunto2/solo-factory) — 9 skills and 3 agents for shipping startups faster. [PyPI](https://pypi.org/project/solograph/)

Install skills + MCP together:
```bash
# Option 1: Skills for any agent (Claude Code, Cursor, Copilot, Gemini CLI, etc.)
npx skills add fortunto2/solo-factory --all

# Option 2: Claude Code plugin (skills + agents + MCP auto-start)
claude plugin marketplace add fortunto2/solo-factory
claude plugin install solo --scope user
```

Or use solograph standalone — just add to `.mcp.json` as shown above.

## License

MIT
