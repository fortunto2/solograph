# solograph

Code intelligence MCP server for Claude Code: a tree-sitter code graph across all your projects, plus full-text search over your Claude Code session history, a markdown knowledge base, and indexed external sources (YouTube, ProductHunt, TrustMRR, Telegram).

> PyPI: `pip install solograph` / `uvx solograph`

Two stores, both embedded, no Docker, no API keys, no embedding models:

- **Code graph** — FalkorDBLite. Projects, files, symbols, packages, imports, calls, inheritance, queried with Cypher (read-only).
- **Search** — SQLite FTS5 with BM25 ranking and Snowball stemming for Russian and English. Each index syncs against the disk before it answers, so there is no reindex step to forget.

Search was vector-based until v0.7. It was replaced after measuring it: the session index had zero overlap with the transcripts on disk, KB search took 14 s cold and scored results by rank position, and source search hung past two minutes. The FTS version answers in tens of milliseconds and finds 32 of 39 sessions in a known-answer test that the vector index found none of. The environment went from 862 MB to under 150 MB.

## Install

```bash
uv tool install solograph              # → solograph + solograph-cli in PATH
claude mcp add -s user solograph -- uvx solograph
solograph-cli init ~/my-projects --deep # registry + code graph with imports/calls
```

## 13 MCP tools

| Tool | What it answers |
|------|-----------------|
| `codegraph_query` | Read-only Cypher over the code graph |
| `codegraph_stats` | Counts of projects, files, symbols, packages, sessions, edges |
| `codegraph_explain` | Architecture overview of one project: stack, layers, hub files, dependencies |
| `codegraph_repomap` | Aider-style map of a project's most important files and symbols |
| `codegraph_shared` | Packages used by more than one project |
| `project_info` | Registry entry for a project, or the whole registry |
| `session_search` | What was asked and answered in past Claude Code sessions, including ones Claude Code already deleted |
| `kb_search` | Markdown knowledge base, by frontmatter title, type, tags and body |
| `source_search` | Indexed YouTube transcripts (timestamped links), ProductHunt, TrustMRR, Telegram |
| `source_list` / `source_tags` / `source_related` | What is indexed, topic tags, videos sharing tags |
| `web_fetch` | Fetch a URL as text |

Web search is not here: it lives in the separate [searxng MCP server](https://github.com/fortunto2/searxng-docker-tavily-adapter), so a broken import in one cannot take the other down.

### How the search indexes behave

- **Lexical, stemmed.** "графами" finds "граф", "indexing" finds "index". There is no cross-language matching: a Russian conversation is found by Russian words.
- **Never stale.** Every search stats the source files and re-reads only what changed. Transcripts are append-only, so a grown one is read from its last byte offset — live transcripts reach hundreds of MB, and a resync costs ~10 ms.
- **Sessions are archived.** Claude Code deletes transcripts after 30 days (`cleanupPeriodDays`). The index keeps their text and marks them `transcript: deleted`; live ones come with a `claude --resume` command. Batch runs started from temp directories (`claude -p` in `/tmp`, agent scratchpads) are left out.
- **Private folders stay private.** A session whose working directory is inside a folder holding `.solograph-private` (or Spotlight's `.metadata_never_index`) is never indexed, and is purged if it was.
- **Scores are BM25** — higher is better, comparable within one answer only.

## CLI

```bash
solograph-cli scan --deep --incremental     # rescan what git says changed
solograph-cli registry                      # merge new checkouts into the registry
solograph-cli explain my-app                # architecture overview
solograph-cli xray ~/my-projects            # portfolio X-ray
solograph-cli diagram my-app                # Mermaid diagram
solograph-cli query "MATCH (p:Project) RETURN p.name LIMIT 5"
solograph-cli session-search "webhook retry" -p my-app
solograph-cli scan-sessions                 # sessions into the graph (Session→File edges) and the index
solograph-cli index-kb                      # warm the KB index (searches also sync on their own)
solograph-cli source-search "pricing tiers" -s youtube
solograph-cli index-youtube -c GregIsenberg -n 10
```

`scripts/refresh-changed.sh` is the nightly job: registry merge, incremental deep scan of every project, session-index sync. It is meant for launchd or cron.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEGRAPH_DB_PATH` | `~/.solo/codegraph.db` | Code graph |
| `CODEGRAPH_REGISTRY` | `~/.solo/registry.yaml` | Project registry |
| `CODEGRAPH_SCAN_PATH` | `~/startups/active` | Comma-separated directories `registry` looks for checkouts in |
| `KB_PATH` | (none) | Knowledge base root; index at `$KB_PATH/.solo/kb/search.db` |
| `CLAUDE_SESSIONS_DIR` | `~/.claude` | Claude config dir whose `projects/` holds transcripts |
| `CODEGRAPH_SESSIONS_DB` | `~/.solo/sessions` | Session index directory (`search.db`) |
| `SOURCES_ROOT` | `~/.solo/sources` | Sources index (`search.db`) and YouTube VTT cache |

Every store is per-registration, so one install serves several profiles (say a personal one and a client one) from separate MCP entries with different env.

## Graph schema

| Node | Key properties |
|------|----------------|
| `Project` | name, stack, path |
| `File` | path, lang, lines, project |
| `Symbol` | name, kind, file, line |
| `Package` | name, version, source (npm/pip/spm/gradle/cargo) |
| `Session` | session_id, project_name, started_at |

| Edge | From → To |
|------|-----------|
| `HAS_FILE` | Project → File |
| `DEFINES` | File → Symbol |
| `IMPORTS` | File → File / Package |
| `CALLS` | File → Symbol |
| `INHERITS` | Symbol → Symbol |
| `DEPENDS_ON` | Project → Package |
| `MODIFIED` | Session → File (git history) |
| `TOUCHED` / `EDITED` / `CREATED` | Session → File |
| `IN_PROJECT` | Session → Project |

```cypher
-- What breaks if I change this file?
MATCH (f:File {path: 'lib/utils.ts'})<-[:IMPORTS*1..3]-(dep:File) RETURN dep.path

-- Most edited files across sessions
MATCH (s:Session)-[:EDITED]->(f:File) RETURN f.path, COUNT(s) AS n ORDER BY n DESC LIMIT 10
```

## Sources

One FTS index at `$SOURCES_ROOT/search.db`, documents keyed `<source>/<id>`, plus `videos` and `video_tags` tables.

- **YouTube** — `index-youtube` discovers videos via SearXNG, fetches metadata and VTT with yt-dlp, chunks by chapter. Each chunk is a document, so a hit links to its second (`&t=140s`); a search returns each video once, at its best chunk. Topic tags come from keyword matching against the `TOPICS` table in `sources.py` (title hits count double).
- **ProductHunt / TrustMRR** — scrapers into the same index; upvotes nudge ranking by at most 15%.
- **Telegram** — `scripts/index_telegram.py` in the consuming repo.

`solograph-search` serves the sources index over HTTP (`GET /search?q=...&source=...&n=5`, `/sources`, `/health`, port `SOLOGRAPH_SEARCH_PORT`, default 8002) for a SearXNG engine.

## Part of Solo Factory

Solograph is the MCP backend for [Solo Factory](https://github.com/fortunto2/solo-factory). It also works standalone.

## License

MIT
