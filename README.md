# solograph

MCP server for a portfolio of projects: symbol search across every repository you own, plus full-text search over your Claude Code session history, a markdown knowledge base, and indexed external sources (YouTube, ProductHunt, TrustMRR, Telegram).

> PyPI: `pip install solograph` / `uvx solograph`

Everything is SQLite with FTS5: BM25 ranking, Snowball stemming for Russian and English, no graph database, no daemon, no embedding models, no API keys. Each index checks the disk before it answers, so there is no reindex step to forget.

## What was cut, and why

solograph used to be a FalkorDB code graph with vector search on top. Both were measured on a real machine before they went:

- **Vectors (removed in 0.7).** The session index had zero overlap with the transcripts on disk, KB search took 14 s cold and scored results by rank position, source search hung past two minutes. FTS answers in tens of milliseconds and finds 32 of 39 sessions in a known-answer test the vector index found none of.
- **The structural graph (removed in 0.8).** 781k call edges plus imports, inheritance and session→file edges got zero structural questions from agents in four months: five calls in total, every one "list the projects" or stats. Grep answers those questions as well. On a 1,100-file Rust + Swift repository, `claude -p` with a dedicated code-graph MCP against plain Grep/Read gave the same accuracy in all 8 runs; the call-chain question cost $1.73 against $2.07 (16.5 tool calls against 25), the impact question $0.69 against $0.65.

What stays is what grep cannot do across repositories: one index of every project's definitions, packages and history. The environment went from 862 MB to 89 MB.

## Install

```bash
uv tool install solograph              # → solograph + solograph-cli in PATH
claude mcp add -s user solograph -- uvx solograph
solograph-cli init ~/my-projects       # registry + portfolio index
```

## 12 MCP tools

| Tool | What it answers |
|------|-----------------|
| `symbol_search` | Where something is defined, across every project: "montage profile" finds `resolve_montage_profile` and `MontageProfile` |
| `codegraph_explain` | What one project is made of: stack, languages, directory layers, patterns, dependencies, files with the most definitions |
| `codegraph_shared` | Packages used by more than one project — or every project using a given package |
| `codegraph_stats` | Counts: projects, files, symbols by kind, packages |
| `project_info` | Registry entry for a project, or the whole registry |
| `session_search` | What was asked and answered in past Claude Code sessions, including ones Claude Code already deleted |
| `kb_search` | Markdown knowledge base, by frontmatter title, type, tags and body |
| `source_search` | Indexed YouTube transcripts (timestamped links), ProductHunt, TrustMRR, Telegram |
| `source_list` / `source_tags` / `source_related` | What is indexed, topic tags, videos sharing tags |
| `web_fetch` | Fetch a URL as text |

Web search is not here: it lives in the separate [searxng MCP server](https://github.com/fortunto2/searxng-docker-tavily-adapter), so a broken import in one cannot take the other down.

### How the indexes behave

- **Symbols by their words.** Names are split on camelCase and snake_case (as codebase-memory-mcp's tokenizer does), so a phrase finds every spelling. All words rank above some words; an exact name comes first. Definitions only — functions, classes, protocols — from tree-sitter grammars for Python, Swift, TypeScript, JavaScript, Kotlin, Rust, Go, Java, Ruby, C, C++, PHP, HCL, Bash and SQL.
- **Lexical, stemmed.** "графами" finds "граф", "indexing" finds "index". No cross-language matching: a Russian conversation is found by Russian words.
- **Never stale.** A scan re-reads a file only when its mtime or size changed; a session search re-reads only transcripts that grew, from their last byte offset — live transcripts reach hundreds of MB, and a resync costs ~10 ms.
- **Sessions are archived.** Claude Code deletes transcripts after 30 days (`cleanupPeriodDays`). The index keeps their text and marks them `transcript: deleted`; live ones come with a `claude --resume` command. Batch runs started from temp directories (`claude -p` in `/tmp`, agent scratchpads) are left out.
- **Private folders stay private.** A session whose working directory is inside a folder holding `.solograph-private` (or Spotlight's `.metadata_never_index`) is never indexed, and is purged if it was.
- **Scores are BM25** — higher is better, comparable within one answer only.

## CLI

```bash
solograph-cli registry                      # merge new checkouts into the registry
solograph-cli scan [-p my-app]              # index what changed (every scan is incremental)
solograph-cli symbols "auth middleware"     # definitions across every project
solograph-cli explain my-app                # what a project is made of
solograph-cli shared [react]                # shared packages, or who uses one
solograph-cli deps my-app                   # one project's dependencies
solograph-cli session-search "webhook retry" -p my-app
solograph-cli scan-sessions                 # read new transcripts into the session index
solograph-cli index-kb                      # warm the KB index (searches also sync on their own)
solograph-cli source-search "pricing tiers" -s youtube
solograph-cli index-youtube -c GregIsenberg -n 10
```

`scripts/refresh-changed.sh` is the nightly job: registry merge, a scan of every project, session-index sync. It is meant for launchd or cron.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEGRAPH_DB_PATH` | `~/.solo/codegraph.db` | Only its directory is used now: the portfolio store is `portfolio.db` next to it |
| `SOLOGRAPH_PORTFOLIO_DB` | (none) | Names the portfolio store outright |
| `CODEGRAPH_REGISTRY` | `~/.solo/registry.yaml` | Project registry |
| `CODEGRAPH_SCAN_PATH` | `~/startups/active` | Comma-separated directories `registry` looks for checkouts in |
| `KB_PATH` | (none) | Knowledge base root; index at `$KB_PATH/.solo/kb/search.db` |
| `CLAUDE_SESSIONS_DIR` | `~/.claude` | Claude config dir whose `projects/` holds transcripts |
| `CODEGRAPH_SESSIONS_DB` | `~/.solo/sessions` | Session index directory (`search.db`) |
| `SOURCES_ROOT` | `~/.solo/sources` | Sources index (`search.db`) and YouTube VTT cache |

Every store is per-registration, so one install serves several profiles (say a personal one and a client one) from separate MCP entries with different env.

## Portfolio store

`portfolio.db`: `projects` (from the registry), `packages` (npm, pip, SPM, Gradle, Cargo manifests), `files` (path, language, lines, mtime, size), `symbols` (name, kind, line) and an FTS5 index over the symbols' words.

## Sources

One FTS index at `$SOURCES_ROOT/search.db`, documents keyed `<source>/<id>`, plus `videos` and `video_tags` tables.

- **YouTube** — `index-youtube` discovers videos via SearXNG, fetches metadata and VTT with yt-dlp, chunks by chapter. Each chunk is a document, so a hit links to its second (`&t=140s`); a search returns each video once, at its best chunk. Topic tags come from keyword matching against the `TOPICS` table in `sources.py` (title hits count double).
- **ProductHunt / TrustMRR** — scrapers into the same index; upvotes nudge ranking by at most 15%.
- **Telegram** — `scripts/index_telegram.py` in the consuming repo.

`solograph-search` serves the sources index over HTTP (`GET /search?q=...&source=...&n=5`, `/sources`, `/health`, port `SOLOGRAPH_SEARCH_PORT`, default 8002) for a SearXNG engine.

## Rust core

`rust/` builds one `solograph` binary: the MCP server (the same 12 tools, arguments and result shapes) plus the commands the nightly job needs — `scan [--project NAME]`, `registry`, `sync-sessions`, `index-kb`. It reads and writes the same SQLite files with the same schemas, so Python and Rust can take turns on one store; the scrapers and indexers (YouTube, ProductHunt, TrustMRR, Telegram) stay in Python and write the sources index Rust reads.

```bash
make rust-test       # unit tests + an MCP contract test over stdio
make rust-install    # → ~/.solo/bin/solograph, apart from the Python entry points
```

Measured on 27 Oct 2026 against copies of the real stores, each engine on its own copy:

| | Python 0.8 | Rust |
|---|---|---|
| Same top-5, same order — 54 session, 8 KB, 8 symbol, 6 source queries | | 76 / 76 |
| Session recall@5, 40 known-answer sessions | 30 | 30 |
| stats, explain, shared, project_info, source_tags, source_related | | identical JSON |
| Query stems, 28,576 RU+EN words from the KB and sessions | snowballstemmer 3.1.1 | 0 differ |
| Portfolio from scratch, 77 projects: files / packages / symbols | 8,935 / 1,350 / 63,585 | 8,935 / 1,350 / 63,582 |
| Full scan from scratch | 32.7 s | 9.8 s, registry merge included |
| MCP start → initialize answered | 280–300 ms | 4 ms |
| First / repeat `session_search` | 24 / 14 ms | 15 / 10 ms |
| On disk | 97 MB venv | 33 MB binary |

The English stemmer is a port of Snowball 3.1.1 (`english.rs`): rust-stemmers implements Snowball 2, and 24 of ~13,000 English words stemmed differently ("organization": organ vs organiz), which is a different prefix query. The three symbols Rust does not find are in files that do not parse cleanly (C++ inside a `.h` read with the C grammar, one Swift `async` method), where tree-sitter's error recovery differs between the core Python bundles (0.26.0) and the one on crates.io (0.26.13). The first launch after a rebuild takes ~0.9 s while macOS checks the new binary.

## Part of Solo Factory

Solograph is the MCP backend for [Solo Factory](https://github.com/fortunto2/solo-factory). It also works standalone.

## License

MIT
