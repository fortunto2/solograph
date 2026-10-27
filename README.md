# solograph

MCP server for a portfolio of projects: symbol search across every repository you own, plus full-text search over your Claude Code session history, a markdown knowledge base, and indexed external sources (YouTube, ProductHunt, TrustMRR, Telegram).

One Rust binary does all of it — the MCP server, every read on the command line, an HTTP API for the sources, the nightly scan. Python is left only for the scrapers that fill the sources index (yt-dlp, Scrapy, Playwright live there).

Everything is SQLite with FTS5: BM25 ranking, Snowball stemming for Russian and English, no graph database, no daemon, no embedding models, no API keys. Each index checks the disk before it answers, so there is no reindex step to forget.

## What was cut, and why

solograph used to be a FalkorDB code graph with vector search on top. Both were measured on a real machine before they went:

- **Vectors (removed in 0.7).** The session index had zero overlap with the transcripts on disk, KB search took 14 s cold and scored results by rank position, source search hung past two minutes. FTS answers in tens of milliseconds and finds 32 of 39 sessions in a known-answer test the vector index found none of.
- **The structural graph (removed in 0.8).** 781k call edges plus imports, inheritance and session→file edges got zero structural questions from agents in four months: five calls in total, every one "list the projects" or stats. Grep answers those questions as well. On a 1,100-file Rust + Swift repository, `claude -p` with a dedicated code-graph MCP against plain Grep/Read gave the same accuracy in all 8 runs; the call-chain question cost $1.73 against $2.07 (16.5 tool calls against 25), the impact question $0.69 against $0.65.

- **The second implementation (removed in 0.9).** For a day the same 12 tools existed in Rust and in Python. Two implementations of one search drift — the class of bug that kept `session_search` failing for seven weeks — so every read now comes from the binary the MCP server runs, and Python keeps only what writes the sources index.

What stays is what grep cannot do across repositories: one index of every project's definitions, packages and history. The Python environment went from 862 MB to 34 MB; the binary is 33 MB.

## Install

```bash
git clone https://github.com/fortunto2/solograph && cd solograph
make rust-install                                  # → ~/.solo/bin/solograph
claude mcp add -s user solograph -- ~/.solo/bin/solograph
CODEGRAPH_SCAN_PATH=~/my-projects ~/.solo/bin/solograph scan   # registry + portfolio index
```

The scrapers are a separate, optional Python CLI: `uv tool install --editable .` gives `solograph-cli index-youtube` and the rest.

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
solograph symbols "auth middleware" [-p my-app]    # definitions across every project
solograph sessions "webhook retry" [-p my-app]     # past sessions, archived ones too
solograph kb "privacy architecture"                # the knowledge base (KB_PATH)
solograph sources search "pricing tiers" -s youtube
solograph sources list | tags | related VIDEO
solograph explain my-app | deps my-app | stats | shared [react]
solograph dupes escape                             # names defined in several places
solograph scan [-p my-app]                         # registry merge + index what changed
solograph sync-sessions | index-kb                 # warm an index (searches sync on their own)
solograph http                                     # sources over HTTP, port 8002
```

Every read takes `--json` and then prints exactly what the matching MCP tool returns, so a script and an agent see the same thing.

Python, for the sources index only: `solograph-cli index-youtube -c GregIsenberg -n 10`, `index-producthunt`, `index-trustmrr`, `source-delete`, and `compact` for the KB inbox.

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

`solograph http` serves the sources index over HTTP (`GET /search?q=...&source=...&n=5`, `/sources`, `/health`, port `SOLOGRAPH_SEARCH_PORT`, default 8002) for the SearXNG engine `sources_local`; `Dockerfile.search` builds it into an image (`make docker-search`). Same endpoints and JSON as the Python `solograph-search` it replaced.

## Rust core

`rust/` builds the `solograph` binary. The scrapers in `src/` write the sources index through `sources.py`; the binary reads it — the tables are the contract, and `tests/test_sources.py` checks it the only way that means anything: Python writes, the real binary reads.

```bash
make rust-test       # unit tests + an MCP contract test over stdio
make test            # builds the binary, then the Python write → Rust read tests
make rust-install    # → ~/.solo/bin/solograph
```

### Session ranking

All the query's words before some of them: a session that has every word comes first, then the rest by BM25. Measured on 37 sessions with a known answer (queries are three words from the session's first prompt), on a frozen copy of the index:

| Variant | recall@1 | recall@5 |
|---|---|---|
| OR, title ×3 (0.8) | 26 | 30 |
| **all words first, then OR (0.9)** | **29** | **32** |
| OR, title ×1 | 27 | 30 |
| OR, title ×10 | 24 | 30 |
| all words first, title ×10 | 28 | 32 |
| OR + recency boost | 26 | 30 |

Titles skip the lines Claude Code writes itself: 82 archived sessions were titled "[Request interrupted by user for tool use]", and prompts lose their "[Image #N]" markers.

### Equivalence with the Python 0.8 it replaced

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
