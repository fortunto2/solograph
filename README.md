# solograph

MCP server for a portfolio of projects: symbol search across every repository you own, plus full-text search over your Claude Code session history, a markdown knowledge base, and indexed external sources (YouTube, ProductHunt, TrustMRR, Telegram).

**Write-up:** [Solograph, rebuilt: one Rust binary over SQLite instead of a graph database and vectors](https://rustman.org/posts/solograph-rebuilt-one-binary) — what was cut and why, what was borrowed from which repositories, and how to set up your own.

One Rust binary does all of it — the MCP server, every read on the command line, an HTTP API for the sources, the nightly scan. Python is left only for the scrapers that fill the sources index (yt-dlp, Scrapy, Playwright live there).

Everything is SQLite with FTS5: BM25 ranking, Snowball stemming for Russian and English, no graph database, no daemon, no embedding models, no API keys. Each index checks the disk before it answers, so there is no reindex step to forget.

## What was cut, and why

solograph used to be a FalkorDB code graph with vector search on top. Both were measured on a real machine before they went:

- **Vectors (removed in 0.7).** The session index had zero overlap with the transcripts on disk, KB search took 14 s cold and scored results by rank position, source search hung past two minutes. FTS answers in tens of milliseconds and finds 32 of 39 sessions in a known-answer test the vector index found none of.
- **The structural graph (removed in 0.8).** 781k call edges plus imports, inheritance and session→file edges got zero structural questions from agents in four months: five calls in total, every one "list the projects" or stats. Grep answers those questions as well. On a 1,100-file Rust + Swift repository, `claude -p` with a dedicated code-graph MCP against plain Grep/Read gave the same accuracy in all 8 runs; the call-chain question cost $1.73 against $2.07 (16.5 tool calls against 25), the impact question $0.69 against $0.65.

- **Seven of twelve tools (removed in 0.17).** Below, under 5 MCP tools.
- **The second implementation (removed in 0.9).** For a day the same 12 tools existed in Rust and in Python. Two implementations of one search drift — the class of bug that kept `session_search` failing for seven weeks — so every read now comes from the binary the MCP server runs, and Python keeps only what writes the sources index.

What stays is what grep cannot do across repositories: one index of every project's definitions, packages and history. The Python environment went from 862 MB to 34 MB; the binary is 31 MB.

## Install

A prebuilt binary from the [latest release](https://github.com/fortunto2/solograph/releases/latest), macOS (Apple silicon) or Linux (x86_64):

```bash
mkdir -p ~/.solo/bin
curl -fL -o ~/.solo/bin/solograph \
  https://github.com/fortunto2/solograph/releases/latest/download/solograph-macos-arm64   # or solograph-linux-x86_64
chmod +x ~/.solo/bin/solograph && ~/.solo/bin/solograph --version
```

Or from source with a Rust toolchain: `cargo install --git https://github.com/fortunto2/solograph solograph --root ~/.solo`, or `make rust-install` in a clone.

Then hand it to your agent and index your projects:

```bash
claude mcp add -s user solograph -e KB_PATH=$HOME/notes -- ~/.solo/bin/solograph
CODEGRAPH_SCAN_PATH=~/projects ~/.solo/bin/solograph scan     # every git checkout under ~/projects
~/.solo/bin/solograph symbols "auth middleware"               # what the agent will get
```

Sessions need nothing: the index reads `~/.claude/projects` on the first search. For new code by morning, cron or launchd `solograph registry && solograph scan`: `registry` adds checkouts that appeared, `scan` only reads the registry, so a project removed from it on purpose stays out.

### Your knowledge base

A folder of markdown with YAML frontmatter, the linked pages in `wiki/` (or wherever `.solograph.yaml` says, below). `KB_PATH` above points `kb_search` at it.

```bash
cd ~/notes
~/.solo/bin/solograph kb --pages "privacy architecture"   # search the wiki
~/.solo/bin/solograph kb-graph --check                    # orphans, broken links; exit 1 when any
```

Two languages: add `wiki/_aliases.yaml`, one group per line, the term your pages use last — `- [песочница, sandbox]`. The hook, in `~/.claude/settings.json`, registered for both events:

```json
"hooks": {
  "SessionStart": [
    { "hooks": [ { "type": "command", "timeout": 3,
      "command": "$HOME/.solo/bin/solograph hook --kb-path $HOME/notes" } ] }
  ],
  "UserPromptSubmit": [
    { "hooks": [ { "type": "command", "timeout": 3,
      "command": "$HOME/.solo/bin/solograph hook --kb-path $HOME/notes" } ] }
  ]
}
```

On a prompt it gives the wiki hint, silent unless the prompt names something the wiki is specifically about. Once per session per project, at SessionStart or on the first prompt that starts work ("plan", "fix", "сделай", "почини"), it gives a map of the registered project holding `cwd`: stack, size, languages, layers, the first dependencies, and which tool answers what. A second profile adds `--db-path` for its own portfolio store. Any failure is silence with exit 0; 50-85 ms warm.

The scrapers are a separate, optional Python CLI: `uv tool install --editable .` gives `solograph-cli index-youtube` and the rest.

## 5 MCP tools

| Tool | What it answers |
|------|-----------------|
| `session_search` | What was asked and answered in past Claude Code sessions, including ones Claude Code already deleted. The session asking is left out. With a result's `session_id`, it reads that session: the passages that match the query |
| `kb_search` | Markdown knowledge base: title and summary fields weigh above the body; `path` scopes it (`wiki/`), `expand_graph` adds up to 3 [[wikilinked]] pages |
| `symbol_search` | Where something is defined, across every project: "montage profile" finds `resolve_montage_profile` and `MontageProfile` |
| `source_search` | Indexed YouTube transcripts (timestamped links), ProductHunt, TrustMRR, Telegram; `related` gives videos sharing a video's topics, no query at all lists the sources and topics |
| `project_info` | The projects: all of them with stacks and counts, one project's make-up (stack, languages, layers, dependencies), or every project using a `package` |

Twelve until 0.17. With tool search on, Claude Code's default, the agent sees only tool names and the server's instructions until it loads a tool, so it picked from twelve bare names, four of them `codegraph_*` for a graph that no longer existed, and solograph sent no instructions. Two months of transcripts from both profiles hold 72 calls: `session_search` 34, `kb_search` 11, `symbol_search` 7, `source_search` 4, the other eight tools 11 between them (`codegraph_shared` and `source_related` none, `web_fetch` one), and 5 calls to tools removed earlier. The tools with similar products are 2-5 (episodic-memory 2, qmd 5, claude-context 4). Now the server says when to reach for each tool (848 characters of instructions), `session_search` and `kb_search` load with the session (`anthropic/alwaysLoad`), the descriptions are 5.4k characters instead of 7.9k, and answers are compact JSON, 13-46% smaller than the indented JSON before. `web_fetch` went with its HTTP client, 113 of the 253 crates in the build, and the binary shrank by 3 MB: Claude Code's own WebFetch and the searxng server's `web_extract` do that job.

The session asking was left out because it came first: in 18 agent searches that found something, the session doing the search was in the results 10 times and first 7 times, since its prompt holds the very words searched for.

Web search is not here: it lives in the separate [searxng MCP server](https://github.com/fortunto2/searxng-docker-tavily-adapter), so a broken import in one cannot take the other down.

### How the indexes behave

- **Symbols by their words.** Names are split on camelCase and snake_case (as codebase-memory-mcp's tokenizer does), so a phrase finds every spelling. All words rank above some words; an exact name comes first. Definitions only — functions, classes, protocols — from tree-sitter grammars for Python, Swift, TypeScript, JavaScript, Kotlin, Rust, Go, Java, Ruby, C, C++, PHP, HCL, Bash and SQL.
- **Lexical, stemmed.** "графами" finds "граф", "indexing" finds "index". No cross-language matching by itself: a Russian conversation is found by Russian words.
- **Never stale.** A scan re-reads a file only when its mtime or size changed; a session search re-reads only transcripts that grew, from their last byte offset — live transcripts reach hundreds of MB, and a resync costs ~10 ms.
- **Sessions are archived.** Claude Code deletes transcripts after 30 days (`cleanupPeriodDays`). The index keeps their text and marks them `transcript: deleted`; live ones come with a `claude --resume` command. Batch runs started from temp directories (`claude -p` in `/tmp`, agent scratchpads) are left out.
- **Private folders stay private.** A session whose working directory is inside a folder holding `.solograph-private` (or Spotlight's `.metadata_never_index`) is never indexed, and is purged if it was.
- **Scores are BM25** — higher is better, comparable within one answer only.
- **Secrets never enter an index.** Every text the session and KB indexes store passes through `secrets.rs` first: API keys and tokens (AWS, GitHub, OpenAI, Anthropic, Google, Slack, Stripe, Hugging Face, Telegram bots, JWTs), private key blocks, passwords in connection strings, and `*_TOKEN=` / `*_SECRET=` values, which keep their name and lose the value. The rules are solo-factory's `secret_guard` hook (adapted from ScreenShield), placeholders such as `your-key` or `${VAR}` left alone. Indexes written earlier are cleaned once on open: the session index here held 23 secrets in 18 sessions, now 0. The sources index, written by the Python scrapers, is not covered yet.
- **Every call is logged, for the next change.** One JSON line per MCP tool call (arguments, result count, first three ids, size, milliseconds, the client's name and Claude Code's session id) and per prompt-hook run (session, pages pointed at, never the prompt), in `usage/` next to the portfolio store, so a second profile's queries stay in its own store. `solograph usage` reads it back, below. Nothing is written from a process started in a folder marked `.solograph-private`, and the CLI is not logged, so evals and scripts do not drown it.

### Is it useful? Five numbers

`solograph usage` opens with them. They count only calls made inside a Claude Code session: the first log held 58 calls, 47 of them from scripts, which would have read as use.

| Number | Counts | Before 0.17 |
|---|---|---|
| reach | sessions where the agent searched at least once, of sessions seen | 12 of 681 (1.8%) |
| hints | pointers and nudges from the hook followed by a call before the next prompt | not logged |
| found | searches that returned something | 31 of 45 (69%) |
| opened | of those, searches after which the agent opened a returned file or page, or read a returned session with `session_id`, before the next prompt (from the transcript, while it is on disk) | 11 of 31 (35%) |
| retried | searches asked again by the same tool within 2 minutes: the first answer did not do | 8 of 45 (18%) |

The last column is two months of transcripts (one profile) counted with the same definitions. 13 of the 45 searches were errors: `session_search` failed on every call from 27 September to 25 October in the Python server it replaced, and nothing counted errors then. Reach is the number to move: a search that is never made helps no one, and hooks that put the pointer into the prompt are what other tools found agents act on (a skill the agent must remember was never invoked in 56% of Vercel's evals; injected context was used 66% of the time against 6% in Codeminer42's).

A first check of the change, before any log: the same five prompts that lean on earlier work ("what did we decide about vectors", "where is MontageProfile defined"), `claude -p` from a temp directory with nothing but the user's CLAUDE.md, the server swapped between 0.16 and 0.17. The agent called solograph on 1 of 5 prompts with 0.16 and on 5 of 5 with 0.17; the answers were right in both (the wiki held them, and with 0.16 the agent grepped it), at $0.30 and $0.32 a prompt. Five prompts, one run each: a direction, not a rate. Where grep is enough, solograph adds a step; what grep cannot reach is the sessions, deleted ones included.

Search quality itself is measured offline, on known answers: a query and the page or session it should find, recall@1 and @5 before and after a change (the numbers in Session ranking below).

### The knowledge base

One index (`$KB_PATH/.solo/kb/kb.db`) serves search, the link graph and the prompt hint.
It replaced three things that read the same markdown separately: a Python FTS index, a
JSON link graph only as fresh as the last script run (it sat at 36 nodes for eight months
while the wiki passed 160 pages), and the hint's own index.

The engine knows no KB's layout. A KB states it in `.solograph.yaml` at its root:

```yaml
kb:
  pages: wiki/                  # the wiki: [[links]], index.md, _aliases.yaml (default wiki/)
  stacks: 1-methodology/stacks  # YAML templates for "which stack for X?" hints (default none)
  exclude: [archive, clients/acme]   # a bare name anywhere, a path from the root
  memory_gate: harness/memory-gate.json  # weights for the memory nudge (default none)
```

A search with no `path` takes two lanes in turn, the whole KB and the wiki, so a Russian question still reaches an English wiki page past Russian notes elsewhere (6 → 16 of 22 in the top 5; 40 questions about notes outside the wiki: same at #1, 34 → 33 in the top 5). `kb-graph`, `kb-hint` and `kb --pages` scope to `pages`; a typo in the section is an
error, not a silent default. The same file's `scope:` key is the portfolio scan's.

- **Fields.** Title; then index_line, description, tags and the file name; then the body.
  BM25 weights 2/2/1, measured on a 75-query known-answer set: top-1 29→31/40 on body
  words, 18→22/25 on summary words, against the old title-plus-body index.
- **Links.** `[[slug]]` and `[[slug|label]]` outside code are recorded at sync, so the graph
  is never stale. `solograph kb-graph` lists hubs, connectors, orphans, broken links and pages
  missing from `index.md`; `expand_graph` adds the pages most linked to the hits (Σ 1/rank,
  catalog and `_*` pages excluded, at most 3).
- **Aliases.** A YAML table in the KB, `wiki/_aliases.yaml`: each group lists other spellings
  of a term, the term the KB writes **last** — `- [песочница, sandbox]`. A query word matching
  an earlier member (by stem) also searches the last; the last expands to nothing. One way on
  purpose: expanding both ways sent English queries to Russian documents (top-1 fell 31→24/40).
  On 22 Russian questions for English pages: 0 found before, 21 in the top 5 within `wiki/`.
- **Hint.** `solograph kb-hint PROMPT --session ID` names a page when the prompt holds a word
  rare in the wiki that names it and the page covers the prompt's topic; silent otherwise,
  each page once per session. `solograph hook` runs it on every prompt
  (`kb-hint --hook` still works alone): it reads the hook's JSON and answers it, syncs only `wiki/`,
  and is silent with exit 0 on any failure. The rules are in `src/kb/hint.rs`.
- **Memory nudge.** With `memory_gate:` set, the hook also scores each prompt of three words
  or more for leaning on the user's earlier work ("как мы делали", "у нас свой крейт…") and,
  past the file's threshold, tells the agent to look it up with `session_search` and
  `kb_search`. It says *that*, not *what*: the agent writes a better query than the raw
  prompt, and injecting results for the prompt itself was half noise. Before the nudge the
  agent searched after 17% of such prompts. The scorer is a logistic regression over words
  and character 3-4-grams (`src/kb/memgate.rs`); the weights come from your own labelled
  prompts, so they hold fragments of them and stay in your KB, not here. On a 290-prompt
  hard set it scored AUC 0.908 against 0.889 for the three small neural models it replaced
  (258 MB resident, ~100 ms a prompt); this adds under a millisecond. Each run logs `gate`
  and `nudged` to the usage log; `solograph usage` counts the nudges.

## CLI

```bash
solograph symbols "auth middleware" [-p my-app]    # definitions across every project
solograph sessions "webhook retry" [-p my-app]     # past sessions, archived ones too
solograph sessions "webhook retry" --session ID    # read one found session for the query
solograph kb "privacy architecture" [--path wiki/] # the knowledge base (KB_PATH, else cwd)
solograph kb-graph [--path wiki/] [--page slug]    # hubs, orphans, broken links, backlinks
solograph kb-graph --path wiki/ --check            # only the problems; fails when there are any
solograph kb-hint "prompt" --session ID            # the UserPromptSubmit pointer, or nothing
solograph sources search "pricing tiers" -s youtube
solograph sources list | tags | related VIDEO
solograph explain my-app | deps my-app | stats | shared [react]
solograph dupes escape                             # names defined in several places
solograph usage [--days 14]                        # the five numbers, then calls, what found nothing, slowest
solograph scan [-p my-app]                         # registry merge + index what changed
solograph sync-sessions | index-kb                 # warm an index (searches sync on their own)
solograph http                                     # sources over HTTP, port 8002
```

Every read takes `--json` and then prints what the matching MCP tool returns, so a script and an agent see the same thing (`project_info` and `source_search` join `explain`/`stats`/`shared` and `sources list`/`tags`/`related` into one answer each).

Python, for the sources index only: `solograph-cli index-youtube -c GregIsenberg -n 10`, `index-producthunt`, `index-trustmrr`, `source-delete`, and `compact` for the KB inbox.

`scripts/refresh-changed.sh` is the nightly job: registry merge, a scan of every project, session-index sync. It is meant for launchd or cron, and needs only the binary: `SOLOGRAPH_SCOPE="a b"` or a `.solograph.yaml` narrows it, and a name the registry lacks fails the run.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEGRAPH_DB_PATH` | `~/.solo/codegraph.db` | Only its directory is used now: the portfolio store is `portfolio.db` next to it |
| `SOLOGRAPH_PORTFOLIO_DB` | (none) | Names the portfolio store outright |
| `CODEGRAPH_REGISTRY` | `~/.solo/registry.yaml` | Project registry |
| `CODEGRAPH_SCAN_PATH` | `~/startups/active` | Comma-separated directories `registry` looks for checkouts in |
| `KB_PATH` | current dir | Knowledge base root; index at `$KB_PATH/.solo/kb/kb.db` (before Oct 2026: `search.db`, left for binaries still running) |
| `SOLOGRAPH_LOG_DIR` | `usage/` next to the portfolio store | Usage log, one JSON line per MCP call and per hook run (`YYYY-MM.jsonl`) |
| `SOLOGRAPH_LOG` | on | `off` stops the usage log |
| `SOLOGRAPH_KB_WEIGHTS` | `2,2,1` | BM25 weights for title, summary fields, body, to re-measure |
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

**Typos.** Every word is searched as a stemmed prefix, so one wrong letter ("солграф",
"технопакр") matched nothing and the word dropped out. A word in fewer than 3 documents
whose neighbour one edit away (two from 8 letters) is in at least three times as many is
also searched as that neighbour (`src/fts/spell.rs`, the index's own vocabulary through
fts5vocab; a KB adds its alias spellings, so "солграф" reaches "solograph"). Sessions take
three passes in turn: every word as written, every word in either spelling, any word, so a
session holding a typo verbatim still comes first. One typo planted in the longest word of
each known-answer query, 0.15 → 0.16:

| Set | clean @1 / @5 | with a typo, before | with a typo, after |
|---|---|---|---|
| KB, 40 body-word questions | 30 / 38 (same) | 19 / 31 | 27 / 38 |
| KB, 40 questions outside the wiki | 25 / 33 (same) | 17 / 23 | 23 / 31 |
| KB, 22 Russian questions for English pages | 0 / 16 (same) | 0 / 14 | 0 / 15 |
| Sessions, 40 known answers | 27 / 31 (same) | 17 / 24 | 21 / 27 |

Without typos the known answer keeps its rank on all 199 queries, and 196 of 199 top-5
lists are identical; in the other 3, places 2-5 change. On the words of 300 real prompts
(1358 distinct words of 5+ letters) the rule respelt 69: 51 right ("првоерку" → "проверку"),
14 real but rare words wrongly ("тачке" → "точке", "брокер" → "блокер") and 4 typos to the
wrong word ("заплонирован" → "заблокирован"). A wrong respelling adds a term but removes
none, so it costs places lower in the list. Not caught: words under 5 letters, a typo in the
first letter, transliteration ("фази сеарч" for "fuzzy search"). A query costs 0-14 ms more.

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
