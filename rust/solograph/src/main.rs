//! solograph — the core as one binary: the MCP server (default), every read the
//! CLI offers, the HTTP API for sources, and the commands the nightly refresh
//! needs. Python keeps only the scrapers and indexers (YouTube, ProductHunt,
//! TrustMRR, Telegram), which write the sources index this binary reads.

mod aliases;
mod cli;
mod deps;
mod english;
mod fts;
mod git;
mod hook;
mod http;
mod kb;
mod paths;
mod portfolio;
mod registry;
mod scanner;
mod secrets;
mod server;
mod sessions;
mod sources;
mod stem;
mod text;
mod usage;
mod yamlval;

use std::path::Path;
use std::process::ExitCode;

use anyhow::{Result, bail};
use rmcp::ServiceExt;

const USAGE: &str = "solograph — code portfolio, sessions, KB and sources search over SQLite FTS5

Usage:
  solograph [serve]                      MCP server on stdio (the default)
  solograph http [--port N]              sources over HTTP: /search /sources /health (port 8002)

  solograph symbols QUERY [-p P] [-k KIND] [-n N]   where something is defined, every project
  solograph sessions QUERY [-p P] [-n N] [--session ID]   past Claude Code sessions, full
                                          text; --session reads that one found session for QUERY
  solograph kb QUERY [--path P | --pages] [--type T] [--graph] [-n N]   the knowledge base:
                                          --kb-path, else KB_PATH, else the current dir;
                                          --path limits it to files under a prefix, --pages
                                          to the wiki (kb.pages in .solograph.yaml, wiki/)
  solograph kb-graph [--path P] [--page SLUG] [-n N]   [[wikilink]] graph of the wiki, or of
                                          --path P (an empty P: the whole KB): hubs, connectors,
                                          orphans, broken links, pages missing from index.md;
                                          --check prints only those and fails when there are any
  solograph hook [--kb-path DIR] [--db-path FILE]   the Claude Code hook for SessionStart and
                                          UserPromptSubmit: the wiki hint on prompts (when a KB
                                          is set), and a map of the project the session is in,
                                          once per session, at start or when work begins
  solograph kb-hint PROMPT [--session ID]   the wiki page a prompt is specifically about,
                                          if any (the UserPromptSubmit hint; silent otherwise);
                                          --hook reads the hook's JSON on stdin and answers it
  solograph sources search QUERY [-s SRC] [-n N]    YouTube / ProductHunt / TrustMRR / Telegram
  solograph sources list | tags [-s SRC] | related VIDEO [-s SRC]
  solograph explain PROJECT | deps PROJECT | stats | shared [PACKAGE]
  solograph dupes [NAME] [-p P] [--min N] [-n N] [--show N]   names defined in several places
  solograph usage [--days N]             what agents asked over MCP and the hook decided:
                                         calls per tool, what found nothing, the slowest
                                         (the log: usage/ next to the portfolio store)
  Every read takes --json: the value the MCP tool returns (project_info and source_search
  join explain/stats/shared and sources list/tags/related into one answer).

  solograph scan [-p NAME]               merge new checkouts into the registry, then index
                                         every project; with -p, index just that one
  solograph registry                     merge new checkouts into the registry only
  solograph registry --list [--scope F | --only NAMES]   name|path per project, read-only;
                                         --scope keeps the names in F's `scope:` list (a
                                         .solograph.yaml), --only the ones given; an unknown
                                         name fails
  solograph sync-sessions [--claude-dir DIR]   read new and grown transcripts into the index
  solograph index-kb [--kb-path DIR]     sync the knowledge-base index now
  --db-path FILE                         the portfolio store for symbols/explain/deps/stats/
                                         shared/dupes/scan (default: SOLOGRAPH_PORTFOLIO_DB)

Environment: CODEGRAPH_REGISTRY, CODEGRAPH_DB_PATH / SOLOGRAPH_PORTFOLIO_DB, CODEGRAPH_SCAN_PATH,
KB_PATH, CLAUDE_SESSIONS_DIR, CODEGRAPH_SESSIONS_DB, SOURCES_ROOT, SOLOGRAPH_SEARCH_PORT.";

fn print_registry_report(r: &registry::RefreshReport, path: &Path) {
    println!("Registry saved: {}", path.display());
    println!("✓ {} projects in {}", r.total, path.display());
    for (label, list) in [
        ("added", &r.added),
        ("removed", &r.removed),
        ("name taken", &r.collisions),
    ] {
        if !list.is_empty() {
            println!("  {label}: {}", list.join(", "));
        }
    }
}

fn scan(a: &cli::Args) -> Result<()> {
    let reg_path = a.path("registry").unwrap_or_else(registry::default_path);
    let project = a.val("project").map(String::from);
    // `scan` reads the registry; `solograph registry` is what merges new checkouts.
    // It used to merge here too, which re-added repos a client profile had removed
    // on purpose (review, 28 Oct 2026). The one exception is a first run: no
    // registry yet, and the folders to look in named explicitly.
    if project.is_none() && !reg_path.exists() {
        if paths::env_path("CODEGRAPH_SCAN_PATH").is_none() {
            bail!(
                "no registry at {}: set CODEGRAPH_SCAN_PATH to the folders holding your \
                 checkouts, or run `solograph registry` first",
                reg_path.display()
            );
        }
        let report = registry::refresh(&reg_path, None)?;
        print_registry_report(&report, &reg_path);
    }
    let Some(reg) = registry::load(&reg_path)? else {
        bail!("Registry not found: {}", reg_path.display())
    };
    let mut pf = portfolio::Portfolio::open(cli::db_path(a))?;
    // The registry is the whole truth even when scanning one project.
    for name in pf.sync_projects(&reg.projects, true)? {
        println!("  {name}: left the registry, removed");
    }
    let targets: Vec<&registry::Project> = reg
        .projects
        .iter()
        .filter(|p| project.as_deref().is_none_or(|n| p.name == n))
        .collect();
    if let Some(n) = &project
        && targets.is_empty()
    {
        bail!("Project not found: {n}");
    }
    println!("Scanning {} project(s)\n", targets.len());
    let mut ex = scanner::SymbolExtractor::default();
    let mut failed: Vec<&str> = Vec::new();
    for p in targets {
        let root = Path::new(&p.path);
        if !root.exists() {
            println!("  Skipping {} (path not found)", p.name);
            continue;
        }
        // One project that cannot be scanned must not stop the rest.
        let r = match pf.scan_project(&p.name, root, &mut ex) {
            Ok(r) => r,
            Err(e) => {
                println!("  {}: FAILED {e:#}", p.name);
                failed.push(&p.name);
                continue;
            }
        };
        if r.read == 0 && r.removed == 0 {
            // refresh-changed.sh counts these lines; keep the wording.
            println!("  {}: unchanged", p.name);
        } else {
            println!(
                "  {}: {} read, {} unchanged, {} removed, {} symbols",
                p.name, r.read, r.unchanged, r.removed, r.symbols
            );
        }
    }
    println!("\n{}", serde_json::to_string(&pf.stats()?)?);
    if !failed.is_empty() {
        bail!("{} project(s) failed: {}", failed.len(), failed.join(", "));
    }
    Ok(())
}

fn sync_sessions(a: &cli::Args) -> Result<()> {
    let root = a.path("claude-dir");
    let mut s = sessions::SessionSearch::open(None, root.as_deref())?;
    match s.sync()? {
        Some((r, _, _)) => {
            println!(
                "sessions — {} read, {} indexed, {} archived, {} failed",
                r.read,
                r.indexed,
                r.gone,
                r.failed.len()
            );
            Ok(())
        }
        None => bail!(
            "transcript directory unreadable: {}",
            a.path("claude-dir")
                .unwrap_or_else(sessions::claude_dir)
                .join("projects")
                .display()
        ),
    }
}

fn index_kb(a: &cli::Args) -> Result<()> {
    let root = kb::root_or_cwd(a.path("kb-path"))?;
    let stats = kb::KnowledgeBase::open(&root, None)?.stats()?;
    println!(
        "KB index: {} documents, {} read now, {} alias groups",
        stats["documents"], stats["read_now"], stats["alias_groups"]
    );
    Ok(())
}

async fn serve() -> Result<()> {
    let service = server::Solograph::new()
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}

/// Rust ignores SIGPIPE, so `solograph symbols x | head` panicked on the closed
/// pipe. A CLI read should end quietly there like any Unix tool. Not for `serve`:
/// the MCP server reports a vanished client as an error, it does not just die.
fn quiet_on_closed_pipe() {
    // SAFETY: called once at startup, before any thread exists; restores the
    // default disposition of one signal.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !matches!(
        args.first().map(String::as_str),
        None | Some("serve" | "http")
    ) {
        quiet_on_closed_pipe();
    }
    let rest = args.get(1..).unwrap_or_default();
    let parsed = || cli::Args::parse(rest);
    let result = match args.first().map(String::as_str) {
        None | Some("serve") => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(Into::into)
            .and_then(|rt| rt.block_on(serve())),
        Some("http") => parsed().and_then(|a| {
            let port = match a.val("port") {
                Some(p) => p
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--port must be a number"))?,
                None => http::port(),
            };
            http::serve(port)
        }),
        Some("scan") => parsed().and_then(|a| scan(&a)),
        Some("registry") => parsed().and_then(|a| {
            let path = a.path("registry").unwrap_or_else(registry::default_path);
            if !a.on("list") {
                return registry::refresh(&path, None).map(|r| print_registry_report(&r, &path));
            }
            let reg = registry::load(&path)?
                .ok_or_else(|| anyhow::anyhow!("no registry at {}", path.display()))?;
            let scope = match (a.path("scope"), a.val("only")) {
                (Some(f), _) => Some(registry::load_scope(&f)?),
                (None, Some(names)) => Some(
                    names
                        .split([' ', ','])
                        .filter(|n| !n.is_empty())
                        .map(String::from)
                        .collect::<Vec<_>>(),
                ),
                (None, None) => None,
            };
            for line in registry::listing(&reg, scope.as_deref())? {
                println!("{line}");
            }
            Ok(())
        }),
        Some("sync-sessions") => parsed().and_then(|a| sync_sessions(&a)),
        Some("index-kb") => parsed().and_then(|a| index_kb(&a)),
        // Debugging aid, not in the usage text: stems of stdin words, one per line,
        // the way queries are stemmed — for checking against Python's snowballstemmer.
        Some("stems") => {
            use std::io::BufRead;
            for w in std::io::stdin().lock().lines().map_while(Result::ok) {
                let cyr = w.chars().any(|c| ('а'..='я').contains(&c));
                let s = if cyr {
                    stem::russian(&w)
                } else {
                    stem::english(&w)
                };
                println!("{w}\t{s}");
            }
            Ok(())
        }
        // Debugging aid, not in the usage text: the definitions one file yields.
        Some("extract-symbols") if args.len() == 3 => {
            let lang = scanner::lang_of(scanner::suffix(&args[2])).unwrap_or("");
            let lang: &'static str = if args[1] == "auto" {
                lang
            } else {
                Box::leak(args[1].clone().into_boxed_str())
            };
            for s in scanner::SymbolExtractor::default().extract(Path::new(&args[2]), lang) {
                println!("{}\t{}\t{}", s.line, s.kind, s.name);
            }
            Ok(())
        }
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("-V" | "--version" | "version") => {
            println!("solograph {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(cmd) => cli::run(cmd, rest)
            .unwrap_or_else(|| Err(anyhow::anyhow!("unknown command: {cmd}\n\n{USAGE}"))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
