//! solograph — the core as one binary: the MCP server (default) and the
//! commands the nightly refresh needs. Python keeps the scrapers and indexers
//! (YouTube, ProductHunt, TrustMRR, Telegram); both read the same SQLite files.

mod deps;
mod english;
mod fts;
mod kb;
mod paths;
mod portfolio;
mod registry;
mod scanner;
mod server;
mod sessions;
mod sources;
mod stem;
mod text;
mod web;
mod yamlval;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use rmcp::ServiceExt;

const USAGE: &str = "solograph — code portfolio, sessions, KB and sources search over SQLite FTS5

Usage:
  solograph [serve]                    MCP server on stdio (the default)
  solograph scan [--project NAME]      merge new checkouts into the registry, then index
                                       every project; with --project, index just that one
  solograph registry                   merge new checkouts into the registry only
  solograph sync-sessions              read new and grown transcripts into the session index
  solograph index-kb [--kb-path DIR]   sync the knowledge-base index now

Environment: CODEGRAPH_REGISTRY, CODEGRAPH_DB_PATH / SOLOGRAPH_PORTFOLIO_DB, CODEGRAPH_SCAN_PATH,
KB_PATH, CLAUDE_SESSIONS_DIR, CODEGRAPH_SESSIONS_DB, SOURCES_ROOT — the same as the Python package.";

fn flag(args: &[String], name: &str) -> Result<Option<String>> {
    match args.iter().position(|a| a == name) {
        Some(i) => args
            .get(i + 1)
            .cloned()
            .map(Some)
            .with_context(|| format!("{name} needs a value")),
        None => Ok(None),
    }
}

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

fn scan(args: &[String]) -> Result<()> {
    let reg_path = flag(args, "--registry")?.map_or_else(registry::default_path, PathBuf::from);
    let project = flag(args, "--project")?;
    if project.is_none() {
        let report = registry::refresh(&reg_path, None)?;
        print_registry_report(&report, &reg_path);
    }
    let Some(reg) = registry::load(&reg_path)? else {
        bail!("Registry not found: {}", reg_path.display())
    };
    let mut pf = portfolio::Portfolio::open(None)?;
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
    for p in targets {
        let root = Path::new(&p.path);
        if !root.exists() {
            println!("  Skipping {} (path not found)", p.name);
            continue;
        }
        let r = pf.scan_project(&p.name, root, &mut ex)?;
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
    Ok(())
}

fn sync_sessions() -> Result<()> {
    let mut s = sessions::SessionSearch::open(None, None)?;
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
            sessions::claude_dir().join("projects").display()
        ),
    }
}

fn index_kb(args: &[String]) -> Result<()> {
    let root = flag(args, "--kb-path")?
        .map(PathBuf::from)
        .or_else(|| paths::env_path("KB_PATH"));
    let Some(root) = root else {
        bail!("Set KB_PATH env var or pass --kb-path")
    };
    let stats = kb::KnowledgeBase::open(&root, None)?.stats()?;
    println!(
        "KB index: {} documents, {} read now",
        stats["documents"], stats["read_now"]
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None | Some("serve") => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(Into::into)
            .and_then(|rt| rt.block_on(serve())),
        Some("scan") => scan(&args[1..]),
        Some("registry") => {
            let path = registry::default_path();
            registry::refresh(&path, None).map(|r| print_registry_report(&r, &path))
        }
        Some("sync-sessions") => sync_sessions(),
        Some("index-kb") => index_kb(&args[1..]),
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
        Some("symbols") if args.len() == 3 => {
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
        Some(other) => Err(anyhow::anyhow!("unknown command: {other}\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
