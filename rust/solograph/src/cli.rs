//! The read commands: what the Python CLI answered until 0.9, now from the same
//! code the MCP server runs, so the two cannot drift. Human-readable by default,
//! `--json` for the exact value the matching MCP tool returns.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::kb::KnowledgeBase;
use crate::paths::env_path;
use crate::portfolio::Portfolio;
use crate::portfolio::queries::DupesQuery;
use crate::sessions::SessionSearch;
use crate::sources::SourceIndex;

/// Options that take a value, by canonical name, with their spellings.
const VALUED: &[(&str, &[&str])] = &[
    ("project", &["-p", "--project"]),
    ("kind", &["-k", "--kind"]),
    ("limit", &["-n", "--limit"]),
    ("source", &["-s", "--source"]),
    ("type", &["--type"]),
    ("min", &["--min"]),
    ("show", &["--show"]),
    ("port", &["--port"]),
    ("kb-path", &["--kb-path"]),
    ("db-path", &["--db-path"]),
    ("registry", &["-r", "--registry"]),
    ("claude-dir", &["--claude-dir"]),
];
/// Switches. `--deep` and `--incremental` are accepted and ignored: post-commit
/// hooks installed in client repos still pass them.
const SWITCHES: &[(&str, &[&str])] = &[
    ("json", &["--json"]),
    ("graph", &["--graph"]),
    ("deep", &["--deep"]),
    ("incremental", &["-i", "--incremental"]),
];

pub struct Args {
    pub pos: Vec<String>,
    vals: HashMap<&'static str, String>,
    switches: HashSet<&'static str>,
}

impl Args {
    pub fn parse(raw: &[String]) -> Result<Self> {
        let mut a = Self {
            pos: Vec::new(),
            vals: HashMap::new(),
            switches: HashSet::new(),
        };
        let mut it = raw.iter();
        while let Some(arg) = it.next() {
            if let Some((name, _)) = VALUED.iter().find(|(_, s)| s.contains(&arg.as_str())) {
                let v = it.next().with_context(|| format!("{arg} needs a value"))?;
                a.vals.insert(name, v.clone());
            } else if let Some((name, _)) = SWITCHES.iter().find(|(_, s)| s.contains(&arg.as_str()))
            {
                a.switches.insert(name);
            } else if arg.starts_with('-') && arg.len() > 1 {
                bail!("unknown option: {arg}");
            } else {
                a.pos.push(arg.clone());
            }
        }
        Ok(a)
    }

    pub fn val(&self, name: &str) -> Option<&str> {
        self.vals.get(name).map(String::as_str)
    }

    pub fn path(&self, name: &str) -> Option<PathBuf> {
        self.val(name).map(PathBuf::from)
    }

    fn num(&self, name: &str, default: usize) -> Result<usize> {
        self.val(name).map_or(Ok(default), |v| {
            v.parse()
                .with_context(|| format!("--{name} must be a number, got {v}"))
        })
    }

    fn json(&self) -> bool {
        self.switches.contains("json")
    }

    fn arg(&self, i: usize, what: &str) -> Result<&str> {
        self.pos
            .get(i)
            .map(String::as_str)
            .with_context(|| format!("missing {what}"))
    }
}

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
}

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

fn clip(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

fn portfolio(a: &Args) -> Result<Portfolio> {
    Portfolio::open(a.path("db-path").as_deref())
}

/// `solograph <command> …` for the read commands; None when `cmd` is not one.
pub fn run(cmd: &str, raw: &[String]) -> Option<Result<()>> {
    let a = match Args::parse(raw) {
        Ok(a) => a,
        Err(e) => return Some(Err(e)),
    };
    Some(match cmd {
        "symbols" => symbols(&a),
        "sessions" => sessions(&a),
        "explain" => explain(&a),
        "stats" => stats(&a),
        "shared" => shared(&a),
        "deps" => deps(&a),
        "dupes" => dupes(&a),
        "kb" => kb(&a),
        "sources" => sources(&a),
        _ => return None,
    })
}

fn symbols(a: &Args) -> Result<()> {
    let hits = portfolio(a)?.search_symbols(
        a.arg(0, "query")?,
        a.num("limit", 20)?,
        a.val("project"),
        a.val("kind"),
    )?;
    if a.json() {
        println!("{}", pretty(&Value::Array(hits)));
    } else if hits.is_empty() {
        println!("No definitions found.");
    } else {
        for h in &hits {
            println!(
                "  {:40} {:9} {}  {}:{}",
                s(h, "name"),
                s(h, "kind"),
                s(h, "project"),
                s(h, "file"),
                h["line"]
            );
        }
    }
    Ok(())
}

fn sessions(a: &Args) -> Result<()> {
    let root = a.path("claude-dir");
    let out = SessionSearch::open(None, root.as_deref())?.search(
        a.arg(0, "query")?,
        a.num("limit", 5)?,
        a.val("project"),
    )?;
    if a.json() {
        println!("{}", pretty(&out));
        return Ok(());
    }
    let results = out["results"].as_array().cloned().unwrap_or_default();
    if results.is_empty() {
        println!("No sessions found. ({})", s(&out["coverage"], "status"));
        return Ok(());
    }
    for (i, r) in results.iter().enumerate() {
        let date = clip(s(r, "started_at"), 10);
        println!(
            "{}. [{}] {}  score {}  {}",
            i + 1,
            s(r, "project"),
            if date.is_empty() { "?".into() } else { date },
            r["score"],
            clip(s(r, "title"), 80)
        );
        let how = r
            .get("resume")
            .or_else(|| r.get("transcript"))
            .and_then(Value::as_str);
        println!("   {}", how.unwrap_or(""));
        println!("   {}\n", clip(&s(r, "snippet").replace('\n', " "), 160));
    }
    Ok(())
}

fn explain(a: &Args) -> Result<()> {
    let project = a.arg(0, "project name")?;
    let v = match portfolio(a)?.explain(project)? {
        Ok(v) => v,
        Err(msg) if a.json() => serde_json::json!({ "error": msg }),
        Err(msg) => bail!(msg),
    };
    if a.json() {
        println!("{}", pretty(&v));
        return Ok(());
    }
    let c = &v["counts"];
    println!("{} — {}", s(&v, "name"), s(&v, "stack"));
    if !s(&v, "description").is_empty() {
        println!("{}", s(&v, "description"));
    }
    println!(
        "{} files | {} symbols | {} packages",
        c["files"], c["symbols"], c["packages"]
    );
    let rows = |k: &str| v[k].as_array().cloned().unwrap_or_default();
    let section = |title: &str, items: Vec<String>| {
        if !items.is_empty() {
            println!("\n{title}");
            for i in items {
                println!("  {i}");
            }
        }
    };
    let cell = |r: &Value, i: usize| r[i].as_str().map_or_else(|| r[i].to_string(), String::from);
    section(
        "Languages",
        rows("languages")
            .iter()
            .map(|r| format!("{:<14} {:>6} files {:>10} lines", cell(r, 0), r[1], r[2]))
            .collect(),
    );
    section(
        "Layers",
        rows("layers")
            .iter()
            .map(|r| format!("{:<40} {:>5} files {:>6} symbols", cell(r, 0), r[1], r[2]))
            .collect(),
    );
    section(
        "Key patterns",
        rows("patterns")
            .iter()
            .map(|p| p.as_str().unwrap_or("").to_string())
            .collect(),
    );
    let deps: Vec<String> = rows("dependencies")
        .iter()
        .map(|r| format!("{} ({})", cell(r, 0), cell(r, 1)))
        .collect();
    if !deps.is_empty() {
        println!("\nTop dependencies\n  {}", deps.join(", "));
    }
    section(
        "Most definitions",
        rows("largest_files")
            .iter()
            .map(|r| format!("{} — {} symbols", cell(r, 0), r[1]))
            .collect(),
    );
    Ok(())
}

fn stats(a: &Args) -> Result<()> {
    let v = portfolio(a)?.stats()?;
    if a.json() {
        println!("{}", pretty(&v));
        return Ok(());
    }
    println!("Solograph portfolio");
    for (k, val) in v.as_object().into_iter().flatten() {
        match val.as_object() {
            Some(inner) => {
                for (ik, iv) in inner {
                    println!("  {:<28} {:>8}", format!("{k}: {ik}"), iv);
                }
            }
            None => println!("  {k:<28} {val:>8}"),
        }
    }
    Ok(())
}

fn shared(a: &Args) -> Result<()> {
    let rows = portfolio(a)?.shared_packages(a.pos.first().map(String::as_str))?;
    if a.json() {
        println!("{}", pretty(&Value::Array(rows)));
    } else if rows.is_empty() {
        println!("No shared packages found");
    } else {
        for r in &rows {
            let projects: Vec<&str> = r["projects"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            println!(
                "  {:32} {:7} {}",
                s(r, "package"),
                s(r, "source"),
                projects.join(", ")
            );
        }
    }
    Ok(())
}

fn deps(a: &Args) -> Result<()> {
    let project = a.arg(0, "project name")?;
    let rows = portfolio(a)?.dependencies(project)?;
    if a.json() {
        println!("{}", pretty(&Value::Array(rows)));
    } else if rows.is_empty() {
        println!("No dependencies found for {project}");
    } else {
        println!("\nDependencies of {project} ({}):\n", rows.len());
        for d in &rows {
            let ver = s(d, "version");
            let ver = if ver.is_empty() {
                String::new()
            } else {
                format!(" v{ver}")
            };
            println!("  [{}] {}{ver}", s(d, "source"), s(d, "name"));
        }
    }
    Ok(())
}

fn dupes(a: &Args) -> Result<()> {
    let rows = portfolio(a)?.dupes(&DupesQuery {
        matching: a.pos.first().map(String::as_str),
        project: a.val("project"),
        min: a.num("min", 3)?,
        limit: a.num("limit", 20)?,
    })?;
    if a.json() {
        println!("{}", pretty(&Value::Array(rows)));
        return Ok(());
    }
    let show = a.num("show", 5)?;
    for r in &rows {
        let repos = r["repos"].as_i64().unwrap_or(0);
        println!(
            "{:>3}× {}  ({}, {repos} repo{})",
            r["count"],
            s(r, "name"),
            s(r, "kind"),
            if repos > 1 { "s" } else { "" }
        );
        for p in r["places"].as_array().into_iter().flatten().take(show) {
            println!("       {}", p.as_str().unwrap_or(""));
        }
    }
    Ok(())
}

fn kb(a: &Args) -> Result<()> {
    let root = a.path("kb-path").or_else(|| env_path("KB_PATH"));
    let Some(root) = root else {
        bail!("Set KB_PATH env var or pass --kb-path")
    };
    let hits = KnowledgeBase::open(&root, None)?.search(
        a.arg(0, "query")?,
        a.num("limit", 5)?,
        a.val("type"),
        a.switches.contains("graph"),
    )?;
    if a.json() {
        println!("{}", pretty(&Value::Array(hits)));
    } else if hits.is_empty() {
        println!("No results found.");
    } else {
        for h in &hits {
            let score = h["score"]
                .as_f64()
                .map_or_else(String::new, |x| format!("{x:6.2}"));
            println!("{score:>6}  {}  — {}", s(h, "file"), s(h, "title"));
        }
    }
    Ok(())
}

fn sources(a: &Args) -> Result<()> {
    let idx = SourceIndex::open(None)?;
    let sub = a.arg(0, "sources command: search, list, tags or related")?;
    let source = a.val("source");
    let out = match sub {
        "search" => Value::Array(idx.search(a.arg(1, "query")?, source, a.num("limit", 5)?)?),
        "list" => Value::Array(idx.list_sources()?),
        "tags" => Value::Array(idx.list_tags(source.unwrap_or("youtube"))?),
        "related" => Value::Array(idx.related_videos(
            source.unwrap_or("youtube"),
            &crate::web::youtube_id(a.arg(1, "video id or URL")?),
        )?),
        other => bail!("unknown sources command: {other} (search, list, tags, related)"),
    };
    if a.json() {
        println!("{}", pretty(&out));
        return Ok(());
    }
    let rows = out.as_array().cloned().unwrap_or_default();
    match sub {
        "search" => print_source_hits(a.arg(1, "query")?, source, &rows),
        "list" => print_source_list(&rows),
        "tags" => print_tags(source.unwrap_or("youtube"), &rows),
        _ => print_related(a.arg(1, "video")?, &rows),
    }
    Ok(())
}

fn print_source_hits(query: &str, source: Option<&str>, rows: &[Value]) {
    if rows.is_empty() {
        println!("No results found.");
        return;
    }
    let scope = source.map_or_else(|| "[all sources]".to_string(), |s| format!("[{s}]"));
    println!("\nSearch: '{query}' {scope} ({} results):\n", rows.len());
    for (i, r) in rows.iter().enumerate() {
        let mut chapter = String::new();
        if !s(r, "chapter").is_empty() {
            chapter = format!(" [{}", s(r, "chapter"));
            if !s(r, "start_time").is_empty() {
                chapter += &format!(" @ {}", s(r, "start_time"));
            }
            chapter += "]";
        }
        println!(
            "{}. [{}] {}{chapter}  (score {})",
            i + 1,
            s(r, "source_type"),
            clip(s(r, "title"), 80),
            r["score"]
        );
        for (prefix, key, n) in [
            ("", "url", 500),
            ("Tags: ", "tags", 500),
            ("", "content", 120),
        ] {
            if !s(r, key).is_empty() {
                println!("   {prefix}{}", clip(s(r, key), n));
            }
        }
        println!();
    }
}

fn print_source_list(rows: &[Value]) {
    if rows.is_empty() {
        println!("No sources indexed. Run: index-telegram or index-youtube");
        return;
    }
    println!("\nIndexed sources ({}):\n", rows.len());
    let mut total = 0;
    for r in rows {
        let mut extra = String::new();
        if let Some(v) = r.get("videos") {
            extra = format!("  ({v} videos, {} chunks)", r["video_chunks"]);
        }
        if let Some(m) = r.get("makers") {
            extra += &format!("  ({m} makers)");
        }
        let count = r["count"].as_i64().unwrap_or(0);
        total += count;
        println!(
            "  {:12} {count:4} docs{extra}  {}",
            s(r, "source"),
            s(r, "path")
        );
    }
    println!("\n  Total: {total} documents");
}

fn print_tags(source: &str, rows: &[Value]) {
    if rows.is_empty() {
        println!("No tags found. Re-index videos to generate auto-tags.");
        return;
    }
    println!("\nTopics in [{source}] ({} tags):\n", rows.len());
    for t in rows {
        let conf = t["avg_confidence"]
            .as_f64()
            .map_or_else(String::new, |c| format!("  (avg {:.0}%)", c * 100.0));
        println!("  {:3}  {}{conf}", t["count"], s(t, "name"));
    }
}

fn print_related(video: &str, rows: &[Value]) {
    let id = crate::web::youtube_id(video);
    if rows.is_empty() {
        println!("No related videos found for {id}");
        return;
    }
    println!("\nRelated videos for {id} ({}):\n", rows.len());
    for r in rows {
        let rel = r["relevance"]
            .as_f64()
            .filter(|x| *x != 0.0)
            .map_or_else(String::new, |x| format!(" relevance: {x:.2}"));
        println!(
            "  [{} shared{rel}] {}",
            r["overlap"],
            clip(s(r, "title"), 70)
        );
        let tags: Vec<&str> = r["shared_tags"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if !tags.is_empty() {
            println!("    Tags: {}", tags.join(", "));
        }
        if !s(r, "url").is_empty() {
            println!("    {}", s(r, "url"));
        }
        println!();
    }
}

/// The path a subcommand points the portfolio at, for commands in main.rs.
pub fn db_path(a: &Args) -> Option<&Path> {
    a.vals.get("db-path").map(Path::new)
}
