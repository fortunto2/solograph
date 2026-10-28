//! The usage log: one JSON line per MCP tool call and per prompt-hook run, so the
//! next change to solograph starts from what agents actually asked, what found
//! nothing, and what was slow, instead of from a guess.
//!
//! Until Oct 2026 the only record was the transcripts, and answering "is anyone
//! using kb_search?" meant parsing 14 days of them by hand.
//!
//! - **Where:** `usage/YYYY-MM.jsonl` next to the portfolio store the process
//!   reads (`~/.solo` by default), so a client profile's queries stay in that
//!   client's store. `SOLOGRAPH_LOG_DIR` overrides it; `SOLOGRAPH_LOG=off` stops it.
//! - **Not logged:** anything from a process started under a folder marked private
//!   (the markers sessions honour), the CLI (scripts and evals would drown the
//!   signal), and prompt text: the hook records only what it decided.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use rmcp::model::CallToolResult;
use serde_json::{Map, Value, json};

/// Fields that name a result, most specific first.
const ID_FIELDS: [&str; 7] = [
    "file",
    "session_id",
    "doc_id",
    "name",
    "url",
    "project",
    "source",
];

/// Where the log lives: SOLOGRAPH_LOG_DIR, else `usage/` next to the portfolio store.
pub fn location() -> PathBuf {
    crate::paths::env_path("SOLOGRAPH_LOG_DIR").unwrap_or_else(|| {
        let db = crate::portfolio::default_db_path();
        db.parent().unwrap_or(Path::new(".")).join("usage")
    })
}

/// Where this process writes, or None when logging is off or the process was
/// started in a private folder.
fn dir() -> Option<PathBuf> {
    if std::env::var("SOLOGRAPH_LOG").is_ok_and(|v| v == "off" || v == "0") {
        return None;
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    if crate::sessions::is_private(&cwd.to_string_lossy()) {
        return None;
    }
    Some(location())
}

/// How many results a tool returned and the ids of the first three.
pub fn summarize(result: &CallToolResult) -> (usize, Vec<String>) {
    let items: Vec<Value> = match result
        .structured_content
        .as_ref()
        .and_then(|s| s.get("result"))
    {
        Some(Value::Array(a)) => a.clone(),
        _ => {
            let blocks: Vec<Value> = result
                .content
                .iter()
                .filter_map(|c| c.as_text())
                .map(|t| serde_json::from_str(&t.text).unwrap_or(Value::Null))
                .collect();
            match blocks.as_slice() {
                [Value::Object(o)] => match o.get("results") {
                    Some(Value::Array(a)) => a.clone(),
                    _ => blocks,
                },
                _ => blocks,
            }
        }
    };
    let top = items
        .iter()
        .take(3)
        .filter_map(|v| ID_FIELDS.iter().find_map(|k| v.get(*k)?.as_str()))
        .map(String::from)
        .collect();
    let n = items.iter().filter(|v| !v.is_null()).count();
    (n, top)
}

fn append(dir: &Path, line: &Value) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let file = dir.join(format!("{}.jsonl", chrono::Utc::now().format("%Y-%m")));
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)?;
    // One write per line: concurrent processes append whole lines (O_APPEND).
    f.write_all(format!("{line}\n").as_bytes())?;
    Ok(())
}

fn base(via: &str, tool: &str, elapsed: Duration) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(
        "ts".into(),
        json!(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
    );
    m.insert("via".into(), json!(via));
    m.insert("tool".into(), json!(tool));
    m.insert("ms".into(), json!(elapsed.as_millis() as u64));
    let cwd = std::env::current_dir().unwrap_or_default();
    m.insert("cwd".into(), json!(cwd.to_string_lossy()));
    m
}

/// One MCP tool call. Logging never fails a call: errors are dropped.
pub fn record_tool(
    tool: &str,
    args: Option<&Map<String, Value>>,
    result: Option<&CallToolResult>,
    elapsed: Duration,
) {
    let Some(dir) = dir() else { return };
    let mut m = base("mcp", tool, elapsed);
    m.insert(
        "args".into(),
        args.map_or(Value::Null, |a| Value::Object(a.clone())),
    );
    match result {
        Some(r) => {
            let (n, top) = summarize(r);
            m.insert("n".into(), json!(n));
            m.insert("top".into(), json!(top));
            m.insert("error".into(), json!(r.is_error.unwrap_or(false)));
        }
        None => {
            m.insert("error".into(), json!(true));
        }
    }
    let _ = append(&dir, &Value::Object(m));
}

/// One run of the prompt hook: whether it pointed anywhere, never the prompt.
pub fn record_hint(session: &str, pages: &[String], elapsed: Duration) {
    let Some(dir) = dir() else { return };
    let mut m = base("hook", "kb_hint", elapsed);
    m.insert("session".into(), json!(session));
    m.insert("n".into(), json!(pages.len()));
    m.insert("top".into(), json!(pages));
    let _ = append(&dir, &Value::Object(m));
}

/// Every line of the log written in the last `days`, oldest first.
pub fn read(dir: &Path, days: i64) -> Vec<Value> {
    let since = chrono::Utc::now() - chrono::Duration::days(days);
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    files
        .iter()
        .flat_map(|f| {
            std::fs::read_to_string(f)
                .unwrap_or_default()
                .lines()
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .filter(|v| {
            v["ts"]
                .as_str()
                .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
                .is_some_and(|t| t >= since)
        })
        .collect()
}

/// What the log says: calls per tool with their median time and how often they
/// found nothing, the queries that found nothing, the slowest calls, and how
/// often the hook pointed somewhere.
pub fn report(lines: &[Value]) -> Value {
    let mut by_tool: HashMap<String, Vec<&Value>> = HashMap::new();
    for l in lines {
        by_tool
            .entry(l["tool"].as_str().unwrap_or("?").to_string())
            .or_default()
            .push(l);
    }
    let mut tools: Vec<Value> = by_tool
        .iter()
        .map(|(tool, ls)| {
            let mut ms: Vec<u64> = ls.iter().filter_map(|l| l["ms"].as_u64()).collect();
            ms.sort_unstable();
            let empty = ls.iter().filter(|l| l["n"].as_u64() == Some(0)).count();
            let errors = ls
                .iter()
                .filter(|l| l["error"].as_bool() == Some(true))
                .count();
            json!({
                "tool": tool,
                "calls": ls.len(),
                "median_ms": ms.get(ms.len() / 2).copied().unwrap_or(0),
                "empty": empty,
                "errors": errors,
            })
        })
        .collect();
    tools.sort_by(|a, b| b["calls"].as_u64().cmp(&a["calls"].as_u64()));
    let query = |l: &Value| -> String {
        ["query", "name", "project", "package", "url"]
            .iter()
            .find_map(|k| l["args"][*k].as_str())
            .unwrap_or("")
            .to_string()
    };
    let empty: Vec<Value> = lines
        .iter()
        .rev()
        .filter(|l| l["via"] == "mcp" && l["n"].as_u64() == Some(0))
        .take(20)
        .map(|l| json!({"ts": l["ts"], "tool": l["tool"], "query": query(l)}))
        .collect();
    let mut slow: Vec<&Value> = lines.iter().filter(|l| l["via"] == "mcp").collect();
    slow.sort_by(|a, b| b["ms"].as_u64().cmp(&a["ms"].as_u64()));
    let slowest: Vec<Value> = slow
        .iter()
        .take(5)
        .map(|l| json!({"ts": l["ts"], "tool": l["tool"], "ms": l["ms"], "query": query(l)}))
        .collect();
    let hooks: Vec<&Value> = lines.iter().filter(|l| l["via"] == "hook").collect();
    let fired = hooks
        .iter()
        .filter(|l| l["n"].as_u64().unwrap_or(0) > 0)
        .count();
    let mut projects: HashMap<String, usize> = HashMap::new();
    for l in lines.iter().filter(|l| l["via"] == "mcp") {
        let cwd = l["cwd"].as_str().unwrap_or("");
        let name = Path::new(cwd)
            .file_name()
            .map_or_else(|| cwd.to_string(), |n| n.to_string_lossy().into_owned());
        *projects.entry(name).or_default() += 1;
    }
    let mut projects: Vec<(String, usize)> = projects.into_iter().collect();
    projects.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    json!({
        "lines": lines.len(),
        "tools": tools,
        "found_nothing": empty,
        "slowest": slowest,
        "hook": {"runs": hooks.len(), "pointed": fired},
        "projects": projects.iter().take(10).map(|(p, n)| json!({"project": p, "calls": n})).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use rmcp::model::{CallToolResult, ContentBlock};
    use serde_json::json;

    use super::{read, report, summarize};

    #[test]
    fn a_list_result_counts_its_items_and_names_the_first_three() {
        let mut r = CallToolResult::success(vec![]);
        r.structured_content = Some(json!({"result": [
            {"file": "wiki/a.md"}, {"file": "wiki/b.md"}, {"file": "wiki/c.md"}, {"file": "wiki/d.md"}
        ]}));
        assert_eq!(
            summarize(&r),
            (
                4,
                vec!["wiki/a.md".into(), "wiki/b.md".into(), "wiki/c.md".into()]
            )
        );
    }

    #[test]
    fn a_dict_with_results_counts_those_and_an_empty_one_counts_zero() {
        let r = CallToolResult::success(vec![ContentBlock::text(
            json!({"results": [{"session_id": "s1"}], "query": "x"}).to_string(),
        )]);
        assert_eq!(summarize(&r), (1, vec!["s1".into()]));
        let mut empty = CallToolResult::success(vec![]);
        empty.structured_content = Some(json!({"result": []}));
        assert_eq!(summarize(&empty).0, 0);
    }

    #[test]
    fn the_report_names_what_found_nothing_and_what_was_slow() {
        let t = tempfile::tempdir().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let lines = [
            json!({"ts": now, "via": "mcp", "tool": "kb_search", "ms": 12, "n": 3, "args": {"query": "sandbox"}, "cwd": "/w/solopreneur"}),
            json!({"ts": now, "via": "mcp", "tool": "kb_search", "ms": 900, "n": 0, "args": {"query": "лайа на маке"}, "cwd": "/w/solopreneur"}),
            json!({"ts": now, "via": "hook", "tool": "kb_hint", "ms": 20, "n": 1, "top": ["wiki/x.md"]}),
            json!({"ts": "2020-01-01T00:00:00Z", "via": "mcp", "tool": "kb_search", "ms": 1, "n": 1}),
        ];
        let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(t.path().join("2026-10.jsonl"), text + "not json\n").unwrap();
        let read = read(t.path(), 30);
        assert_eq!(read.len(), 3, "old lines and junk are left out");
        let r = report(&read);
        assert_eq!(r["tools"][0]["tool"], "kb_search");
        assert_eq!(r["tools"][0]["calls"], 2);
        assert_eq!(r["tools"][0]["empty"], 1);
        assert_eq!(r["found_nothing"][0]["query"], "лайа на маке");
        assert_eq!(r["slowest"][0]["ms"], 900);
        assert_eq!(r["hook"], json!({"runs": 1, "pointed": 1}));
        assert_eq!(
            r["projects"][0],
            json!({"project": "solopreneur", "calls": 2})
        );
    }
}
