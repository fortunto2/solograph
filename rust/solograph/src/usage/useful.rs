//! Whether solograph helps, in five numbers from the agents' own calls: how many
//! sessions searched at all, whether a hint from the hook was followed, whether a
//! search found something, whether the agent then opened what it found, and
//! whether it had to ask again. Only lines with a Claude Code session count:
//! scripts and evals run without one, and they made 47 of 58 calls in the first
//! log (Oct 2026), which would have read as usage.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

/// A search redone this soon, by the same tool in the same session, means the
/// first answer did not do. Sooner than `PARALLEL_SECS` it is one batch of
/// searches sent together, not a second try.
const RETRY_SECS: i64 = 120;
const PARALLEL_SECS: i64 = 3;

fn secs(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|t| t.timestamp())
}

/// An agent's call: an MCP line written inside a Claude Code session.
fn agent_call(l: &Value) -> bool {
    l["via"] == "mcp" && l["session"].as_str().is_some_and(|s| !s.is_empty())
}

/// What one transcript shows after a moment: the tool inputs the agent sent until
/// the user's next prompt.
pub struct Turns {
    /// (unix seconds, tool name, input as JSON text) or a prompt (name "").
    events: Vec<(i64, String, String)>,
}

/// A user line the person typed: not a tool result, not text Claude Code inserted
/// (`isMeta` skill bodies and images, `<task-notification>`, `<local-command-…>`).
fn typed(o: &Value) -> bool {
    if o["type"] != "user" || o["isMeta"] == true {
        return false;
    }
    let own = |t: &str| !t.starts_with('<') && !t.starts_with("[Image");
    match &o["message"]["content"] {
        Value::String(t) => own(t),
        Value::Array(bs) => {
            !bs.iter().any(|b| b["type"] == "tool_result")
                && bs
                    .iter()
                    .any(|b| b["type"] == "text" && b["text"].as_str().is_some_and(own))
        }
        _ => false,
    }
}

impl Turns {
    /// A session's transcript, with the tool calls of its subagents
    /// (`<session>/subagents/*.jsonl`): their searches are logged under it.
    pub fn read(path: &Path) -> Option<Self> {
        let mut events = Vec::new();
        Self::scan(path, true, &mut events)?;
        let subagents = path.with_extension("").join("subagents");
        for f in std::fs::read_dir(subagents).into_iter().flatten().flatten() {
            let _ = Self::scan(&f.path(), false, &mut events);
        }
        events.sort_by_key(|e| e.0);
        Some(Self { events })
    }

    fn scan(path: &Path, prompts: bool, events: &mut Vec<(i64, String, String)>) -> Option<()> {
        use std::io::BufRead;
        let file = std::io::BufReader::new(std::fs::File::open(path).ok()?);
        for line in file.lines().map_while(Result::ok) {
            // Live transcripts reach hundreds of MB, most of it tool results: parse
            // only the lines that can be a tool call or a prompt.
            let call = line.contains("\"tool_use\"");
            let prompt = line.contains("\"type\":\"user\"") && !line.contains("\"tool_result\"");
            if !call && !prompt {
                continue;
            }
            let Ok(o) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            let Some(t) = o["timestamp"].as_str().and_then(secs) else {
                continue;
            };
            if prompts && typed(&o) {
                events.push((t, String::new(), String::new()));
            }
            for b in o["message"]["content"].as_array().into_iter().flatten() {
                if b["type"] == "tool_use" {
                    let name = b["name"].as_str().unwrap_or("").to_string();
                    events.push((t, name, b["input"].to_string()));
                }
            }
        }
        Some(())
    }

    /// Whether, after `at` and before the next prompt, one of `ids` was handed to
    /// a tool: a page read, a file opened, a found session read with
    /// `session_search(session_id=…)`. Another search by `tool` itself counts only
    /// in that last form: its query is words, not what it found.
    pub fn opened(&self, at: i64, tool: &str, ids: &[&str]) -> bool {
        self.events
            .iter()
            .skip_while(|(t, _, _)| *t < at)
            .take_while(|(_, name, _)| !name.is_empty())
            .any(|(_, name, input)| {
                (!name.ends_with(tool) || input.contains("\"session_id\""))
                    && ids.iter().any(|id| input.contains(id))
            })
    }
}

/// The transcript of `session` under `claude_dir`, if it is still on disk.
pub fn transcript(claude_dir: &Path, session: &str) -> Option<PathBuf> {
    std::fs::read_dir(claude_dir.join("projects"))
        .ok()?
        .flatten()
        .map(|d| d.path().join(format!("{session}.jsonl")))
        .find(|p| p.exists())
}

/// The five numbers. `turns` gives a session's transcript when it can be read.
pub fn report(lines: &[Value], turns: &dyn Fn(&str) -> Option<Turns>) -> Value {
    let calls: Vec<&Value> = lines.iter().filter(|l| agent_call(l)).collect();
    let hooks: Vec<&Value> = lines.iter().filter(|l| l["via"] == "hook").collect();
    let session = |l: &Value| l["session"].as_str().unwrap_or("").to_string();
    let seen: HashSet<String> = hooks.iter().chain(&calls).map(|l| session(l)).collect();
    let searched: HashSet<String> = calls.iter().map(|l| session(l)).collect();

    // A hint is followed when the agent calls solograph before the next prompt.
    let mut followed = 0;
    let mut hinted = 0;
    for (i, h) in hooks.iter().enumerate() {
        if h["n"].as_u64().unwrap_or(0) == 0 && h["nudged"] != true {
            continue;
        }
        hinted += 1;
        let (s, from) = (session(h), h["ts"].as_str().and_then(secs).unwrap_or(0));
        let until = hooks[i + 1..]
            .iter()
            .find(|n| session(n) == s)
            .and_then(|n| n["ts"].as_str().and_then(secs))
            .unwrap_or(i64::MAX);
        followed += usize::from(calls.iter().any(|c| {
            session(c) == s
                && c["ts"]
                    .as_str()
                    .and_then(secs)
                    .is_some_and(|t| t >= from && t < until)
        }));
    }

    // Reading a found session (`session_id`) is the use of a search, not one.
    let searches: Vec<&&Value> = calls
        .iter()
        .filter(|l| l["tool"].as_str().is_some_and(|t| t.ends_with("_search")))
        .filter(|l| l["args"]["session_id"].is_null())
        .collect();
    let found: Vec<&&Value> = searches
        .iter()
        .filter(|l| l["n"].as_u64().unwrap_or(0) > 0)
        .copied()
        .collect();
    let retried = searches
        .iter()
        .enumerate()
        .filter(|(i, l)| {
            let t = l["ts"].as_str().and_then(secs).unwrap_or(0);
            searches[i + 1..].iter().any(|n| {
                session(n) == session(l)
                    && n["tool"] == l["tool"]
                    && n["args"] != l["args"]
                    && n["ts"]
                        .as_str()
                        .and_then(secs)
                        .is_some_and(|nt| (PARALLEL_SECS..=RETRY_SECS).contains(&(nt - t)))
            })
        })
        .count();

    let mut cache: HashMap<String, Option<Turns>> = HashMap::new();
    let (mut checked, mut opened) = (0, 0);
    for l in &found {
        let s = session(l);
        let Some(t) = cache.entry(s.clone()).or_insert_with(|| turns(&s)) else {
            continue;
        };
        let ids: Vec<&str> = l["top"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        checked += 1;
        let at = l["ts"].as_str().and_then(secs).unwrap_or(0);
        opened += usize::from(t.opened(at, l["tool"].as_str().unwrap_or(""), &ids));
    }
    json!({
        "sessions": seen.len(),
        "searched": searched.len(),
        "hinted": hinted,
        "followed": followed,
        "searches": searches.len(),
        "found": found.len(),
        "checked": checked,
        "opened": opened,
        "retried": retried,
        "errors": calls.iter().filter(|l| l["error"] == true).count(),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Turns, report};

    fn turns(lines: &[serde_json::Value]) -> Turns {
        let t = tempfile::NamedTempFile::new().unwrap();
        let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(t.path(), text).unwrap();
        Turns::read(t.path()).unwrap()
    }

    #[test]
    fn opening_a_found_page_before_the_next_prompt_counts_and_after_it_does_not() {
        let t = turns(&[
            json!({"type": "user", "timestamp": "2026-10-29T10:00:00Z", "message": {"content": "как мы делали песочницу?"}}),
            json!({"type": "assistant", "timestamp": "2026-10-29T10:00:05Z", "message": {"content": [
                {"type": "tool_use", "name": "mcp__solograph__kb_search", "input": {"query": "sandbox-only-here"}}]}}),
            json!({"type": "assistant", "timestamp": "2026-10-29T10:00:09Z", "message": {"content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/kb/wiki/sandbox.md"}}]}}),
            json!({"type": "user", "timestamp": "2026-10-29T10:01:00Z", "message": {"content": [{"type": "text", "text": "ок"}]}}),
            json!({"type": "assistant", "timestamp": "2026-10-29T10:01:05Z", "message": {"content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/kb/wiki/other.md"}}]}}),
        ]);
        let at = chrono::DateTime::parse_from_rfc3339("2026-10-29T10:00:06Z")
            .unwrap()
            .timestamp();
        assert!(t.opened(at, "kb_search", &["wiki/sandbox.md"]));
        // The search's own input is not an opening of what it found.
        assert!(!t.opened(at - 2, "kb_search", &["sandbox-only-here"]));
        // A page read only after the next prompt was not this search's doing.
        assert!(!t.opened(at, "kb_search", &["wiki/other.md"]));

        let skill = turns(&[
            json!({"type": "assistant", "timestamp": "2026-10-29T10:00:07Z", "message": {"content": [
                {"type": "tool_use", "name": "Skill", "input": {"skill": "x"}}]}}),
            json!({"type": "user", "isMeta": true, "timestamp": "2026-10-29T10:00:08Z", "message": {"content": [{"type": "text", "text": "Base directory for this skill"}]}}),
            json!({"type": "user", "timestamp": "2026-10-29T10:00:08Z", "message": {"content": "<task-notification>done</task-notification>"}}),
            json!({"type": "assistant", "timestamp": "2026-10-29T10:00:09Z", "message": {"content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/kb/wiki/sandbox.md"}}]}}),
        ]);
        assert!(
            skill.opened(at, "kb_search", &["wiki/sandbox.md"]),
            "text Claude Code inserts is not the next prompt"
        );

        let read = turns(&[
            json!({"type": "assistant", "timestamp": "2026-10-29T10:00:09Z", "message": {"content": [
                {"type": "tool_use", "name": "mcp__solograph__session_search", "input": {"query": "векторы", "session_id": "abc123"}}]}}),
        ]);
        assert!(
            read.opened(at, "session_search", &["abc123"]),
            "reading a found session is opening it"
        );
    }

    #[test]
    fn the_numbers_count_only_agent_calls_and_follow_hints_within_the_turn() {
        let call = |ts: &str, s: &str, q: &str, n: u64| json!({"ts": ts, "via": "mcp", "tool": "kb_search", "session": s, "args": {"query": q}, "n": n, "top": if n > 0 { json!(["wiki/a.md"]) } else { json!([]) }});
        let lines = [
            json!({"ts": "2026-10-29T10:00:00Z", "via": "hook", "tool": "kb_hint", "session": "a", "n": 1}),
            call("2026-10-29T10:00:10Z", "a", "песочница", 0),
            call("2026-10-29T10:00:40Z", "a", "sandbox", 1),
            json!({"ts": "2026-10-29T10:05:00Z", "via": "hook", "tool": "kb_hint", "session": "a", "n": 0, "nudged": true}),
            json!({"ts": "2026-10-29T10:09:00Z", "via": "hook", "tool": "kb_hint", "session": "a", "n": 0}),
            json!({"ts": "2026-10-29T10:00:00Z", "via": "hook", "tool": "kb_hint", "session": "b", "n": 0}),
            // An eval's call: no session, not an agent.
            json!({"ts": "2026-10-29T10:00:00Z", "via": "mcp", "tool": "kb_search", "args": {"query": "x"}, "n": 0}),
        ];
        let mut lines = lines.to_vec();
        // Reading the found session is not a search, and not a retry of one.
        lines.push(json!({"ts": "2026-10-29T10:00:50Z", "via": "mcp", "tool": "kb_search", "session": "a", "args": {"query": "sandbox", "session_id": "x"}, "n": 2}));
        // Two searches sent together are one batch, not a retry.
        lines.push(call("2026-10-29T10:20:00Z", "b", "one", 1));
        lines.push(call("2026-10-29T10:20:01Z", "b", "two", 1));
        let r = report(&lines, &|_| None);
        assert_eq!(r["sessions"], 2);
        assert_eq!(r["searched"], 2);
        assert_eq!(
            (r["hinted"].clone(), r["followed"].clone()),
            (json!(2), json!(1))
        );
        assert_eq!(
            (r["searches"].clone(), r["found"].clone()),
            (json!(4), json!(3))
        );
        assert_eq!(
            r["retried"], 1,
            "only the empty search, asked again in 30 s"
        );
        assert_eq!(
            r["checked"], 0,
            "no transcript, nothing claimed about opening"
        );
    }
}
