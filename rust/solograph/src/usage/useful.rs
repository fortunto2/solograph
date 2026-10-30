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
pub const RETRY_SECS: i64 = 120;
const PARALLEL_SECS: i64 = 3;

pub fn secs(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|t| t.timestamp())
}

/// An agent's call: an MCP line written inside a Claude Code session.
fn agent_call(l: &Value) -> bool {
    l["via"] == "mcp" && l["session"].as_str().is_some_and(|s| !s.is_empty())
}

fn session(l: &Value) -> &str {
    l["session"].as_str().unwrap_or("")
}

fn ts(l: &Value) -> Option<i64> {
    l["ts"].as_str().and_then(secs)
}

/// A transcript line's time, read without parsing the line: the last
/// `"timestamp"` key, since Claude Code writes it after the message.
fn stamp(line: &str) -> Option<i64> {
    let at = line.rfind("\"timestamp\":\"")? + "\"timestamp\":\"".len();
    secs(&line[at..at + line[at..].find('"')?])
}

/// A tool call in a transcript: its input as JSON text, and the session it
/// reads when it is `session_search(session_id=…)`.
struct Call {
    name: String,
    input: String,
    reads: Option<String>,
}

/// What one transcript shows after a moment: the tool calls the agent made until
/// the user's next prompt.
pub struct Turns {
    /// (unix seconds, the call), or a prompt (None), in time order.
    events: Vec<(i64, Option<Call>)>,
}

/// A user line the person typed (`sessions::typed`): not a tool result, not an
/// `isMeta` line (skill bodies, image notes).
fn typed(o: &Value) -> bool {
    if o["type"] != "user" || o["isMeta"] == true {
        return false;
    }
    let own = crate::sessions::typed;
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
    /// A session's transcript from `since` (unix seconds) on, with the tool calls
    /// of its subagents (`<session>/subagents/*.jsonl`): their searches are logged
    /// under it. Nothing earlier can be what a later search led to.
    pub fn read(path: &Path, since: i64) -> Option<Self> {
        let mut events = Vec::new();
        Self::scan(path, since, true, &mut events)?;
        let subagents = path.with_extension("").join("subagents");
        for f in std::fs::read_dir(subagents).into_iter().flatten().flatten() {
            let written = f
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok());
            if written.is_none_or(|d| d.as_secs() as i64 >= since) {
                let _ = Self::scan(&f.path(), since, false, &mut events);
            }
        }
        events.sort_by_key(|e| e.0);
        Some(Self { events })
    }

    fn scan(
        path: &Path,
        since: i64,
        prompts: bool,
        events: &mut Vec<(i64, Option<Call>)>,
    ) -> Option<()> {
        use std::io::BufRead;
        let file = std::io::BufReader::new(std::fs::File::open(path).ok()?);
        for line in file.lines().map_while(Result::ok) {
            // Live transcripts reach hundreds of MB, most of it tool results and
            // pasted images: parse only the lines that can be an agent's tool call or
            // a typed prompt, and none from before `since`.
            let call = line.contains("\"tool_use\"") && line.contains("\"type\":\"assistant\"");
            let prompt = prompts
                && line.contains("\"type\":\"user\"")
                && !line.contains("\"tool_result\"")
                && !line.contains("\"isMeta\":true");
            if !(call || prompt) || stamp(&line).is_some_and(|t| t < since) {
                continue;
            }
            let Ok(o) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            let Some(t) = o["timestamp"].as_str().and_then(secs) else {
                continue;
            };
            if prompt && typed(&o) {
                events.push((t, None));
            }
            for b in o["message"]["content"].as_array().into_iter().flatten() {
                if b["type"] == "tool_use"
                    && let Some(name) = b["name"].as_str()
                {
                    let reads = b["input"]["session_id"]
                        .as_str()
                        .filter(|s| s.len() >= 6)
                        .map(str::to_string);
                    let input = b["input"].to_string();
                    events.push((
                        t,
                        Some(Call {
                            name: name.to_string(),
                            input,
                            reads,
                        }),
                    ));
                }
            }
        }
        Some(())
    }

    /// Whether, after `at` and before the next prompt, one of `ids` was handed to
    /// a tool: a page read, a file opened, a found session read with
    /// `session_search(session_id=…)`. Another search by `tool` itself counts only
    /// in that last form: its query is words, not what it found. A session may be
    /// read by the first characters of its id, as the tool allows.
    pub fn opened(&self, at: i64, tool: &str, ids: &[&str]) -> bool {
        self.events
            .iter()
            .skip_while(|(t, _)| *t < at)
            .map_while(|(_, call)| call.as_ref())
            .any(|c| {
                (!c.name.ends_with(tool) || c.reads.is_some())
                    && ids.iter().any(|id| {
                        c.input.contains(id)
                            || c.reads.as_deref().is_some_and(|r| id.starts_with(r))
                    })
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

/// The five numbers. `turns(session, since)` gives a session's transcript from
/// `since` on, when it can be read.
pub fn report(lines: &[Value], turns: &dyn Fn(&str, i64) -> Option<Turns>) -> Value {
    let calls: Vec<&Value> = lines.iter().filter(|l| agent_call(l)).collect();
    let hooks: Vec<&Value> = lines.iter().filter(|l| l["via"] == "hook").collect();
    let seen: HashSet<&str> = hooks.iter().chain(&calls).map(|l| session(l)).collect();
    let searched: HashSet<&str> = calls.iter().map(|l| session(l)).collect();

    // A hint is followed when the agent calls solograph before the next prompt.
    let mut followed = 0;
    let mut hinted = 0;
    for (i, h) in hooks.iter().enumerate() {
        if h["n"].as_u64().unwrap_or(0) == 0 && h["nudged"] != true {
            continue;
        }
        hinted += 1;
        let (s, from) = (session(h), ts(h).unwrap_or(0));
        let until = hooks[i + 1..]
            .iter()
            .find(|n| session(n) == s)
            .and_then(|n| ts(n))
            .unwrap_or(i64::MAX);
        followed += usize::from(
            calls
                .iter()
                .any(|c| session(c) == s && ts(c).is_some_and(|t| t >= from && t < until)),
        );
    }

    // Reading a found session (`session_id`) is the use of a search, not one.
    let searches: Vec<&Value> = calls
        .iter()
        .copied()
        .filter(|l| l["tool"].as_str().is_some_and(|t| t.ends_with("_search")))
        .filter(|l| l["args"]["session_id"].is_null())
        .collect();
    let found: Vec<&Value> = searches
        .iter()
        .copied()
        .filter(|l| l["n"].as_u64().unwrap_or(0) > 0)
        .collect();
    let retried = searches
        .iter()
        .enumerate()
        .filter(|(i, l)| {
            let t = ts(l).unwrap_or(0);
            searches[i + 1..].iter().any(|n| {
                session(n) == session(l)
                    && n["tool"] == l["tool"]
                    && n["args"] != l["args"]
                    && ts(n).is_some_and(|nt| (PARALLEL_SECS..=RETRY_SECS).contains(&(nt - t)))
            })
        })
        .count();

    // Each transcript is read once, from its session's first search that found
    // something.
    let mut since: HashMap<&str, i64> = HashMap::new();
    for l in &found {
        let t = ts(l).unwrap_or(0);
        since
            .entry(session(l))
            .and_modify(|s| *s = (*s).min(t))
            .or_insert(t);
    }
    let transcripts: HashMap<&str, Turns> = since
        .into_iter()
        .filter_map(|(s, t)| Some((s, turns(s, t)?)))
        .collect();
    let (mut checked, mut opened) = (0, 0);
    for l in &found {
        let Some(t) = transcripts.get(session(l)) else {
            continue;
        };
        let ids: Vec<&str> = l["top"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        checked += 1;
        opened += usize::from(t.opened(ts(l).unwrap_or(0), l["tool"].as_str().unwrap_or(""), &ids));
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

    fn turns_since(lines: &[serde_json::Value], since: i64) -> Turns {
        let t = tempfile::NamedTempFile::new().unwrap();
        let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(t.path(), text).unwrap();
        Turns::read(t.path(), since).unwrap()
    }

    fn turns(lines: &[serde_json::Value]) -> Turns {
        turns_since(lines, 0)
    }

    #[test]
    fn opening_a_found_page_before_the_next_prompt_counts_and_after_it_does_not() {
        let lines = [
            json!({"type": "user", "timestamp": "2026-10-29T10:00:00Z", "message": {"content": "как мы делали песочницу?"}}),
            json!({"type": "assistant", "timestamp": "2026-10-29T10:00:05Z", "message": {"content": [
                {"type": "tool_use", "name": "mcp__solograph__kb_search", "input": {"query": "sandbox-only-here"}}]}}),
            json!({"type": "assistant", "timestamp": "2026-10-29T10:00:09Z", "message": {"content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/kb/wiki/sandbox.md"}}]}}),
            json!({"type": "user", "timestamp": "2026-10-29T10:01:00Z", "message": {"content": [{"type": "text", "text": "ок"}]}}),
            json!({"type": "assistant", "timestamp": "2026-10-29T10:01:05Z", "message": {"content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/kb/wiki/other.md"}}]}}),
        ];
        let t = turns(&lines);
        let at = chrono::DateTime::parse_from_rfc3339("2026-10-29T10:00:06Z")
            .unwrap()
            .timestamp();
        assert!(t.opened(at, "kb_search", &["wiki/sandbox.md"]));
        // The search's own input is not an opening of what it found.
        assert!(!t.opened(at - 2, "kb_search", &["sandbox-only-here"]));
        // A page read only after the next prompt was not this search's doing.
        assert!(!t.opened(at, "kb_search", &["wiki/other.md"]));
        // Read from a later moment, the earlier lines are not there at all.
        assert!(!turns_since(&lines, at + 10).opened(at, "kb_search", &["wiki/sandbox.md"]));

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

        for typed in [
            "<command-message>simplify</command-message> <command-name>/simplify</command-name>",
            "[Image #1] поправь чтоб не мешало",
        ] {
            let next = turns(&[
                json!({"type": "user", "timestamp": "2026-10-29T10:00:08Z", "message": {"content": typed}}),
                json!({"type": "assistant", "timestamp": "2026-10-29T10:00:09Z", "message": {"content": [
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "/kb/wiki/sandbox.md"}}]}}),
            ]);
            assert!(
                !next.opened(at, "kb_search", &["wiki/sandbox.md"]),
                "a slash command or a prompt with an image is the next prompt: {typed}"
            );
        }

        let read = turns(&[
            json!({"type": "assistant", "timestamp": "2026-10-29T10:00:09Z", "message": {"content": [
                {"type": "tool_use", "name": "mcp__solograph__session_search", "input": {"query": "векторы", "session_id": "abc123"}}]}}),
        ]);
        assert!(
            read.opened(at, "session_search", &["abc123"]),
            "reading a found session is opening it"
        );
        assert!(
            read.opened(at, "session_search", &["abc123-full-uuid"]),
            "by the first characters of its id too"
        );
        assert!(!read.opened(at, "session_search", &["zzz999-other"]));
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
        let r = report(&lines, &|_, _| None);
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
