//! The usage log as a running server and hook write it: a line per MCP tool call
//! with the arguments, the result count and the first ids; a line per hook run
//! without the prompt; nothing at all from a private folder or with the log off.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

fn kb(root: &Path) {
    let wiki = root.join("wiki");
    std::fs::create_dir_all(&wiki).unwrap();
    std::fs::write(
        wiki.join("sandbox.md"),
        "---\ntitle: Agent sandbox\n---\nRun the agent in a sandbox.\n",
    )
    .unwrap();
}

/// Starts the server in `cwd`, calls kb_search for "sandbox" and for a word no
/// page has, and returns the log lines.
fn session(cwd: &Path, log: &Path, extra: &[(&str, &str)]) -> Vec<Value> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_solograph"));
    cmd.current_dir(cwd)
        .env("KB_PATH", cwd.join("kb"))
        .env("SOLOGRAPH_LOG_DIR", log)
        .env("HOME", cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    // A request id of 0 is a notification: sent, no answer read.
    let mut rpc = |id: u64, method: &str, params: Value| {
        let msg = if id == 0 {
            json!({"jsonrpc": "2.0", "method": method})
        } else {
            json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
        };
        writeln!(stdin, "{msg}").unwrap();
        if id != 0 {
            let mut line = String::new();
            stdout.read_line(&mut line).unwrap();
        }
    };
    rpc(
        1,
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}),
    );
    rpc(0, "notifications/initialized", Value::Null);
    rpc(
        2,
        "tools/call",
        json!({"name": "kb_search", "arguments": {"query": "sandbox"}}),
    );
    rpc(
        3,
        "tools/call",
        json!({"name": "kb_search", "arguments": {"query": "zzqx"}}),
    );
    drop(stdin);
    child.wait().unwrap();
    read(log)
}

fn read(log: &Path) -> Vec<Value> {
    std::fs::read_dir(log)
        .into_iter()
        .flatten()
        .flatten()
        .flat_map(|e| {
            std::fs::read_to_string(e.path())
                .unwrap_or_default()
                .lines()
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .map(|l| serde_json::from_str(&l).unwrap())
        .collect()
}

#[test]
fn every_tool_call_is_one_line_with_what_it_found() {
    let t = tempfile::tempdir().unwrap();
    kb(&t.path().join("kb"));
    let lines = session(t.path(), &t.path().join("log"), &[]);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(lines[0]["via"], "mcp");
    assert_eq!(lines[0]["tool"], "kb_search");
    assert_eq!(lines[0]["args"]["query"], "sandbox");
    assert_eq!(lines[0]["n"], 1);
    assert_eq!(lines[0]["top"], json!(["wiki/sandbox.md"]));
    assert_eq!(
        lines[1]["n"], 0,
        "a query that found nothing is the point of the log"
    );
    assert!(lines[0]["ms"].is_u64() && lines[0]["ts"].is_string());
}

#[test]
fn a_private_folder_or_log_off_writes_nothing() {
    let t = tempfile::tempdir().unwrap();
    kb(&t.path().join("kb"));
    std::fs::write(t.path().join(".solograph-private"), "").unwrap();
    assert!(session(t.path(), &t.path().join("log"), &[]).is_empty());

    let u = tempfile::tempdir().unwrap();
    kb(&u.path().join("kb"));
    assert!(session(u.path(), &u.path().join("log"), &[("SOLOGRAPH_LOG", "off")]).is_empty());
}

#[test]
fn the_hook_logs_its_decision_never_the_prompt() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("kb");
    kb(&root);
    let log = t.path().join("log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_solograph"))
        .args(["kb-hint", "--hook", "--kb-path"])
        .arg(&root)
        .env("HOME", t.path())
        .env("SOLOGRAPH_LOG_DIR", &log)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let prompt = "a secret prompt about my sandbox plans";
    child
        .stdin
        .take()
        .unwrap()
        .write_all(
            json!({"prompt": prompt, "session_id": "s1"})
                .to_string()
                .as_bytes(),
        )
        .unwrap();
    child.wait().unwrap();
    let lines = read(&log);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0]["via"], "hook");
    assert_eq!(lines[0]["session"], "s1");
    assert!(!lines[0].to_string().contains("secret"), "{}", lines[0]);
}

#[test]
fn a_key_in_a_query_is_not_kept_in_the_log() {
    let t = tempfile::tempdir().unwrap();
    kb(&t.path().join("kb"));
    let log = t.path().join("log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_solograph"))
        .current_dir(t.path())
        .env("KB_PATH", t.path().join("kb"))
        .env("SOLOGRAPH_LOG_DIR", &log)
        .env("HOME", t.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    for msg in [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "kb_search", "arguments": {"query": "why does sk-ant-api03-abcdefghijklmnopqrstuvwxyz fail"}}}),
    ] {
        writeln!(stdin, "{msg}").unwrap();
        if msg.get("id").is_some() {
            line.clear();
            stdout.read_line(&mut line).unwrap();
        }
    }
    drop(stdin);
    child.wait().unwrap();
    let logged = read(&log);
    assert_eq!(logged.len(), 1);
    let q = logged[0]["args"]["query"].as_str().unwrap();
    assert!(
        q.contains("[redacted:anthropic-key]") && !q.contains("sk-ant-api03"),
        "{q}"
    );
}
