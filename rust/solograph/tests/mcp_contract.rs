//! The binary as a client sees it: spawned, spoken to over stdio, every tool
//! called — the Rust counterpart of tests/test_tool_contracts.py. A tool's
//! result has to arrive in the shape Python's server sent: a list as one JSON
//! block per item plus structuredContent.result, a dict as one JSON block.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    id: u64,
}

impl Client {
    fn start(env: &[(&str, String)]) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_solograph"));
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().expect("binary starts");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut c = Self {
            child,
            stdin,
            stdout,
            id: 0,
        };
        let init = c.rpc("initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}));
        assert_eq!(init["result"]["serverInfo"]["name"], "solograph");
        writeln!(
            c.stdin,
            r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
        )
        .unwrap();
        c
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.id += 1;
        let msg = json!({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params});
        writeln!(self.stdin, "{msg}").unwrap();
        loop {
            let mut line = String::new();
            assert!(
                self.stdout.read_line(&mut line).unwrap() > 0,
                "server closed stdout"
            );
            let v: Value = serde_json::from_str(&line).unwrap();
            if v["id"] == json!(self.id) {
                return v;
            }
        }
    }

    fn call(&mut self, name: &str, args: Value) -> Value {
        let r = self.rpc("tools/call", json!({"name": name, "arguments": args}));
        assert!(r["result"]["isError"] != true, "{name}: {r}");
        r["result"].clone()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn list(result: &Value) -> Vec<Value> {
    let items = result["structuredContent"]["result"]
        .as_array()
        .cloned()
        .expect("a list result");
    assert_eq!(
        result["content"].as_array().unwrap().len(),
        items.len(),
        "one text block per item"
    );
    items
}

fn dict(result: &Value) -> Value {
    assert!(result.get("structuredContent").is_none_or(Value::is_null));
    serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap()
}

fn fixture(t: &Path) -> Vec<(&'static str, String)> {
    let proj = t.join("claude/projects/-w-app");
    std::fs::create_dir_all(&proj).unwrap();
    let line = json!({"type": "user", "cwd": "/w/app", "timestamp": "2026-10-01T10:00:00Z", "message": {"content": "graph rebuild please"}});
    std::fs::write(proj.join("s1.jsonl"), format!("{line}\n")).unwrap();
    std::fs::create_dir_all(t.join("kb/wiki")).unwrap();
    std::fs::write(
        t.join("kb/wiki/note.md"),
        "---\ntitle: Graph notes\ntype: concept\n---\ntree-sitter",
    )
    .unwrap();
    let repo = t.join("active/video");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("montage.py"),
        "class MontageProfile:\n    pass\n\ndef resolve_montage_profile():\n    pass\n",
    )
    .unwrap();
    let s = |p: &Path| p.display().to_string();
    vec![
        ("HOME", s(t)),
        ("CLAUDE_SESSIONS_DIR", s(&t.join("claude"))),
        ("CODEGRAPH_SESSIONS_DB", s(&t.join("db/sessions"))),
        ("KB_PATH", s(&t.join("kb"))),
        ("SOURCES_ROOT", s(&t.join("db/sources"))),
        ("SOLOGRAPH_PORTFOLIO_DB", s(&t.join("db/portfolio.db"))),
        ("CODEGRAPH_REGISTRY", s(&t.join("registry.yaml"))),
        ("CODEGRAPH_SCAN_PATH", s(&t.join("active"))),
    ]
}

#[test]
fn every_tool_answers_in_the_shape_python_did() {
    let t = tempfile::tempdir().unwrap();
    let env = fixture(t.path());
    let scan = Command::new(env!("CARGO_BIN_EXE_solograph"))
        .arg("scan")
        .envs(env.iter().map(|(k, v)| (*k, v)))
        .output()
        .unwrap();
    assert!(
        scan.status.success(),
        "{}",
        String::from_utf8_lossy(&scan.stderr)
    );

    let mut c = Client::start(&env);
    let tools: Vec<String> = c.rpc("tools/list", json!({}))["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    let mut want = vec![
        "kb_search",
        "session_search",
        "symbol_search",
        "codegraph_stats",
        "codegraph_explain",
        "codegraph_shared",
        "project_info",
        "web_fetch",
        "source_search",
        "source_tags",
        "source_related",
        "source_list",
    ];
    want.sort_unstable();
    let mut got = tools.clone();
    got.sort();
    assert_eq!(got, want);

    let sessions = dict(&c.call("session_search", json!({"query": "graph"})));
    assert_eq!(sessions["results"][0]["session_id"], "s1");
    assert_eq!(sessions["coverage"]["status"], "complete");

    assert_eq!(
        list(&c.call("kb_search", json!({"query": "graph"})))[0]["file"],
        "wiki/note.md"
    );

    let symbols = list(&c.call("symbol_search", json!({"query": "montage profile"})));
    let names: Vec<&str> = symbols
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    assert!(
        names.contains(&"MontageProfile") && names.contains(&"resolve_montage_profile"),
        "{names:?}"
    );

    assert_eq!(dict(&c.call("codegraph_stats", json!({})))["projects"], 1);
    assert_eq!(
        dict(&c.call("codegraph_explain", json!({"project": "video"})))["counts"]["symbols"],
        2
    );
    assert_eq!(
        dict(&c.call("codegraph_explain", json!({"project": "Rustam's app"})))["error"],
        "Project 'Rustam's app' not found."
    );
    assert!(list(&c.call("codegraph_shared", json!({}))).is_empty());
    assert_eq!(
        dict(&c.call("project_info", json!({"name": "video"})))["name"],
        "video"
    );
    assert_eq!(list(&c.call("project_info", json!({})))[0]["name"], "video");
    assert!(list(&c.call("source_search", json!({"query": "pricing"}))).is_empty());
    assert!(list(&c.call("source_list", json!({}))).is_empty());
    assert!(list(&c.call("source_tags", json!({}))).is_empty());
    assert!(
        list(&c.call(
            "source_related",
            json!({"video_url": "https://youtu.be/abcdefghijk"})
        ))
        .is_empty()
    );
    let fetched = dict(&c.call(
        "web_fetch",
        json!({"url": "http://127.0.0.1:9/", "timeout": 2}),
    ));
    assert_eq!(
        fetched["status"], 0,
        "an unreachable URL is an error object, not a failed call"
    );
}
