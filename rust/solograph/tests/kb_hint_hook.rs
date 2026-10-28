//! `kb-hint --hook` as Claude Code runs it: the UserPromptSubmit payload on stdin,
//! the hint back as additionalContext, and silence with exit 0 for anything that
//! goes wrong — a broken hint must never break a prompt. Until Oct 2026 a Python
//! wrapper (scripts/wiki_hint.py) did this translation, and its interpreter
//! starting up was most of the hook's 74 ms.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;

fn kb(root: &Path) {
    let wiki = root.join("wiki");
    std::fs::create_dir_all(&wiki).unwrap();
    let page = |slug: &str, title: &str, body: &str| {
        std::fs::write(
            wiki.join(format!("{slug}.md")),
            format!("---\ntitle: \"{title}\"\n---\n\n{body}\n"),
        )
        .unwrap();
    };
    page(
        "laya-open-jev-on-mac",
        "Running an open Jev on a Mac: Laya, measured",
        "Laya is an open reproduction of Jev. Core ML int8 309 MB.",
    );
    for i in 0..30 {
        page(
            &format!("filler-{i}"),
            &format!("Project note {i}"),
            "the project skill agent session build work",
        );
    }
}

fn hook(home: &Path, root: &Path, stdin: &str) -> (bool, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_solograph"))
        .args(["kb-hint", "--hook", "--kb-path"])
        .arg(root)
        .env("HOME", home)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    )
}

#[test]
fn the_hint_comes_back_as_additional_context() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("kb");
    kb(&root);
    let payload = r#"{"prompt": "how would laya run inside the harness?", "session_id": "s1"}"#;
    let (ok, out) = hook(t.path(), &root, payload);
    assert!(ok);
    let v: Value = serde_json::from_str(&out).expect(&out);
    assert_eq!(v["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit");
    let ctx = v["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(ctx.contains("laya-open-jev-on-mac"), "{ctx}");

    // Once per session: the same prompt again says nothing.
    assert_eq!(hook(t.path(), &root, payload), (true, String::new()));
}

#[test]
fn anything_wrong_is_silence_and_exit_zero() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("kb");
    kb(&root);
    for stdin in [
        "not json",
        "",
        r#"{"session_id": "s1"}"#,
        r#"{"prompt": 42}"#,
        r#"{"prompt": "the project skill agent session"}"#,
    ] {
        assert_eq!(
            hook(t.path(), &root, stdin),
            (true, String::new()),
            "{stdin}"
        );
    }
    let nowhere = t.path().join("no-such-kb");
    let payload = r#"{"prompt": "how would laya run inside the harness?"}"#;
    assert_eq!(hook(t.path(), &nowhere, payload), (true, String::new()));
}

/// A KB that has a memory gate: synthetic weights (the real ones are private).
fn kb_with_gate(root: &Path) {
    kb(root);
    std::fs::write(
        root.join("gate.json"),
        r#"{"features": "w+c34/v1", "bias": -1.2, "threshold": 0.5,
           "weights": {"w:нас": 2.0, "c: у ": 0.8, "c:али ": 1.1, "w:мы": 0.9}}"#,
    )
    .unwrap();
    std::fs::write(
        root.join(".solograph.yaml"),
        "kb:\n  memory_gate: gate.json\n",
    )
    .unwrap();
}

#[test]
fn a_prompt_leaning_on_earlier_work_gets_the_memory_nudge() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("kb");
    kb_with_gate(&root);
    let payload = r#"{"prompt": "у нас было такое, мы делали это раньше", "session_id": "g1"}"#;
    let (ok, out) = hook(t.path(), &root, payload);
    assert!(ok);
    let v: Value = serde_json::from_str(&out).expect(&out);
    let ctx = v["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(
        ctx.contains("session_search") && ctx.contains("p=0.7"),
        "{ctx}"
    );
    // A short approval never opens it, whatever its words.
    let (_, out) = hook(
        t.path(),
        &root,
        r#"{"prompt": "да мы", "session_id": "g2"}"#,
    );
    assert_eq!(out, "");
    // Without a gate in the config the hook only names pages, as before.
    let bare = t.path().join("bare");
    kb(&bare);
    assert_eq!(hook(t.path(), &bare, payload), (true, String::new()));
}
