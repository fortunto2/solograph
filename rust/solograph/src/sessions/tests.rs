//! Port of tests/test_session_coverage.py.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{SessionSearch, clean_title, coverage, read_transcript};
use crate::fts::SyncReport;

fn line(v: &Value) -> String {
    format!("{}\n", serde_json::to_string(v).unwrap())
}

fn transcript(path: &Path, prompt: &str, answer: &str) -> PathBuf {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let lines = [
        json!({"type": "user", "cwd": "/w/proj", "timestamp": "2026-10-01T10:00:00Z", "message": {"content": prompt}}),
        json!({"type": "user", "isMeta": true, "message": {"content": [{"type": "text", "text": "Base directory for this skill: /skills/build"}]}}),
        json!({"type": "assistant", "message": {"model": "m", "content": [{"type": "text", "text": answer}]}}),
    ];
    std::fs::write(path, lines.iter().map(line).collect::<String>()).unwrap();
    path.to_path_buf()
}

fn search(tmp: &Path) -> SessionSearch {
    SessionSearch::open(Some(&tmp.join("index.db")), Some(&tmp.join("claude"))).unwrap()
}

fn proj(tmp: &Path) -> PathBuf {
    tmp.join("claude").join("projects").join("-w-proj")
}

fn ids(out: &Value) -> Vec<String> {
    out["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["session_id"].as_str().unwrap().to_string())
        .collect()
}

fn append(path: &Path, text: &str) {
    std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
}

#[test]
fn a_new_transcript_is_found_on_the_next_search() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    transcript(
        &proj(t.path()).join("s1.jsonl"),
        "настрой вебхук HubSpot для сделок",
        "Done.",
    );
    let out = s.search("hubspot вебхуки", 5, None).unwrap();
    assert_eq!(ids(&out), ["s1"]);
    assert_eq!(out["coverage"]["status"], "complete");
    assert_eq!(out["coverage"]["read_now"], 1);
}

#[test]
fn a_deleted_transcript_stays_searchable_and_says_so() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    let path = transcript(
        &proj(t.path()).join("s1.jsonl"),
        "audio to midi model training",
        "Done.",
    );
    s.search("midi", 5, None).unwrap();
    std::fs::remove_file(path).unwrap();
    let out = s.search("midi", 5, None).unwrap();
    assert_eq!(out["results"][0]["session_id"], "s1");
    assert!(
        out["results"][0]["transcript"]
            .as_str()
            .unwrap()
            .contains("only copy")
    );
    assert_eq!(out["coverage"]["archived"], 1);
}

#[test]
fn the_answer_is_searchable_not_just_the_prompt() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    transcript(
        &proj(t.path()).join("s1.jsonl"),
        "почему падает сборка",
        "Причина: ldflags потерял -lsqlite3",
    );
    assert_eq!(ids(&s.search("lsqlite3", 5, None).unwrap()), ["s1"]);
}

#[test]
fn skill_expansions_are_not_the_session() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    transcript(&proj(t.path()).join("s1.jsonl"), "fix the parser", "Done.");
    assert!(ids(&s.search("base directory skill", 5, None).unwrap()).is_empty());
}

#[test]
fn an_unreadable_directory_is_unknown_never_complete() {
    let t = tempfile::tempdir().unwrap();
    let mut s = SessionSearch::open(
        Some(&t.path().join("index.db")),
        Some(&t.path().join("nowhere")),
    )
    .unwrap();
    assert_eq!(
        s.search("anything", 5, None).unwrap()["coverage"]["status"],
        "unknown"
    );
}

#[test]
fn a_transcript_that_cannot_be_read_makes_the_answer_partial() {
    let r = SyncReport {
        read: 2,
        failed: vec!["s3".into()],
        indexed: 2,
        ..Default::default()
    };
    let cov = coverage(Some(&(r, 3, 0)), 2);
    assert_eq!(cov["status"], "partial");
    assert_eq!(cov["failed"], json!(["s3"]));
    assert!(cov["why"].as_str().unwrap().contains("1 transcript"));
}

#[test]
fn complete_only_with_nothing_failed() {
    let r = SyncReport {
        indexed: 3,
        ..Default::default()
    };
    assert_eq!(coverage(Some(&(r, 3, 0)), 3)["status"], "complete");
}

#[test]
fn project_filter() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    transcript(
        &proj(t.path()).join("s1.jsonl"),
        "deploy the landing",
        "Done.",
    );
    transcript(
        &t.path().join("claude/projects/-w-other/s2.jsonl"),
        "deploy the landing",
        "Done.",
    );
    assert_eq!(
        ids(&s.search("deploy landing", 5, Some("other")).unwrap()),
        ["s2"]
    );
}

#[test]
fn a_growing_transcript_is_read_from_where_it_stopped() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    let path = transcript(
        &proj(t.path()).join("s1.jsonl"),
        "start the export",
        "Done.",
    );
    s.search("export", 5, None).unwrap();
    // Proof that only the tail is read: the head changes in place, same length.
    let head = std::fs::read(&path).unwrap();
    let pos = head.windows(6).position(|w| w == b"export").unwrap();
    let mut changed = head.clone();
    changed[pos..pos + 6].copy_from_slice(b"zebras");
    std::fs::write(&path, changed).unwrap();
    append(
        &path,
        &line(
            &json!({"type": "assistant", "message": {"content": [{"type": "text", "text": "Saved as parquet"}]}}),
        ),
    );
    let out = s.search("parquet", 5, None).unwrap();
    assert_eq!(out["results"][0]["session_id"], "s1");
    assert_eq!(out["results"][0]["title"], "start the export");
    assert!(ids(&s.search("zebras", 5, None).unwrap()).is_empty());
    let offset: i64 = s
        .store
        .db
        .query_row("SELECT offset FROM docs WHERE doc_id = 's1'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(offset as u64, std::fs::metadata(&path).unwrap().len());
}

#[test]
fn a_line_still_being_written_is_read_once_it_is_whole() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    let path = transcript(
        &proj(t.path()).join("s1.jsonl"),
        "first prompt here",
        "Done.",
    );
    let tail = line(
        &json!({"type": "assistant", "message": {"content": [{"type": "text", "text": "kubernetes rollout done"}]}}),
    );
    append(&path, &tail[..30]);
    assert!(ids(&s.search("kubernetes", 5, None).unwrap()).is_empty());
    append(&path, &tail[30..]);
    assert_eq!(ids(&s.search("kubernetes", 5, None).unwrap()), ["s1"]);
}

#[test]
fn claude_codes_own_title_wins_over_the_first_prompt() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    let path = transcript(
        &proj(t.path()).join("s1.jsonl"),
        "ну что там с графом",
        "Done.",
    );
    assert_eq!(
        s.search("графом", 5, None).unwrap()["results"][0]["title"],
        "ну что там с графом"
    );
    append(
        &path,
        &line(
            &json!({"type": "ai-title", "aiTitle": "Solograph freshness fix", "sessionId": "s1"}),
        ),
    );
    assert_eq!(
        s.search("графом", 5, None).unwrap()["results"][0]["title"],
        "Solograph freshness fix"
    );
}

#[test]
fn batch_runs_from_temp_directories_are_not_conversations() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    transcript(
        &proj(t.path()).join("s1.jsonl"),
        "label these turns: deploy landing",
        "Done.",
    );
    transcript(
        &t.path().join("claude/projects/-private-tmp/b1.jsonl"),
        "label these turns: deploy landing",
        "Done.",
    );
    assert_eq!(ids(&s.search("deploy landing", 5, None).unwrap()), ["s1"]);
}

#[test]
fn a_session_inside_a_private_folder_is_never_indexed() {
    let t = tempfile::tempdir().unwrap();
    let private_cwd = t.path().join("home/personal/family");
    std::fs::create_dir_all(&private_cwd).unwrap();
    let path = transcript(
        &proj(t.path()).join("s1.jsonl"),
        "семейная история, не для чужих глаз",
        "Done.",
    );
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\"/w/proj\"", &format!("\"{}\"", private_cwd.display()));
    std::fs::write(&path, text).unwrap();
    assert_eq!(
        ids(&search(t.path()).search("семейная", 5, None).unwrap()),
        ["s1"]
    );
    std::fs::write(t.path().join("home/personal/.solograph-private"), "").unwrap();
    let out = search(t.path()).search("семейная", 5, None).unwrap();
    assert!(ids(&out).is_empty());
    assert_eq!(out["coverage"]["indexed"], 0);
}

#[test]
fn the_reader_matches_python_on_meta() {
    let t = tempfile::tempdir().unwrap();
    let path = transcript(
        &proj(t.path()).join("s1.jsonl"),
        "hello there world",
        "Done.",
    );
    let doc = read_transcript(&path, 0).unwrap();
    assert_eq!(doc.kind, "proj");
    assert_eq!(
        Value::Object(doc.meta),
        json!({"session_id": "s1", "project": "proj", "project_path": "/w/proj", "branch": "",
               "started_at": "2026-10-01T10:00:00Z", "ended_at": "2026-10-01T10:00:00Z"})
    );
}

#[test]
fn a_system_line_is_never_the_title_and_image_marks_are_dropped() {
    assert_eq!(
        clean_title("[Request interrupted by user for tool use]"),
        None
    );
    assert_eq!(
        clean_title("[Image #2] [Image #3] и вкладку агента").as_deref(),
        Some("и вкладку агента")
    );
    let t = tempfile::tempdir().unwrap();
    let path = proj(t.path()).join("s1.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let lines = [
        json!({"type": "user", "cwd": "/w/proj", "message": {"content": "[Request interrupted by user]"}}),
        json!({"type": "user", "message": {"content": "[Image #1] почини экспорт видео"}}),
    ];
    std::fs::write(&path, lines.iter().map(line).collect::<String>()).unwrap();
    assert_eq!(
        read_transcript(&path, 0).unwrap().title,
        "почини экспорт видео"
    );
}

#[test]
fn an_archived_title_that_was_a_system_line_is_fixed_on_open() {
    let t = tempfile::tempdir().unwrap();
    let s = search(t.path());
    let body = "[Project: solopreneur | 2026-03-01]\n\nPrompts:\n1. [Request interrupted by user for tool use]\n2. разбери чеки за март\n";
    let doc = crate::fts::Doc {
        title: "[Request interrupted by user for tool use]".into(),
        body: body.into(),
        kind: "solopreneur".into(),
        meta: serde_json::Map::new(),
        end: None,
    };
    crate::fts::FtsStore::put(&s.store.db, "old", Some(&doc), None, None, true).unwrap();
    drop(s);
    let mut s = search(t.path());
    let out = s.search("чеки", 5, None).unwrap();
    assert_eq!(out["results"][0]["title"], "разбери чеки за март");
}

#[test]
fn all_words_rank_above_some_words() {
    let t = tempfile::tempdir().unwrap();
    let mut s = search(t.path());
    // OR alone ranks "some" first: short, two of the words, while "all" buries all
    // three in a long session and the third word is common (every filler has it).
    transcript(
        &proj(t.path()).join("some.jsonl"),
        "migrate migrate billing",
        "ok",
    );
    let long = format!(
        "{} migrate billing webhook",
        "planning notes about the week ".repeat(15)
    );
    transcript(&proj(t.path()).join("all.jsonl"), &long, "ok");
    for i in 0..4 {
        transcript(
            &proj(t.path()).join(format!("f{i}.jsonl")),
            "webhook retries and logs",
            "ok",
        );
    }
    let out = s.search("migrate billing webhook", 2, None).unwrap();
    assert_eq!(ids(&out), ["all", "some"]);
}
