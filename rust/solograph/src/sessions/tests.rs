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
    let s2 = transcript(
        &t.path().join("claude/projects/-w-other/s2.jsonl"),
        "deploy the landing",
        "Done.",
    );
    // A session's project is its working directory, as in a real transcript.
    let text = std::fs::read_to_string(&s2)
        .unwrap()
        .replace("/w/proj", "/w/other");
    std::fs::write(&s2, text).unwrap();
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
    // kind is the working directory now, so a project filter matches the real
    // name; the displayed project is its last part, as before.
    assert_eq!(doc.kind, "/w/proj");
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
        keys: String::new(),
        body: body.into(),
        kind: "solopreneur".into(),
        meta: serde_json::Map::new(),
        end: None,
    };
    crate::fts::FtsStore::put(
        &s.store.db,
        s.store.layout.columns,
        "old",
        Some(&doc),
        None,
        None,
        true,
    )
    .unwrap();
    // An index from before the fix: titles not yet cleaned.
    s.store.db.execute_batch("PRAGMA user_version = 0").unwrap();
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

/// A transcript in `cwd`, with the given id, in the same Claude project dir as
/// every other here.
fn transcript_in(tmp: &Path, id: &str, cwd: &Path, prompt: &str) -> PathBuf {
    let path = transcript(&proj(tmp).join(format!("{id}.jsonl")), prompt, "Done.");
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\"/w/proj\"", &format!("\"{}\"", cwd.display()));
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn privacy_is_judged_per_transcript_not_from_the_first_one_in_the_folder() {
    // Review 28 Oct: the verdict was cached per Claude project dir from its first
    // transcript; a cwd-less first file made a private session searchable.
    let t = tempfile::tempdir().unwrap();
    let private_cwd = t.path().join("home/personal");
    std::fs::create_dir_all(&private_cwd).unwrap();
    std::fs::write(private_cwd.join(".solograph-private"), "").unwrap();
    std::fs::create_dir_all(proj(t.path())).unwrap();
    std::fs::write(
        proj(t.path()).join("a0.jsonl"),
        line(&json!({"type": "summary", "summary": "no cwd here"})),
    )
    .unwrap();
    transcript_in(t.path(), "s1", &private_cwd, "семейная история");
    let out = search(t.path()).search("семейная", 5, None).unwrap();
    assert!(ids(&out).is_empty(), "{out}");
}

#[test]
fn a_marker_added_while_the_server_runs_is_honoured_on_the_next_search() {
    let t = tempfile::tempdir().unwrap();
    let cwd = t.path().join("home/notes");
    std::fs::create_dir_all(&cwd).unwrap();
    transcript_in(t.path(), "s1", &cwd, "семейная история");
    let mut s = search(t.path());
    assert_eq!(ids(&s.search("семейная", 5, None).unwrap()), ["s1"]);
    std::fs::write(cwd.join(".solograph-private"), "").unwrap();
    assert!(ids(&s.search("семейная", 5, None).unwrap()).is_empty());
}

#[test]
fn an_archived_session_from_a_folder_marked_private_later_is_purged() {
    let t = tempfile::tempdir().unwrap();
    let cwd = t.path().join("home/notes");
    std::fs::create_dir_all(&cwd).unwrap();
    let path = transcript_in(t.path(), "s1", &cwd, "семейная история");
    let mut s = search(t.path());
    assert_eq!(ids(&s.search("семейная", 5, None).unwrap()), ["s1"]);
    std::fs::remove_file(path).unwrap(); // Claude Code deleted it: archived
    assert_eq!(ids(&s.search("семейная", 5, None).unwrap()), ["s1"]);
    std::fs::write(cwd.join(".solograph-private"), "").unwrap();
    assert!(ids(&s.search("семейная", 5, None).unwrap()).is_empty());
}

#[test]
fn a_transcript_with_no_cwd_at_all_is_not_indexed() {
    // Fail closed: without a working directory there is no way to tell.
    let t = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(proj(t.path())).unwrap();
    std::fs::write(
        proj(t.path()).join("s1.jsonl"),
        line(&json!({"type": "user", "message": {"content": "семейная история"}})),
    )
    .unwrap();
    assert!(ids(&search(t.path()).search("семейная", 5, None).unwrap()).is_empty());
}

#[test]
fn a_hyphenated_project_is_found_by_its_real_name() {
    // Review 28 Oct: the label came from splitting the Claude dir name on every
    // '-', so `-p video-analyzer` found nothing (30 of 53 real dirs).
    let t = tempfile::tempdir().unwrap();
    let cwd = t.path().join("w/video-analyzer");
    std::fs::create_dir_all(&cwd).unwrap();
    let path = transcript(
        &t.path().join("claude/projects/-w-video-analyzer/s1.jsonl"),
        "montage profile tuning",
        "Done.",
    );
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\"/w/proj\"", &format!("\"{}\"", cwd.display()));
    std::fs::write(&path, text).unwrap();
    let out = search(t.path())
        .search("montage", 5, Some("video-analyzer"))
        .unwrap();
    assert_eq!(ids(&out), ["s1"]);
    assert_eq!(out["results"][0]["project"], "video-analyzer");
}

#[test]
fn a_session_started_with_a_skill_is_found_by_its_arguments() {
    // Review 28 Oct: every user text starting with '<' was dropped, so the
    // arguments of /solo:research never reached the index.
    let t = tempfile::tempdir().unwrap();
    let path = proj(t.path()).join("s1.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let lines = [
        json!({"type": "user", "cwd": "/w/proj", "timestamp": "2026-10-01T10:00:00Z", "message": {"content":
            "<command-message>solo:research</command-message>\n<command-name>/solo:research</command-name>\n<command-args>therapist note-taking marketplace</command-args>"}}),
        json!({"type": "assistant", "message": {"model": "m", "content": [{"type": "text", "text": "Researching."}]}}),
    ];
    std::fs::write(&path, lines.iter().map(line).collect::<String>()).unwrap();
    let out = search(t.path())
        .search("therapist marketplace", 5, None)
        .unwrap();
    assert_eq!(ids(&out), ["s1"]);
    assert!(
        out["results"][0]["title"]
            .as_str()
            .unwrap()
            .contains("therapist"),
        "{out}"
    );
}

#[test]
fn the_resume_command_survives_a_space_in_the_path() {
    let t = tempfile::tempdir().unwrap();
    let cwd = t.path().join("w/My Project");
    std::fs::create_dir_all(&cwd).unwrap();
    let path = transcript(
        &proj(t.path()).join("s1.jsonl"),
        "space test prompt",
        "Done.",
    );
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\"/w/proj\"", &format!("\"{}\"", cwd.display()));
    std::fs::write(&path, text).unwrap();
    let out = search(t.path()).search("space test", 5, None).unwrap();
    let resume = out["results"][0]["resume"].as_str().unwrap();
    assert!(
        resume.starts_with(&format!("cd '{}'", cwd.display())),
        "{resume}"
    );
}

#[test]
fn a_secret_in_a_session_is_never_stored_or_shown() {
    let t = tempfile::tempdir().unwrap();
    transcript(
        &proj(t.path()).join("s1.jsonl"),
        "deploy failed, here is my env: OPENAI_API_KEY=abcd1234EFGH5678ijkl and AKIAZ7Q3LMN4PQR8STUV",
        "Rotate both keys.",
    );
    let mut s = search(t.path());
    let out = s.search("deploy failed env", 5, None).unwrap();
    assert_eq!(ids(&out), ["s1"]);
    let text = out.to_string();
    assert!(
        !text.contains("abcd1234EFGH5678ijkl") && !text.contains("AKIAZ7Q3LMN4PQR8STUV"),
        "{text}"
    );
    // The value itself is not in the index at all.
    assert!(ids(&s.search("AKIAZ7Q3LMN4PQR8STUV", 5, None).unwrap()).is_empty());
    let body: String = s
        .store
        .db
        .query_row("SELECT title || body FROM fts", [], |r| r.get(0))
        .unwrap();
    assert!(
        body.contains("OPENAI_API_KEY=[redacted:env-secret]"),
        "{body}"
    );
}

#[test]
fn an_index_written_before_redaction_is_cleaned_on_open() {
    let t = tempfile::tempdir().unwrap();
    transcript(
        &proj(t.path()).join("s1.jsonl"),
        "old session prompt",
        "Done.",
    );
    {
        let mut s = search(t.path());
        s.search("old", 5, None).unwrap();
        // As an older binary left it: a key straight in the body, version unset.
        s.store
            .db
            .execute_batch(
                "UPDATE fts SET body = body || ' leaked sk-ant-api03-abcdefghijklmnopqrstuvwxyz'; \
                 DROP TABLE IF EXISTS solograph_redaction;",
            )
            .unwrap();
    }
    let s = search(t.path());
    let body: String = s
        .store
        .db
        .query_row("SELECT body FROM fts", [], |r| r.get(0))
        .unwrap();
    assert!(
        body.contains("[redacted:anthropic-key]") && !body.contains("sk-ant-api03"),
        "{body}"
    );
}
