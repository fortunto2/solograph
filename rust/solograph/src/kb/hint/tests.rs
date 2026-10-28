//! Port of tests/test_wiki_hint.py. Contract: a prompt naming something the wiki
//! is specifically about gets one line pointing at the page; a prompt of common
//! words, or a short steer like "да делай", gets nothing; a page is pointed at once
//! per session. Silence is the default: a false hint costs context on every prompt.

use std::path::{Path, PathBuf};

use super::hint;
use crate::kb::KnowledgeBase;

fn pages() -> Vec<(String, String, String)> {
    let mut p: Vec<(String, String, String)> = vec![
        (
            "laya-open-jev-on-mac".into(),
            "Running an open Jev on a Mac: Laya, measured".into(),
            "Laya is an open reproduction of Jev. Core ML int8 309 MB, 13.7 ms per question."
                .into(),
        ),
        (
            "jev-system-one".into(),
            "Jev: System One decision model".into(),
            "Jev returns calibrated probabilities.".into(),
        ),
        (
            "codex-oauth-proxy".into(),
            "Codex OAuth proxy".into(),
            "Route LLM requests through a ChatGPT subscription via oauth.".into(),
        ),
    ];
    // A topic word the wiki mentions often enough not to be rare (5 of 38 pages).
    for i in 0..5 {
        p.push((
            format!("supabase-note-{i}"),
            format!("Supabase note {i}"),
            "supabase auth database rls".into(),
        ));
    }
    // Filler, so that common words are common, the way they are in the real wiki.
    for i in 0..30 {
        p.push((
            format!("filler-{i}"),
            format!("Project note {i}"),
            "the project skill agent session build work".into(),
        ));
    }
    p
}

struct Fixture {
    _t: tempfile::TempDir,
    root: PathBuf,
    state: PathBuf,
    kb: KnowledgeBase,
}

impl Fixture {
    fn new() -> Self {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("kb");
        let wiki = root.join("wiki");
        std::fs::create_dir_all(&wiki).unwrap();
        for (slug, title, body) in pages() {
            let line: String = body.chars().take(60).collect();
            std::fs::write(
                wiki.join(format!("{slug}.md")),
                format!("---\ntitle: \"{title}\"\nindex_line: \"{line}\"\n---\n\n{body}\n"),
            )
            .unwrap();
        }
        std::fs::write(wiki.join("_aliases.yaml"), "- [лайа, лайя, laya]\n").unwrap();
        std::fs::write(
            root.join(".solograph.yaml"),
            "kb:\n  stacks: 1-methodology/stacks\n",
        )
        .unwrap();
        let kb = KnowledgeBase::open(&root, Some(&t.path().join("db"))).unwrap();
        let state = t.path().join("state");
        Self {
            _t: t,
            root,
            state,
            kb,
        }
    }

    fn hint(&mut self, prompt: &str, session: &str) -> String {
        hint(&mut self.kb, prompt, session, &self.state)
            .unwrap()
            .text
    }

    fn write(&self, rel: &str, text: &str) {
        let p: &Path = &self.root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }
}

#[test]
fn a_rare_term_points_at_its_page() {
    let mut f = Fixture::new();
    let out = f.hint("а как применить laya для харнеса?", "s1");
    assert!(out.contains("laya-open-jev-on-mac"), "{out}");
    assert!(
        out.matches('\n').count() <= 2,
        "a pointer, not a dump: {out}"
    );
}

#[test]
fn common_words_stay_silent() {
    let mut f = Fixture::new();
    assert_eq!(
        f.hint("сделай project skill agent session build лучше", "s1"),
        ""
    );
}

#[test]
fn a_short_steer_stays_silent() {
    let mut f = Fixture::new();
    assert_eq!(f.hint("да делай", "s1"), "");
    assert_eq!(
        f.hint("laya", "s1"),
        "",
        "one word is a steer, not a question"
    );
}

#[test]
fn slash_commands_and_system_notifications_stay_silent() {
    let mut f = Fixture::new();
    assert_eq!(
        f.hint("/solo:research laya on device models today", "s1"),
        ""
    );
    assert_eq!(
        f.hint("<task-notification> agent finished, laya results on the mac are in </task-notification>", "t3"),
        ""
    );
    assert_eq!(
        f.hint(
            "<system-reminder>laya on the mac measured</system-reminder>",
            "t3"
        ),
        ""
    );
}

#[test]
fn the_same_page_once_per_session() {
    let mut f = Fixture::new();
    assert!(f.hint("что там с laya на маке?", "s1").contains("laya"));
    assert_eq!(f.hint("ещё раз про laya на маке?", "s1"), "");
    assert!(f.hint("что там с laya на маке?", "s2").contains("laya"));
}

#[test]
fn a_new_page_is_found_without_a_rebuild_step() {
    let mut f = Fixture::new();
    f.hint("что там с laya на маке?", "s1");
    f.write(
        "wiki/zork-engine.md",
        "---\ntitle: \"Zork engine\"\n---\n\nzork engine notes\n",
    );
    assert!(
        f.hint("как устроен zork engine у нас?", "s3")
            .contains("zork-engine")
    );
}

#[test]
fn a_kb_without_a_wiki_is_silent() {
    let t = tempfile::tempdir().unwrap();
    let mut kb = KnowledgeBase::open(&t.path().join("nope"), Some(&t.path().join("db"))).unwrap();
    let out = hint(
        &mut kb,
        "что там с laya на маке?",
        "s1",
        &t.path().join("state"),
    )
    .unwrap()
    .text;
    assert_eq!(out, "");
}

#[test]
fn a_russian_spelling_reaches_the_latin_term_through_the_alias_table() {
    let mut f = Fixture::new();
    assert!(
        f.hint("а что там лайа на маке умеет?", "t1")
            .contains("laya-open-jev-on-mac")
    );
}

#[test]
fn a_stack_question_points_at_the_stack_file() {
    let mut f = Fixture::new();
    f.write(
        "1-methodology/stacks/ios-swift.yaml",
        "name: iOS Swift\nframework: SwiftUI\n",
    );
    f.write(
        "1-methodology/stacks/python-api.yaml",
        "name: Python API\nframework: FastAPI\n",
    );
    let out = f.hint("какой стек взять для ios приложения со swiftui", "t2");
    assert!(out.contains("ios-swift.yaml"), "{out}");
    assert!(!out.contains("python-api"), "{out}");
}

#[test]
fn a_rare_word_alone_does_not_override_the_topic() {
    // "oauth" is rare and names the proxy page, but the prompt is about supabase,
    // which that page never mentions: pointing at it would be a false hint.
    let mut f = Fixture::new();
    let out = f.hint("how to set up supabase oauth with google", "s9");
    assert!(!out.contains("codex-oauth-proxy"), "{out}");
}

#[test]
fn the_wiki_is_wherever_the_kb_config_says() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("kb");
    let notes = root.join("notes");
    std::fs::create_dir_all(&notes).unwrap();
    std::fs::write(root.join(".solograph.yaml"), "kb:\n  pages: notes\n").unwrap();
    for (slug, title, body) in pages() {
        std::fs::write(
            notes.join(format!("{slug}.md")),
            format!("---\ntitle: \"{title}\"\n---\n\n{body}\n"),
        )
        .unwrap();
    }
    let mut kb = KnowledgeBase::open(&root, Some(&t.path().join("db"))).unwrap();
    let out = hint(
        &mut kb,
        "how would laya run inside the harness?",
        "c1",
        &t.path().join("state"),
    )
    .unwrap()
    .text;
    assert!(out.contains("notes/laya-open-jev-on-mac.md"), "{out}");
}

#[test]
fn no_stacks_in_the_config_means_no_stack_hint() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("kb");
    std::fs::create_dir_all(root.join("wiki")).unwrap();
    std::fs::create_dir_all(root.join("1-methodology/stacks")).unwrap();
    std::fs::write(
        root.join("1-methodology/stacks/ios-swift.yaml"),
        "name: iOS Swift\nframework: SwiftUI\n",
    )
    .unwrap();
    let mut kb = KnowledgeBase::open(&root, Some(&t.path().join("db"))).unwrap();
    let out = hint(
        &mut kb,
        "какой стек взять для ios приложения со swiftui",
        "c2",
        &t.path().join("state"),
    )
    .unwrap()
    .text;
    assert_eq!(out, "");
}
