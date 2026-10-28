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

/// Stack templates the way they really differ: one large, mentioning everything
/// in passing, and small focused ones.
fn stack_fixture() -> Fixture {
    let f = Fixture::new();
    let filler: String = (0..300).map(|i| format!("word{i} ")).collect();
    f.write(
        "1-methodology/stacks/rust-native.yaml",
        &format!("name: Rust Native\nnotes: ffi for ios and swiftui apps, python bindings, cli tools\n{filler}\n"),
    );
    f.write(
        "1-methodology/stacks/ios-swift.yaml",
        "name: iOS Swift\nframework: SwiftUI\nplatform: ios iphone app\n",
    );
    f.write(
        "1-methodology/stacks/python-api.yaml",
        "name: Python API\nframework: FastAPI\nauth: token expiry\n",
    );
    f
}

#[test]
fn the_focused_stack_beats_the_large_one_that_mentions_everything() {
    // Review 28 Oct: an unnormalised idf sum sent this to rust-native.yaml.
    let mut f = stack_fixture();
    let out = f.hint("which stack for an ios app with swiftui", "k1");
    assert!(out.contains("ios-swift.yaml"), "{out}");
}

#[test]
fn a_stack_trace_or_an_expiring_token_is_not_a_stack_question() {
    let mut f = stack_fixture();
    for (i, p) in [
        "here is the python stack trace from the failing test",
        "the call stack overflows in the parser again",
        "токен истекает через час и python api падает",
        "стекло на экране треснуло, python app",
    ]
    .iter()
    .enumerate()
    {
        let out = f.hint(p, &format!("k2-{i}"));
        assert!(!out.contains("Stack:"), "{p} -> {out}");
    }
}

#[test]
fn a_russian_stack_question_in_any_case_form_is_one() {
    let mut f = stack_fixture();
    let out = f.hint("какой стек выбрать для ios приложения на swiftui", "k3");
    assert!(out.contains("ios-swift.yaml"), "{out}");
    let out = f.hint("подскажи по стеку для python api с fastapi", "k4");
    assert!(out.contains("python-api.yaml"), "{out}");
}

#[test]
fn two_letter_words_tell_stacks_apart() {
    // "ml" and "ai" were dropped as too short, so "Python ML/AI" looked shorter
    // than "Python API" and won every python question, or lost this one.
    let mut f = stack_fixture();
    f.write(
        "1-methodology/stacks/python-ml.yaml",
        "name: Python ML/AI\nframework: training loops, api for inference\n",
    );
    let out = f.hint("стек для python ml модели", "k5");
    assert!(out.contains("python-ml.yaml"), "{out}");
    let out = f.hint("which stack for a python api with fastapi", "k6");
    assert!(out.contains("python-api.yaml"), "{out}");
}

#[test]
fn a_common_russian_word_does_not_become_a_page_name_through_an_alias() {
    // Replay of 300 real prompts, 28 Oct: "турции" -> turkey named the company page
    // in talk about lenses and chess, "визу" -> visa a visa-photo app in talk about
    // a visa case. A spelling used all over the KB is a word, not a name.
    let mut f = Fixture::new();
    f.write(
        "wiki/visa-photo-app.md",
        "---\ntitle: \"Visa photo app\"\n---\n\nChecks a visa photo against the spec.\n",
    );
    f.write(
        "wiki/_aliases.yaml",
        "- [лайа, лайя, laya]\n- [виза, visa]\n",
    );
    for i in 0..10 {
        f.write(
            &format!("crm/note-{i}.md"),
            &format!("# Заметка {i}\n\nвиза, документы для визы, визовый центр\n"),
        );
    }
    f.kb.sync().unwrap(); // the hint reads KB-wide counts from the index
    let out = f.hint("кейс на визу надо подготовить и отправить юристу", "a1");
    assert!(!out.contains("visa-photo-app"), "{out}");
    // A rare spelling still reaches its page.
    let out = f.hint("а что там лайа на маке умеет?", "a2");
    assert!(out.contains("laya-open-jev-on-mac"), "{out}");
}

#[test]
fn a_word_only_in_the_summary_needs_the_topic_to_back_it() {
    // Replay of 300 real prompts, 28 Oct: a street address with "Irvine" pointed
    // at a hackathon held in Irvine, named only in its index_line.
    let mut f = Fixture::new();
    f.write(
        "wiki/brocheck-hackathon.md",
        "---\ntitle: \"BroCheck: a 50-minute build\"\nindex_line: \"Post: Lightning Hackathon at EvoNexus Irvine, decision models\"\n---\n\nA build.\n",
    );
    let out = f.hint("413 Saint Vincent Irvine, они дома будут после 11го", "i1");
    assert!(!out.contains("brocheck"), "{out}");
    // In the title or file name, the word is a name and needs no backing.
    let out = f.hint("а что там лайа на маке умеет?", "i2");
    assert!(out.contains("laya-open-jev-on-mac"), "{out}");
}

#[test]
fn a_page_whose_file_name_is_spelled_out_is_named() {
    // "kill iterate scale" names kill-iterate-scale.md though no word is rare
    // (8 of 15 named prompts were missed this way, 28 Oct).
    let mut f = Fixture::new();
    f.write(
        "wiki/kill-iterate-scale.md",
        "---\ntitle: \"Kill, iterate or scale\"\n---\n\nWeekly rules for the project skill agent session.\n",
    );
    // Common words here, as in a real wiki: none of them is rare enough to name it.
    for i in 0..12 {
        f.write(
            &format!("wiki/note-{i}.md"),
            &format!("---\ntitle: \"Note {i}\"\n---\n\nWe kill, iterate and scale things.\n"),
        );
    }
    let out = f.hint("правила kill iterate scale для продуктов напомни", "s1");
    assert!(out.contains("kill-iterate-scale"), "{out}");
    // Two of the three words are not the name.
    let out = f.hint("kill the build and scale the workers", "s2");
    assert!(!out.contains("kill-iterate-scale"), "{out}");
}
