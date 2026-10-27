//! Port of tests/test_fts.py.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde_json::Value;

use super::{Doc, FtsStore};

fn doc(path: &Path) -> Result<Option<Doc>> {
    let text = std::fs::read_to_string(path)?;
    let mut meta = serde_json::Map::new();
    meta.insert(
        "name".into(),
        Value::String(path.file_name().unwrap().to_string_lossy().into()),
    );
    Ok(Some(Doc {
        title: text.lines().next().unwrap_or("").to_string(),
        body: text.clone(),
        kind: "notes".into(),
        meta,
        end: None,
    }))
}

fn files(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), p))
        .collect();
    out.sort();
    out
}

fn setup() -> (tempfile::TempDir, PathBuf, FtsStore) {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let store = FtsStore::open(&tmp.path().join("index.db")).unwrap();
    (tmp, src, store)
}

fn ids(hits: &[serde_json::Map<String, Value>]) -> Vec<String> {
    hits.iter()
        .map(|h| h["doc_id"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_new_file_is_found_and_an_unrelated_word_is_not() {
    let (_t, d, mut store) = setup();
    std::fs::write(
        d.join("a.txt"),
        "Граф кода\nСтроим граф кода на tree-sitter",
    )
    .unwrap();
    std::fs::write(d.join("b.txt"), "Рецепт\nПлов с бараниной").unwrap();
    store.sync(&files(&d), &doc, false, None).unwrap();
    assert_eq!(ids(&store.search("графом", 5, None).unwrap()), ["a.txt"]);
    assert!(store.search("пицца", 5, None).unwrap().is_empty());
}

#[test]
fn rare_term_outranks_common_one() {
    let (_t, d, mut store) = setup();
    for i in 0..5 {
        std::fs::write(
            d.join(format!("c{i}.txt")),
            format!("Заметка {i}\nвидео монтаж видео"),
        )
        .unwrap();
    }
    std::fs::write(
        d.join("rare.txt"),
        "Азалия\nвидео про Азалию и машину времени",
    )
    .unwrap();
    store.sync(&files(&d), &doc, false, None).unwrap();
    assert_eq!(
        ids(&store.search("видео Азалия", 5, None).unwrap())[0],
        "rare.txt"
    );
}

#[test]
fn an_edited_file_is_reread_and_an_untouched_one_is_not() {
    let (_t, d, mut store) = setup();
    std::fs::write(d.join("a.txt"), "first\nold words").unwrap();
    std::fs::write(d.join("b.txt"), "second\nstable words").unwrap();
    assert_eq!(store.sync(&files(&d), &doc, false, None).unwrap().read, 2);
    std::fs::write(d.join("a.txt"), "first\nfresh vocabulary here").unwrap();
    assert_eq!(store.sync(&files(&d), &doc, false, None).unwrap().read, 1);
    assert_eq!(
        ids(&store.search("vocabulary", 5, None).unwrap()),
        ["a.txt"]
    );
    assert!(store.search("old", 5, None).unwrap().is_empty());
}

#[test]
fn a_vanished_file_keeps_its_text_when_asked_to() {
    let (_t, d, mut store) = setup();
    std::fs::write(d.join("a.txt"), "transcript\nhubspot webhook subscription").unwrap();
    store.sync(&files(&d), &doc, true, None).unwrap();
    std::fs::remove_file(d.join("a.txt")).unwrap();
    let report = store.sync(&files(&d), &doc, true, None).unwrap();
    assert_eq!(report.gone, 1);
    let hit = &store.search("webhook", 5, None).unwrap()[0];
    assert_eq!(hit["doc_id"], "a.txt");
    assert_eq!(hit["gone"], true);
}

#[test]
fn a_vanished_file_leaves_by_default() {
    let (_t, d, mut store) = setup();
    std::fs::write(d.join("a.txt"), "note\nkept nowhere").unwrap();
    store.sync(&files(&d), &doc, false, None).unwrap();
    std::fs::remove_file(d.join("a.txt")).unwrap();
    store.sync(&files(&d), &doc, false, None).unwrap();
    assert!(store.search("nowhere", 5, None).unwrap().is_empty());
    assert_eq!(store.count().unwrap(), 0);
}

#[test]
fn a_file_the_parser_rejects_is_counted_not_hidden() {
    let (_t, d, mut store) = setup();
    std::fs::write(d.join("a.txt"), "ok\nfine").unwrap();
    std::fs::write(d.join("b.txt"), "bad\nbroken").unwrap();
    let parse = |p: &Path| -> Result<Option<Doc>> {
        if p.ends_with("b.txt") {
            bail!("cannot parse")
        }
        doc(p)
    };
    let report = store.sync(&files(&d), &parse, false, None).unwrap();
    assert_eq!(report.read, 1);
    assert_eq!(report.failed, ["b.txt"]);
}

#[test]
fn kind_filter_is_a_substring_match() {
    let (_t, d, mut store) = setup();
    std::fs::write(d.join("a.txt"), "one\nshared word").unwrap();
    std::fs::write(d.join("b.txt"), "two\nshared word").unwrap();
    let parse = |p: &Path| -> Result<Option<Doc>> {
        let mut dd = doc(p)?.unwrap();
        dd.kind = if p.ends_with("a.txt") {
            "life2film-ios".into()
        } else {
            "other".into()
        };
        Ok(Some(dd))
    };
    store.sync(&files(&d), &parse, false, None).unwrap();
    assert_eq!(
        ids(&store.search("shared", 5, Some("life2film")).unwrap()),
        ["a.txt"]
    );
}

#[test]
fn score_is_bm25_not_a_rank_dressed_as_relevance() {
    let (_t, d, mut store) = setup();
    std::fs::write(d.join("a.txt"), "one\nalpha alpha alpha beta").unwrap();
    std::fs::write(
        d.join("b.txt"),
        "two\nalpha gamma delta epsilon zeta eta theta",
    )
    .unwrap();
    for name in ["c", "d", "e"] {
        std::fs::write(d.join(format!("{name}.txt")), format!("{name}\nomega")).unwrap();
    }
    store.sync(&files(&d), &doc, false, None).unwrap();
    let scores: Vec<f64> = store
        .search("alpha", 5, None)
        .unwrap()
        .iter()
        .map(|h| h["score"].as_f64().unwrap())
        .collect();
    assert!(scores[0] > scores[1] && scores[1] > 0.0);
    assert_ne!(scores[0], 1.0);
}
