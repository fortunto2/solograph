//! Port of tests/test_kb.py.

use std::path::{Path, PathBuf};

use super::{KnowledgeBase, frontmatter, markdown_files};

fn kb_root(tmp: &Path) -> PathBuf {
    let root = tmp.join("kb");
    std::fs::create_dir_all(root.join("wiki")).unwrap();
    std::fs::write(
        root.join("wiki/solograph.md"),
        "---\ntitle: Solograph code graph\ntype: concept\ntags: [tree-sitter, graph]\n---\nTree-sitter parses every project into a graph.",
    )
    .unwrap();
    std::fs::write(
        root.join("wiki/visa.md"),
        "---\ntype: project\n---\n# O-1A visa\n\nAttorney shortlist.",
    )
    .unwrap();
    let hidden = root.join("crm/.playwright-mcp");
    std::fs::create_dir_all(&hidden).unwrap();
    std::fs::write(
        hidden.join("amazon-nas.md"),
        "- generic graph code tree sitter button",
    )
    .unwrap();
    root
}

fn kb(tmp: &Path) -> KnowledgeBase {
    KnowledgeBase::open(&kb_root(tmp), Some(&tmp.join("db"))).unwrap()
}

#[test]
fn hidden_directories_are_not_the_knowledge_base() {
    let t = tempfile::tempdir().unwrap();
    let mut names: Vec<String> = markdown_files(&kb_root(t.path()))
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    names.sort();
    assert_eq!(names, ["wiki/solograph.md", "wiki/visa.md"]);
}

#[test]
fn frontmatter_title_and_tags_are_searchable() {
    let t = tempfile::tempdir().unwrap();
    let hits = kb(t.path())
        .search("tree-sitter graph", 5, None, false)
        .unwrap();
    assert_eq!(hits[0]["file"], "wiki/solograph.md");
    assert_eq!(hits[0]["type"], "concept");
}

#[test]
fn heading_is_the_title_when_frontmatter_has_none() {
    let t = tempfile::tempdir().unwrap();
    assert_eq!(
        kb(t.path()).search("attorney", 5, None, false).unwrap()[0]["title"],
        "O-1A visa"
    );
}

#[test]
fn type_filter() {
    let t = tempfile::tempdir().unwrap();
    assert_eq!(
        kb(t.path())
            .search("graph visa", 5, Some("project"), false)
            .unwrap()[0]["file"],
        "wiki/visa.md"
    );
}

#[test]
fn a_new_note_is_found_without_a_reindex_step() {
    let t = tempfile::tempdir().unwrap();
    let mut k = kb(t.path());
    k.search("anything", 5, None, false).unwrap();
    std::fs::write(
        t.path().join("kb/wiki/new.md"),
        "# Caretta nests\n\nTemperature decides sex.",
    )
    .unwrap();
    assert_eq!(
        k.search("caretta", 5, None, false).unwrap()[0]["file"],
        "wiki/new.md"
    );
}

#[test]
fn broken_frontmatter_is_still_a_document() {
    let t = tempfile::tempdir().unwrap();
    let mut k = kb(t.path());
    std::fs::write(
        t.path().join("kb/wiki/bad.md"),
        "---\ntitle: [unclosed\n---\nmangosteen harvest",
    )
    .unwrap();
    assert_eq!(
        k.search("mangosteen", 5, None, false).unwrap()[0]["file"],
        "wiki/bad.md"
    );
}

#[test]
fn frontmatter_body_is_stripped_like_python_frontmatter() {
    let (meta, body) = frontmatter("\n---\ntitle: X\n---\n\n  Body text \n").unwrap();
    assert_eq!(meta["title"], "X");
    assert_eq!(body, "Body text");
    let (meta, body) = frontmatter("no front matter\n---\n").unwrap();
    assert!(meta.is_empty());
    assert_eq!(body, "no front matter\n---");
}
