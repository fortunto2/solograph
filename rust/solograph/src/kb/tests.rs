//! Port of tests/test_kb.py.

use std::path::{Path, PathBuf};

use super::{KnowledgeBase, frontmatter, walk};

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
    let root = kb_root(t.path());
    let mut found = Vec::new();
    walk(&root, &root, &super::Config::default(), &mut found);
    let mut names: Vec<String> = found.into_iter().map(|(k, _)| k).collect();
    names.sort();
    assert_eq!(names, ["wiki/solograph.md", "wiki/visa.md"]);
}

#[test]
fn frontmatter_title_and_tags_are_searchable() {
    let t = tempfile::tempdir().unwrap();
    let hits = kb(t.path())
        .search("tree-sitter graph", 5, None, false, None)
        .unwrap();
    assert_eq!(hits[0]["file"], "wiki/solograph.md");
    assert_eq!(hits[0]["type"], "concept");
}

#[test]
fn heading_is_the_title_when_frontmatter_has_none() {
    let t = tempfile::tempdir().unwrap();
    assert_eq!(
        kb(t.path())
            .search("attorney", 5, None, false, None)
            .unwrap()[0]["title"],
        "O-1A visa"
    );
}

#[test]
fn type_filter() {
    let t = tempfile::tempdir().unwrap();
    assert_eq!(
        kb(t.path())
            .search("graph visa", 5, Some("project"), false, None)
            .unwrap()[0]["file"],
        "wiki/visa.md"
    );
}

#[test]
fn a_new_note_is_found_without_a_reindex_step() {
    let t = tempfile::tempdir().unwrap();
    let mut k = kb(t.path());
    k.search("anything", 5, None, false, None).unwrap();
    std::fs::write(
        t.path().join("kb/wiki/new.md"),
        "# Caretta nests\n\nTemperature decides sex.",
    )
    .unwrap();
    assert_eq!(
        k.search("caretta", 5, None, false, None).unwrap()[0]["file"],
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
        k.search("mangosteen", 5, None, false, None).unwrap()[0]["file"],
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

#[test]
fn a_path_prefix_limits_the_search_literally() {
    let t = tempfile::tempdir().unwrap();
    let root = kb_root(t.path());
    std::fs::create_dir_all(root.join("3-inbox")).unwrap();
    std::fs::write(
        root.join("3-inbox/graph.md"),
        "# Graph idea\n\ngraph graph graph",
    )
    .unwrap();
    std::fs::create_dir_all(root.join("wikiX")).unwrap();
    std::fs::write(root.join("wikiX/graph.md"), "# Graph\n\ngraph").unwrap();
    let mut k = KnowledgeBase::open(&root, Some(&t.path().join("db"))).unwrap();
    let all = k.search("graph", 10, None, false, None).unwrap();
    assert!(all.iter().any(|h| h["file"] == "3-inbox/graph.md"));
    let wiki = k.search("graph", 10, None, false, Some("wiki/")).unwrap();
    assert!(!wiki.is_empty());
    assert!(
        wiki.iter()
            .all(|h| h["file"].as_str().unwrap().starts_with("wiki/")),
        "{wiki:?}"
    );
    // `_` would be a LIKE wildcard; a prefix is compared as written.
    assert!(
        k.search("graph", 10, None, false, Some("wik_/"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_word_in_the_summary_outranks_the_same_word_in_a_long_body() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("kb");
    std::fs::create_dir_all(root.join("wiki")).unwrap();
    let filler = "filler words here. ".repeat(10);
    std::fs::write(
        root.join("wiki/a.md"),
        format!("---\ntitle: Notes\n---\n{filler} quokka once."),
    )
    .unwrap();
    std::fs::write(
        root.join("wiki/b.md"),
        format!("---\ntitle: Other notes\nindex_line: \"quokka population counts\"\n---\n{filler}"),
    )
    .unwrap();
    let mut k = KnowledgeBase::open(&root, Some(&t.path().join("db"))).unwrap();
    assert_eq!(
        k.search("quokka", 5, None, false, None).unwrap()[0]["file"],
        "wiki/b.md"
    );
}

#[test]
fn an_alias_table_in_the_kb_lets_a_russian_word_find_an_english_page() {
    let t = tempfile::tempdir().unwrap();
    let root = kb_root(t.path());
    let mut k = KnowledgeBase::open(&root, Some(&t.path().join("db"))).unwrap();
    assert!(
        k.search("поверенный", 5, None, false, None)
            .unwrap()
            .is_empty()
    );
    std::fs::write(
        root.join("wiki/_aliases.yaml"),
        "- [поверенный, attorney]\n",
    )
    .unwrap();
    assert_eq!(
        k.search("поверенного", 5, None, false, None).unwrap()[0]["file"],
        "wiki/visa.md"
    );
}

#[test]
fn an_index_in_the_old_layout_is_rebuilt_not_misread() {
    let t = tempfile::tempdir().unwrap();
    let root = kb_root(t.path());
    let db = t.path().join("db");
    {
        let old = crate::fts::FtsStore::open(&db.join(super::DB_FILE)).unwrap();
        old.db
            .execute_batch("INSERT INTO docs(doc_id) VALUES ('ghost.md')")
            .unwrap();
    }
    let mut k = KnowledgeBase::open(&root, Some(&db)).unwrap();
    let hits = k.search("attorney", 5, None, false, None).unwrap();
    assert_eq!(hits[0]["file"], "wiki/visa.md");
    assert!(!k.ids().unwrap().contains("ghost.md"));
}

#[test]
fn a_scoped_sync_reads_and_drops_only_under_its_prefix() {
    let t = tempfile::tempdir().unwrap();
    let mut k = kb(t.path());
    std::fs::write(k.root.join("notes.md"), "# Notes\n\nOutside the wiki.").unwrap();
    assert_eq!(k.sync().unwrap().indexed, 3);

    std::fs::remove_file(k.root.join("wiki/visa.md")).unwrap();
    std::fs::remove_file(k.root.join("notes.md")).unwrap();
    let r = k.sync_under(Some("wiki/")).unwrap();
    // The wiki page that vanished is gone; the file outside the scope waits for a
    // full sync instead of being dropped because the scoped walk did not list it.
    assert_eq!((r.removed, r.indexed), (1, 2));
    assert_eq!(k.sync().unwrap().indexed, 1);
}

#[test]
fn a_folder_marked_private_is_not_read_into_a_kb_above_it() {
    // Review 28 Oct: `cd ~ && solograph kb family` read ~/personal into ~/.solo/kb.
    let t = tempfile::tempdir().unwrap();
    let root = kb_root(t.path());
    let personal = root.join("personal");
    std::fs::create_dir_all(&personal).unwrap();
    std::fs::write(personal.join(".solograph-private"), "").unwrap();
    std::fs::write(personal.join("family.md"), "# Family\n\nprivate notes").unwrap();
    let mut found = Vec::new();
    walk(&root, &root, &super::Config::default(), &mut found);
    assert!(
        found.iter().all(|(id, _)| !id.starts_with("personal/")),
        "{found:?}"
    );
}

#[test]
fn the_home_directory_is_never_an_implicit_kb() {
    let home = crate::paths::home();
    assert!(super::implicit_root(&home).is_none());
    assert!(super::implicit_root(Path::new("/")).is_none());
    let t = tempfile::tempdir().unwrap();
    assert_eq!(super::implicit_root(t.path()).as_deref(), Some(t.path()));
}
