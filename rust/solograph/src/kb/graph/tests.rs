//! Port of tests/test_kb_graph.py, plus what the graph now also answers: orphans,
//! broken links, the catalog check, and capped neighbour expansion.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::json;

use super::{Graph, is_meta};
use crate::kb::KnowledgeBase;

fn page(root: &Path, rel: &str, body: &str, title: &str, tags: &str) {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        format!("---\ntitle: \"{title}\"\ntype: concept\ntags: [{tags}]\n---\n{body}\n"),
    )
    .unwrap();
}

fn fixture(root: &Path) {
    page(
        root,
        "wiki/a.md",
        "See [[b]], [[B|label]], [[missing]] and [[a]].",
        "A",
        "x, y",
    );
    page(
        root,
        "wiki/b.md",
        "Back to [[a]].\n```toml\n[[c]]\n```\nInline `[[c]]`.",
        "B",
        "",
    );
    page(root, "wiki/c.md", "Nobody links here outside code.", "", "");
    page(
        root,
        "wiki/d.md",
        "Linked by nobody, missing from the catalog.",
        "D",
        "",
    );
    page(
        root,
        "1-methodology/raw.md",
        "Raw doc points at [[c]].",
        "Raw",
        "",
    );
    page(
        root,
        "wiki/index.md",
        "Catalog: [[a]] [[b]] [[c]] [[gone]]",
        "Index",
        "",
    );
    page(
        root,
        "wiki/_schema.md",
        "Tooling: [[a]], e.g. [[page-name]]",
        "Schema",
        "",
    );
}

fn graph(t: &Path) -> Graph {
    fixture(&t.join("kb"));
    KnowledgeBase::open(&t.join("kb"), Some(&t.join("db")))
        .unwrap()
        .graph()
        .unwrap()
}

#[test]
fn edges_follow_written_links_only() {
    let t = tempfile::tempdir().unwrap();
    let g = graph(t.path());
    let pairs: BTreeSet<(String, String)> = g
        .edges
        .iter()
        .filter(|(a, b)| !is_meta(a) && !is_meta(b))
        .map(|(a, b)| {
            if a < b {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            }
        })
        .collect();
    let want: BTreeSet<(String, String)> = [
        ("1-methodology/raw.md", "wiki/c.md"),
        ("wiki/a.md", "wiki/b.md"),
    ]
    .into_iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect();
    assert_eq!(pairs, want);
    assert_eq!(g.pages["wiki/a.md"].tags, ["x", "y"]);
    assert_eq!(g.pages["wiki/c.md"].title, "c");
}

#[test]
fn the_report_names_orphans_broken_links_and_catalog_gaps() {
    let t = tempfile::tempdir().unwrap();
    let r = graph(t.path()).report(Some("wiki/"), Some("a"), 10);
    assert_eq!(r["orphans"], json!(["wiki/d.md"]));
    assert_eq!(
        r["broken"],
        json!([{"from": "wiki/a.md", "link": "missing"}, {"from": "wiki/index.md", "link": "gone"}])
    );
    assert_eq!(r["not_in_index"], json!(["d"]));
    assert_eq!(r["pages"], 4);
    assert_eq!(r["page"]["backlinks"], json!(["wiki/b.md"]));
    assert_eq!(r["page"]["links"], json!(["wiki/b.md"]));
    // The catalog is not a hub's reason: a has one real backlink, not two.
    assert_eq!(r["hubs"][0]["in"], 1);
}

#[test]
fn neighbours_skip_the_catalog_and_stop_at_the_cap() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().join("kb");
    let links: String = (0..6).map(|i| format!("[[n{i}]] ")).collect();
    page(&root, "wiki/hub.md", &links, "Hub", "");
    for i in 0..6 {
        page(
            &root,
            &format!("wiki/n{i}.md"),
            "leaf",
            &format!("N{i}"),
            "",
        );
    }
    page(&root, "wiki/n5.md", "leaf, and back to [[hub]]", "N5", "");
    page(&root, "wiki/index.md", "[[hub]] [[n0]]", "Index", "");
    let mut k = KnowledgeBase::open(&root, Some(&t.path().join("db"))).unwrap();
    let g = k.graph().unwrap();
    let n = g.neighbours(&[json!({"file": "wiki/hub.md"})], None, 3);
    assert_eq!(n.len(), 3);
    assert!(n.iter().all(|h| h["file"] != "wiki/index.md"));
    // Linked both ways from the hit: n5 ranks first.
    assert_eq!(n[0]["file"], "wiki/n5.md");
    assert_eq!(n[0]["graph_source"], "via link: Hub");
}

#[test]
fn expand_graph_appends_marked_neighbours_after_the_hits() {
    let t = tempfile::tempdir().unwrap();
    fixture(&t.path().join("kb"));
    let mut k = KnowledgeBase::open(&t.path().join("kb"), Some(&t.path().join("db"))).unwrap();
    let hits = k.search("label", 1, None, true, None).unwrap();
    assert_eq!(hits.len(), 2, "{hits:?}");
    assert_eq!(hits[0]["file"], "wiki/a.md");
    assert!(hits[0].get("graph_source").is_none());
    assert_eq!(hits[1]["file"], "wiki/b.md");
    assert_eq!(hits[1]["graph_source"], "via link: A");
    // c's only real neighbour is outside wiki/, so a wiki-scoped search adds none.
    assert_eq!(
        k.search("outside code", 1, None, true, None).unwrap().len(),
        2
    );
    assert_eq!(
        k.search("outside code", 1, None, true, Some("wiki/"))
            .unwrap()
            .len(),
        1
    );
}
