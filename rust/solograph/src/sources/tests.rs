//! Read-side port of tests/test_sources.py. Rows are written in the exact shape
//! Python's upsert_one / write_video / upsert_maker write them.

use serde_json::{Map, Value, json};

use super::SourceIndex;
use crate::fts::{Doc, FtsStore};

fn idx(tmp: &std::path::Path) -> SourceIndex {
    SourceIndex::open(Some(tmp)).unwrap()
}

fn put(idx: &SourceIndex, doc_id: &str, title: &str, body: &str, kind: &str, meta: Value) {
    let Value::Object(meta) = meta else { panic!() };
    let doc = Doc {
        title: title.into(),
        keys: String::new(),
        body: body.into(),
        kind: kind.into(),
        meta,
        end: None,
    };
    FtsStore::put(
        &idx.store.db,
        idx.store.layout.columns,
        doc_id,
        Some(&doc),
        None,
        None,
        false,
    )
    .unwrap();
}

fn doc(idx: &SourceIndex, source: &str, id: &str, title: &str, content: &str, popularity: i64) {
    let meta = json!({"type": "doc", "source_type": source, "source_name": source, "url": format!("https://t.me/x/{id}"),
                      "tags": "", "created": "2026-01-01", "popularity": popularity, "content": content});
    put(idx, &format!("{source}/{id}"), title, content, source, meta);
}

fn video(
    idx: &SourceIndex,
    vid: &str,
    title: &str,
    chunks: &[(&str, &str, f64)],
    tags: &[(&str, f64)],
) {
    idx.store
        .db
        .execute(
            "INSERT OR REPLACE INTO videos(source, video_id, title, url, channel, channel_name, created, tags) \
             VALUES ('youtube', ?1, ?2, ?3, '@indiechannel', 'Indie', '2026-01-01', '')",
            rusqlite::params![vid, title, format!("https://youtube.com/watch?v={vid}")],
        )
        .unwrap();
    for (t, w) in tags {
        idx.store
            .db
            .execute(
                "INSERT INTO video_tags VALUES (?, ?, ?)",
                rusqlite::params![vid, t, w],
            )
            .unwrap();
    }
    for (i, (chapter, text, secs)) in chunks.iter().enumerate() {
        let meta = json!({"type": "video_chunk", "video_id": vid, "source_name": "@indiechannel",
            "url": format!("https://youtube.com/watch?v={vid}&t={}s", *secs as i64), "created": "2026-01-01", "tags": "",
            "chapter": chapter, "start_time": "", "start_seconds": secs, "chunk_type": "transcript"});
        put(
            idx,
            &format!("youtube/{vid}:{i}"),
            &format!("{title} — {chapter}"),
            text,
            "youtube",
            meta,
        );
    }
}

fn ids(hits: &[Value], key: &str) -> Vec<String> {
    hits.iter()
        .map(|h| h[key].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_document_is_found_with_its_source() {
    let t = tempfile::tempdir().unwrap();
    let i = idx(t.path());
    doc(
        &i,
        "telegram",
        "p1",
        "Bootstrapped to 10k MRR",
        "no investors, just customers",
        0,
    );
    let hit = &i.search("investors", None, 5).unwrap()[0];
    assert_eq!(hit["doc_id"], "p1");
    assert_eq!(hit["source_name"], "telegram");
    assert!(
        ["title", "url", "content", "score"]
            .iter()
            .all(|k| hit.get(*k).is_some())
    );
}

#[test]
fn source_filter() {
    let t = tempfile::tempdir().unwrap();
    let i = idx(t.path());
    doc(&i, "telegram", "p1", "Churn notes", "churn churn", 0);
    doc(&i, "trustmrr", "p2", "Churn study", "churn", 0);
    assert_eq!(
        ids(&i.search("churn", Some("trustmrr"), 5).unwrap(), "doc_id"),
        ["p2"]
    );
}

#[test]
fn popularity_breaks_a_tie() {
    let t = tempfile::tempdir().unwrap();
    let i = idx(t.path());
    doc(
        &i,
        "producthunt",
        "a",
        "Invoice generator",
        "invoice generator",
        1,
    );
    doc(
        &i,
        "producthunt",
        "b",
        "Invoice generator",
        "invoice generator",
        5000,
    );
    for k in 0..4 {
        doc(&i, "producthunt", &format!("z{k}"), "Other", "unrelated", 0);
    }
    assert_eq!(
        i.search("invoice generator", None, 5).unwrap()[0]["doc_id"],
        "b"
    );
}

#[test]
fn video_chunks_come_back_with_a_timestamp_once_per_video() {
    let t = tempfile::tempdir().unwrap();
    let i = idx(t.path());
    video(
        &i,
        "abcdefghijk",
        "Pricing a SaaS",
        &[
            ("ch0", "intro talk", 0.0),
            ("ch1", "tiered pricing anchors", 10.0),
            ("ch2", "pricing pages", 20.0),
        ],
        &[],
    );
    let hits = i.search("pricing", None, 5).unwrap();
    assert_eq!(hits.len(), 1);
    let url = hits[0]["url"].as_str().unwrap();
    assert!(url.ends_with("&t=10s") || url.ends_with("&t=20s"));
    assert!(["ch1", "ch2"].contains(&hits[0]["chapter"].as_str().unwrap()));
}

#[test]
fn tags_related_and_list() {
    let t = tempfile::tempdir().unwrap();
    let i = idx(t.path());
    video(
        &i,
        "aaaaaaaaaaa",
        "AI agents for pricing",
        &[],
        &[("AI agents and automation", 0.4)],
    );
    video(
        &i,
        "bbbbbbbbbbb",
        "Agents that automate ops",
        &[],
        &[("AI agents and automation", 0.5)],
    );
    video(&i, "ccccccccccc", "Borscht", &[], &[]);
    let names: Vec<String> = i
        .list_tags("youtube")
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().into())
        .collect();
    assert!(names.contains(&"AI agents and automation".to_string()));
    assert_eq!(
        ids(
            &i.related_videos("youtube", "aaaaaaaaaaa").unwrap(),
            "video_id"
        ),
        ["bbbbbbbbbbb"]
    );
}

#[test]
fn makers_link_to_their_profile() {
    let t = tempfile::tempdir().unwrap();
    let i = idx(t.path());
    let meta = json!({"type": "maker", "username": "jane", "headline": "Builds invoice tools"});
    put(
        &i,
        "producthunt/maker:jane",
        "Jane (@jane)",
        "Builds invoice tools",
        "producthunt",
        meta,
    );
    assert_eq!(
        i.search("invoice", None, 5).unwrap()[0]["url"],
        "https://www.producthunt.com/@jane"
    );
}

#[test]
fn list_counts_documents_videos_and_chunks() {
    let t = tempfile::tempdir().unwrap();
    let i = idx(t.path());
    doc(&i, "telegram", "p1", "alpha", "alpha", 0);
    video(&i, "abcdefghijk", "Video", &[("", "alpha beta", 5.0)], &[]);
    let listed: Map<String, Value> = i
        .list_sources()
        .unwrap()
        .into_iter()
        .map(|e| (e["source"].as_str().unwrap().to_string(), e))
        .collect();
    assert_eq!(listed["telegram"]["count"], 1);
    assert_eq!(listed["youtube"]["videos"], 1);
    assert_eq!(listed["youtube"]["video_chunks"], 1);
}

#[test]
fn one_long_video_does_not_crowd_out_the_others() {
    let t = tempfile::tempdir().unwrap();
    let i = idx(t.path());
    let long: Vec<(&str, &str, f64)> = (0..30).map(|_| ("", "ai agents ai agents", 1.0)).collect();
    video(&i, "aaaaaaaaaaa", "Agents course", &long, &[]);
    video(
        &i,
        "bbbbbbbbbbb",
        "Short clip",
        &[("", "ai agents in one line", 1.0)],
        &[],
    );
    let mut got = ids(
        &i.search("ai agents", Some("youtube"), 2).unwrap(),
        "video_id",
    );
    got.sort();
    assert_eq!(got, ["aaaaaaaaaaa", "bbbbbbbbbbb"]);
}
