//! The HTTP contract search_server.py had: paths, status codes and JSON keys.

use serde_json::json;

use super::{decode, handle, param};
use crate::fts::{Doc, FtsStore};
use crate::sources::SourceIndex;

fn idx_with_docs(tmp: &std::path::Path) -> SourceIndex {
    let idx = SourceIndex::open(Some(tmp)).unwrap();
    for (id, title, body) in [
        ("1", "Pricing tiers", "how to price a saas product in tiers"),
        ("2", "Cold email", "first customers from cold email"),
    ] {
        let meta = json!({"type": "doc", "source_type": "telegram", "source_name": "telegram",
                          "url": format!("https://t.me/x/{id}"), "tags": "", "created": "2026-01-01"});
        let serde_json::Value::Object(meta) = meta else {
            panic!()
        };
        let doc = Doc {
            title: title.into(),
            body: body.into(),
            kind: "telegram".into(),
            meta,
            end: None,
        };
        FtsStore::put(
            &idx.store.db,
            &format!("telegram/{id}"),
            Some(&doc),
            None,
            None,
            false,
        )
        .unwrap();
    }
    idx
}

#[test]
fn query_strings_decode_like_parse_qs() {
    assert_eq!(decode("AI+tool%20x%C3%A9"), "AI tool xé");
    assert_eq!(param("q=&n=3", "q"), None);
    assert_eq!(
        param("n=3&q=pricing+tiers", "q").as_deref(),
        Some("pricing tiers")
    );
}

#[test]
fn search_returns_results_query_and_source() {
    let t = tempfile::tempdir().unwrap();
    let idx = idx_with_docs(t.path());
    let (status, body) = handle(&idx, "/search?q=price&source=telegram&n=1");
    assert_eq!(status, 200);
    assert_eq!(body["query"], "price");
    assert_eq!(body["source"], "telegram");
    assert_eq!(body["results"][0]["doc_id"], "1");
    let (_, all) = handle(&idx, "/search?q=price");
    assert!(all["source"].is_null());
}

#[test]
fn errors_keep_their_codes() {
    let t = tempfile::tempdir().unwrap();
    let idx = idx_with_docs(t.path());
    assert_eq!(
        handle(&idx, "/search?n=2"),
        (400, json!({"error": "missing q param"}))
    );
    let (status, body) = handle(&idx, "/search?q=x&source=nowhere");
    assert_eq!(status, 400);
    assert_eq!(body["available"], json!(["telegram"]));
    assert_eq!(handle(&idx, "/nope").0, 404);
}

#[test]
fn sources_and_health_count_documents() {
    let t = tempfile::tempdir().unwrap();
    let idx = idx_with_docs(t.path());
    let (_, sources) = handle(&idx, "/sources");
    assert_eq!(sources["sources"][0]["count"], 2);
    assert_eq!(
        handle(&idx, "/health"),
        (200, json!({"status": "ok", "total": 2, "sources": 1}))
    );
}
