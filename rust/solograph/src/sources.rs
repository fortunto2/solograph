//! External sources, read side. Python's sources.py writes the index (YouTube,
//! ProductHunt, TrustMRR, Telegram scrapers stay in Python); this reads the same
//! <sources_root>/search.db with the same ranking. The schema is the contract.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use rusqlite::params;
use serde_json::{Map, Value, json};

use crate::fts::{FtsStore, round3};
use crate::paths::{env_path, home};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS videos(
    video_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT, url TEXT, channel TEXT, channel_name TEXT,
    created TEXT, tags TEXT, description TEXT,
    duration_seconds INTEGER, chapters TEXT
);
CREATE INDEX IF NOT EXISTS videos_source ON videos(source, channel);
CREATE TABLE IF NOT EXISTS video_tags(
    video_id TEXT NOT NULL, tag TEXT NOT NULL, weight REAL,
    PRIMARY KEY (video_id, tag)
);
";

pub fn default_root() -> PathBuf {
    env_path("SOURCES_ROOT").unwrap_or_else(|| home().join(".solo").join("sources"))
}

pub struct SourceIndex {
    store: FtsStore,
}

fn get_str(h: &Map<String, Value>, k: &str) -> Value {
    h.get(k).cloned().unwrap_or_else(|| json!(""))
}

impl SourceIndex {
    pub fn open(root: Option<&Path>) -> Result<Self> {
        let root = root.map_or_else(default_root, Path::to_path_buf);
        let store = FtsStore::open(&root.join("search.db"))?;
        store.db.execute_batch(SCHEMA)?;
        Ok(Self { store })
    }

    /// BM25 over every source, or one. A video appears once, at its best chunk;
    /// the window widens until it holds n videos or the hits run out.
    pub fn search(&self, query: &str, source: Option<&str>, n: usize) -> Result<Vec<Value>> {
        let mut fetch = n * 4;
        loop {
            let hits = self.store.search(query, fetch, source)?;
            let mut out = rank(&hits);
            if out.len() >= n || hits.len() < fetch {
                out.truncate(n);
                return Ok(out);
            }
            fetch *= 4;
        }
    }

    pub fn list_tags(&self, source: &str) -> Result<Vec<Value>> {
        let mut stmt = self.store.db.prepare(
            "SELECT t.tag, count(*), avg(t.weight) FROM video_tags t \
             JOIN videos v ON v.video_id = t.video_id WHERE v.source = ? \
             GROUP BY t.tag ORDER BY count(*) DESC",
        )?;
        let rows = stmt.query_map([source], |r| {
            let avg: Option<f64> = r.get(2)?;
            Ok(json!({
                "name": r.get::<_, String>(0)?,
                "count": r.get::<_, i64>(1)?,
                "avg_confidence": avg.filter(|a| *a != 0.0).map(round3),
            }))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn related_videos(&self, source: &str, video_id: &str) -> Result<Vec<Value>> {
        let mut stmt = self.store.db.prepare(
            "SELECT v.video_id, v.title, v.url, group_concat(b.tag), count(*), sum(a.weight + b.weight) / 2 AS rel \
             FROM video_tags a JOIN video_tags b ON b.tag = a.tag AND b.video_id != a.video_id \
             JOIN videos v ON v.video_id = b.video_id AND v.source = ? \
             WHERE a.video_id = ? GROUP BY v.video_id ORDER BY rel DESC",
        )?;
        let rows = stmt.query_map(params![source, video_id], |r| {
            let tags: Option<String> = r.get(3)?;
            let rel: Option<f64> = r.get(5)?;
            Ok(json!({
                "video_id": r.get::<_, String>(0)?,
                "title": r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                "url": r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                "shared_tags": tags.unwrap_or_default().split(',').collect::<Vec<_>>(),
                "overlap": r.get::<_, i64>(4)?,
                "relevance": rel.filter(|x| *x != 0.0).map_or(json!(0), |x| json!(round3(x))),
            }))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Per source: documents + videos + makers, with the video/chunk split.
    pub fn list_sources(&self) -> Result<Vec<Value>> {
        let db = &self.store.db;
        let mut counts: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
        let mut stmt = db.prepare(
            "SELECT kind, json_extract(meta, '$.type'), count(*) FROM docs GROUP BY 1, 2",
        )?;
        for row in stmt.query_map([], |r| {
            Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })? {
            let (kind, typ, n) = row?;
            counts
                .entry(kind.unwrap_or_default())
                .or_default()
                .insert(typ.unwrap_or_default(), n);
        }
        let mut videos: BTreeMap<String, i64> = BTreeMap::new();
        let mut stmt = db.prepare("SELECT source, count(*) FROM videos GROUP BY source")?;
        for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (src, n) = row?;
            counts.entry(src.clone()).or_default();
            videos.insert(src, n);
        }
        let path = self.store.path.display().to_string();
        Ok(counts
            .iter()
            .map(|(src, c)| {
                let get = |k: &str| c.get(k).copied().unwrap_or(0);
                let vids = videos.get(src).copied().unwrap_or(0);
                let mut e = Map::new();
                e.insert("source".into(), json!(src));
                e.insert("count".into(), json!(get("doc") + get("maker") + vids));
                e.insert("path".into(), json!(path));
                if vids != 0 {
                    e.insert("videos".into(), json!(vids));
                    e.insert("video_chunks".into(), json!(get("video_chunk")));
                }
                if get("maker") != 0 {
                    e.insert("makers".into(), json!(get("maker")));
                }
                Value::Object(e)
            })
            .collect())
    }
}

fn rank(hits: &[Map<String, Value>]) -> Vec<Value> {
    let mut seen_videos: HashSet<String> = HashSet::new();
    let mut out: Vec<(f64, Map<String, Value>)> = Vec::new();
    for hit in hits {
        let kind = hit["kind"].as_str().unwrap_or("");
        let doc_id = hit["doc_id"].as_str().unwrap_or("");
        let doc_id = doc_id.strip_prefix(&format!("{kind}/")).unwrap_or(doc_id);
        let mut score = hit["score"].as_f64().unwrap_or(0.0);
        let title: String = hit["title"]
            .as_str()
            .unwrap_or("")
            .chars()
            .take(100)
            .collect();
        let mut e = Map::new();
        e.insert("doc_id".into(), json!(doc_id));
        e.insert("source_type".into(), get_str(hit, "source_type"));
        e.insert(
            "source_name".into(),
            hit.get("source_name")
                .cloned()
                .unwrap_or_else(|| json!(kind)),
        );
        e.insert("title".into(), json!(title));
        e.insert("url".into(), get_str(hit, "url"));
        e.insert("content".into(), hit["snippet"].clone());
        e.insert("created".into(), get_str(hit, "created"));
        e.insert("tags".into(), get_str(hit, "tags"));
        match hit.get("type").and_then(Value::as_str) {
            Some("video_chunk") => {
                let vid = hit
                    .get("video_id")
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                if !seen_videos.insert(vid) {
                    continue;
                }
                let ct = hit
                    .get("chunk_type")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("transcript");
                e.insert("source_type".into(), json!(format!("youtube-{ct}")));
                for k in ["video_id", "chapter", "start_time", "start_seconds"] {
                    e.insert(k.into(), hit.get(k).cloned().unwrap_or(Value::Null));
                }
            }
            Some("maker") => {
                e.insert("source_type".into(), json!(format!("{kind}-maker")));
                let user = hit.get("username").and_then(Value::as_str).unwrap_or("");
                e.insert(
                    "url".into(),
                    json!(format!("https://www.producthunt.com/@{user}")),
                );
            }
            _ => {
                if let Some(pop) = hit
                    .get("popularity")
                    .and_then(Value::as_f64)
                    .filter(|p| *p != 0.0)
                {
                    // Upvotes add at most 15%, as the vector ranking did.
                    score *= 1.0 + ((pop + 1.0).log10() / 25.0).min(0.15);
                }
            }
        }
        let score = round3(score);
        e.insert("score".into(), json!(score));
        out.push((score, e));
    }
    // Stable, like Python's sort: equal scores keep BM25 order.
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    out.into_iter().map(|(_, e)| Value::Object(e)).collect()
}

#[cfg(test)]
mod tests;
