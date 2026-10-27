//! Knowledge base search: markdown with YAML frontmatter, ranked by BM25. Port of kb.py.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::fts::{Doc, FtsStore, SyncReport};
use crate::yamlval::{python_str, to_json};

const SKIP_DIRS: [&str; 6] = [
    "node_modules",
    "__pycache__",
    "archive",
    "blog",
    "invoices",
    "solo-factory",
];
const SKIP_FILES: [&str; 2] = ["README.md", "INDEX.md"];

/// (relative path, path) for every markdown file outside hidden and vendored dirs.
///
/// Walks like os.walk: a symlink to a directory is listed but not entered, a
/// symlink to a file is a file.
pub fn markdown_files(root: &Path) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    walk(root, root, &mut found);
    found
}

fn walk(root: &Path, dir: &Path, found: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(ft) = entry.file_type() else { continue };
        let is_dir = if ft.is_symlink() {
            path.is_dir()
        } else {
            ft.is_dir()
        };
        if is_dir {
            if !ft.is_symlink() && !name.starts_with('.') && !SKIP_DIRS.contains(&name.as_str()) {
                subdirs.push(path);
            }
        } else if name.ends_with(".md") && !SKIP_FILES.contains(&name.as_str()) {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            found.push((rel, path));
        }
    }
    for sub in subdirs {
        walk(root, &sub, found);
    }
}

static BOUNDARY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^-{3,}\s*$").expect("static regex"));

/// python-frontmatter's parse: strip, split on the first two `---` lines, YAML
/// the middle. Returns (metadata, body); no frontmatter means the stripped text.
pub fn frontmatter(text: &str) -> Result<(Map<String, Value>, String)> {
    let text = text.trim();
    let mut bounds = BOUNDARY.find_iter(text);
    let (Some(open), Some(close)) = (bounds.next(), bounds.next()) else {
        return Ok((Map::new(), text.to_string()));
    };
    if open.start() != 0 {
        return Ok((Map::new(), text.to_string()));
    }
    let fm: serde_yaml::Value = serde_yaml::from_str(&text[open.end()..close.start()])?;
    let meta = match to_json(&fm) {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    Ok((meta, text[close.end()..].trim().to_string()))
}

pub fn parse_markdown(path: &Path) -> Result<Option<Doc>> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes).replace('\u{FFFD}', "");
    let (meta, body) = frontmatter(&text).unwrap_or_else(|_| (Map::new(), text.clone()));
    let mut title = meta
        .get("title")
        .filter(|v| crate::fts::truthy(v))
        .map(python_str)
        .unwrap_or_default();
    if title.is_empty() {
        let heading = body.lines().find(|l| l.starts_with("# ")).unwrap_or("");
        title = heading.get(2..).unwrap_or("").trim().to_string();
        if title.is_empty() {
            title = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
        }
    }
    let tags = match meta.get("tags") {
        Some(Value::Array(a)) => a.iter().map(python_str).collect::<Vec<_>>().join(","),
        Some(v) if crate::fts::truthy(v) => python_str(v),
        _ => String::new(),
    };
    let doc_type = meta
        .get("type")
        .filter(|v| crate::fts::truthy(v))
        .map_or_else(|| "unknown".to_string(), python_str);
    let mut m = Map::new();
    m.insert("type".into(), json!(doc_type));
    m.insert("tags".into(), json!(tags));
    Ok(Some(Doc {
        title,
        body: format!("{tags}\n{body}"),
        kind: doc_type,
        meta: m,
        end: None,
    }))
}

pub struct KnowledgeBase {
    pub root: PathBuf,
    pub store: FtsStore,
}

impl KnowledgeBase {
    pub fn open(root: &Path, db_dir: Option<&Path>) -> Result<Self> {
        let dir = db_dir.map_or_else(|| root.join(".solo").join("kb"), Path::to_path_buf);
        Ok(Self {
            root: root.to_path_buf(),
            store: FtsStore::open(&dir.join("search.db"))?,
        })
    }

    pub fn sync(&mut self) -> Result<SyncReport> {
        let files = markdown_files(&self.root);
        self.store.sync(&files, &parse_markdown, false, None)
    }

    pub fn search(
        &mut self,
        query: &str,
        n: usize,
        doc_type: Option<&str>,
        expand_graph: bool,
    ) -> Result<Vec<Value>> {
        self.sync()?;
        let mut results: Vec<Value> = self
            .store
            .search(query, n, doc_type)?
            .into_iter()
            .map(|h| {
                json!({
                    "file": h["doc_id"],
                    "title": h["title"],
                    "type": h.get("type").cloned().unwrap_or_else(|| h["kind"].clone()),
                    "tags": h.get("tags").cloned().unwrap_or_else(|| json!("")),
                    "score": h["score"],
                    "snippet": h["snippet"],
                })
            })
            .collect();
        if expand_graph {
            let extra = self.graph_neighbors(&results);
            results.extend(extra);
        }
        Ok(results)
    }

    /// 1-hop neighbours of the top three hits from .metadata/knowledge_graph.json.
    fn graph_neighbors(&self, results: &[Value]) -> Vec<Value> {
        let path = self.root.join(".metadata").join("knowledge_graph.json");
        let Some(graph) = std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        else {
            return Vec::new();
        };
        let mut adj: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for e in graph["edges"].as_array().into_iter().flatten() {
            let typ = e["type"].as_str().unwrap_or("");
            if typ == "explicit" || typ == "shared_tags" {
                let (s, t) = (
                    e["source"].as_str().unwrap_or(""),
                    e["target"].as_str().unwrap_or(""),
                );
                adj.entry(s.into())
                    .or_default()
                    .push((t.into(), typ.into()));
                adj.entry(t.into())
                    .or_default()
                    .push((s.into(), typ.into()));
            }
        }
        let mut seen: HashSet<String> = results
            .iter()
            .filter_map(|r| r["file"].as_str().map(String::from))
            .collect();
        let mut extra = Vec::new();
        for r in results.iter().take(3) {
            let file = r["file"].as_str().unwrap_or("");
            for (neighbor, edge_type) in adj.get(file).into_iter().flatten() {
                if !seen.insert(neighbor.clone()) {
                    continue;
                }
                let node = &graph["nodes"][neighbor.as_str()];
                let tags: Vec<&str> = node["tags"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .collect();
                extra.push(json!({
                    "file": neighbor,
                    "title": node.get("title").cloned().unwrap_or_else(|| json!(neighbor)),
                    "type": node.get("type").cloned().unwrap_or_else(|| json!("unknown")),
                    "tags": tags.join(","),
                    "graph_source": format!("via {edge_type}: {}", r["title"].as_str().unwrap_or("")),
                }));
            }
        }
        extra
    }

    pub fn stats(&mut self) -> Result<Value> {
        let r = self.sync()?;
        Ok(
            json!({"documents": r.indexed, "read_now": r.read, "storage": self.store.path.display().to_string()}),
        )
    }
}

#[cfg(test)]
mod tests;
