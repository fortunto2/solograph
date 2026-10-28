//! Knowledge base: markdown with YAML frontmatter, ranked by BM25, and the
//! [[wikilink]] graph between the pages, read in the same pass over the files.
//!
//! One engine for everything that used to index the same wiki separately (Oct 2026):
//! a Python FTS index for `make wiki-search`, another for the prompt hint, a JSON
//! graph rebuilt by a script for `expand_graph`, and grep for orphans and hubs.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::SystemTime;

use anyhow::Result;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::aliases::Aliases;
use crate::fts::{Doc, FtsStore, Layout, SyncReport};
use crate::paths::env_path;
use crate::yamlval::{python_str, to_json};

pub mod graph;
pub mod hint;

/// Its own file: the three-column layout is not the two-column `search.db` that
/// solograph wrote there until Oct 2026, which binaries still running keep reading.
pub const DB_FILE: &str = "kb.db";
const LAYOUT_VERSION: i64 = 1;
/// title, keys (index_line, description, tags, file name), body. Measured on a
/// known-answer set of 75 queries (Oct 2026): 2/2/1 beat Epiphan's 5/3/1 and every
/// title weight from 1 to 12; summary words count as much as title words. Override
/// with SOLOGRAPH_KB_WEIGHTS=t,k,b to re-measure.
const WEIGHTS: [f64; 3] = [2.0, 2.0, 1.0];

const SKIP_DIRS: [&str; 6] = [
    "node_modules",
    "__pycache__",
    "archive",
    "blog",
    "invoices",
    "solo-factory",
];
const SKIP_FILES: [&str; 2] = ["README.md", "INDEX.md"];

pub fn layout() -> Layout {
    let weights = std::env::var("SOLOGRAPH_KB_WEIGHTS")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|x| x.trim().parse().ok())
                .collect::<Option<Vec<f64>>>()
        })
        .filter(|w| w.len() == 3)
        .unwrap_or_else(|| WEIGHTS.to_vec());
    Layout {
        columns: &["title", "keys", "body"],
        weights,
        version: LAYOUT_VERSION,
    }
}

/// The KB root: KB_PATH, else the current directory when it looks like one (an
/// existing index, or markdown with frontmatter one level down).
pub fn detect_root() -> Option<PathBuf> {
    if let Some(p) = env_path("KB_PATH") {
        return Some(p);
    }
    let cwd = std::env::current_dir().ok()?;
    if cwd.join(".solo").join("kb").exists() {
        return Some(cwd);
    }
    let mut checked = 0;
    for dir in std::fs::read_dir(&cwd)
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
    {
        for f in std::fs::read_dir(dir.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            if checked >= 5 {
                return None;
            }
            if f.path().extension().is_some_and(|x| x == "md") {
                checked += 1;
                let head = std::fs::read(f.path()).unwrap_or_default();
                if head.starts_with(b"---") {
                    return Some(cwd);
                }
            }
        }
    }
    None
}

/// An explicit path, else KB_PATH, else the current directory.
pub fn root_or_cwd(explicit: Option<PathBuf>) -> PathBuf {
    explicit
        .or_else(|| env_path("KB_PATH"))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// (relative path, path) for every markdown file outside hidden and vendored dirs.
///
/// Walks like os.walk: a symlink to a directory is listed but not entered, a
/// symlink to a file is a file.
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
static LINK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\[([^\]|\\#]+)").expect("static regex"));
static FENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(```|~~~)").expect("static regex"));
static INLINE_CODE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`[^`]*`").expect("static regex"));

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

/// The pages a text links to: `[[slug]]` and `[[slug|label]]`, outside fenced
/// blocks and inline code (`[[agent.menu]]` in a TOML example is not a link),
/// as slugs — lowercased, spaces to hyphens, an `#anchor` dropped. Each once.
pub fn wikilinks(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut fence: Option<&str> = None;
    for line in text.lines() {
        if let Some(m) = FENCE.captures(line) {
            let mark = m.get(1).map_or("", |x| x.as_str());
            fence = match fence {
                Some(f) if f == mark => None,
                Some(f) => Some(f),
                None => Some(mark),
            };
            continue;
        }
        if fence.is_some() {
            continue;
        }
        let plain = INLINE_CODE.replace_all(line, "");
        for c in LINK.captures_iter(&plain) {
            let slug = c[1].trim().to_lowercase().replace(' ', "-");
            if !slug.is_empty() && !out.contains(&slug) {
                out.push(slug);
            }
        }
    }
    out
}

fn field(meta: &Map<String, Value>, key: &str) -> String {
    meta.get(key)
        .filter(|v| crate::fts::truthy(v))
        .map(python_str)
        .unwrap_or_default()
}

pub fn parse_markdown(path: &Path) -> Result<Option<Doc>> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes).replace('\u{FFFD}', "");
    let (meta, body) = frontmatter(&text).unwrap_or_else(|_| (Map::new(), text.clone()));
    let mut title = field(&meta, "title");
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
    // The file name is a name too: `code-graph-mcp-landscape` says "landscape" where
    // the title does not.
    let slug = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .replace(['-', '_'], " ");
    // index_line first: the hint reads it back as the page's one-line summary.
    let keys = [
        field(&meta, "index_line"),
        field(&meta, "description"),
        tags.replace(',', " "),
        slug,
    ]
    .join("\n");
    let mut m = Map::new();
    m.insert("type".into(), json!(doc_type));
    m.insert("tags".into(), json!(tags));
    m.insert("links".into(), json!(wikilinks(&text)));
    Ok(Some(Doc {
        title,
        keys,
        body,
        kind: doc_type,
        meta: m,
        end: None,
    }))
}

pub struct KnowledgeBase {
    pub root: PathBuf,
    pub store: FtsStore,
    aliases: Option<(PathBuf, SystemTime, Aliases)>,
}

impl KnowledgeBase {
    pub fn open(root: &Path, db_dir: Option<&Path>) -> Result<Self> {
        let dir = db_dir.map_or_else(|| root.join(".solo").join("kb"), Path::to_path_buf);
        Ok(Self {
            root: root.to_path_buf(),
            store: FtsStore::open_with(&dir.join(DB_FILE), layout())?,
            aliases: None,
        })
    }

    pub fn sync(&mut self) -> Result<SyncReport> {
        self.sync_under(None)
    }

    /// Only the files under `prefix` (`wiki/`): the hint reads nothing else, and the
    /// whole tree was 7x the directories (a client checkout alone held 949 of 1187).
    pub fn sync_under(&mut self, prefix: Option<&str>) -> Result<SyncReport> {
        let mut files = Vec::new();
        let dir = prefix.map_or_else(|| self.root.clone(), |p| self.root.join(p));
        walk(&self.root, &dir, &mut files);
        self.store
            .sync_under(&files, &parse_markdown, false, None, prefix)
    }

    /// The KB's alias table, re-read when the file changes; None without one.
    pub fn aliases(&mut self) -> Option<&Aliases> {
        let found = Aliases::locate(&self.root);
        let fresh = match (&found, &self.aliases) {
            (Some((p, t)), Some((cp, ct, _))) => p == cp && t == ct,
            (None, None) => true,
            _ => false,
        };
        if !fresh {
            self.aliases = found.map(|(p, t)| {
                let a = Aliases::load(&p).unwrap_or_default();
                (p, t, a)
            });
        }
        self.aliases
            .as_ref()
            .map(|(_, _, a)| a)
            .filter(|a| !a.is_empty())
    }

    /// The FTS5 expression for a query: its own terms, plus every alias of a term
    /// the KB's alias table lists.
    pub fn expression(&mut self, query: &str) -> String {
        match self.aliases() {
            Some(a) => a.expression(query),
            None => crate::text::match_expression(query),
        }
    }

    pub fn search(
        &mut self,
        query: &str,
        n: usize,
        doc_type: Option<&str>,
        expand_graph: bool,
        path: Option<&str>,
    ) -> Result<Vec<Value>> {
        self.sync()?;
        let expr = self.expression(query);
        let mut results: Vec<Value> = self
            .store
            .search_scoped(&expr, n, doc_type, path)?
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
        if expand_graph && !results.is_empty() {
            let g = self.link_graph()?; // synced above
            let extra = g.neighbours(&results, path, 3);
            results.extend(extra);
        }
        Ok(results)
    }

    pub fn stats(&mut self) -> Result<Value> {
        let r = self.sync()?;
        let aliases = self.aliases().map_or(0, Aliases::len);
        Ok(json!({
            "documents": r.indexed,
            "read_now": r.read,
            "alias_groups": aliases,
            "storage": self.store.path.display().to_string(),
        }))
    }

    /// Doc ids currently indexed (synced first).
    #[cfg(test)]
    pub fn ids(&mut self) -> Result<std::collections::HashSet<String>> {
        self.sync()?;
        let mut stmt = self.store.db.prepare("SELECT doc_id FROM docs")?;
        let ids = stmt
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(ids)
    }
}

#[cfg(test)]
mod tests;
