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
use crate::fts::{Column, Doc, FtsStore, Layout, SyncReport};
use crate::paths::env_path;
use crate::yamlval::{python_str, to_json};

pub mod config;
pub mod graph;
pub mod hint;
pub mod memgate;

pub use config::Config;

/// Its own file: the three-column layout is not the two-column `search.db` that
/// solograph wrote there until Oct 2026, which binaries still running keep reading.
pub const DB_FILE: &str = "kb.db";
const LAYOUT_VERSION: i64 = 1;
/// title, keys (index_line, description, tags, file name), body. Measured on a
/// known-answer set of 75 queries (Oct 2026): 2/2/1 beat Epiphan's 5/3/1 and every
/// title weight from 1 to 12; summary words count as much as title words. Override
/// with SOLOGRAPH_KB_WEIGHTS=t,k,b to re-measure.
const WEIGHTS: [f64; 3] = [2.0, 2.0, 1.0];

/// Never knowledge, in any KB; what else to leave out is the KB's own `exclude`.
const SKIP_DIRS: [&str; 2] = ["node_modules", "__pycache__"];
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
        columns: &[Column::Title, Column::Keys, Column::Body],
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
    let cwd = implicit_root(&std::env::current_dir().ok()?)?;
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

/// A path filter as a folder: "wiki" and "./wiki" mean "wiki/". Without the slash
/// it matched wiki-archive/ too, and made the catalog "wikiindex.md".
pub fn prefix(p: &str) -> String {
    let p = p.trim().trim_start_matches("./");
    if p.is_empty() || p.ends_with('/') || p.ends_with(".md") {
        p.to_string()
    } else {
        format!("{p}/")
    }
}

/// An explicit path, else KB_PATH, else the current directory unless it is the
/// home directory or `/`: from there the walk would read everything below it.
pub fn root_or_cwd(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit.or_else(|| env_path("KB_PATH")) {
        return Ok(p);
    }
    let cwd = std::env::current_dir()?;
    implicit_root(&cwd).ok_or_else(|| {
        anyhow::anyhow!(
            "no knowledge base here ({}): pass --kb-path DIR or set KB_PATH",
            cwd.display()
        )
    })
}

/// The current directory as a KB root, unless it is home or the filesystem root.
fn implicit_root(cwd: &Path) -> Option<PathBuf> {
    let home = crate::paths::home();
    (cwd != home && cwd.parent().is_some()).then(|| cwd.to_path_buf())
}

/// (relative path, path) for every markdown file outside hidden, vendored and
/// excluded dirs.
///
/// Walks like os.walk: a symlink to a directory is listed but not entered, a
/// symlink to a file is a file.
fn walk(root: &Path, dir: &Path, cfg: &Config, found: &mut Vec<(String, PathBuf)>) {
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
            let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy();
            // A folder marked private never reaches a KB rooted above it.
            let private = || {
                crate::sessions::PRIVATE_MARKERS
                    .iter()
                    .any(|m| path.join(m).exists())
            };
            if !ft.is_symlink()
                && !name.starts_with('.')
                && !SKIP_DIRS.contains(&name.as_str())
                && !cfg.excludes(&rel, &name)
                && !private()
            {
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
        walk(root, &sub, cfg, found);
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
    let raw = &text[open.end()..close.start()];
    let fm: serde_yaml::Value = match serde_yaml::from_str(raw) {
        Ok(v) => v,
        // A repeated key: keep the last, as PyYAML did, rather than lose it all.
        Err(e) if e.to_string().contains("duplicate") => serde_yaml::from_str(&last_key_wins(raw))?,
        Err(e) => return Err(e.into()),
    };
    let meta = match to_json(&fm) {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    Ok((meta, text[close.end()..].trim().to_string()))
}

/// YAML with each top-level key kept once, at its last block (the key's line and
/// the indented or list lines under it).
fn last_key_wins(yaml: &str) -> String {
    let mut blocks: Vec<(String, String)> = Vec::new();
    for line in yaml.lines() {
        let top = !line.starts_with([' ', '\t', '-', '#']) && line.contains(':');
        match (top, blocks.last_mut()) {
            (true, _) => {
                let key = line.split(':').next().unwrap_or("").trim().to_string();
                blocks.retain(|(k, _)| *k != key);
                blocks.push((key, format!("{line}\n")));
            }
            (false, Some((_, text))) => {
                text.push_str(line);
                text.push('\n');
            }
            (false, None) => {}
        }
    }
    blocks.into_iter().map(|(_, t)| t).collect()
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
    pub config: Config,
    /// When `.solograph.yaml` was read: a long-lived server re-reads it on change.
    config_stamp: Option<SystemTime>,
    aliases: Option<(PathBuf, SystemTime, Aliases)>,
}

/// a[0], b[0], a[1], b[1], …, each document once, n at most.
fn interleave(
    a: Vec<Map<String, Value>>,
    b: Vec<Map<String, Value>>,
    n: usize,
) -> Vec<Map<String, Value>> {
    let mut out: Vec<Map<String, Value>> = Vec::new();
    let (mut a, mut b) = (a.into_iter(), b.into_iter());
    loop {
        let (x, y) = (a.next(), b.next());
        if x.is_none() && y.is_none() {
            return out;
        }
        for h in [x, y].into_iter().flatten() {
            if out.len() < n && !out.iter().any(|o| o["doc_id"] == h["doc_id"]) {
                out.push(h);
            }
        }
    }
}

fn config_stamp(root: &Path) -> Option<SystemTime> {
    std::fs::metadata(root.join(config::FILE))
        .and_then(|m| m.modified())
        .ok()
}

impl KnowledgeBase {
    pub fn open(root: &Path, db_dir: Option<&Path>) -> Result<Self> {
        let dir = db_dir.map_or_else(|| root.join(".solo").join("kb"), Path::to_path_buf);
        let mut store = FtsStore::open_with(&dir.join(DB_FILE), layout())?;
        store.redact_existing()?;
        Ok(Self {
            root: root.to_path_buf(),
            store,
            config: Config::load(root)?,
            config_stamp: config_stamp(root),
            aliases: None,
        })
    }

    pub fn sync(&mut self) -> Result<SyncReport> {
        self.sync_under(None)
    }

    /// Only the files under `prefix` (the wiki): the hint reads nothing else, and the
    /// whole tree was 7x the directories (a client checkout alone held 949 of 1187).
    pub fn sync_under(&mut self, prefix: Option<&str>) -> Result<SyncReport> {
        // The MCP server keeps one KnowledgeBase for its life; a config it never
        // re-read fought every CLI run over what the index holds (review, 28 Oct).
        let stamp = config_stamp(&self.root);
        if stamp != self.config_stamp {
            self.config = Config::load(&self.root)?;
            self.config_stamp = stamp;
        }
        let mut files = Vec::new();
        let dir = prefix.map_or_else(|| self.root.clone(), |p| self.root.join(p));
        walk(&self.root, &dir, &self.config, &mut files);
        self.store
            .sync_under(&files, &parse_markdown, false, None, prefix)
    }

    /// The KB's alias table, re-read when the file changes; None without one.
    pub fn aliases(&mut self) -> Option<&Aliases> {
        let found = Aliases::locate(self.config.aliases_file(&self.root));
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
        let path = path.map(prefix);
        let path = path.as_deref();
        let expr = self.expression(query);
        let wiki = &self.config.pages;
        let hits = if path.is_none() && !wiki.is_empty() && self.root.join(wiki).is_dir() {
            // Two lanes, taken in turn from the whole KB first. A Russian question for
            // an English wiki page lost to Russian notes elsewhere: 6 of 22 in the top
            // 5, against 20 with the wiki alone. Interleaved: 16 of 22 (14 of 22 on a
            // held-out set), with 40 questions about notes outside the wiki unchanged
            // at #1 and one fewer in the top 5 (28 Oct 2026).
            let whole = self.store.search_scoped(&expr, n, doc_type, None)?;
            let pages = self.store.search_scoped(&expr, n, doc_type, Some(wiki))?;
            interleave(whole, pages, n)
        } else {
            self.store.search_scoped(&expr, n, doc_type, path)?
        };
        let mut results: Vec<Value> = hits
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
