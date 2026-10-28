//! One SQLite file per corpus: `docs` remembers each source file's mtime, size
//! and read offset, `fts` is the FTS5 table ranked by BM25. Port of fts.py, on
//! the same schema — Python and Rust open the same files.

use std::collections::{HashMap, HashSet};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde_json::{Map, Value};

use crate::text::fold;

const DOCS: &str = "
CREATE TABLE IF NOT EXISTS docs(
    id INTEGER PRIMARY KEY,
    doc_id TEXT UNIQUE NOT NULL,
    kind TEXT,
    mtime REAL,
    size INTEGER,
    offset INTEGER,
    gone INTEGER NOT NULL DEFAULT 0,
    meta TEXT
);
";

/// An FTS column, filled from the document field of the same name. An enum, not a
/// string: a misspelt column name used to index the body without a word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    Title,
    Keys,
    Body,
}

impl Column {
    pub fn name(self) -> &'static str {
        match self {
            Self::Title => "title",
            Self::Keys => "keys",
            Self::Body => "body",
        }
    }

    fn text(self, d: &Doc) -> &str {
        match self {
            Self::Title => &d.title,
            Self::Keys => &d.keys,
            Self::Body => &d.body,
        }
    }
}

/// Which text columns the FTS table has and how BM25 weighs them; snippets come
/// from the body. `version` 0 is the shared on-disk schema
/// Python also writes (sessions, sources) and is never dropped. A layout with a
/// version of its own owns its file outright: a mismatch drops the tables, and the
/// next sync re-reads the corpus from disk.
#[derive(Debug, Clone)]
pub struct Layout {
    pub columns: &'static [Column],
    pub weights: Vec<f64>,
    pub version: i64,
}

impl Layout {
    /// `bm25(fts, w1, w2, …)` with this layout's column weights.
    pub fn bm25(&self) -> String {
        let w: Vec<String> = self.weights.iter().map(f64::to_string).collect();
        format!("bm25(fts, {})", w.join(", "))
    }

    fn names(&self) -> String {
        let n: Vec<&str> = self.columns.iter().map(|c| c.name()).collect();
        n.join(", ")
    }
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            columns: &[Column::Title, Column::Body],
            weights: vec![3.0, 1.0],
            version: 0,
        }
    }
}

/// One parsed document. `end` is the byte offset a parser read up to.
#[derive(Debug, Default, Clone)]
pub struct Doc {
    pub title: String,
    /// Short fields worth more than the body (frontmatter summary, tags), for a
    /// layout with a `keys` column; the two-column layout ignores it.
    pub keys: String,
    pub body: String,
    pub kind: String,
    pub meta: Map<String, Value>,
    pub end: Option<i64>,
}

#[derive(Debug, Default)]
pub struct SyncReport {
    pub read: usize,
    pub unchanged: usize,
    pub failed: Vec<String>,
    pub removed: usize,
    pub gone: i64,
    pub indexed: i64,
}

pub type Parse<'a> = &'a dyn Fn(&Path) -> Result<Option<Doc>>;
pub type Append<'a> = &'a dyn Fn(&Path, i64) -> Result<Doc>;

/// st_mtime exactly as CPython computes it (`sec + nsec * 1e-9` in double), so a
/// file Python indexed is "unchanged" to Rust and the two never re-read each other's work.
pub fn mtime_of(meta: &std::fs::Metadata) -> f64 {
    meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9
}

/// Python's `round(x, 3)`: the exact binary value rounded to 3 decimals, ties to
/// even. `(x * 1000).round()` is not it — the multiplication itself rounds, and
/// 1.5874999… came out 1.588 where Python says 1.587.
pub fn round3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap_or(x)
}

#[test]
fn round3_rounds_the_exact_value_like_python() {
    assert_eq!(round3(2.675), 2.675);
    assert_eq!(round3(1.0005), 1.0);
    assert_eq!(round3(0.1235), 0.123);
    assert_eq!(round3(-4.2291), -4.229);
}

/// Bodies longer than this (bytes) get a window snippet instead of FTS5's.
const WINDOW_SNIPPET_ABOVE: usize = 256 * 1024;

pub struct FtsStore {
    pub path: PathBuf,
    pub db: Connection,
    pub layout: Layout,
}

type Known = (Option<f64>, Option<i64>, i64, Option<i64>);

impl FtsStore {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with(path, Layout::default())
    }

    pub fn open_with(path: &Path, layout: Layout) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let db = Connection::open(path)?;
        db.busy_timeout(std::time::Duration::from_secs(30))?;
        db.pragma_update(None, "journal_mode", "WAL")?;
        if layout.version > 0 {
            let have: i64 = db.query_row("PRAGMA user_version", [], |r| r.get(0))?;
            if have != layout.version {
                db.execute_batch("DROP TABLE IF EXISTS fts; DROP TABLE IF EXISTS docs;")?;
                db.pragma_update(None, "user_version", layout.version)?;
            }
        }
        db.execute_batch(DOCS)?;
        db.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5({}, tokenize='unicode61 remove_diacritics 2');",
            layout.names()
        ))?;
        Ok(Self {
            path: path.to_path_buf(),
            db,
            layout,
        })
    }

    /// The FTS row for a document, in the layout's column order.
    fn insert_fts(db: &Connection, columns: &[Column], rowid: i64, d: &Doc) -> Result<()> {
        let values: Vec<String> = columns.iter().map(|c| fold(c.text(d))).collect();
        let marks = vec!["?"; columns.len()].join(", ");
        let mut args: Vec<&dyn rusqlite::ToSql> = vec![&rowid];
        args.extend(values.iter().map(|v| v as &dyn rusqlite::ToSql));
        db.execute(
            &format!(
                "INSERT INTO fts(rowid, {}) VALUES (?, {marks})",
                columns
                    .iter()
                    .map(|c| c.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            args.as_slice(),
        )?;
        Ok(())
    }

    /// Insert or replace one document; `None` records the file as seen but empty.
    pub fn put(
        db: &Connection,
        columns: &[Column],
        doc_id: &str,
        doc: Option<&Doc>,
        mtime: Option<f64>,
        size: Option<i64>,
        gone: bool,
    ) -> Result<()> {
        let kind = doc.map_or("", |d| d.kind.as_str());
        let meta = doc.map_or_else(
            || "{}".to_string(),
            |d| Value::Object(d.meta.clone()).to_string(),
        );
        let offset = doc.and_then(|d| d.end);
        let row: Option<i64> = db
            .query_row("SELECT id FROM docs WHERE doc_id = ?", [doc_id], |r| {
                r.get(0)
            })
            .optional()?;
        let rowid = if let Some(rowid) = row {
            db.execute(
                "UPDATE docs SET kind = ?, mtime = ?, size = ?, offset = ?, gone = ?, meta = ? WHERE id = ?",
                params![kind, mtime, size, offset, gone as i64, meta, rowid],
            )?;
            db.execute("DELETE FROM fts WHERE rowid = ?", [rowid])?;
            rowid
        } else {
            db.execute(
                "INSERT INTO docs(doc_id, kind, mtime, size, offset, gone, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![doc_id, kind, mtime, size, offset, gone as i64, meta],
            )?;
            db.last_insert_rowid()
        };
        if let Some(d) = doc {
            Self::insert_fts(db, columns, rowid, d)?;
        }
        Ok(())
    }

    /// Bring the index in line with `files` ({doc_id: path}): re-read only what
    /// changed, drop what vanished — or with `keep_gone` keep it, marked gone.
    /// `append(path, offset)` reads only the tail of an append-only file that grew.
    pub fn sync(
        &mut self,
        files: &[(String, PathBuf)],
        parse: Parse,
        keep_gone: bool,
        append: Option<Append>,
    ) -> Result<SyncReport> {
        self.sync_under(files, parse, keep_gone, append, None)
    }

    /// [`Self::sync`] for one part of the corpus: `files` lists only the documents
    /// whose id starts with `under`, and only those can vanish.
    pub fn sync_under(
        &mut self,
        files: &[(String, PathBuf)],
        parse: Parse,
        keep_gone: bool,
        append: Option<Append>,
        under: Option<&str>,
    ) -> Result<SyncReport> {
        let mut report = SyncReport::default();
        let columns = self.layout.columns;
        // IMMEDIATE: the write lock is taken up front, so a second writer waits on
        // busy_timeout. A deferred transaction that read first could not upgrade
        // under WAL and failed at once with SQLITE_BUSY (review, 28 Oct 2026).
        let mut tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let known: HashMap<String, Known> = {
            let mut stmt = tx.prepare("SELECT doc_id, mtime, size, gone, offset FROM docs")?;
            stmt.query_map([], |r| {
                Ok((r.get(0)?, (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
            })?
            .collect::<rusqlite::Result<_>>()?
        };
        let listed: HashSet<&str> = files.iter().map(|(id, _)| id.as_str()).collect();
        for (doc_id, path) in files {
            let Ok(st) = std::fs::metadata(path) else {
                report.failed.push(doc_id.clone());
                continue;
            };
            let (mtime, size) = (mtime_of(&st), st.len() as i64);
            let prev = known.get(doc_id);
            if let Some(&(pm, ps, pg, _)) = prev
                && pm == Some(mtime)
                && ps == Some(size)
                && pg == 0
            {
                report.unchanged += 1;
                continue;
            }
            let grown = |p: &Known| {
                p.3.is_some_and(|o| o != 0) && p.2 == 0 && p.1.is_some_and(|s| size > s)
            };
            // One savepoint per document: a failure half-way leaves nothing of it,
            // instead of a row with a fresh mtime and no text ("unchanged" forever).
            let sp = tx.savepoint()?;
            let outcome = match (append, prev) {
                (Some(append), Some(p)) if grown(p) => append(path, p.3.unwrap_or(0))
                    .and_then(|d| Self::append_tail(&sp, doc_id, &d, mtime, size)),
                _ => parse(path).and_then(|d| {
                    Self::put(
                        &sp,
                        columns,
                        doc_id,
                        d.as_ref(),
                        Some(mtime),
                        Some(size),
                        false,
                    )
                }),
            };
            match outcome {
                Ok(()) => {
                    sp.commit()?;
                    report.read += 1;
                }
                Err(_) => report.failed.push(doc_id.clone()),
            }
        }
        let vanished = known
            .keys()
            .filter(|id| !listed.contains(id.as_str()) && under.is_none_or(|p| id.starts_with(p)));
        for doc_id in vanished {
            if keep_gone {
                // Only the newly gone: re-marking 638 archived rows made every
                // session search a writer.
                tx.execute(
                    "UPDATE docs SET gone = 1 WHERE doc_id = ? AND gone = 0",
                    [doc_id],
                )?;
            } else {
                tx.execute(
                    "DELETE FROM fts WHERE rowid = (SELECT id FROM docs WHERE doc_id = ?)",
                    [doc_id],
                )?;
                tx.execute("DELETE FROM docs WHERE doc_id = ?", [doc_id])?;
                report.removed += 1;
            }
        }
        report.gone = tx.query_row("SELECT count(*) FROM docs WHERE gone = 1", [], |r| r.get(0))?;
        report.indexed = tx.query_row("SELECT count(*) FROM docs", [], |r| r.get(0))?;
        tx.commit()?;
        Ok(report)
    }

    /// Remove documents outright — text included, archive or not.
    pub fn forget(&mut self, doc_ids: &HashSet<String>) -> Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        for doc_id in doc_ids {
            tx.execute(
                "DELETE FROM fts WHERE rowid = (SELECT id FROM docs WHERE doc_id = ?)",
                [doc_id],
            )?;
            tx.execute("DELETE FROM docs WHERE doc_id = ?", [doc_id])?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Add a parsed tail to an existing document: body appended, meta updated.
    fn append_tail(
        db: &Connection,
        doc_id: &str,
        delta: &Doc,
        mtime: f64,
        size: i64,
    ) -> Result<()> {
        let (rowid, meta): (i64, Option<String>) = db.query_row(
            "SELECT id, meta FROM docs WHERE doc_id = ?",
            [doc_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let mut merged: Map<String, Value> =
            serde_json::from_str(meta.as_deref().unwrap_or("{}")).unwrap_or_default();
        for (k, v) in &delta.meta {
            if truthy(v) {
                merged.insert(k.clone(), v.clone());
            }
        }
        db.execute(
            "UPDATE docs SET mtime = ?, size = ?, offset = ?, meta = ? WHERE id = ?",
            params![
                mtime,
                size,
                delta.end,
                Value::Object(merged).to_string(),
                rowid
            ],
        )?;
        if !delta.body.is_empty() || !delta.title.is_empty() {
            let (title, body): (String, String) = db
                .query_row(
                    "SELECT title, body FROM fts WHERE rowid = ?",
                    [rowid],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?
                .unwrap_or_default();
            db.execute("DELETE FROM fts WHERE rowid = ?", [rowid])?;
            let title = if delta.title.is_empty() {
                title
            } else {
                fold(&delta.title)
            };
            db.execute(
                "INSERT INTO fts(rowid, title, body) VALUES (?, ?, ?)",
                params![rowid, title, format!("{body}\n\n{}", fold(&delta.body))],
            )?;
        }
        Ok(())
    }

    /// BM25 hits for free text: doc_id, kind, title, snippet, meta, score, gone.
    #[cfg(test)]
    pub fn search(
        &self,
        query: &str,
        n_results: usize,
        kind: Option<&str>,
    ) -> Result<Vec<Map<String, Value>>> {
        self.search_expr(&crate::text::match_expression(query), n_results, kind)
    }

    /// The same, for an FTS5 expression built by the caller (see SessionSearch).
    pub fn search_expr(
        &self,
        expr: &str,
        n_results: usize,
        kind: Option<&str>,
    ) -> Result<Vec<Map<String, Value>>> {
        self.search_scoped(expr, n_results, kind, None)
    }

    /// The same, limited to documents whose id starts with `prefix` (a path such as
    /// `wiki/`), compared literally: `_` and `%` in a path are not wildcards.
    pub fn search_scoped(
        &self,
        expr: &str,
        n_results: usize,
        kind: Option<&str>,
        prefix: Option<&str>,
    ) -> Result<Vec<Map<String, Value>>> {
        self.search_inner(expr, n_results, kind, prefix, None)
    }

    /// The best match per group (`group` is an SQL expression over `d`, e.g. a
    /// video id from meta), n groups. Choosing groups in SQL replaced widening a
    /// window 4x per pass with a snippet for every chunk: 5.9 s on the real
    /// sources index, against tens of ms (review, 28 Oct 2026).
    pub fn search_grouped(
        &self,
        expr: &str,
        n_results: usize,
        kind: Option<&str>,
        group: &str,
    ) -> Result<Vec<Map<String, Value>>> {
        self.search_inner(expr, n_results, kind, None, Some(group))
    }

    fn search_inner(
        &self,
        expr: &str,
        n_results: usize,
        kind: Option<&str>,
        prefix: Option<&str>,
        group: Option<&str>,
    ) -> Result<Vec<Map<String, Value>>> {
        if expr.is_empty() {
            return Ok(Vec::new());
        }
        // Two passes. Any FTS column in the ranking query (snippet(), even the title)
        // makes SQLite read the content row — body included, several MB for a long
        // transcript — of every match before it sorts: 0.8–2.6 s a query, against
        // 1–20 ms for ranking from the index alone. Title and snippet come second,
        // for the rows that made the cut.
        let mut sql = format!(
            "SELECT fts.rowid, d.doc_id, d.kind, d.gone, d.meta, \
             {} AS rank FROM fts JOIN docs d ON d.id = fts.rowid WHERE fts MATCH ?1",
            self.layout.bm25()
        );
        let like = kind.map(|k| format!("%{k}%"));
        sql.push_str(" AND (?3 IS NULL OR d.kind LIKE ?3)");
        sql.push_str(" AND (?4 IS NULL OR substr(d.doc_id, 1, length(?4)) = ?4)");
        let sql = match group {
            // SQLite returns the other columns from the row min() picked.
            // MATERIALIZED: flattened into the aggregate, bm25() has no FTS5 context.
            Some(g) => format!(
                "WITH r AS MATERIALIZED ({}) SELECT rowid, doc_id, kind, gone, meta, \
                 min(rank) AS rank FROM r GROUP BY grp ORDER BY rank LIMIT ?2",
                sql.replacen(" AS rank FROM", &format!(" AS rank, {g} AS grp FROM"), 1)
                    .replacen("SELECT fts.rowid,", "SELECT fts.rowid AS rowid,", 1)
            ),
            None => sql + " ORDER BY rank LIMIT ?2",
        };
        let mut stmt = self.db.prepare(&sql)?;
        let map_row = |r: &rusqlite::Row| -> rusqlite::Result<(i64, Map<String, Value>)> {
            let mut hit = Map::new();
            hit.insert("doc_id".into(), Value::String(r.get(1)?));
            hit.insert(
                "kind".into(),
                Value::String(r.get::<_, Option<String>>(2)?.unwrap_or_default()),
            );
            hit.insert("title".into(), Value::String(String::new()));
            hit.insert("snippet".into(), Value::String(String::new()));
            let meta: Option<String> = r.get(4)?;
            if let Ok(Value::Object(m)) = serde_json::from_str(meta.as_deref().unwrap_or("{}")) {
                hit.extend(m);
            }
            let rank: f64 = r.get(5)?;
            hit.insert("score".into(), Value::from(round3(-rank)));
            hit.insert("gone".into(), Value::Bool(r.get::<_, i64>(3)? != 0));
            Ok((r.get(0)?, hit))
        };
        let rows: Vec<(i64, Map<String, Value>)> = stmt
            .query_map(params![expr, n_results as i64, like, prefix], map_row)?
            .collect::<rusqlite::Result<_>>()?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<String> = rows.iter().map(|(id, _)| id.to_string()).collect();
        // Long bodies get a window snippet built here (see text::window_snippet);
        // the rest keep FTS5's, which picks the densest fragment.
        let mut texts: HashMap<i64, (String, String)> = HashMap::new();
        let mut small: Vec<String> = Vec::new();
        let mut stmt = self.db.prepare(&format!(
            "SELECT rowid, title, body FROM fts WHERE rowid IN ({})",
            ids.join(",")
        ))?;
        for row in stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            ))
        })? {
            let (id, title, body) = row?;
            let snip = if body.len() > WINDOW_SNIPPET_ABOVE {
                crate::text::window_snippet(&body, expr)
            } else {
                None
            };
            if snip.is_none() {
                small.push(id.to_string());
            }
            texts.insert(id, (title, snip.unwrap_or_default()));
        }
        if !small.is_empty() {
            let mut stmt = self.db.prepare(&format!(
                // `+rowid`: a bare rowid IN is handed to FTS5, which re-runs the
                // MATCH per row: 66 ms for 20 snippets against 3 ms.
                "SELECT rowid, snippet(fts, {}, '«', '»', '…', 24) FROM fts \
                 WHERE fts MATCH ?1 AND +rowid IN ({})",
                self.layout
                    .columns
                    .iter()
                    .position(|c| *c == Column::Body)
                    .unwrap_or(self.layout.columns.len() - 1),
                small.join(",")
            ))?;
            for row in stmt.query_map([expr], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?))
            })? {
                let (id, snip) = row?;
                if let Some(entry) = texts.get_mut(&id) {
                    entry.1 = snip.unwrap_or_default();
                }
            }
        }
        Ok(rows
            .into_iter()
            .map(|(id, mut hit)| {
                if let Some((title, snip)) = texts.remove(&id) {
                    hit.insert("title".into(), Value::String(title));
                    hit.insert("snippet".into(), Value::String(snip));
                }
                hit
            })
            .collect())
    }

    pub fn count(&self) -> Result<i64> {
        Ok(self
            .db
            .query_row("SELECT count(*) FROM docs", [], |r| r.get(0))?)
    }
}

/// Python truthiness for the JSON values meta holds.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests;
