//! The portfolio: every project in the registry, its packages, files and symbol
//! definitions, in one SQLite file. Port of portfolio.py, same schema.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use rusqlite::{Connection, params};
use serde_json::{Map, Value, json};

use crate::deps::scan_deps;
use crate::fts::{mtime_of, round3};
use crate::paths::{env_path, home};
use crate::registry::Project;
use crate::scanner::{SymbolExtractor, has_symbols, project_files};
use crate::text::{match_expression, split_identifier};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS projects(
    name TEXT PRIMARY KEY, path TEXT, stacks TEXT, status TEXT,
    last_commit TEXT, description TEXT, scanned_at TEXT
);
CREATE TABLE IF NOT EXISTS packages(
    project TEXT NOT NULL, name TEXT NOT NULL, version TEXT, source TEXT NOT NULL,
    PRIMARY KEY (project, source, name)
);
CREATE INDEX IF NOT EXISTS packages_name ON packages(name);
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY, project TEXT NOT NULL, path TEXT NOT NULL,
    lang TEXT, lines INTEGER, mtime REAL, size INTEGER, UNIQUE (project, path)
);
CREATE TABLE IF NOT EXISTS symbols(
    id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, name TEXT NOT NULL, kind TEXT, line INTEGER
);
CREATE INDEX IF NOT EXISTS symbols_file ON symbols(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    words, tokenize='unicode61 remove_diacritics 2'
);
";

/// $SOLOGRAPH_PORTFOLIO_DB, else portfolio.db next to $CODEGRAPH_DB_PATH (~/.solo/).
pub fn default_db_path() -> PathBuf {
    if let Some(p) = env_path("SOLOGRAPH_PORTFOLIO_DB") {
        return p;
    }
    let graph =
        env_path("CODEGRAPH_DB_PATH").unwrap_or_else(|| home().join(".solo").join("codegraph.db"));
    graph
        .parent()
        .map_or_else(|| PathBuf::from("portfolio.db"), |d| d.join("portfolio.db"))
}

#[derive(Debug, Default)]
pub struct ScanReport {
    pub read: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub symbols: usize,
}

pub struct Portfolio {
    pub db: Connection,
}

fn forget_files(db: &Connection, ids: &[i64]) -> Result<()> {
    for id in ids {
        db.execute(
            "DELETE FROM symbols_fts WHERE rowid IN (SELECT id FROM symbols WHERE file_id = ?)",
            [id],
        )?;
        db.execute("DELETE FROM symbols WHERE file_id = ?", [id])?;
        db.execute("DELETE FROM files WHERE id = ?", [id])?;
    }
    Ok(())
}

fn dir_key(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    match parts.len() {
        n if n >= 3 => format!("{}/{}/", parts[0], parts[1]),
        2 => format!("{}/", parts[0]),
        _ => "(root)".into(),
    }
}

impl Portfolio {
    pub fn open(path: Option<&Path>) -> Result<Self> {
        let path = path.map_or_else(default_db_path, Path::to_path_buf);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let db = Connection::open(&path)?;
        db.busy_timeout(std::time::Duration::from_secs(30))?;
        db.pragma_update(None, "journal_mode", "WAL")?;
        db.execute_batch(SCHEMA)?;
        Ok(Self { db })
    }

    /// Upsert registry entries; with `prune`, drop projects no longer listed. Returns the dropped names.
    pub fn sync_projects(&mut self, entries: &[Project], prune: bool) -> Result<Vec<String>> {
        let tx = self.db.transaction()?;
        for e in entries {
            let desc: String = e
                .description
                .as_deref()
                .unwrap_or("")
                .chars()
                .take(500)
                .collect();
            tx.execute(
                "INSERT INTO projects(name, path, stacks, status, last_commit, description) VALUES (?, ?, ?, ?, ?, ?) \
                 ON CONFLICT(name) DO UPDATE SET path = excluded.path, stacks = excluded.stacks, status = excluded.status, \
                 last_commit = excluded.last_commit, description = excluded.description",
                params![e.name, e.path, e.stacks.join(","), e.status, e.last_commit.clone().unwrap_or_default(), desc],
            )?;
        }
        let mut dropped = Vec::new();
        if prune {
            let listed: HashSet<&str> = entries.iter().map(|e| e.name.as_str()).collect();
            let all: Vec<String> = tx
                .prepare("SELECT name FROM projects")?
                .query_map([], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            for name in all.into_iter().filter(|n| !listed.contains(n.as_str())) {
                let ids: Vec<i64> = tx
                    .prepare("SELECT id FROM files WHERE project = ?")?
                    .query_map([&name], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?;
                forget_files(&tx, &ids)?;
                tx.execute("DELETE FROM packages WHERE project = ?", [&name])?;
                tx.execute("DELETE FROM projects WHERE name = ?", [&name])?;
                dropped.push(name);
            }
        }
        tx.commit()?;
        Ok(dropped)
    }

    /// Bring one project's files, symbols and packages in line with the disk.
    pub fn scan_project(
        &mut self,
        name: &str,
        root: &Path,
        ex: &mut SymbolExtractor,
    ) -> Result<ScanReport> {
        let mut report = ScanReport::default();
        let tx = self.db.transaction()?;
        let known: HashMap<String, (i64, Option<f64>, Option<i64>)> = tx
            .prepare("SELECT id, path, mtime, size FROM files WHERE project = ?")?
            .query_map([name], |r| {
                Ok((r.get::<_, String>(1)?, (r.get(0)?, r.get(2)?, r.get(3)?)))
            })?
            .collect::<rusqlite::Result<_>>()?;
        let mut seen: HashSet<String> = HashSet::new();
        for (fp, lang) in project_files(root) {
            let rel = fp
                .strip_prefix(root)
                .unwrap_or(&fp)
                .to_string_lossy()
                .into_owned();
            seen.insert(rel.clone());
            let Ok(st) = std::fs::metadata(&fp) else {
                continue;
            };
            let (mtime, size) = (mtime_of(&st), st.len() as i64);
            let prev = known.get(&rel);
            if let Some(&(_, pm, ps)) = prev
                && pm == Some(mtime)
                && ps == Some(size)
            {
                report.unchanged += 1;
                continue;
            }
            if let Some(&(id, _, _)) = prev {
                forget_files(&tx, &[id])?;
            }
            let lines = std::fs::read(&fp)
                .map_or(0, |b| b.iter().filter(|&&c| c == b'\n').count() as i64 + 1);
            tx.execute(
                "INSERT INTO files(project, path, lang, lines, mtime, size) VALUES (?, ?, ?, ?, ?, ?)",
                params![name, rel, lang, lines, mtime, size],
            )?;
            let fid = tx.last_insert_rowid();
            let symbols = if has_symbols(lang) {
                ex.extract(&fp, lang)
            } else {
                Vec::new()
            };
            for sym in &symbols {
                tx.execute(
                    "INSERT INTO symbols(file_id, name, kind, line) VALUES (?, ?, ?, ?)",
                    params![fid, sym.name, sym.kind, sym.line],
                )?;
                let sid = tx.last_insert_rowid();
                let words = format!(
                    "{} {}",
                    split_identifier(&sym.name),
                    sym.name.to_lowercase()
                );
                tx.execute(
                    "INSERT INTO symbols_fts(rowid, words) VALUES (?, ?)",
                    params![sid, words],
                )?;
            }
            report.read += 1;
            report.symbols += symbols.len();
        }
        let gone: Vec<i64> = known
            .iter()
            .filter(|(k, _)| !seen.contains(*k))
            .map(|(_, v)| v.0)
            .collect();
        forget_files(&tx, &gone)?;
        report.removed = gone.len();
        tx.execute("DELETE FROM packages WHERE project = ?", [name])?;
        for pkg in scan_deps(root) {
            tx.execute(
                "INSERT OR REPLACE INTO packages(project, name, version, source) VALUES (?, ?, ?, ?)",
                params![name, pkg.name, pkg.version, pkg.source],
            )?;
        }
        let now = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S+00:00")
            .to_string();
        tx.execute(
            "UPDATE projects SET scanned_at = ? WHERE name = ?",
            params![now, name],
        )?;
        tx.commit()?;
        Ok(report)
    }

    /// Definitions whose name contains the query's words, best first; an exact name wins.
    pub fn search_symbols(
        &self,
        query: &str,
        n: usize,
        project: Option<&str>,
        kind: Option<&str>,
    ) -> Result<Vec<Value>> {
        let words: Vec<String> = query
            .split_whitespace()
            .map(|w| {
                let s = split_identifier(w);
                if s.is_empty() { w.to_string() } else { s }
            })
            .collect();
        let expr = match_expression(&words.join(" "));
        if expr.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = String::from(
            "SELECT s.id, s.name, s.kind, s.line, f.path, f.project, bm25(symbols_fts) AS rank \
             FROM symbols_fts JOIN symbols s ON s.id = symbols_fts.rowid JOIN files f ON f.id = s.file_id \
             WHERE symbols_fts MATCH ?1",
        );
        if project.is_some() {
            sql.push_str(" AND f.project = ?3");
        }
        if kind.is_some() {
            sql.push_str(if project.is_some() {
                " AND s.kind = ?4"
            } else {
                " AND s.kind = ?3"
            });
        }
        sql.push_str(" ORDER BY rank LIMIT ?2");
        let exact: String = query
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '_')
            .collect::<String>()
            .to_lowercase();
        let limit = (n * 5).max(50) as i64;
        let mut stmt = self.db.prepare(&sql)?;
        let mut order: Vec<i64> = Vec::new();
        let mut hits: HashMap<i64, (i32, Map<String, Value>)> = HashMap::new();
        let all_words = expr.split(" OR ").collect::<Vec<_>>().join(" AND ");
        for m in [all_words.as_str(), expr.as_str()] {
            let mut binds: Vec<Box<dyn rusqlite::ToSql>> =
                vec![Box::new(m.to_string()), Box::new(limit)];
            binds.extend(project.map(|p| Box::new(p.to_string()) as Box<dyn rusqlite::ToSql>));
            binds.extend(kind.map(|k| Box::new(k.to_string()) as Box<dyn rusqlite::ToSql>));
            let rows = stmt.query_map(
                rusqlite::params_from_iter(binds.iter().map(|b| b.as_ref())),
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, Option<i64>>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, String>(5)?,
                        r.get::<_, f64>(6)?,
                    ))
                },
            )?;
            for row in rows {
                let (sid, name, k, line, path, proj, rank) = row?;
                if hits.contains_key(&sid) {
                    continue;
                }
                let boost = if name.replace('_', "").to_lowercase() == exact {
                    3.0
                } else {
                    1.0
                };
                let tier = if m != expr || !expr.contains(" OR ") {
                    0
                } else {
                    1
                };
                let mut h = Map::new();
                h.insert("name".into(), json!(name));
                h.insert("kind".into(), json!(k));
                h.insert("project".into(), json!(proj));
                h.insert("file".into(), json!(path));
                h.insert("line".into(), json!(line));
                h.insert("score".into(), json!(round3(-rank * boost)));
                hits.insert(sid, (tier, h));
                order.push(sid);
            }
            if hits.len() >= n {
                break;
            }
        }
        let mut ranked: Vec<(i32, Map<String, Value>)> = order
            .into_iter()
            .filter_map(|sid| hits.remove(&sid))
            .collect();
        let score = |h: &Map<String, Value>| h["score"].as_f64().unwrap_or(0.0);
        ranked.sort_by(|a, b| {
            a.0.cmp(&b.0).then(
                score(&b.1)
                    .partial_cmp(&score(&a.1))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        Ok(ranked
            .into_iter()
            .take(n)
            .map(|(_, h)| Value::Object(h))
            .collect())
    }

    /// Packages used by more than one project — or, given a name, every project using it.
    pub fn shared_packages(&self, package: Option<&str>) -> Result<Vec<Value>> {
        let map = |r: &rusqlite::Row| -> rusqlite::Result<Value> {
            let projects: String = r.get(2)?;
            let mut set: Vec<&str> = projects
                .split('|')
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            set.sort_unstable();
            Ok(
                json!({"package": r.get::<_, String>(0)?, "source": r.get::<_, String>(1)?, "projects": set}),
            )
        };
        let rows: Vec<Value> = match package {
            Some(p) => self
                .db
                .prepare(
                    "SELECT name, source, group_concat(project, '|') FROM packages WHERE lower(name) LIKE ? \
                     GROUP BY name, source ORDER BY count(*) DESC, name",
                )?
                .query_map([format!("%{}%", p.to_lowercase())], map)?
                .collect::<rusqlite::Result<_>>()?,
            None => self
                .db
                .prepare(
                    "SELECT name, source, group_concat(project, '|') FROM packages GROUP BY name, source \
                     HAVING count(DISTINCT project) > 1 ORDER BY count(*) DESC, name",
                )?
                .query_map([], map)?
                .collect::<rusqlite::Result<_>>()?,
        };
        Ok(rows)
    }

    pub fn stats(&self) -> Result<Value> {
        let count = |sql: &str| -> Result<i64> { Ok(self.db.query_row(sql, [], |r| r.get(0))?) };
        let mut by_kind = Map::new();
        let mut stmt = self
            .db
            .prepare("SELECT kind, count(*) FROM symbols GROUP BY kind ORDER BY 2 DESC")?;
        for row in stmt.query_map([], |r| {
            Ok((r.get::<_, Option<String>>(0)?, r.get::<_, i64>(1)?))
        })? {
            let (k, n) = row?;
            by_kind.insert(k.unwrap_or_else(|| "null".into()), json!(n));
        }
        Ok(json!({
            "projects": count("SELECT count(*) FROM projects")?,
            "files": count("SELECT count(*) FROM files")?,
            "symbols": count("SELECT count(*) FROM symbols")?,
            "packages": count("SELECT count(DISTINCT name) FROM packages")?,
            "scanned_projects": count("SELECT count(*) FROM projects WHERE scanned_at IS NOT NULL")?,
            "symbols_by_kind": by_kind,
        }))
    }
}

mod explain;
pub mod queries;

#[cfg(test)]
mod tests;
