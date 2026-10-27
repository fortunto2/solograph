//! `codegraph_explain`: what a project is made of. Port of Portfolio.explain.

use std::collections::HashMap;

use anyhow::Result;
use rusqlite::OptionalExtension;
use serde_json::{Map, Value, json};

use super::{Portfolio, dir_key};

/// Insertion-ordered counter, so ties sort the way Python's dicts would.
#[derive(Default)]
struct Counts {
    order: Vec<String>,
    map: HashMap<String, (i64, i64)>,
}

impl Counts {
    fn add(&mut self, key: &str, a: i64, b: i64) {
        let e = self.map.entry(key.to_string()).or_insert_with(|| {
            self.order.push(key.to_string());
            (0, 0)
        });
        e.0 += a;
        e.1 += b;
    }
    /// (key, a, b) sorted by a descending, stable.
    fn ranked(&self) -> Vec<(String, i64, i64)> {
        let mut v: Vec<(String, i64, i64)> = self
            .order
            .iter()
            .map(|k| (k.clone(), self.map[k].0, self.map[k].1))
            .collect();
        v.sort_by_key(|x| -x.1);
        v
    }
}

impl Portfolio {
    /// Stack, languages, directory layers, patterns, dependencies, largest files.
    /// Err(String) when the project is unknown, as Python returned a message.
    pub fn explain(&self, project: &str) -> Result<std::result::Result<Value, String>> {
        let db = &self.db;
        let row: Option<(Option<String>, Option<String>)> = db
            .query_row(
                "SELECT stacks, description FROM projects WHERE name = ?",
                [project],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((stacks, description)) = row else {
            return Ok(Err(format!("Project '{project}' not found.")));
        };
        let files: Vec<(String, Option<String>, Option<i64>)> = db
            .prepare("SELECT id, path, lang, lines FROM files WHERE project = ?")?
            .query_map([project], |r| Ok((r.get(1)?, r.get(2)?, r.get(3)?)))?
            .collect::<rusqlite::Result<_>>()?;
        let symbols: Vec<(String, Option<String>, String)> = db
            .prepare("SELECT s.name, s.kind, f.path FROM symbols s JOIN files f ON f.id = s.file_id WHERE f.project = ?")?
            .query_map([project], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        let deps: Vec<(String, String)> = db
            .prepare("SELECT name, source FROM packages WHERE project = ? ORDER BY source, name")?
            .query_map([project], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        let stack = stacks
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "(no stack)".into());
        let mut data = Map::new();
        data.insert("name".into(), json!(project));
        data.insert("stack".into(), json!(stack));
        data.insert("description".into(), json!(description.unwrap_or_default()));
        data.insert(
            "counts".into(),
            json!({"files": files.len(), "symbols": symbols.len(), "packages": deps.len()}),
        );
        let (mut langs, mut dirs) = (Counts::default(), Counts::default());
        for (path, lang, lines) in &files {
            if let Some(lang) = lang.as_deref().filter(|l| !l.is_empty()) {
                langs.add(lang, 1, lines.unwrap_or(0));
            }
            dirs.add(&dir_key(path), 1, 0);
        }
        let mut per_file = Counts::default();
        for (_, _, path) in &symbols {
            dirs.add(&dir_key(path), 0, 1);
            per_file.add(path, 1, 0);
        }
        let triples = |v: Vec<(String, i64, i64)>| -> Value {
            json!(
                v.into_iter()
                    .map(|(k, a, b)| json!([k, a, b]))
                    .collect::<Vec<_>>()
            )
        };
        if !langs.order.is_empty() {
            data.insert("languages".into(), triples(langs.ranked()));
        }
        if !dirs.order.is_empty() {
            data.insert(
                "layers".into(),
                triples(dirs.ranked().into_iter().take(12).collect()),
            );
        }
        let class = |k: &Option<String>| k.as_deref() == Some("class");
        let mut patterns = Vec::new();
        let mut mixins: Vec<&str> = symbols
            .iter()
            .filter(|(n, k, _)| class(k) && n.contains("Mixin"))
            .map(|(n, _, _)| n.as_str())
            .collect();
        mixins.sort_unstable();
        if !mixins.is_empty() {
            patterns.push(format!(
                "Mixins: {}",
                mixins
                    .iter()
                    .take(8)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let crud = symbols
            .iter()
            .filter(|(n, k, _)| {
                class(k)
                    && ["Create", "Read", "Update", "Schema"]
                        .iter()
                        .any(|s| n.ends_with(s))
            })
            .count();
        if crud > 3 {
            patterns.push(format!(
                "CRUD schemas: {crud} Create/Read/Update/Schema classes"
            ));
        }
        let protocols = symbols
            .iter()
            .filter(|(_, k, _)| matches!(k.as_deref(), Some("protocol" | "interface")))
            .count();
        if protocols > 0 {
            patterns.push(format!("Protocols/Interfaces: {protocols}"));
        }
        if !patterns.is_empty() {
            data.insert("patterns".into(), json!(patterns));
        }
        if !deps.is_empty() {
            data.insert(
                "dependencies".into(),
                json!(
                    deps.iter()
                        .take(15)
                        .map(|(n, s)| json!([n, s]))
                        .collect::<Vec<_>>()
                ),
            );
        }
        let biggest: Vec<(String, i64, i64)> = per_file.ranked().into_iter().take(8).collect();
        if !biggest.is_empty() {
            let kept: Vec<Value> = biggest
                .into_iter()
                .filter(|(_, n, _)| *n >= 3)
                .map(|(p, n, _)| json!([p, n]))
                .collect();
            data.insert("largest_files".into(), json!(kept));
        }
        Ok(Ok(Value::Object(data)))
    }
}
