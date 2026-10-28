//! The prompt hint: point a session at a wiki page when the prompt names something
//! the wiki is specifically about.
//!
//! The wiki holds measured experience (models tried, stacks chosen, limits hit) that
//! a fresh session does not know exists. A rule saying "check the wiki" is
//! forgotten; a UserPromptSubmit hook notices instead. It stays silent unless a
//! prompt holds a word that is rare in the wiki AND names a page (slug, title or
//! index_line), on a page that also covers the prompt's topical words; it points at
//! two pages at most, and at a page once per session. A false hint costs context on
//! every prompt, so silence is the default.
//!
//! Russian words reach the English wiki through the KB's alias table, the same one
//! search uses ("лайа" → laya). Until Oct 2026 this was scripts/wiki_hint.py with
//! its own FTS index and its own hardcoded alias dict.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use super::KnowledgeBase;
use super::graph::stem;
use crate::text::all_terms as expr;
use anyhow::Result;
use regex::Regex;
use rusqlite::params;

const MIN_WORDS: usize = 3;
const MAX_PAGES: usize = 2;
const STOP: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "what", "how", "why", "are", "was", "you",
    "can", "not", "but", "all", "any", "use", "using", "into", "about", "make", "makes", "made",
    "get", "got", "has", "have", "had", "will", "would", "should", "could", "just", "also", "more",
    "most", "some", "only", "than", "then", "them", "they", "there", "their", "our", "its",
];
const STACK_WORDS: [&str; 2] = ["стек", "stack"];
const HEADER: &str =
    "The wiki has measured notes on this — read before answering or searching the web:";

static TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z][a-z0-9+#.\-]{2,}").expect("static regex"));

/// Where per-session "already pointed at" lists live.
pub fn state_dir() -> PathBuf {
    crate::paths::home().join(".cache").join("wiki-hint")
}

/// Latin words of a prompt worth looking up, each once, trimmed of `.` and `-`.
fn tokens(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in TOKEN.find_iter(text) {
        let t = m
            .as_str()
            .trim_matches(|c| c == '.' || c == '-')
            .to_string();
        if t.len() >= 3 && !STOP.contains(&t.as_str()) && !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

struct Candidate {
    path: String,
    title: String,
    line: String,
}

/// Rows under the wiki folder; which of them are pages is `Config::is_page`.
const UNDER: &str = "d.gone = 0 AND substr(d.doc_id, 1, length(?2)) = ?2";

impl KnowledgeBase {
    /// Wiki pages holding every word of `expr`; every wiki page without one.
    fn wiki_count(&self, expr: Option<&str>) -> usize {
        let sql = match expr {
            // No ?1 here; binding it anyway is fine, SQLite counts up to the largest index.
            None => format!("SELECT d.doc_id FROM docs d WHERE {UNDER}"),
            Some(_) => format!(
                "SELECT d.doc_id FROM fts JOIN docs d ON d.id = fts.rowid \
                 WHERE fts MATCH ?1 AND {UNDER}"
            ),
        };
        let ids = self.store.db.prepare(&sql).and_then(|mut s| {
            s.query_map(params![expr, self.config.pages], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()
        });
        ids.unwrap_or_default()
            .iter()
            .filter(|id| self.config.is_page(id))
            .count()
    }

    /// Rare words (few pages: they name one) and topical ones (up to a fifth of the
    /// pages: too common to name a page, specific enough to be the topic).
    fn rarity(&self, toks: &[String]) -> (Vec<String>, Vec<String>) {
        let n = self.wiki_count(None);
        let limit = 2.max((n as f64 * 0.05).round() as usize);
        let topical_max = limit.max((n as f64 * 0.2) as usize);
        let (mut rare, mut topical) = (Vec::new(), Vec::new());
        for t in toks {
            let df = self.wiki_count(Some(&expr(t)));
            if df > 0 && df <= limit {
                rare.push(t.clone());
            } else if df > limit && df <= topical_max {
                topical.push(t.clone());
            }
        }
        (rare, topical)
    }

    fn candidates(&self, rare: &[String]) -> Result<Vec<(i64, Candidate)>> {
        let q = rare
            .iter()
            .map(|t| format!("({})", expr(t)))
            .collect::<Vec<_>>()
            .join(" OR ");
        let mut stmt = self.store.db.prepare(&format!(
            "SELECT d.id, d.doc_id, fts.title, fts.keys FROM fts JOIN docs d ON d.id = fts.rowid \
             WHERE fts MATCH ?1 AND {UNDER} ORDER BY {} LIMIT 20",
            self.store.layout.bm25()
        ))?;
        let rows: Vec<(i64, Candidate)> = stmt
            .query_map(params![q, self.config.pages], |r| {
                let keys: String = r.get(3)?;
                Ok((
                    r.get(0)?,
                    Candidate {
                        path: r.get(1)?,
                        title: r.get(2)?,
                        line: keys.lines().next().unwrap_or("").to_string(),
                    },
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows
            .into_iter()
            .filter(|(_, c)| self.config.is_page(&c.path))
            .take(5)
            .collect())
    }

    /// A page found by a rare word must also hold half the topical words, or
    /// "supabase oauth" points at a note about some other oauth.
    fn covers(&self, rowid: i64, topical: &[String]) -> bool {
        if topical.is_empty() {
            return true;
        }
        let have = topical
            .iter()
            .filter(|t| {
                self.store
                    .db
                    .query_row(
                        "SELECT count(*) FROM fts WHERE fts MATCH ?1 AND rowid = ?2",
                        params![expr(t), rowid],
                        |r| r.get::<_, i64>(0),
                    )
                    .is_ok_and(|n| n > 0)
            })
            .count();
        have * 2 >= topical.len()
    }
}

/// "Which stack for X?": the stack template sharing the most (and rarest) of the
/// prompt's words; rarity in the wiki does not apply.
fn stack(dir: &Path, toks: &[String]) -> Option<Candidate> {
    let mut files: Vec<(PathBuf, String, HashSet<String>)> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
        .filter_map(|p| {
            let text = std::fs::read_to_string(&p).ok()?;
            let words = tokens(&text.to_lowercase()).into_iter().collect();
            let name = text
                .lines()
                .find_map(|l| l.strip_prefix("name:"))
                .map_or_else(
                    || {
                        p.file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned()
                    },
                    |n| n.trim().to_string(),
                );
            Some((p, name, words))
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let n = files.len() as f64;
    let score = |words: &HashSet<String>| -> f64 {
        toks.iter()
            .filter(|t| words.contains(*t))
            .map(|t| {
                let df = files.iter().filter(|f| f.2.contains(t)).count() as f64;
                ((n + 1.0) / (df + 0.5)).ln()
            })
            .sum()
    };
    let best = files
        .iter()
        .map(|f| (score(&f.2), f))
        .filter(|(s, _)| *s > 0.0)
        .max_by(|a, b| a.0.total_cmp(&b.0).then_with(|| b.1.0.cmp(&a.1.0)))?;
    Some(Candidate {
        path: best.1.0.to_string_lossy().into_owned(),
        title: format!("Stack: {}", best.1.1),
        line: String::new(),
    })
}

fn seen_file(state: &Path, session: &str) -> PathBuf {
    let clean: String = session
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    let name = if clean.is_empty() {
        "none".into()
    } else {
        clean
    };
    state.join(format!("{name}.json"))
}

/// The hint for one prompt, empty when silent. Pages pointed at are remembered
/// under `state` for the session.
pub fn hint(kb: &mut KnowledgeBase, prompt: &str, session: &str, state: &Path) -> Result<String> {
    let silent = String::new();
    let p = prompt.trim();
    if p.starts_with('/') || p.starts_with('<') || p.split_whitespace().count() < MIN_WORDS {
        return Ok(silent);
    }
    let pages = kb.config.pages.clone();
    if !kb.root.join(&pages).is_dir() {
        return Ok(silent);
    }
    let low = p.to_lowercase();
    let extra = kb
        .aliases()
        .map(|a| a.equivalents(&low).join(" "))
        .unwrap_or_default();
    let text = format!("{low} {extra}");
    let toks = tokens(&text);
    // Most Russian prompts have no Latin word and no alias: nothing to look up.
    if toks.is_empty() {
        return Ok(silent);
    }
    kb.sync_under(Some(&pages))?;
    let sf = seen_file(state, session);
    let seen: HashSet<String> = std::fs::read_to_string(&sf)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let abs = |rel: &str| kb.root.join(rel).to_string_lossy().into_owned();
    let mut picked: Vec<Candidate> = Vec::new();
    if STACK_WORDS.iter().any(|w| text.contains(w))
        && let Some(dir) = kb.config.stacks.as_ref().map(|s| kb.root.join(s))
        && let Some(c) = stack(&dir, &toks)
        && !seen.contains(&c.path)
    {
        picked.push(c);
    }
    let (rare, topical) = kb.rarity(&toks);
    if !rare.is_empty() {
        for (rowid, c) in kb.candidates(&rare)? {
            if picked.len() == MAX_PAGES {
                break;
            }
            let path = abs(&c.path);
            let named = format!("{} {} {}", stem(&c.path).replace('-', " "), c.title, c.line)
                .to_lowercase();
            if seen.contains(&path)
                || picked.iter().any(|x| x.path == path)
                || !rare.iter().any(|t| named.contains(t.as_str()))
                || !kb.covers(rowid, &topical)
            {
                continue;
            }
            picked.push(Candidate { path, ..c });
        }
    }
    if picked.is_empty() {
        return Ok(silent);
    }
    let mut all: Vec<String> = seen.into_iter().collect();
    all.extend(picked.iter().map(|c| c.path.clone()));
    all.sort();
    all.dedup();
    std::fs::create_dir_all(state)?;
    std::fs::write(&sf, serde_json::to_string(&all)?)?;
    let mut lines = vec![HEADER.to_string()];
    for c in &picked {
        let about = if c.line.is_empty() { &c.title } else { &c.line };
        lines.push(format!(
            "- {} — {}",
            c.path,
            about.chars().take(160).collect::<String>()
        ));
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests;
