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

use std::collections::{HashMap, HashSet};
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
const HEADER: &str =
    "The wiki has measured notes on this — read before answering or searching the web:";

static TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z][a-z0-9+#.\-]{2,}").expect("static regex"));
/// "stack" and "стек" as words, in any Russian case; not "стекло" or "истекает".
static STACK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bstacks?\b|\bстек(?:а|е|у|ом|и|ов|ам|ами|ах)?\b").expect("static regex")
});
/// Two-letter words that say nothing about a stack.
const SHORT_STOP: &[&str] = &[
    "an", "in", "on", "to", "of", "is", "it", "be", "or", "as", "at", "by", "we", "me", "my", "do",
    "if", "so", "up", "no", "us",
];
/// Words for matching stack templates: two letters count here ("ml", "ai", "go").
static STACK_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z][a-z0-9+#.\-]+").expect("static regex"));
/// The same word meaning something else: a trace, a frame, an overflow.
static NOT_STACK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bcall\s+stack|\bstack\s*(?:trace|frame|overflow)").expect("static regex")
});

/// Whether a prompt asks which stack to use (review, 28 Oct 2026: a substring
/// check sent "stack trace" and "токен истекает" to a stack template).
fn asks_for_a_stack(text: &str) -> bool {
    STACK.is_match(text) && !NOT_STACK.is_match(text)
}

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

    /// Whether an FTS expression matches more than a twentieth of the whole KB,
    /// as rarity is judged for wiki words below: a word, not a name.
    fn common_in_kb(&self, expr: &str) -> bool {
        let count = |sql: &str, p: Option<&str>| -> i64 {
            match p {
                Some(e) => self.store.db.query_row(sql, [e], |r| r.get(0)),
                None => self.store.db.query_row(sql, [], |r| r.get(0)),
            }
            .unwrap_or(0)
        };
        let total = count("SELECT count(*) FROM docs WHERE gone = 0", None);
        let df = count(
            "SELECT count(*) FROM fts JOIN docs d ON d.id = fts.rowid \
             WHERE fts MATCH ?1 AND d.gone = 0",
            Some(expr),
        );
        df as f64 > (total as f64 * 0.05).max(2.0)
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

    /// Pages whose file name the prompt spells out: every word of the slug of three
    /// letters or more ("project" and years aside), at least two of them. "kill
    /// iterate scale" names kill-iterate-scale.md though no word of it is rare.
    fn slug_named(&self, toks: &[String]) -> Result<Vec<Candidate>> {
        let mut stmt = self.store.db.prepare(&format!(
            "SELECT d.doc_id, fts.title, fts.keys FROM fts JOIN docs d ON d.id = fts.rowid \
             WHERE {UNDER}"
        ))?;
        let rows: Vec<(String, String, String)> = stmt
            .query_map(params![Option::<String>::None, self.config.pages], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows
            .into_iter()
            .filter(|(id, _, _)| self.config.is_page(id))
            .filter(|(id, _, _)| {
                let words: Vec<&str> = stem(id)
                    .split('-')
                    .filter(|w| {
                        w.len() >= 3 && *w != "project" && !w.chars().all(|c| c.is_ascii_digit())
                    })
                    .collect();
                words.len() >= 2 && words.iter().all(|w| toks.iter().any(|t| t == w))
            })
            .map(|(path, title, keys)| Candidate {
                path,
                title,
                line: keys.lines().next().unwrap_or("").to_string(),
            })
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

/// "Which stack for X?": the template BM25 ranks first for the prompt's words,
/// over two fields scored separately, its name (with the file name) and its text,
/// as the Python hint's FTS5 columns were. The port's unnormalised idf sum always
/// chose the largest template (review, 28 Oct 2026).
fn stack(dir: &Path, prompt: &str) -> Option<Candidate> {
    let mut toks: Vec<String> = Vec::new();
    for m in STACK_WORD.find_iter(&prompt.to_lowercase()) {
        let t = m
            .as_str()
            .trim_matches(|c| c == '.' || c == '-')
            .to_string();
        if !STOP.contains(&t.as_str()) && !SHORT_STOP.contains(&t.as_str()) && !toks.contains(&t) {
            toks.push(t);
        }
    }
    #[derive(Default)]
    struct Field {
        tf: HashMap<String, f64>,
        len: f64,
    }
    impl Field {
        fn of(text: &str) -> Self {
            let low = text.to_lowercase();
            let mut f = Self::default();
            for m in STACK_WORD.find_iter(&low) {
                let t = m.as_str().trim_matches(|c| c == '.' || c == '-');
                *f.tf.entry(t.to_string()).or_default() += 1.0;
            }
            // Length in every word, short ones too: counting only the matchable
            // ones made "Python ML/AI" look shorter than "Python API" and win.
            f.len = low
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .count() as f64;
            f
        }
    }
    struct Template {
        path: PathBuf,
        name: String,
        fields: [Field; 2],
    }
    let mut docs: Vec<Template> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
        .filter_map(|p| {
            let text = std::fs::read_to_string(&p).ok()?;
            let stem = p
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let name = text
                .lines()
                .find_map(|l| l.strip_prefix("name:"))
                .map_or_else(|| stem.clone(), |n| n.trim().to_string());
            let title = Field::of(&format!("{name} {}", stem.replace('-', " ")));
            Some(Template {
                path: p,
                name,
                fields: [title, Field::of(&text)],
            })
        })
        .collect();
    docs.sort_by(|a, b| a.path.cmp(&b.path));
    let n = docs.len() as f64;
    let avg: Vec<f64> = (0..2)
        .map(|i| docs.iter().map(|d| d.fields[i].len).sum::<f64>() / n.max(1.0))
        .collect();
    let (k1, b) = (1.2, 0.75);
    let score = |d: &Template| -> f64 {
        toks.iter()
            .map(|t| {
                let df = docs
                    .iter()
                    .filter(|x| x.fields.iter().any(|f| f.tf.contains_key(t)))
                    .count() as f64;
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let per_field: f64 = d
                    .fields
                    .iter()
                    .zip(&avg)
                    .filter_map(|(f, avg)| {
                        let tf = *f.tf.get(t)?;
                        Some(tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * f.len / avg)))
                    })
                    .sum();
                idf * per_field
            })
            .sum()
    };
    let best = docs
        .iter()
        .map(|d| (score(d), d))
        .filter(|(s, _)| *s > 0.0)
        .max_by(|a, b| a.0.total_cmp(&b.0).then_with(|| b.1.path.cmp(&a.1.path)))?;
    Some(Candidate {
        path: best.1.path.to_string_lossy().into_owned(),
        title: format!("Stack: {}", best.1.name),
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

/// What the hint said, and the pages it pointed at (for the usage log).
#[derive(Debug, Default)]
pub struct Hint {
    pub text: String,
    pub pages: Vec<String>,
}

/// The hint for one prompt, empty when silent. Pages pointed at are remembered
/// under `state` for the session.
pub fn hint(kb: &mut KnowledgeBase, prompt: &str, session: &str, state: &Path) -> Result<Hint> {
    let silent = Hint::default();
    let p = prompt.trim();
    if p.starts_with('/') || p.starts_with('<') || p.split_whitespace().count() < MIN_WORDS {
        return Ok(silent);
    }
    let pages = kb.config.pages.clone();
    if !kb.root.join(&pages).is_dir() {
        return Ok(silent);
    }
    let low = p.to_lowercase();
    // An alias only names a page when its spelling is rare across the whole KB:
    // "лайа" is a name, "виза" and "турция" are words that, through visa and
    // turkey, pointed at a visa-photo app and the company page in talk about a
    // visa case and contact lenses (replay of 300 real prompts, 28 Oct 2026).
    let used = kb.aliases().map(|a| a.used(&low)).unwrap_or_default();
    let extra: Vec<String> = used
        .into_iter()
        .filter(|(spelling, _)| !kb.common_in_kb(spelling))
        .map(|(_, term)| term)
        .collect();
    let extra = extra.join(" ");
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
    if asks_for_a_stack(&text)
        && let Some(dir) = kb.config.stacks.as_ref().map(|s| kb.root.join(s))
        && let Some(c) = stack(&dir, &text)
        && !seen.contains(&c.path)
    {
        picked.push(c);
    }
    for c in kb.slug_named(&toks)? {
        let path = abs(&c.path);
        if picked.len() < MAX_PAGES
            && !seen.contains(&path)
            && !picked.iter().any(|x| x.path == path)
        {
            picked.push(Candidate { path, ..c });
        }
    }
    let (rare, topical) = kb.rarity(&toks);
    if !rare.is_empty() {
        for (rowid, c) in kb.candidates(&rare)? {
            if picked.len() == MAX_PAGES {
                break;
            }
            let path = abs(&c.path);
            // The file name and title name a page; its summary line only describes
            // it. A word found in the summary alone needs the prompt's topic behind
            // it: "Irvine" in an address pointed at a hackathon held there (replay
            // of 300 real prompts, 28 Oct 2026).
            let strong = format!("{} {}", stem(&c.path).replace('-', " "), c.title);
            let named = if topical.is_empty() {
                strong.to_lowercase()
            } else {
                format!("{strong} {}", c.line).to_lowercase()
            };
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
    Ok(Hint {
        text: lines.join("\n"),
        pages: picked.into_iter().map(|c| c.path).collect(),
    })
}

#[cfg(test)]
mod tests;
