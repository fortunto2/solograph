//! Typos in a query. Each word becomes a stemmed prefix term, so one wrong letter
//! ("солграф" for "солограф", "технопакр") matches nothing and the word silently
//! drops out of the search. A word the index has (almost) never seen is respelt as
//! its commonest neighbour one or two edits away in the index's own vocabulary.
//! Callers search both spellings, so a rare word spelt right loses nothing.

use std::sync::LazyLock;

use anyhow::Result;
use regex::Regex;
use rusqlite::params;

use super::FtsStore;

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+").expect("static regex"));

/// A word found in fewer documents than this is suspect.
const RARE: i64 = 3;
/// Its respelling must be in this many times more documents (counting the word's own + 1).
const COMMONER: i64 = 3;
/// Shorter words have too many neighbours to guess from.
const MIN_LEN: usize = 5;

impl FtsStore {
    /// The query with each suspect word replaced by its respelling, or None when no
    /// word was replaced. `known` are words to respell to although the index may not
    /// hold them (an alias table's spellings).
    pub fn respell(&self, query: &str, known: &[String]) -> Result<Option<String>> {
        self.db.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp.vocab USING fts5vocab(main, fts, 'row');",
        )?;
        let folded = crate::text::fold(&query.to_lowercase());
        let mut words: Vec<String> = Vec::new();
        let mut changed = false;
        for m in WORD.find_iter(&folded) {
            let w = m.as_str();
            match self.neighbour(w, known)? {
                Some(n) => {
                    words.push(n);
                    changed = true;
                }
                None => words.push(w.to_string()),
            }
        }
        Ok(changed.then(|| words.join(" ")))
    }

    fn neighbour(&self, word: &str, known: &[String]) -> Result<Option<String>> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() < MIN_LEN || !chars.iter().all(|c| c.is_alphabetic()) {
            return Ok(None);
        }
        let (key, _) = crate::text::terms_of(word)
            .pop()
            .unwrap_or_else(|| (word.to_string(), String::new()));
        let own = docs_under(self, &key)?;
        if own >= RARE {
            return Ok(None);
        }
        let most = if chars.len() >= 8 { 2 } else { 1 };
        let first = chars[0].to_string();
        let floor = COMMONER * (own + 1);
        let mut st = self.db.prepare_cached(
            "SELECT term, doc FROM temp.vocab WHERE term >= ?1 AND term < ?2 AND doc >= ?3",
        )?;
        let rows = st.query_map(params![first, format!("{first}\u{10FFFF}"), floor], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        let declared = known
            .iter()
            .filter(|k| k.starts_with(&first))
            .map(|k| Ok((k.clone(), floor)));
        let mut best: Option<(usize, i64, String)> = None;
        for row in rows.map(|r| r.map_err(anyhow::Error::from)).chain(declared) {
            let (term, doc) = row?;
            let t: Vec<char> = term.chars().collect();
            if t.len().abs_diff(chars.len()) > most {
                continue;
            }
            let Some(d) = distance(&chars, &t, most) else {
                continue;
            };
            if best
                .as_ref()
                .is_none_or(|(bd, bdoc, _)| (d, -doc) < (*bd, -bdoc))
            {
                best = Some((d, doc, term));
            }
        }
        Ok(best.map(|(_, _, t)| t))
    }
}

/// Documents holding any term that starts with `key` (the largest per-term count,
/// a lower bound that is cheap).
fn docs_under(store: &FtsStore, key: &str) -> Result<i64> {
    let mut st = store.db.prepare_cached(
        "SELECT coalesce(max(doc), 0) FROM temp.vocab WHERE term >= ?1 AND term < ?2",
    )?;
    Ok(st.query_row(params![key, format!("{key}\u{10FFFF}")], |r| r.get(0))?)
}

/// Optimal-string-alignment distance (a swap of two neighbours is one edit), or None
/// once it must exceed `most`.
fn distance(a: &[char], b: &[char], most: usize) -> Option<usize> {
    let mut prev2: Vec<usize> = Vec::new();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                cur[j] = cur[j].min(prev2[j - 2] + 1);
            }
        }
        if cur.iter().min().is_some_and(|&m| m > most) {
            return None;
        }
        prev2 = std::mem::replace(&mut prev, cur);
    }
    Some(prev[b.len()]).filter(|&d| d <= most)
}

#[cfg(test)]
mod tests {
    use super::distance;
    use crate::fts::{Doc, FtsStore};

    fn d(a: &str, b: &str) -> Option<usize> {
        let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
        distance(&a, &b, 2)
    }

    #[test]
    fn a_swap_is_one_edit_and_far_words_are_cut_off() {
        assert_eq!(d("технопакр", "технопарк"), Some(1));
        assert_eq!(d("солграф", "солограф"), Some(1));
        assert_eq!(d("лайффильим", "лайффильм"), Some(1));
        assert_eq!(d("graph", "graph"), Some(0));
        assert_eq!(d("sandbox", "solograph"), None);
    }

    fn store(docs: &[(String, String)]) -> (tempfile::TempDir, FtsStore) {
        let t = tempfile::tempdir().unwrap();
        let mut files = Vec::new();
        for (id, body) in docs {
            let p = t.path().join(id);
            std::fs::write(&p, body).unwrap();
            files.push((id.clone(), p));
        }
        let read = |p: &std::path::Path| -> anyhow::Result<Option<Doc>> {
            let body = std::fs::read_to_string(p)?;
            Ok(Some(Doc {
                body,
                ..Default::default()
            }))
        };
        let mut s = FtsStore::open(&t.path().join("s.db")).unwrap();
        s.sync(&files, &read, false, None).unwrap();
        (t, s)
    }

    #[test]
    fn a_typo_is_respelt_as_the_common_word_and_a_right_rare_word_is_not() {
        let mut docs: Vec<(String, String)> = (0..6)
            .map(|i| (format!("d{i}"), "солограф ищет по сессиям".to_string()))
            .collect();
        docs.push(("x".into(), "солографический редкий термин".into()));
        let (_t, s) = store(&docs);
        assert_eq!(
            s.respell("как солграф ищет", &[]).unwrap().as_deref(),
            Some("как солограф ищет")
        );
        // Common words and short ones stay; nothing replaced is None.
        assert_eq!(s.respell("солограф ищет", &[]).unwrap(), None);
        assert_eq!(s.respell("zzzzzzzz qq", &[]).unwrap(), None);
        // A word the index lacks but an alias table declares.
        assert_eq!(
            s.respell("технопакр", &["технопарк".into()])
                .unwrap()
                .as_deref(),
            Some("технопарк")
        );
    }
}
