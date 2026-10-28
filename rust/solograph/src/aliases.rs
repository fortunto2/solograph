//! Term aliases: other spellings of a term the KB uses, so a query in those
//! spellings also searches the term.
//!
//! Lexical search does not cross languages: a Russian question finds nothing in an
//! English wiki (0 of 22 on the known-answer set, Oct 2026). Embeddings were the
//! old bridge and were removed for cost and drift; the Epiphan knowledge agent
//! closes the same gap with a term-alias table, and so does this.
//!
//! The table lives in the KB, next to the pages it describes — the vocabulary is the
//! KB's, not the tool's. YAML, one group per line; the LAST member is the term the
//! KB writes, the ones before it are spellings a query may use instead:
//!
//! ```yaml
//! - [антихрупкость, antifragile]
//! - [песочница, sandbox]
//! - [инженерия контекста, context engineering]
//! ```
//!
//! A query word matches a member by its stemmed key, so inflected Russian ("песочницу") still matches "песочница".
//! A phrase member matches when every one of its words is in the query, and is
//! searched as all its words together.
//!
//! One way only, measured: expanding in both directions sent English queries to
//! Russian documents through the same groups (top-1 on the English sets fell from
//! 31/40 to 24/40, from 7/10 to 2/10 on real agent queries).

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;

use crate::text::{all_terms, match_expression, terms_of};

/// One member as written, and its words' (key, term) pairs.
#[derive(Debug)]
struct Member {
    text: String,
    words: Vec<(String, String)>,
}

impl Member {
    fn new(text: &str) -> Self {
        Self {
            text: text.to_lowercase(),
            words: terms_of(text),
        }
    }

    /// The FTS5 term: one word, or all of a phrase's words.
    fn term(&self) -> String {
        let all = all_terms(&self.text);
        if self.words.len() == 1 {
            all
        } else {
            format!("({all})")
        }
    }
}

#[derive(Debug)]
struct Group {
    spellings: Vec<Member>,
    term: Member,
}

#[derive(Debug, Default)]
pub struct Aliases {
    groups: Vec<Group>,
}

/// Two stemmed keys name the same word: equal, or one a prefix of the other when
/// both are long enough that the prefix is not a different word.
fn same(a: &str, b: &str) -> bool {
    a == b
        || (a.chars().count() >= 5
            && b.chars().count() >= 5
            && (a.starts_with(b) || b.starts_with(a)))
}

impl Aliases {
    /// The alias file of a KB root, `wiki/_aliases.yaml`, with its mtime, so a
    /// caller can reload on change.
    pub fn locate(root: &Path) -> Option<(PathBuf, SystemTime)> {
        let p = root.join("wiki").join("_aliases.yaml");
        let t = std::fs::metadata(&p).and_then(|m| m.modified()).ok()?;
        Some((p, t))
    }

    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self::parse(&std::fs::read_to_string(path)?))
    }

    pub fn parse(text: &str) -> Self {
        let list: serde_yaml::Value = serde_yaml::from_str(text).unwrap_or_default();
        let groups = list
            .as_sequence()
            .into_iter()
            .flatten()
            .filter_map(serde_yaml::Value::as_sequence)
            .filter_map(|g| {
                let mut members: Vec<Member> = g
                    .iter()
                    .filter_map(|m| m.as_str())
                    .map(Member::new)
                    .filter(|m| !m.words.is_empty())
                    .collect();
                let term = members.pop()?;
                (!members.is_empty()).then_some(Group {
                    spellings: members,
                    term,
                })
            })
            .collect();
        Self { groups }
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// The groups whose other spellings the query uses: a spelling with all its
    /// words in the query.
    fn named<'a>(&'a self, query: &str) -> impl Iterator<Item = &'a Group> {
        let words = terms_of(query);
        self.groups.iter().filter(move |g| {
            g.spellings.iter().any(|m| {
                m.words
                    .iter()
                    .all(|(k, _)| words.iter().any(|(w, _)| same(w, k)))
            })
        })
    }

    /// The KB's own terms for the spellings a query uses, as written (lowercased).
    pub fn equivalents(&self, query: &str) -> Vec<&str> {
        self.named(query).map(|g| g.term.text.as_str()).collect()
    }

    /// The query's own terms, OR the KB's term for each other spelling it uses.
    pub fn expression(&self, query: &str) -> String {
        let base = match_expression(query);
        if base.is_empty() {
            return base;
        }
        let mut extra: Vec<String> = Vec::new();
        for g in self.named(query) {
            let t = g.term.term();
            if !base.split(" OR ").any(|b| b == t) && !extra.contains(&t) {
                extra.push(t);
            }
        }
        if extra.is_empty() {
            base
        } else {
            format!("{base} OR {}", extra.join(" OR "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Aliases;

    const TABLE: &str = "- [антихрупкость, antifragile]\n- [песочница, sandbox]\n- [инженерия контекста, context engineering]\n- [лайа, lyna, laya]\n";

    #[test]
    fn a_russian_word_also_searches_the_kbs_term() {
        let e = Aliases::parse(TABLE).expression("антихрупкость портфель");
        assert!(e.contains("\"antifragil\"*"), "{e}");
        assert!(e.contains("\"портфел\"*"), "{e}");
    }

    #[test]
    fn an_inflected_form_matches_by_stem() {
        let e = Aliases::parse(TABLE).expression("запусти в песочнице");
        assert!(e.contains("\"sandbox\"*"), "{e}");
    }

    #[test]
    fn a_phrase_spelling_needs_all_its_words_and_expands_as_and() {
        let a = Aliases::parse(TABLE);
        assert!(!a.expression("инженерия зданий").contains("context"));
        let e = a.expression("инженерия контекста для агентов");
        assert!(e.contains("(\"context\"* AND \"engin\"*)"), "{e}");
    }

    #[test]
    fn the_kbs_own_term_expands_to_nothing() {
        // Both directions sent English queries to Russian documents.
        let a = Aliases::parse(TABLE);
        for q in ["sandbox escape", "laya on a mac", "tree sitter"] {
            assert_eq!(a.expression(q), crate::text::match_expression(q), "{q}");
        }
    }

    #[test]
    fn every_other_spelling_reaches_the_term() {
        let a = Aliases::parse(TABLE);
        assert_eq!(a.equivalents("что умеет лайа"), ["laya"]);
        assert_eq!(a.equivalents("lyna benchmarks"), ["laya"]);
        assert!(a.equivalents("laya benchmarks").is_empty());
    }

    #[test]
    fn one_member_groups_and_bad_yaml_are_handled() {
        assert_eq!(Aliases::parse("- [a1x, b2y]\n").len(), 1);
        assert!(Aliases::parse("- [alone]\n").is_empty());
        assert!(Aliases::parse(": not yaml [").is_empty());
    }
}
