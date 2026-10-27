//! Text primitives shared by every index: ё folding, the FTS5 query built from
//! stemmed prefix terms, and identifier splitting. Port of fts.py and
//! portfolio.split_identifier; the same input must give the same query string,
//! because Python and Rust read the same .db files.

use std::sync::LazyLock;

use regex::Regex;

use crate::stem;

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+").expect("static regex"));

/// unicode61 lowercases but does not fold ё into е, so stored text and queries are folded here.
pub fn fold(text: &str) -> String {
    text.replace('ё', "е").replace('Ё', "Е")
}

fn is_cyrillic(token: &str) -> bool {
    token.chars().any(|c| ('а'..='я').contains(&c))
}

/// Free text → an FTS5 OR-query of stemmed prefix terms (fts.match_expression).
///
/// OR, not AND: BM25 already ranks documents that match more and rarer terms
/// higher. Short tokens and tokens with digits stay whole.
pub fn match_expression(query: &str) -> String {
    let folded = fold(&query.to_lowercase());
    let mut terms: Vec<String> = Vec::new();
    for m in WORD.find_iter(&folded) {
        let token = m.as_str();
        let len = token.chars().count();
        let term = if len < 3 || token.chars().any(char::is_numeric) {
            format!("\"{token}\"")
        } else {
            let stemmed = if is_cyrillic(token) {
                stem::russian(token)
            } else {
                stem::english(token)
            };
            // A Snowball stem is not always a prefix ("vocabulary" -> "vocabulari"),
            // and a prefix query needs one.
            let prefix: String = stemmed
                .chars()
                .zip(token.chars())
                .take_while(|(a, b)| a == b)
                .map(|(a, _)| a)
                .collect();
            if prefix.chars().count() >= 3 {
                format!("\"{prefix}\"*")
            } else {
                format!("\"{token}\"")
            }
        };
        if !terms.contains(&term) {
            terms.push(term);
        }
    }
    terms.join(" OR ")
}

/// `resolve_montage_profile`, `MontageProfile`, `HTTPServer` → their lowercase words.
///
/// Same tokens as the Python regex `[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+`,
/// written out because the regex crate has no lookahead.
pub fn split_identifier(name: &str) -> String {
    let c: Vec<char> = name.chars().collect();
    let (up, low) = (
        |ch: char| ch.is_ascii_uppercase(),
        |ch: char| ch.is_ascii_lowercase(),
    );
    let mut words = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let run = c[i..].iter().take_while(|&&ch| up(ch)).count();
        // 1. An uppercase run followed by Upper+lower: the longest such prefix.
        let acronym = (1..=run)
            .rev()
            .find(|&k| i + k + 1 < c.len() && up(c[i + k]) && low(c[i + k + 1]));
        if let Some(k) = acronym {
            words.push(c[i..i + k].iter().collect::<String>());
            i += k;
            continue;
        }
        // 2. An optional capital then lowercase letters.
        let start_low = if up(c[i]) { i + 1 } else { i };
        let lows = c
            .get(start_low..)
            .map_or(0, |s| s.iter().take_while(|&&ch| low(ch)).count());
        if lows > 0 {
            words.push(c[i..start_low + lows].iter().collect::<String>());
            i = start_low + lows;
            continue;
        }
        // 3. A plain uppercase run, 4. digits.
        if run > 0 {
            words.push(c[i..i + run].iter().collect::<String>());
            i += run;
            continue;
        }
        let digits = c[i..].iter().take_while(|ch| ch.is_numeric()).count();
        if digits > 0 {
            words.push(c[i..i + digits].iter().collect::<String>());
            i += digits;
            continue;
        }
        i += 1;
    }
    words
        .iter()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

static TERM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""([^"]+)"(\*)?"#).expect("static regex"));

/// A snippet without FTS5: the first word matching a term of `expr` (a
/// match_expression() string), with ~8 words before it and ~16 after, matched words
/// in «», cut ends marked …. FTS5's snippet() tokenises the whole body to choose its
/// fragment — 900 ms on the two 3–4 MB transcripts a three-word query returned —
/// while this stops at the first hit. Words are lowercased one at a time, so byte
/// offsets into `body` stay valid whatever it contains. None when nothing matches.
pub fn window_snippet(body: &str, expr: &str) -> Option<String> {
    let terms: Vec<(String, bool)> = TERM
        .captures_iter(expr)
        .map(|c| (c[1].to_string(), c.get(2).is_some()))
        .collect();
    let hit = |w: &str| {
        let w = fold(&w.to_lowercase());
        terms.iter().any(|(t, prefix)| {
            if *prefix {
                w.starts_with(t.as_str())
            } else {
                &w == t
            }
        })
    };
    let first = WORD.find_iter(body).find(|m| hit(m.as_str()))?;
    // Words in a byte window around the hit, then 8 before and 16 after it.
    let floor = |mut i: usize| {
        while !body.is_char_boundary(i) {
            i -= 1;
        }
        i
    };
    let from = floor(first.start().saturating_sub(800));
    let to = floor((first.end() + 1600).min(body.len()));
    let words: Vec<regex::Match> = WORD.find_iter(&body[from..to]).collect();
    let at = words
        .iter()
        .position(|m| m.start() + from == first.start())?;
    let (lo, hi) = (at.saturating_sub(8), (at + 16).min(words.len() - 1));
    let (start, end) = (from + words[lo].start(), from + words[hi].end());
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    let mut cursor = start;
    for m in &words[lo..=hi] {
        let (ws, we) = (from + m.start(), from + m.end());
        out.push_str(&body[cursor..ws]);
        if hit(m.as_str()) {
            out.push('«');
            out.push_str(&body[ws..we]);
            out.push('»');
        } else {
            out.push_str(&body[ws..we]);
        }
        cursor = we;
    }
    if end < body.len() {
        out.push('…');
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn russian_inflections_meet_at_the_stem() {
        assert_eq!(match_expression("графами"), "\"граф\"*");
        assert_eq!(match_expression("сессий"), "\"сесс\"*");
    }

    #[test]
    fn english_stems_and_short_words_are_kept_whole() {
        assert_eq!(
            match_expression("indexing graphs"),
            "\"index\"* OR \"graph\"*"
        );
        assert_eq!(
            match_expression("O-1A visa"),
            "\"o\" OR \"1a\" OR \"visa\"*"
        );
    }

    #[test]
    fn a_stem_that_is_not_a_prefix_is_cut_back_to_one() {
        assert_eq!(match_expression("vocabulary"), "\"vocabular\"*");
        assert_eq!(match_expression("studies"), "\"studi\"*");
    }

    #[test]
    fn yo_is_folded_because_unicode61_does_not() {
        assert_eq!(match_expression("ёлка"), match_expression("елка"));
    }

    #[test]
    fn query_with_only_punctuation_matches_nothing() {
        assert_eq!(match_expression("*** ()"), "");
    }

    #[test]
    fn identifiers_split_into_their_words() {
        assert_eq!(
            split_identifier("resolve_montage_profile"),
            "resolve montage profile"
        );
        assert_eq!(split_identifier("MontageProfile"), "montage profile");
        assert_eq!(split_identifier("HTTPServer"), "http server");
        assert_eq!(split_identifier("parseJSON2"), "parse json 2");
    }

    #[test]
    fn a_window_snippet_marks_the_first_hit_and_cuts_both_ends() {
        let body = format!(
            "{} The billing webhook failed twice. {}",
            "word ".repeat(30),
            "tail ".repeat(30)
        );
        let expr = match_expression("webhooks billing");
        let snip = window_snippet(&body, &expr).unwrap();
        assert!(snip.starts_with('…') && snip.ends_with('…'), "{snip}");
        assert!(snip.contains("The «billing» «webhook» failed"), "{snip}");
        assert_eq!(window_snippet("nothing relevant here", &expr), None);
    }

    #[test]
    fn a_window_snippet_handles_cyrillic() {
        let body = "Сначала разобрали логи. Потом переписали вебхуки для сделок HubSpot.";
        let snip = window_snippet(body, &match_expression("вебхук")).unwrap();
        assert!(snip.contains("«вебхуки»"), "{snip}");
    }

    #[test]
    fn a_window_snippet_survives_text_whose_lowercase_changes_length() {
        // "İ" lowercases to two code points; a whole-body lowercase shifted offsets.
        let body = format!("İstanbul notes. {} then the webhook fired.", "x ".repeat(5));
        let snip = window_snippet(&body, &match_expression("webhook")).unwrap();
        assert!(snip.contains("«webhook» fired"), "{snip}");
    }
}
