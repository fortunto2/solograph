//! The memory gate: does a prompt lean on the user's earlier work ("у нас", "мы делали",
//! "как в прошлый раз", a project of theirs named in passing)? When it does, the prompt
//! hook tells the agent to look it up with session_search and kb_search, and leaves the
//! looking to the agent: a raw prompt is a poor query (injecting results was half noise).
//!
//! A logistic regression over the prompt's words and character 3-4-grams. The weights are
//! the KB's, not the tool's: trained on the user's own labeled prompts, they hold fragments
//! of them, so they live in a private repo and `.solograph.yaml` points here
//! (`kb: memory_gate: <file>`). Measured against a three-model neural server it replaced
//! (Oct 2026): AUC 0.908 against 0.889 on a hard set, no server, microseconds a prompt.
//!
//! Features must match the trainer exactly: lowercase, ё -> е, whitespace collapsed, the
//! first 700 characters; "w:<word>" per \w+ word, "c:<gram>" per 3- and 4-character gram
//! of " word "; binary, scaled by 1 / max(1, sqrt(number of known features present)).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Deserialize;

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+").expect("static regex"));
static SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("static regex"));

/// What the hook says when the gate opens: that memory matters and where to look.
pub fn nudge(p: f64) -> String {
    format!(
        "This message seems to lean on the user's earlier work (memory gate p={p:.2}). Before \
         answering from scratch, look it up: mcp__solograph__session_search and \
         mcp__solograph__kb_search with the specific names and terms in the message."
    )
}

#[derive(Deserialize)]
pub struct MemoryGate {
    features: String,
    bias: f64,
    pub threshold: f64,
    weights: HashMap<String, f64>,
}

impl MemoryGate {
    pub fn load(path: &Path) -> Result<Self> {
        let g: Self = serde_json::from_str(
            &std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?,
        )?;
        if g.features != "w+c34/v1" {
            bail!(
                "{}: features {} are not w+c34/v1",
                path.display(),
                g.features
            );
        }
        Ok(g)
    }

    /// Probability that the prompt leans on earlier work.
    pub fn score(&self, prompt: &str) -> f64 {
        let known: Vec<f64> = features(prompt)
            .iter()
            .filter_map(|f| self.weights.get(f).copied())
            .collect();
        let norm = (known.len() as f64).sqrt().max(1.0);
        let z = self.bias + known.iter().sum::<f64>() / norm;
        1.0 / (1.0 + (-z).exp())
    }
}

fn features(text: &str) -> HashSet<String> {
    let low = text.to_lowercase().replace('ё', "е");
    let collapsed = SPACE.replace_all(&low, " ");
    let t: String = collapsed.chars().take(700).collect();
    let mut out = HashSet::new();
    for w in WORD.find_iter(&t) {
        out.insert(format!("w:{}", w.as_str()));
        let padded: Vec<char> = format!(" {} ", w.as_str()).chars().collect();
        for n in [3, 4] {
            for g in padded.windows(n) {
                out.insert(format!("c:{}", g.iter().collect::<String>()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::MemoryGate;

    /// Synthetic weights; the expected scores come from the trainer's own features()
    /// (harness-decisions/scripts/memgate.py), so the two stay in step.
    fn gate() -> MemoryGate {
        serde_json::from_str(
            r#"{"features": "w+c34/v1", "bias": -1.2, "threshold": 0.5, "weights":
               {"w:нас": 2.0, "c: у ": 0.8, "c:али ": 1.1, "w:мы": 0.9, "w:погода": -1.5,
                "c:ёжи": 5.0, "c:ежи": 0.3, "w:a_b2": 0.7}}"#,
        )
        .unwrap()
    }

    #[test]
    fn scores_match_the_trainer() {
        let g = gate();
        let long = format!("{} у нас", "x".repeat(900));
        for (text, want) in [
            ("у нас было такое, мы делали", 0.768525),
            ("Погода   завтра", 0.062973),
            ("ЁЖИК и ежи", 0.28905),
            ("a_b2 test", 0.377541),
            ("", 0.231475),
            (long.as_str(), 0.231475),
        ] {
            let got = g.score(text);
            assert!((got - want).abs() < 1e-5, "{text:?}: {got} vs {want}");
        }
    }

    #[test]
    fn a_file_with_other_features_is_refused() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("g.json");
        std::fs::write(
            &p,
            r#"{"features": "w/v9", "bias": 0, "threshold": 0.5, "weights": {}}"#,
        )
        .unwrap();
        assert!(MemoryGate::load(&p).is_err());
    }
}
