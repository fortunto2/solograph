//! `solograph hook`: one Claude Code hook for SessionStart and UserPromptSubmit that hands a
//! session what the portfolio already knows, instead of a rule asking it to look.
//!
//! Two things, both silent unless they have something:
//! - the wiki hint (`kb-hint`), on every prompt, when a KB is configured;
//! - a map of the project the session works in (stack, size, languages, layers, main
//!   dependencies, and which tools answer what), once per session per project: at
//!   SessionStart, or on the first prompt that starts or plans a piece of work.
//!
//! The project is the registered one whose path contains the hook's `cwd`, so a
//! profile picks its own portfolio store the same way its MCP entry does (env vars or
//! `--db-path`). Any failure is silence with exit 0: a broken hint must never break a
//! prompt.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{Value, json};

use crate::portfolio::Portfolio;

/// Word starts that mark a prompt as starting or planning work. Prefixes, so "plan"
/// covers planning and "реализ" covers реализуй / реализовать.
const START_WORDS: &[&str] = &[
    "plan",
    "implement",
    "build",
    "refactor",
    "design",
    "feature",
    "scaffold",
    "migrate",
    "rewrite",
    "fix",
    "debug",
    "план",
    "спланир",
    "сделай",
    "сделать",
    "реализ",
    "задач",
    "почин",
    "добав",
    "рефактор",
    "спроектир",
    "начн",
    "начина",
    "фич",
    "перепиш",
    "мигрир",
    "разбер",
];

pub fn starts_work(prompt: &str) -> bool {
    prompt
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| !w.is_empty() && START_WORDS.iter().any(|s| w.starts_with(s)))
}

/// The registered project whose path holds `cwd`, the deepest one when they nest.
pub fn project_for(p: &Portfolio, cwd: &Path) -> Result<Option<String>> {
    let mut stmt = p.db.prepare("SELECT name, path FROM projects")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    let mut best: Option<(usize, String)> = None;
    for row in rows {
        let (name, path) = row?;
        if path.is_empty() || !cwd.starts_with(&path) {
            continue;
        }
        if best.as_ref().is_none_or(|(len, _)| path.len() > *len) {
            best = Some((path.len(), name));
        }
    }
    Ok(best.map(|(_, n)| n))
}

fn cell(r: &Value, i: usize) -> String {
    r[i].as_str().map_or_else(|| r[i].to_string(), String::from)
}

/// The project map, compact: what `explain` says, cut to what changes a plan.
pub fn map_text(v: &Value) -> String {
    let rows = |k: &str, n: usize| -> Vec<Value> {
        v[k].as_array()
            .map(|a| a.iter().take(n).cloned().collect())
            .unwrap_or_default()
    };
    let c = &v["counts"];
    let mut out = vec![format!(
        "Project map (solograph) — {}: {}. {} files, {} symbols, {} packages.",
        v["name"].as_str().unwrap_or(""),
        v["stack"].as_str().unwrap_or("stack unknown"),
        c["files"],
        c["symbols"],
        c["packages"]
    )];
    let langs: Vec<String> = rows("languages", 4)
        .iter()
        .map(|r| format!("{} {}", cell(r, 0), cell(r, 1)))
        .collect();
    if !langs.is_empty() {
        out.push(format!("Languages (files): {}.", langs.join(", ")));
    }
    let layers: Vec<String> = rows("layers", 6)
        .iter()
        .map(|r| format!("{} ({} files)", cell(r, 0), cell(r, 1)))
        .collect();
    if !layers.is_empty() {
        out.push(format!("Layers: {}.", layers.join(", ")));
    }
    let deps: Vec<String> = rows("dependencies", 10)
        .iter()
        .map(|r| cell(r, 0))
        .collect();
    if !deps.is_empty() {
        // `explain` lists packages by source then name, so these are the first ten
        // alphabetically, not the most important ones; say so.
        out.push(format!(
            "Dependencies (first {} of {}, by name): {}.",
            deps.len(),
            c["packages"],
            deps.join(", ")
        ));
    }
    out.push(
        "Before writing new code: symbol_search finds where a name is already defined \
         across every project, session_search finds the past session that hit the same \
         problem, kb_search the notes. Inside this repo, Grep is as good."
            .into(),
    );
    out.join("\n")
}

fn seen_file(state: &Path, session: &str) -> PathBuf {
    let safe: String = session
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    state.join(format!("map-{safe}"))
}

/// True the first time `project` is asked for in `session`; records it.
pub fn first_time(state: &Path, session: &str, project: &str) -> bool {
    if session.is_empty() {
        return true;
    }
    let f = seen_file(state, session);
    let seen = std::fs::read_to_string(&f).unwrap_or_default();
    if seen.lines().any(|l| l == project) {
        return false;
    }
    let _ = std::fs::create_dir_all(state);
    let _ = std::fs::write(&f, format!("{seen}{project}\n"));
    true
}

/// The map for this hook call, if one is due: a known project, the right moment, and
/// not shown yet in this session.
pub fn map_for(
    db: Option<&Path>,
    cwd: &Path,
    event: &str,
    prompt: &str,
    session: &str,
    state: &Path,
) -> Result<Option<String>> {
    if event != "SessionStart" && !starts_work(prompt) {
        return Ok(None);
    }
    let p = Portfolio::open(db)?;
    let Some(name) = project_for(&p, cwd)? else {
        return Ok(None);
    };
    let Ok(v) = p.explain(&name)? else {
        return Ok(None);
    };
    if !first_time(state, session, &name) {
        return Ok(None);
    }
    Ok(Some(map_text(&v)))
}

pub fn output(event: &str, parts: &[String]) -> Option<Value> {
    let text = parts
        .iter()
        .filter(|p| !p.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n");
    (!text.is_empty())
        .then(|| json!({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_words_are_word_starts_not_substrings() {
        assert!(starts_work("давай спланируем миграцию"));
        assert!(starts_work("Implement the booking link"));
        assert!(starts_work("почини синк"));
        assert!(!starts_work("what is the prefix here?"));
        assert!(!starts_work("тихо, 20:04 UTC"));
    }

    #[test]
    fn a_map_is_shown_once_per_session_and_project() {
        let t = tempfile::tempdir().unwrap();
        assert!(first_time(t.path(), "s1", "alpha"));
        assert!(!first_time(t.path(), "s1", "alpha"));
        assert!(first_time(t.path(), "s1", "beta"));
        assert!(first_time(t.path(), "s2", "alpha"));
    }

    #[test]
    fn the_map_keeps_what_changes_a_plan() {
        let v = json!({"name": "app", "stack": "nextjs", "counts": {"files": 3, "symbols": 9, "packages": 2},
            "languages": [["typescript", 3, 120]], "layers": [["src/app/", 2, 7]],
            "dependencies": [["next", "npm"], ["zod", "npm"]]});
        let m = map_text(&v);
        assert!(
            m.starts_with("Project map (solograph) — app: nextjs. 3 files, 9 symbols, 2 packages.")
        );
        assert!(m.contains("Languages (files): typescript 3."));
        assert!(m.contains("Layers: src/app/ (2 files)."));
        assert!(m.contains("Dependencies (first 2 of 2, by name): next, zod."));
    }

    #[test]
    fn nothing_to_say_is_no_output() {
        assert!(output("UserPromptSubmit", &[String::new()]).is_none());
        let o = output("SessionStart", &["map".into()]).unwrap();
        assert_eq!(o["hookSpecificOutput"]["hookEventName"], "SessionStart");
    }
}
