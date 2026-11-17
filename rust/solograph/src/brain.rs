//! The brain: one entry an agent reads first, and one append-only way to write back.
//! Both are the KB's choice, not the engine's: a top-level `brain:` section of
//! `.solograph.yaml`. Top level on purpose: `kb:` refuses unknown keys, and a binary
//! from before the brain must keep reading a file that has one.
//!
//! ```yaml
//! brain:
//!   orient: harness/brain/ORIENT.md   # what brain_start returns
//!   capture: 3-inbox/brain            # where capture writes, one new file per note
//! ```
//!
//! Without the section neither tool is offered. Why only these two: Rinat Abdullin's
//! BitGN Brain ended at two tools (start, run) after a dozen; here `run` is the
//! agent's own shell, so what is left is the entry and the write path
//! (wiki/company-brain.md in the author's KB).

use std::io::Write;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// The map an agent reads first, relative to the KB root.
    pub orient: Option<String>,
    /// The folder `capture` writes into, relative to the KB root.
    pub capture: Option<String>,
}

#[derive(Deserialize, Default)]
struct File {
    #[serde(default)]
    brain: Option<Config>,
}

/// A note longer than this is a document: write it as a file and capture where it is.
pub const MAX_TEXT: usize = 20_000;
pub const MAX_TITLE: usize = 200;
/// What a capture is; the nightly loop and the inbox sort by it.
pub const KINDS: [&str; 5] = ["note", "decision", "finding", "handoff", "idea"];

impl Config {
    /// The KB's brain section; None when the file or the section is absent. A section
    /// that does not parse is an error: a typo must not silently hide both tools.
    pub fn load(root: &Path) -> Result<Option<Self>> {
        let path = root.join(crate::kb::config::FILE);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(None);
        };
        let file: Option<File> =
            serde_yaml::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
        Ok(file.and_then(|f| f.brain))
    }
}

/// The KB root and its brain section, when both exist.
pub fn detect() -> Option<(PathBuf, Config)> {
    let root = crate::kb::detect_root()?;
    let cfg = Config::load(&root).ok()??;
    Some((root, cfg))
}

/// A configured path: relative, and inside the root once joined.
fn inside(root: &Path, rel: &str) -> Result<PathBuf> {
    let p = Path::new(rel.trim());
    let plain = p
        .components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    if p.as_os_str().is_empty() || p.is_absolute() || !plain {
        bail!("a brain path must be relative and stay inside the KB: {rel:?}");
    }
    Ok(root.join(p))
}

/// The orient file's text, and a small header: where it is, when it changed, where
/// captures go.
pub fn start(root: &Path, cfg: &Config) -> Result<(String, Value)> {
    let Some(rel) = cfg.orient.as_deref() else {
        bail!("no `brain: orient:` in {}", crate::kb::config::FILE);
    };
    let path = inside(root, rel)?;
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let updated = std::fs::metadata(&path)
        .and_then(|m| m.modified())
        .ok()
        .map(|t| {
            chrono::DateTime::<chrono::Local>::from(t)
                .format("%Y-%m-%d %H:%M")
                .to_string()
        });
    let header = json!({"orient": rel, "updated": updated, "capture": cfg.capture});
    Ok((text, header))
}

pub struct Note<'a> {
    pub title: &'a str,
    pub text: &'a str,
    pub tags: &'a [String],
    pub kind: &'a str,
    pub client: Option<&'a str>,
}

#[derive(Serialize)]
struct Front<'a> {
    #[serde(rename = "type")]
    doc_type: &'a str,
    status: &'a str,
    title: &'a str,
    created: String,
    tags: Vec<String>,
    kind: &'a str,
    source: &'a str,
    client: &'a str,
}

/// File name from a title: letters and digits in any script, the rest one dash,
/// at most 60 characters.
fn slug(title: &str) -> String {
    let mut s = String::new();
    for c in title.chars().flat_map(char::to_lowercase) {
        if s.chars().count() >= 60 {
            break;
        }
        if c.is_alphanumeric() {
            s.push(c);
        } else if !s.is_empty() && !s.ends_with('-') {
            s.push('-');
        }
    }
    let s = s.trim_end_matches('-');
    if s.is_empty() {
        "note".into()
    } else {
        s.into()
    }
}

/// Writes one new file and never touches an existing one: a second capture with the
/// same title the same day gets `-2`. Returns the path relative to the root.
pub fn capture(
    root: &Path,
    cfg: &Config,
    note: &Note,
    now: chrono::DateTime<chrono::Local>,
) -> Result<Value> {
    let Some(rel) = cfg.capture.as_deref() else {
        bail!("no `brain: capture:` in {}", crate::kb::config::FILE);
    };
    let title = note.title.trim();
    if title.is_empty() {
        bail!("a capture needs a title: the words someone would search for it by");
    }
    if title.chars().count() > MAX_TITLE {
        bail!("title over {MAX_TITLE} characters");
    }
    let text = note.text.trim();
    if text.is_empty() {
        bail!("a capture needs text");
    }
    if text.chars().count() > MAX_TEXT {
        bail!(
            "text over {MAX_TEXT} characters: write the long thing to a file and capture where it is"
        );
    }
    let kind = match note.kind.trim() {
        "" => "note",
        k if KINDS.contains(&k) => k,
        k => bail!("kind must be one of {KINDS:?}, not {k:?}"),
    };
    let mut tags = vec!["brain".to_string()];
    for t in note.tags {
        let t = t.trim().to_lowercase();
        if !t.is_empty() && !tags.contains(&t) {
            tags.push(t);
        }
    }
    let front = Front {
        doc_type: "capture",
        status: "draft",
        title,
        created: now.format("%Y-%m-%d").to_string(),
        tags,
        kind,
        source: "brain-mcp",
        client: note.client.unwrap_or("unknown"),
    };
    let body = format!("---\n{}---\n\n{text}\n", serde_yaml::to_string(&front)?);
    let dir = inside(root, rel)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let stem = format!("{}-{}", front.created, slug(title));
    for n in 1..100 {
        let name = if n == 1 {
            format!("{stem}.md")
        } else {
            format!("{stem}-{n}.md")
        };
        let path = dir.join(&name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                f.write_all(body.as_bytes())
                    .with_context(|| format!("writing {}", path.display()))?;
                let shown = Path::new(rel.trim().trim_end_matches('/')).join(&name);
                return Ok(json!({
                    "path": shown.to_string_lossy(),
                    "kind": kind,
                    "chars": text.chars().count(),
                }));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).with_context(|| format!("creating {}", path.display())),
        }
    }
    bail!("99 captures named {stem} today already")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kb(yaml: &str) -> tempfile::TempDir {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(t.path().join(crate::kb::config::FILE), yaml).unwrap();
        t
    }

    fn at() -> chrono::DateTime<chrono::Local> {
        use chrono::TimeZone;
        chrono::Local
            .with_ymd_and_hms(2026, 11, 17, 9, 30, 0)
            .unwrap()
    }

    fn note<'a>(title: &'a str, text: &'a str, tags: &'a [String]) -> Note<'a> {
        Note {
            title,
            text,
            tags,
            kind: "",
            client: Some("claude-code"),
        }
    }

    const BRAIN: &str = "kb:\n  pages: wiki/\nbrain:\n  orient: harness/brain/ORIENT.md\n  capture: 3-inbox/brain\n";

    #[test]
    fn no_section_means_no_brain_and_a_typo_is_an_error() {
        let t = tempfile::tempdir().unwrap();
        assert!(Config::load(t.path()).unwrap().is_none());
        assert!(
            Config::load(kb("kb:\n  pages: wiki/\n").path())
                .unwrap()
                .is_none()
        );
        assert!(Config::load(kb("brain:\n  orientt: x.md\n").path()).is_err());
        let c = Config::load(kb(BRAIN).path()).unwrap().unwrap();
        assert_eq!(c.capture.as_deref(), Some("3-inbox/brain"));
    }

    #[test]
    fn the_kb_section_still_parses_next_to_a_brain_section() {
        // `kb:` denies unknown fields; the brain must not trip it.
        let t = kb(BRAIN);
        assert_eq!(crate::kb::Config::load(t.path()).unwrap().pages, "wiki/");
    }

    #[test]
    fn start_returns_the_orient_text_and_where_captures_go() {
        let t = kb(BRAIN);
        let cfg = Config::load(t.path()).unwrap().unwrap();
        std::fs::create_dir_all(t.path().join("harness/brain")).unwrap();
        std::fs::write(
            t.path().join("harness/brain/ORIENT.md"),
            "# Map\nwiki/ is the wiki\n",
        )
        .unwrap();
        let (text, header) = start(t.path(), &cfg).unwrap();
        assert!(text.starts_with("# Map"));
        assert_eq!(header["capture"], "3-inbox/brain");
        assert!(header["updated"].is_string());
    }

    #[test]
    fn paths_outside_the_kb_are_refused() {
        let t = tempfile::tempdir().unwrap();
        for bad in ["../x.md", "/etc/passwd", "a/../../x", ""] {
            let cfg = Config {
                orient: Some(bad.into()),
                capture: Some(bad.into()),
            };
            assert!(start(t.path(), &cfg).is_err(), "{bad}");
            assert!(
                capture(t.path(), &cfg, &note("t", "x", &[]), at()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn capture_writes_a_new_file_with_frontmatter_and_never_overwrites() {
        let t = kb(BRAIN);
        let cfg = Config::load(t.path()).unwrap().unwrap();
        let tags = vec!["ECOM2".to_string(), "brain".to_string(), " ".to_string()];
        let first = capture(
            t.path(),
            &cfg,
            &note("Решение: ECOM2 / рой", "why", &tags),
            at(),
        )
        .unwrap();
        assert_eq!(
            first["path"],
            "3-inbox/brain/2026-11-17-решение-ecom2-рой.md"
        );
        let second = capture(
            t.path(),
            &cfg,
            &note("Решение: ECOM2 / рой", "again", &[]),
            at(),
        )
        .unwrap();
        assert_eq!(
            second["path"],
            "3-inbox/brain/2026-11-17-решение-ecom2-рой-2.md"
        );
        let body = std::fs::read_to_string(t.path().join(first["path"].as_str().unwrap())).unwrap();
        assert!(body.starts_with("---\ntype: capture\n"), "{body}");
        assert!(body.contains("tags:\n- brain\n- ecom2\n"), "{body}");
        assert!(body.contains("client: claude-code"), "{body}");
        assert!(body.ends_with("---\n\nwhy\n"), "{body}");
        let front: serde_yaml::Value =
            serde_yaml::from_str(body.split("---").nth(1).unwrap()).unwrap();
        assert_eq!(front["title"].as_str(), Some("Решение: ECOM2 / рой"));
        assert_eq!(front["kind"].as_str(), Some("note"));
    }

    #[test]
    fn capture_refuses_what_it_should() {
        let t = kb(BRAIN);
        let cfg = Config::load(t.path()).unwrap().unwrap();
        let long = "x".repeat(MAX_TEXT + 1);
        assert!(capture(t.path(), &cfg, &note(" ", "x", &[]), at()).is_err());
        assert!(capture(t.path(), &cfg, &note("t", " \n", &[]), at()).is_err());
        assert!(capture(t.path(), &cfg, &note("t", &long, &[]), at()).is_err());
        let mut n = note("t", "x", &[]);
        n.kind = "todo";
        assert!(capture(t.path(), &cfg, &n, at()).is_err());
        n.kind = "decision";
        assert_eq!(
            capture(t.path(), &cfg, &n, at()).unwrap()["kind"],
            "decision"
        );
    }

    #[test]
    fn a_title_with_no_letters_still_gets_a_name() {
        assert_eq!(slug("!!! ???"), "note");
        assert_eq!(slug("A  b--c"), "a-b-c");
        assert_eq!(slug(&"я".repeat(80)).chars().count(), 60);
    }
}
