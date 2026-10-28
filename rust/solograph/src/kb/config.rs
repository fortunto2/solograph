//! A knowledge base's own settings: the `kb:` section of `.solograph.yaml` at its
//! root. The engine knows nothing of any one KB's layout; this file is where a KB
//! says where its wiki is and what the walk leaves out.
//!
//! ```yaml
//! kb:
//!   pages: wiki/                  # the wiki: [[links]], index.md, _aliases.yaml
//!   stacks: 1-methodology/stacks  # YAML templates for "which stack for X?" hints
//!   exclude: [archive, clients/acme]
//! ```
//!
//! Without the file every field has its default, which fits a KB with a `wiki/`.
//! The same file may hold `scope:` for the portfolio scan; each reads its own key.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

pub const FILE: &str = ".solograph.yaml";

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// The wiki folder, with a trailing `/`: the scope of kb-graph, kb-hint and
    /// `kb --pages`, and where `index.md` and `_aliases.yaml` live.
    pub pages: String,
    /// Stack templates the hint matches "which stack" prompts against; none unset.
    pub stacks: Option<String>,
    /// Directories the walk does not enter: a bare name anywhere, a path with a `/`
    /// from the root. Hidden directories, node_modules and __pycache__ always.
    pub exclude: Vec<String>,
    /// Weights of the prompt hook's memory gate (kb/memgate.rs), relative to the root;
    /// private by nature. Without it the hook only names wiki pages.
    pub memory_gate: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pages: "wiki/".into(),
            stacks: None,
            exclude: Vec::new(),
            memory_gate: None,
        }
    }
}

#[derive(Deserialize, Default)]
struct File {
    #[serde(default)]
    kb: Option<Config>,
}

impl Config {
    /// The root's settings; a file that does not parse is an error, not a silent
    /// default, because a wrong scope would look like an empty wiki.
    pub fn load(root: &Path) -> Result<Self> {
        let path = root.join(FILE);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(Self::default());
        };
        let file: Option<File> =
            serde_yaml::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
        let mut c = file.and_then(|f| f.kb).unwrap_or_default();
        let pages = c
            .pages
            .trim()
            .trim_start_matches("./")
            .trim_end_matches('/');
        c.pages = if pages.is_empty() {
            String::new()
        } else {
            format!("{pages}/")
        };
        Ok(c)
    }

    pub fn aliases_file(&self, root: &Path) -> PathBuf {
        root.join(&self.pages).join("_aliases.yaml")
    }

    /// A top-level wiki page: under `pages`, not in a subfolder, not the catalog
    /// or a `_` tooling file.
    pub fn is_page(&self, id: &str) -> bool {
        id.strip_prefix(&self.pages)
            .is_some_and(|rest| !rest.contains('/') && rest.ends_with(".md"))
            && !super::graph::is_meta(id)
    }

    /// Whether the walk skips the directory at `rel` (relative to the root).
    pub fn excludes(&self, rel: &str, name: &str) -> bool {
        self.exclude.iter().any(|e| {
            let e = e.trim_end_matches('/');
            if e.contains('/') { rel == e } else { name == e }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    fn load(yaml: &str) -> anyhow::Result<Config> {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(t.path().join(super::FILE), yaml).unwrap();
        Config::load(t.path())
    }

    #[test]
    fn no_file_or_no_kb_section_is_the_default() {
        let t = tempfile::tempdir().unwrap();
        assert_eq!(Config::load(t.path()).unwrap().pages, "wiki/");
        assert_eq!(load("scope: [a, b]\n").unwrap().pages, "wiki/");
    }

    #[test]
    fn pages_is_normalised_to_a_prefix() {
        assert_eq!(load("kb:\n  pages: ./notes\n").unwrap().pages, "notes/");
        let c = load("kb:\n  pages: notes/\n").unwrap();
        assert!(c.is_page("notes/a.md"));
        for id in [
            "notes/sub/a.md",
            "notes/index.md",
            "notes/_aliases.md",
            "wiki/a.md",
        ] {
            assert!(!c.is_page(id), "{id}");
        }
    }

    #[test]
    fn a_typo_is_an_error_not_a_default() {
        assert!(load("kb:\n  page: notes/\n").is_err());
        assert!(load("kb: [not, a, map]\n").is_err());
    }

    #[test]
    fn exclude_is_a_name_anywhere_or_a_path_from_the_root() {
        let c = load("kb:\n  exclude: [invoices, clients/acme/]\n").unwrap();
        assert!(c.excludes("crm/x/invoices", "invoices"));
        assert!(c.excludes("clients/acme", "acme"));
        assert!(!c.excludes("other/acme", "acme"));
    }
}
