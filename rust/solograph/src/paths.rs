//! Home directory and `~`-expanded paths from the environment, as Python's
//! `Path(os.environ[...]).expanduser()` reads them.

use std::path::PathBuf;

pub fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

pub fn expand(p: &str) -> PathBuf {
    match p.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => {
            home().join(rest.trim_start_matches('/'))
        }
        _ => PathBuf::from(p),
    }
}

/// The env var as a path, or None when unset or empty.
pub fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| expand(&v))
}
