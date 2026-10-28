//! The project registry (registry.yaml): load, merge with the disk, save.
//! Port of registry.py's refresh_registry and detect_stacks.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use serde::Deserialize;

use crate::paths::{env_path, expand, home};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Project {
    pub name: String,
    pub path: String,
    #[serde(default = "active")]
    pub status: String,
    #[serde(default)]
    pub stacks: Vec<String>,
    #[serde(default)]
    pub has_git: bool,
    #[serde(default)]
    pub has_claude_md: bool,
    #[serde(default)]
    pub last_commit: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub prd_path: Option<String>,
}

fn active() -> String {
    "active".into()
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Registry {
    #[serde(default)]
    pub projects: Vec<Project>,
    #[serde(default)]
    pub scan_path: Option<String>,
    #[serde(default)]
    pub last_scan: Option<String>,
}

pub fn default_path() -> PathBuf {
    env_path("CODEGRAPH_REGISTRY").unwrap_or_else(|| home().join(".solo").join("registry.yaml"))
}

pub fn load(path: &Path) -> Result<Option<Registry>> {
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_yaml::from_str(&std::fs::read_to_string(path)?)?))
}

#[derive(Deserialize, Default)]
struct ScopeFile {
    #[serde(default)]
    scope: Vec<String>,
}

/// The `scope:` names of a `.solograph.yaml`: which repositories are one product.
pub fn load_scope(path: &Path) -> Result<Vec<String>> {
    let f: Option<ScopeFile> = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
    Ok(f.unwrap_or_default().scope)
}

/// `name|path` per project, in registry order, for the nightly script. With a
/// scope, only those names, and a name the registry lacks is an error: it used
/// to be skipped in silence.
pub fn listing(reg: &Registry, scope: Option<&[String]>) -> Result<Vec<String>> {
    if let Some(names) = scope {
        let missing: Vec<&str> = names
            .iter()
            .filter(|n| !reg.projects.iter().any(|p| &p.name == *n))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            anyhow::bail!("not in the registry: {}", missing.join(", "));
        }
    }
    Ok(reg
        .projects
        .iter()
        .filter(|p| scope.is_none_or(|names| names.contains(&p.name)))
        .map(|p| format!("{}|{}", p.name, p.path))
        .collect())
}

fn q(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

fn q_opt(s: &Option<String>) -> String {
    s.as_deref().map_or_else(|| "null".into(), q)
}

/// Every string double-quoted: PyYAML reads an unquoted ISO date as a datetime,
/// and the Python model wants a str.
pub fn to_yaml(reg: &Registry) -> String {
    let mut y = String::from("projects:\n");
    if reg.projects.is_empty() {
        y = "projects: []\n".into();
    }
    for p in &reg.projects {
        y += &format!(
            "- name: {}\n  path: {}\n  status: {}\n",
            q(&p.name),
            q(&p.path),
            q(&p.status)
        );
        if p.stacks.is_empty() {
            y += "  stacks: []\n";
        } else {
            y += "  stacks:\n";
            for s in &p.stacks {
                y += &format!("  - {}\n", q(s));
            }
        }
        y += &format!(
            "  has_git: {}\n  has_claude_md: {}\n  last_commit: {}\n  description: {}\n  prd_path: {}\n",
            p.has_git,
            p.has_claude_md,
            q_opt(&p.last_commit),
            q(p.description.as_deref().unwrap_or("")),
            q_opt(&p.prd_path)
        );
    }
    y += &format!(
        "scan_path: {}\nlast_scan: {}\n",
        q_opt(&reg.scan_path),
        q_opt(&reg.last_scan)
    );
    y
}

const STACK_DETECTORS: [(&str, usize, &str); 10] = [
    ("*.xcodeproj", 3, "ios-swift"),
    ("project.yml", 2, "ios-swift"),
    ("build.gradle.kts", 3, "kotlin-android"),
    ("astro.config.*", 2, "astro-static"),
    ("wrangler.toml", 2, "cloudflare-workers"),
    ("wrangler.jsonc", 2, "cloudflare-workers"),
    ("pyproject.toml", 2, "python-ml"),
    ("Cargo.toml", 2, "rust-native"),
    ("Package.swift", 2, "ios-swift"),
    ("next.config.*", 2, "nextjs-supabase"),
];
const VENDORED: [&str; 4] = ["node_modules", ".next", "dist", "build"];

/// fnmatch with `*` only — all the detectors need.
fn glob(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == name,
        Some((pre, post)) => {
            name.len() >= pre.len() + post.len() && name.starts_with(pre) && name.ends_with(post)
        }
    }
}

/// Every entry up to `max_depth` below root, never entering hidden or vendored dirs.
fn shallow_entries(root: &Path, max_depth: usize) -> Vec<(usize, PathBuf)> {
    fn walk(d: &Path, depth: usize, max: usize, out: &mut Vec<(usize, PathBuf)>) {
        let Ok(entries) = std::fs::read_dir(d) else {
            return;
        };
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            out.push((depth, e.path()));
            let real_dir = e.file_type().is_ok_and(|t| t.is_dir());
            if depth < max
                && real_dir
                && !name.starts_with('.')
                && !VENDORED.contains(&name.as_str())
            {
                walk(&e.path(), depth + 1, max, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, 1, max_depth, &mut out);
    out
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

pub fn detect_stacks(root: &Path) -> Vec<String> {
    let entries = shallow_entries(root, 3);
    let mut detected: HashSet<&str> = HashSet::new();
    for (pattern, max_depth, stack) in STACK_DETECTORS {
        if entries
            .iter()
            .any(|(d, p)| *d <= max_depth && glob(pattern, &file_name(p)))
        {
            detected.insert(stack);
        }
    }
    let mut pkgs: Vec<&(usize, PathBuf)> = entries
        .iter()
        .filter(|(d, p)| *d <= 2 && file_name(p) == "package.json")
        .collect();
    pkgs.sort();
    let deps = pkgs.iter().find_map(|(_, p)| {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
        let mut keys: HashSet<String> = HashSet::new();
        for section in ["dependencies", "devDependencies"] {
            keys.extend(
                v.get(section)
                    .and_then(|s| s.as_object())
                    .into_iter()
                    .flatten()
                    .map(|(k, _)| k.clone()),
            );
        }
        Some(keys)
    });
    if let Some(deps) = deps
        && deps.contains("next")
    {
        detected.insert("nextjs-supabase");
        if deps.contains("ai") {
            detected.insert("nextjs-ai-agents");
        }
    }
    let fastapi = entries.iter().any(|(d, p)| {
        *d <= 2
            && file_name(p) == "pyproject.toml"
            && std::fs::read_to_string(p).is_ok_and(|t| t.to_lowercase().contains("fastapi"))
    });
    if detected.contains("python-ml") && fastapi {
        detected.remove("python-ml");
        detected.insert("python-api");
    }
    let mut out: Vec<String> = detected.into_iter().map(String::from).collect();
    out.sort();
    out
}

fn last_commit(root: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["log", "-1", "--format=%ci"])
        .current_dir(root)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

fn description(root: &Path) -> String {
    for f in ["CLAUDE.md", "README.md"] {
        if let Ok(text) = std::fs::read_to_string(root.join(f)) {
            let line = text
                .split('\n')
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("---"));
            if let Some(l) = line {
                return l.chars().take(120).collect();
            }
        }
    }
    String::new()
}

/// A real git checkout: not hidden, `.git` a directory (a worktree's is a file).
pub fn is_project_dir(p: &Path) -> bool {
    p.is_dir() && !file_name(p).starts_with('.') && p.join(".git").is_dir()
}

fn project_info(dir: &Path, status: &str) -> Project {
    Project {
        name: file_name(dir),
        path: dir.display().to_string(),
        status: status.into(),
        stacks: detect_stacks(dir),
        has_git: dir.join(".git").is_dir(),
        has_claude_md: dir.join("CLAUDE.md").is_file(),
        last_commit: last_commit(dir),
        description: Some(description(dir)),
        prd_path: None,
    }
}

#[derive(Debug, Default)]
pub struct RefreshReport {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub collisions: Vec<String>,
    pub backup: Option<PathBuf>,
    pub total: usize,
}

const KEEP_BACKUPS: usize = 5;

/// $CODEGRAPH_SCAN_PATH, else the paths the registry recorded last time — the
/// nightly job runs with neither configured — else ~/startups/active.
pub fn resolve_scan_paths(env: Option<String>, recorded: Option<String>) -> Vec<PathBuf> {
    let raw = env
        .filter(|v| !v.is_empty())
        .or(recorded.filter(|v| !v.is_empty()))
        .unwrap_or_else(|| home().join("startups/active").display().to_string());
    raw.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(expand)
        .collect()
}

/// Bring the registry up to date with the disk without losing what was put there.
pub fn refresh(path: &Path, scan_paths: Option<Vec<PathBuf>>) -> Result<RefreshReport> {
    let existing = load(path)?.unwrap_or_default();
    let scan_paths = scan_paths.unwrap_or_else(|| {
        resolve_scan_paths(
            std::env::var("CODEGRAPH_SCAN_PATH").ok(),
            existing.scan_path.clone(),
        )
    });
    let mut report = RefreshReport::default();
    let mut kept: Vec<Project> = Vec::new();
    for p in existing.projects {
        let d = PathBuf::from(&p.path);
        if !d.exists() {
            report.removed.push(p.name);
            continue;
        }
        let fresh = project_info(&d, &p.status);
        kept.push(Project {
            name: p.name,
            status: p.status,
            prd_path: p.prd_path,
            ..fresh
        });
    }
    let mut known_paths: HashSet<String> = kept.iter().map(|p| p.path.clone()).collect();
    let mut known_names: HashSet<String> = kept.iter().map(|p| p.name.clone()).collect();
    for dir in &scan_paths {
        let Ok(entries) = std::fs::read_dir(dir) else {
            eprintln!("Scan path not found: {}", dir.display());
            continue;
        };
        let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        dirs.sort();
        for d in dirs {
            let key = d.display().to_string();
            if known_paths.contains(&key) || !is_project_dir(&d) {
                continue;
            }
            let name = file_name(&d);
            if known_names.contains(&name) {
                report.collisions.push(key);
                continue;
            }
            kept.push(project_info(&d, "active"));
            known_paths.insert(key);
            known_names.insert(name.clone());
            report.added.push(name);
        }
    }
    if path.exists() {
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S-%6f");
        let base = file_name(path);
        let backup = path.with_file_name(format!("{base}.{stamp}.bak"));
        std::fs::copy(path, &backup)?;
        report.backup = Some(backup);
        let mut olds: Vec<PathBuf> = std::fs::read_dir(path.parent().unwrap_or(Path::new(".")))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                file_name(p).starts_with(&format!("{base}.")) && file_name(p).ends_with(".bak")
            })
            .collect();
        olds.sort();
        let excess = olds.len().saturating_sub(KEEP_BACKUPS);
        for old in &olds[..excess] {
            std::fs::remove_file(old)?;
        }
    }
    let reg = Registry {
        projects: kept,
        scan_path: Some(
            scan_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(","),
        ),
        last_scan: Some(
            chrono::Local::now()
                .format("%Y-%m-%dT%H:%M:%S%.6f")
                .to_string(),
        ),
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, to_yaml(&reg))?;
    report.total = reg.projects.len();
    Ok(report)
}

#[cfg(test)]
mod tests;
