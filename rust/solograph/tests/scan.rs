//! `solograph scan` against real git repositories, for three things the review of
//! 28 Oct 2026 reproduced: a hook's GIT_DIR read another worktree's index, a
//! conflicted path failed the scan, and `scan` rewrote a registry it was only
//! meant to read (re-adding repos a client profile had excluded).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn git(dir: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.email=t@example.com", "-c", "user.name=t"])
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .unwrap();
    // A merge that conflicts exits 1 on purpose.
    assert!(
        ok.status.success() || args[0] == "merge",
        "git {args:?}: {}",
        String::from_utf8_lossy(&ok.stderr)
    );
}

fn repo(dir: &Path, file: &str, code: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    git(dir, &["init", "-q", "-b", "main"]);
    std::fs::write(dir.join(file), code).unwrap();
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "-qm", "init"]);
    dir.to_path_buf()
}

fn solograph(t: &Path, args: &[&str], env: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_solograph"));
    cmd.args(args)
        .env("HOME", t)
        .env("CODEGRAPH_REGISTRY", t.join("registry.yaml"))
        .env("SOLOGRAPH_PORTFOLIO_DB", t.join("portfolio.db"))
        .env("SOLOGRAPH_LOG", "off")
        .env_remove("CODEGRAPH_SCAN_PATH");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}

fn registry(t: &Path, projects: &[(&str, &Path)]) {
    let mut y = String::from("projects:\n");
    for (name, path) in projects {
        y += &format!("- name: \"{name}\"\n  path: \"{}\"\n", path.display());
    }
    std::fs::write(t.join("registry.yaml"), y).unwrap();
}

fn symbols(t: &Path, q: &str) -> String {
    String::from_utf8_lossy(&solograph(t, &["symbols", q], &[]).stdout).into_owned()
}

#[test]
fn a_hooks_git_dir_does_not_redirect_the_scan() {
    let t = tempfile::tempdir().unwrap();
    let main = repo(
        &t.path().join("main"),
        "a.py",
        "def main_only_function():\n    pass\n",
    );
    let other = repo(
        &t.path().join("other"),
        "b.py",
        "def other_function():\n    pass\n",
    );
    registry(t.path(), &[("main", &main)]);
    let out = solograph(
        t.path(),
        &["scan", "-p", "main"],
        &[
            ("GIT_DIR", &other.join(".git")),
            ("GIT_INDEX_FILE", &other.join(".git/index")),
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(symbols(t.path(), "main only function").contains("main_only_function"));
}

#[test]
fn a_conflicted_path_mid_merge_does_not_fail_the_scan() {
    let t = tempfile::tempdir().unwrap();
    let r = repo(
        &t.path().join("r"),
        "a.py",
        "def shared_function():\n    return 0\n",
    );
    git(&r, &["checkout", "-qb", "side"]);
    std::fs::write(r.join("a.py"), "def shared_function():\n    return 1\n").unwrap();
    git(&r, &["commit", "-qam", "side"]);
    git(&r, &["checkout", "-q", "main"]);
    std::fs::write(r.join("a.py"), "def shared_function():\n    return 2\n").unwrap();
    git(&r, &["commit", "-qam", "main"]);
    git(&r, &["merge", "-q", "side"]);
    registry(t.path(), &[("r", &r)]);
    let out = solograph(t.path(), &["scan", "-p", "r"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn scan_reads_the_registry_and_never_rewrites_it() {
    let t = tempfile::tempdir().unwrap();
    let projects = t.path().join("projects");
    let kept = repo(
        &projects.join("kept"),
        "a.py",
        "def kept_function():\n    pass\n",
    );
    repo(
        &projects.join("excluded"),
        "b.py",
        "def excluded_function():\n    pass\n",
    );
    registry(t.path(), &[("kept", &kept)]);
    let before = std::fs::read_to_string(t.path().join("registry.yaml")).unwrap();
    let out = solograph(t.path(), &["scan"], &[("CODEGRAPH_SCAN_PATH", &projects)]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(t.path().join("registry.yaml")).unwrap(),
        before
    );
    assert!(!symbols(t.path(), "excluded function").contains("excluded_function"));
}

#[test]
fn without_a_registry_scan_builds_one_only_from_named_folders() {
    let t = tempfile::tempdir().unwrap();
    let out = solograph(t.path(), &["scan"], &[]);
    assert!(!out.status.success());
    assert!(!t.path().join("registry.yaml").exists());

    let projects = t.path().join("projects");
    repo(
        &projects.join("app"),
        "a.py",
        "def app_function():\n    pass\n",
    );
    let out = solograph(t.path(), &["scan"], &[("CODEGRAPH_SCAN_PATH", &projects)]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(symbols(t.path(), "app function").contains("app_function"));
}
