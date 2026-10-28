//! Port of tests/test_registry_merge.py. Fixtures are written the way PyYAML
//! writes them (unquoted), so reading a Python-written registry is tested too.

use std::path::{Path, PathBuf};

use super::{detect_stacks, listing, load, load_scope, refresh, resolve_scan_paths};

fn repo(path: &Path, files: &[(&str, &str)]) -> PathBuf {
    std::fs::create_dir_all(path.join(".git")).unwrap();
    for (rel, text) in files {
        let f = path.join(rel);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, text).unwrap();
    }
    path.to_path_buf()
}

fn python_registry(path: &Path, projects: &[(&str, &Path)], scan_path: &str) {
    let mut y = if projects.is_empty() {
        "projects: []\n".to_string()
    } else {
        "projects:\n".to_string()
    };
    for (name, p) in projects {
        y += &format!(
            "- name: {name}\n  path: {}\n  status: active\n  stacks: []\n  has_git: false\n  has_claude_md: false\n  last_commit: 2026-10-25 19:54:14 -0700\n  description: ''\n  prd_path: null\n",
            p.display()
        );
    }
    y += &format!("scan_path: '{scan_path}'\nlast_scan: ''\n");
    std::fs::write(path, y).unwrap();
}

fn names(path: &Path) -> Vec<String> {
    load(path)
        .unwrap()
        .unwrap()
        .projects
        .into_iter()
        .map(|p| p.name)
        .collect()
}

#[test]
fn new_checkouts_are_appended_and_outside_entries_kept() {
    let t = tempfile::tempdir().unwrap();
    let active = t.path().join("active");
    repo(&active.join("old-one"), &[]);
    repo(&active.join("new-one"), &[]);
    let nested = repo(&t.path().join("elsewhere/nested"), &[]);
    let reg = t.path().join("registry.yaml");
    python_registry(
        &reg,
        &[("old-one", &active.join("old-one")), ("nested", &nested)],
        "",
    );
    let report = refresh(&reg, Some(vec![active])).unwrap();
    assert_eq!(names(&reg), ["old-one", "nested", "new-one"]);
    assert_eq!(report.added, ["new-one"]);
    assert!(report.removed.is_empty());
}

#[test]
fn an_entry_whose_directory_is_gone_is_dropped_and_named() {
    let t = tempfile::tempdir().unwrap();
    let here = repo(&t.path().join("here"), &[]);
    let reg = t.path().join("registry.yaml");
    python_registry(
        &reg,
        &[("here", &here), ("deleted", &t.path().join("deleted"))],
        "",
    );
    let report = refresh(&reg, Some(vec![])).unwrap();
    assert_eq!(names(&reg), ["here"]);
    assert_eq!(report.removed, ["deleted"]);
}

#[test]
fn only_real_checkouts_count_as_new_projects() {
    let t = tempfile::tempdir().unwrap();
    let active = t.path().join("active");
    repo(&active.join("repo"), &[]);
    std::fs::create_dir_all(active.join("no-git")).unwrap();
    std::fs::create_dir_all(active.join("worktree")).unwrap();
    std::fs::write(
        active.join("worktree/.git"),
        "gitdir: /somewhere/.git/worktrees/w\n",
    )
    .unwrap();
    std::fs::write(active.join("demo.zip"), "PK").unwrap();
    std::fs::create_dir_all(active.join(".hidden/.git")).unwrap();
    let reg = t.path().join("registry.yaml");
    refresh(&reg, Some(vec![active])).unwrap();
    assert_eq!(names(&reg), ["repo"]);
}

#[test]
fn a_new_directory_never_takes_an_existing_name() {
    let t = tempfile::tempdir().unwrap();
    let active = t.path().join("active");
    repo(&active.join("api"), &[]);
    let first = repo(&t.path().join("other/api"), &[]);
    let reg = t.path().join("registry.yaml");
    python_registry(&reg, &[("api", &first)], "");
    let report = refresh(&reg, Some(vec![active.clone()])).unwrap();
    let projects = load(&reg).unwrap().unwrap().projects;
    assert_eq!(projects.len(), 1);
    assert_eq!(projects[0].path, first.display().to_string());
    assert_eq!(
        report.collisions,
        [active.join("api").display().to_string()]
    );
}

#[test]
fn the_previous_file_is_backed_up_and_backups_do_not_pile_up() {
    let t = tempfile::tempdir().unwrap();
    let active = t.path().join("active");
    repo(&active.join("a"), &[]);
    let reg = t.path().join("registry.yaml");
    python_registry(&reg, &[], "");
    let before = std::fs::read_to_string(&reg).unwrap();
    let report = refresh(&reg, Some(vec![active.clone()])).unwrap();
    assert_eq!(
        std::fs::read_to_string(report.backup.unwrap()).unwrap(),
        before
    );
    for _ in 0..8 {
        refresh(&reg, Some(vec![active.clone()])).unwrap();
    }
    let baks = std::fs::read_dir(t.path())
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".bak"))
        .count();
    assert_eq!(baks, 5);
}

#[test]
fn scan_paths_default_to_the_ones_the_registry_recorded() {
    assert_eq!(
        resolve_scan_paths(None, Some("/a,/b".into())),
        [PathBuf::from("/a"), PathBuf::from("/b")]
    );
    assert_eq!(
        resolve_scan_paths(Some("/env".into()), Some("/a".into())),
        [PathBuf::from("/env")]
    );
}

#[test]
fn stack_detection_sees_shallow_markers_and_ignores_node_modules() {
    let t = tempfile::tempdir().unwrap();
    let proj = repo(
        &t.path().join("app"),
        &[
            (
                "web/package.json",
                r#"{"dependencies": {"next": "15", "ai": "4"}}"#,
            ),
            ("node_modules/x/pyproject.toml", "[project]\nname='x'\n"),
            ("ios/App/App.xcodeproj/project.pbxproj", ""),
        ],
    );
    assert_eq!(
        detect_stacks(&proj),
        ["ios-swift", "nextjs-ai-agents", "nextjs-supabase"]
    );
}

#[test]
fn rust_swift_package_and_next_projects_get_a_stack() {
    let t = tempfile::tempdir().unwrap();
    for (name, marker, stack) in [
        ("r", "Cargo.toml", "rust-native"),
        ("s", "Package.swift", "ios-swift"),
        ("n", "next.config.ts", "nextjs-supabase"),
    ] {
        let d = repo(&t.path().join(name), &[(marker, "")]);
        assert!(detect_stacks(&d).contains(&stack.to_string()), "{marker}");
    }
}

#[test]
fn the_written_registry_quotes_dates_for_pyyaml() {
    let t = tempfile::tempdir().unwrap();
    let reg = t.path().join("registry.yaml");
    refresh(&reg, Some(vec![])).unwrap();
    let text = std::fs::read_to_string(&reg).unwrap();
    assert!(
        text.lines().any(|l| l.starts_with("last_scan: \"20")),
        "{text}"
    );
}

/// The nightly script read the registry with the project's Python venv; on
/// 28 Oct the venv was being rebuilt at 04:10 and the run failed. The binary
/// that already parses the registry now answers the listing itself.
#[test]
fn listing_is_name_pipe_path_and_a_scope_narrows_it_loudly() {
    let t = tempfile::tempdir().unwrap();
    let reg = t.path().join("registry.yaml");
    let (a, b) = (t.path().join("a"), t.path().join("b"));
    python_registry(&reg, &[("alpha", &a), ("beta", &b)], "");
    let r = load(&reg).unwrap().unwrap();

    let all = listing(&r, None).unwrap();
    assert_eq!(
        all,
        vec![
            format!("alpha|{}", a.display()),
            format!("beta|{}", b.display())
        ]
    );

    let scope_file = t.path().join(".solograph.yaml");
    std::fs::write(&scope_file, "# comment\nscope:\n  - beta   # why\n").unwrap();
    let scope = load_scope(&scope_file).unwrap();
    assert_eq!(
        listing(&r, Some(&scope)).unwrap(),
        vec![format!("beta|{}", b.display())]
    );

    // A name only on disk used to be skipped in silence ("Project not found").
    let err = listing(&r, Some(&["gamma".to_string()]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("gamma"), "{err}");
}

/// Review 28 Oct: serde_json leaves DEL, C1 controls and U+FFFE raw, and YAML
/// refuses them, so one odd byte in one README broke every registry read.
#[test]
fn control_characters_in_a_description_still_load() {
    let t = tempfile::tempdir().unwrap();
    let reg = super::Registry {
        projects: vec![super::Project {
            name: "odd".into(),
            path: "/x/odd".into(),
            description: Some("a\u{7f}b\u{9d}c\u{fffe}d\u{1}e".into()),
            ..Default::default()
        }],
        ..Default::default()
    };
    let path = t.path().join("registry.yaml");
    std::fs::write(&path, super::to_yaml(&reg)).unwrap();
    let back = load(&path).unwrap().unwrap();
    assert_eq!(
        back.projects[0].description.as_deref(),
        Some("a\u{7f}b\u{9d}c\u{fffe}d\u{1}e")
    );
}
