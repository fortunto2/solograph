//! Port of tests/test_portfolio.py.

use std::path::Path;

use serde_json::{Value, json};

use super::Portfolio;
use crate::registry::Project;
use crate::scanner::SymbolExtractor;

const VIDEO: [(&str, &str); 3] = [
    (
        "src/montage.py",
        "class MontageProfile:\n    pass\n\n\ndef resolve_montage_profile(name):\n    return name\n",
    ),
    ("src/render.py", "def render_frame(x):\n    return x\n"),
    (
        "pyproject.toml",
        "[project]\nname = \"video\"\ndependencies = [\"requests>=2\", \"numpy\"]\n",
    ),
];

fn project(tmp: &Path, name: &str, files: &[(&str, &str)]) -> Project {
    let root = tmp.join(name);
    std::fs::create_dir_all(&root).unwrap();
    for (rel, text) in files {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }
    Project {
        name: name.into(),
        path: root.display().to_string(),
        status: "active".into(),
        stacks: vec!["python".into()],
        ..Default::default()
    }
}

fn scanned(tmp: &Path, projects: &[Project]) -> Portfolio {
    let mut pf = Portfolio::open(Some(&tmp.join("portfolio.db"))).unwrap();
    pf.sync_projects(projects, true).unwrap();
    for p in projects {
        pf.scan_project(&p.name, Path::new(&p.path), &mut SymbolExtractor::default())
            .unwrap();
    }
    pf
}

fn names(hits: &[Value]) -> Vec<String> {
    hits.iter()
        .map(|h| h["name"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_query_by_words_finds_both_spellings() {
    let t = tempfile::tempdir().unwrap();
    let pf = scanned(t.path(), &[project(t.path(), "video", &VIDEO)]);
    let hits = pf
        .search_symbols("montage profile", 20, None, None)
        .unwrap();
    let mut top: Vec<String> = names(&hits).into_iter().take(2).collect();
    top.sort();
    assert_eq!(top, ["MontageProfile", "resolve_montage_profile"]);
    let hit = hits
        .iter()
        .find(|h| h["name"] == "resolve_montage_profile")
        .unwrap();
    assert_eq!(
        (&hit["project"], &hit["file"], &hit["line"], &hit["kind"]),
        (
            &json!("video"),
            &json!("src/montage.py"),
            &json!(5),
            &json!("function")
        )
    );
}

#[test]
fn an_exact_name_comes_first() {
    let t = tempfile::tempdir().unwrap();
    let pf = scanned(t.path(), &[project(t.path(), "video", &VIDEO)]);
    assert_eq!(
        pf.search_symbols("MontageProfile", 20, None, None).unwrap()[0]["name"],
        "MontageProfile"
    );
    assert_eq!(
        pf.search_symbols("resolve_montage_profile", 20, None, None)
            .unwrap()[0]["name"],
        "resolve_montage_profile"
    );
}

#[test]
fn project_and_kind_filters() {
    let t = tempfile::tempdir().unwrap();
    let pf = scanned(
        t.path(),
        &[
            project(t.path(), "a", &VIDEO),
            project(
                t.path(),
                "b",
                &[("x.py", "def montage_profile():\n    pass\n")],
            ),
        ],
    );
    assert!(
        pf.search_symbols("montage profile", 20, Some("b"), None)
            .unwrap()
            .iter()
            .all(|h| h["project"] == "b")
    );
    assert!(
        pf.search_symbols("montage profile", 20, None, Some("class"))
            .unwrap()
            .iter()
            .all(|h| h["kind"] == "class")
    );
}

#[test]
fn a_second_scan_reads_nothing_and_an_edit_reads_one_file() {
    let t = tempfile::tempdir().unwrap();
    let p = project(t.path(), "video", &VIDEO);
    let mut pf = Portfolio::open(Some(&t.path().join("portfolio.db"))).unwrap();
    pf.sync_projects(std::slice::from_ref(&p), true).unwrap();
    let mut ex = SymbolExtractor::default();
    assert_eq!(
        pf.scan_project("video", Path::new(&p.path), &mut ex)
            .unwrap()
            .read,
        3
    );
    assert_eq!(
        pf.scan_project("video", Path::new(&p.path), &mut ex)
            .unwrap()
            .read,
        0
    );
    std::fs::write(
        t.path().join("video/src/render.py"),
        "def render_thumbnail(x):\n    return x\n",
    )
    .unwrap();
    assert_eq!(
        pf.scan_project("video", Path::new(&p.path), &mut ex)
            .unwrap()
            .read,
        1
    );
    assert!(
        !names(&pf.search_symbols("render frame", 20, None, None).unwrap())
            .contains(&"render_frame".to_string())
    );
    assert_eq!(
        pf.search_symbols("render thumbnail", 20, None, None)
            .unwrap()[0]["name"],
        "render_thumbnail"
    );
}

#[test]
fn a_deleted_file_takes_its_symbols_with_it() {
    let t = tempfile::tempdir().unwrap();
    let p = project(t.path(), "video", &VIDEO);
    let mut pf = scanned(t.path(), std::slice::from_ref(&p));
    std::fs::remove_file(t.path().join("video/src/render.py")).unwrap();
    assert_eq!(
        pf.scan_project("video", Path::new(&p.path), &mut SymbolExtractor::default())
            .unwrap()
            .removed,
        1
    );
    assert!(
        pf.search_symbols("render", 20, None, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn packages_and_who_shares_them() {
    let t = tempfile::tempdir().unwrap();
    let b = project(
        t.path(),
        "b",
        &[(
            "pyproject.toml",
            "[project]\nname = \"b\"\ndependencies = [\"requests\"]\n",
        )],
    );
    let pf = scanned(t.path(), &[project(t.path(), "a", &VIDEO), b]);
    assert_eq!(
        pf.shared_packages(None).unwrap(),
        [json!({"package": "requests", "source": "pip", "projects": ["a", "b"]})]
    );
    assert_eq!(
        pf.shared_packages(Some("numpy")).unwrap(),
        [json!({"package": "numpy", "source": "pip", "projects": ["a"]})]
    );
}

#[test]
fn a_project_dropped_from_the_registry_leaves_the_store() {
    let t = tempfile::tempdir().unwrap();
    let a = project(t.path(), "a", &VIDEO);
    let mut pf = scanned(
        t.path(),
        &[
            a.clone(),
            project(t.path(), "b", &[("x.py", "def only_in_b():\n    pass\n")]),
        ],
    );
    pf.sync_projects(&[a], true).unwrap();
    assert!(
        pf.search_symbols("only in b", 20, None, None)
            .unwrap()
            .is_empty()
    );
    assert_eq!(pf.stats().unwrap()["projects"], 1);
}

#[test]
fn a_name_with_a_quote_is_just_a_name() {
    let t = tempfile::tempdir().unwrap();
    let pf = scanned(t.path(), &[project(t.path(), "Rustam's app", &VIDEO)]);
    assert_eq!(
        pf.explain("Rustam's app").unwrap().unwrap()["counts"]["files"],
        3
    );
    assert!(
        !pf.search_symbols("montage", 20, Some("Rustam's app"), None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn explain_says_what_a_project_is_made_of() {
    let t = tempfile::tempdir().unwrap();
    let pf = scanned(t.path(), &[project(t.path(), "video", &VIDEO)]);
    let data = pf.explain("video").unwrap().unwrap();
    assert_eq!(data["stack"], "python");
    assert_eq!(
        data["counts"],
        json!({"files": 3, "symbols": 3, "packages": 2})
    );
    assert_eq!(data["languages"][0][0], "python");
    assert!(
        data["dependencies"]
            .as_array()
            .unwrap()
            .contains(&json!(["requests", "pip"]))
    );
    assert_eq!(
        pf.explain("nope").unwrap().unwrap_err(),
        "Project 'nope' not found."
    );
}

#[test]
fn all_words_rank_above_some_words() {
    let t = tempfile::tempdir().unwrap();
    let pf = scanned(
        t.path(),
        &[project(
            t.path(),
            "video",
            &[(
                "a.py",
                "def render_frame():\n    pass\n\n\ndef render_page():\n    pass\n",
            )],
        )],
    );
    assert_eq!(
        names(&pf.search_symbols("render frame", 20, None, None).unwrap()),
        ["render_frame", "render_page"]
    );
}
