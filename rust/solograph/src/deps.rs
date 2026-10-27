//! A project's dependency manifests as packages. Port of scanner/deps.py.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub source: &'static str,
}

fn pkg(name: impl Into<String>, version: impl Into<String>, source: &'static str) -> Package {
    Package {
        name: name.into(),
        version: version.into(),
        source,
    }
}

pub fn scan_deps(root: &Path) -> Vec<Package> {
    let mut out = Vec::new();
    for f in find_dep_files(root, "package.json") {
        out.extend(package_json(&f));
    }
    let pyproject = root.join("pyproject.toml");
    if pyproject.exists() {
        out.extend(pyproject_toml(&pyproject));
    }
    let req = root.join("requirements.txt");
    if req.exists() && !pyproject.exists() {
        out.extend(requirements_txt(&req));
    }
    let swift = root.join("Package.swift");
    if swift.exists() {
        out.extend(package_swift(&swift));
    }
    for f in find_dep_files(root, "Cargo.toml") {
        out.extend(cargo_toml(&f));
    }
    let gradle = [
        root.join("app/build.gradle.kts"),
        root.join("build.gradle.kts"),
    ]
    .into_iter()
    .find(|p| p.exists());
    if let Some(g) = gradle {
        out.extend(gradle_kts(&g));
    }
    out
}

/// The file at the root and one level deep (monorepos).
fn find_dep_files(root: &Path, filename: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    if root.join(filename).exists() {
        found.push(root.join(filename));
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return found;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let skip =
            ["node_modules", ".next", "dist", "build", "__pycache__"].contains(&name.as_str());
        if e.path().is_dir() && !name.starts_with('.') && !skip && e.path().join(filename).exists()
        {
            found.push(e.path().join(filename));
        }
    }
    found
}

fn lstrip<'a>(s: &'a str, chars: &str) -> &'a str {
    s.trim_start_matches(|c| chars.contains(c))
}

fn package_json(path: &Path) -> Vec<Package> {
    let Some(data) = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for section in ["dependencies", "devDependencies"] {
        for (name, version) in data
            .get(section)
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            if let Some(v) = version.as_str() {
                out.push(pkg(name.clone(), lstrip(v, "^~>="), "npm"));
            }
        }
    }
    out
}

/// `name>=1.0` → name, as deps.py splits it (no extras or markers handling, on purpose: same output).
fn split_requirement(dep: &str) -> (String, String) {
    let name = dep.split(">=").next().unwrap_or("");
    let name = name.split("==").next().unwrap_or("");
    let name = name.split('<').next().unwrap_or("");
    let name = name.split('>').next().unwrap_or("").trim().to_string();
    let version = lstrip(dep.replace(name.as_str(), "").trim(), ">=<~!").to_string();
    (name, version)
}

fn pyproject_toml(path: &Path) -> Vec<Package> {
    let Some(data) = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let deps = data
        .get("project")
        .and_then(|p| p.get("dependencies"))
        .and_then(toml::Value::as_array);
    for dep in deps.into_iter().flatten().filter_map(toml::Value::as_str) {
        let (name, version) = split_requirement(dep);
        out.push(pkg(name, version, "pip"));
    }
    let poetry = data.get("tool").and_then(|t| t.get("poetry"));
    for section in ["dependencies", "dev-dependencies"] {
        let table = poetry
            .and_then(|p| p.get(section))
            .and_then(toml::Value::as_table);
        for (name, ver) in table.into_iter().flatten() {
            if name == "python" {
                continue;
            }
            let ver = match ver {
                toml::Value::Table(t) => t.get("version").map(toml_str).unwrap_or_default(),
                other => toml_str(other),
            };
            out.push(pkg(name.clone(), lstrip(&ver, "^~>=<!"), "pip"));
        }
    }
    out
}

fn toml_str(v: &toml::Value) -> String {
    v.as_str().map_or_else(|| v.to_string(), str::to_string)
}

fn requirements_txt(path: &Path) -> Vec<Package> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('-'))
        .filter_map(|l| {
            let (name, version) = split_requirement(l);
            (!name.is_empty()).then(|| pkg(name, version, "pip"))
        })
        .collect()
}

static SPM: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\.package\(url:\s*"https?://[^/]+/[^/]+/([^"]+)""#).expect("static regex")
});
static GRADLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"implementation\("([^:]+):([^:]+):([^"]+)"\)"#).expect("static regex")
});

fn package_swift(path: &Path) -> Vec<Package> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    // `.git` is a suffix, removed once. deps.py used rstrip(".git"), which strips
    // any trailing '.', 'g', 'i', 't' — "swift-log" came out "swift-lo".
    SPM.captures_iter(&text)
        .map(|c| pkg(c[1].strip_suffix(".git").unwrap_or(&c[1]), "", "spm"))
        .collect()
}

fn gradle_kts(path: &Path) -> Vec<Package> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    GRADLE
        .captures_iter(&text)
        .map(|c| pkg(format!("{}:{}", &c[1], &c[2]), &c[3], "gradle"))
        .collect()
}

fn cargo_toml(path: &Path) -> Vec<Package> {
    let Some(data) = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
    else {
        return Vec::new();
    };
    let ws_deps = data.get("workspace").and_then(|w| w.get("dependencies"));
    let sections = [
        data.get("dependencies"),
        data.get("dev-dependencies"),
        data.get("build-dependencies"),
        ws_deps,
    ];
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for section in sections
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_table)
    {
        for (name, spec) in section {
            if !seen.insert(name.clone()) {
                continue;
            }
            let version = match spec {
                toml::Value::String(s) => s.clone(),
                toml::Value::Table(t) => match t.get("version") {
                    Some(v) if !toml_str(v).is_empty() => toml_str(v),
                    _ if t
                        .get("path")
                        .is_some_and(|p| p.as_str().is_none_or(|s| !s.is_empty())) =>
                    {
                        "path".into()
                    }
                    _ => String::new(),
                },
                _ => String::new(),
            };
            out.push(pkg(name.clone(), version, "cargo"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_swift_package_keeps_the_g_at_the_end_of_its_name() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("Package.swift");
        std::fs::write(
            &f,
            r#".package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
               .package(url: "https://github.com/x/Alamofire", from: "5.0.0")"#,
        )
        .unwrap();
        let names: Vec<String> = package_swift(&f).into_iter().map(|p| p.name).collect();
        assert_eq!(names, ["swift-log", "Alamofire"]);
    }

    #[test]
    fn pyproject_and_cargo() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("pyproject.toml"),
            "[project]\nname = \"v\"\ndependencies = [\"requests>=2\", \"numpy\"]\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[dependencies]\nserde = \"1\"\nlocal = { path = \"../x\" }\n[workspace.dependencies]\nserde = \"2\"\n",
        )
        .unwrap();
        let got: Vec<(String, String, &str)> = scan_deps(dir.path())
            .into_iter()
            .map(|p| (p.name, p.version, p.source))
            .collect();
        assert!(got.contains(&("requests".into(), "2".into(), "pip")));
        assert!(got.contains(&("numpy".into(), "".into(), "pip")));
        assert!(got.contains(&("serde".into(), "1".into(), "cargo")));
        assert!(got.contains(&("local".into(), "path".into(), "cargo")));
        assert_eq!(got.iter().filter(|p| p.0 == "serde").count(), 1);
    }
}
