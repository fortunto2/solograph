//! A project's source files and the definitions in them. Port of
//! scanner/code.py: same file universe (git-tracked, same skip rules), same
//! tree-sitter queries on the same grammar versions.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor};

/// Extension (with the dot) → language name, as LANG_MAP.
pub fn lang_of(ext: &str) -> Option<&'static str> {
    Some(match ext {
        ".py" => "python",
        ".php" | ".inc" => "php",
        ".tf" | ".hcl" => "hcl",
        ".js" | ".mjs" | ".cjs" => "javascript",
        ".html" | ".htm" => "html",
        ".css" | ".scss" => "css",
        ".yml" | ".yaml" => "yaml",
        ".sh" | ".bash" => "bash",
        ".sql" => "sql",
        ".toml" => "toml",
        ".swift" => "swift",
        ".ts" => "typescript",
        ".tsx" => "tsx",
        ".kt" => "kotlin",
        ".rs" => "rust",
        ".go" => "go",
        ".java" => "java",
        ".rb" => "ruby",
        ".c" | ".h" => "c",
        ".cpp" | ".cc" | ".cxx" | ".hpp" => "cpp",
        _ => return None,
    })
}

const EXTENSIONS: &[&str] = &[
    ".py", ".php", ".inc", ".tf", ".hcl", ".js", ".mjs", ".cjs", ".html", ".htm", ".css", ".scss",
    ".yml", ".yaml", ".sh", ".bash", ".sql", ".toml", ".swift", ".ts", ".tsx", ".kt", ".rs", ".go",
    ".java", ".rb", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
];

const SKIP_DIRS: &[&str] = &[
    ".git",
    ".claude",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".build",
    "target",
    "DerivedData",
    "build",
    ".next",
    "dist",
    ".output",
    ".gradle",
    "Pods",
    ".eggs",
    ".tox",
    ".turbo",
    ".vercel",
    ".wrangler",
    "coverage",
    ".nyc_output",
    "htmlcov",
    ".pytest_cache",
    "samples",
    "Samples",
    "examples",
    "Examples",
    "react-samples",
    "vendor",
    "third_party",
    "third-party",
    "generated",
    ".cache",
    ".parcel-cache",
    ".swc",
    ".idea",
    ".vscode",
];
const SKIP_FILES: [&str; 2] = [".DS_Store", "package-lock.json"];
const SKIP_SUFFIXES: [&str; 2] = [".map", ".lock"];
const SKIP_MARKERS: [&str; 5] = [".min.", ".bundle.", "-min.", ".pack.", "-lock."];

/// Python's `Path.suffix`: the last dot-suffix of the name, "" for dotfiles.
pub fn suffix(name: &str) -> &str {
    match name.rfind('.') {
        Some(0) | None => "",
        Some(i) if i == name.len() - 1 => "",
        Some(i) => &name[i..],
    }
}

fn skipped_dir(path: &Path) -> bool {
    path.components().any(|c| match c {
        Component::Normal(s) => {
            let s = s.to_string_lossy();
            SKIP_DIRS.contains(&s.as_ref())
        }
        _ => false,
    })
}

pub fn is_skipped_file(name: &str) -> bool {
    SKIP_FILES.contains(&name)
        || SKIP_SUFFIXES.iter().any(|s| name.ends_with(s))
        || SKIP_MARKERS.iter().any(|m| name.contains(m))
}

fn keep(fp: &Path) -> bool {
    let name = fp.file_name().unwrap_or_default().to_string_lossy();
    !skipped_dir(fp) && !is_skipped_file(&name) && fp.is_file()
}

/// The files git tracks, or None when it is not a checkout.
fn tracked_files(root: &Path) -> Option<Vec<PathBuf>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(
        text.split('\0')
            .filter(|n| !n.is_empty())
            .map(|n| root.join(n))
            .collect(),
    )
}

/// (path, language) for every source file in a project, git-tracked when possible.
pub fn project_files(root: &Path) -> Vec<(PathBuf, &'static str)> {
    if let Some(tracked) = tracked_files(root) {
        return tracked
            .into_iter()
            .filter_map(|fp| {
                let name = fp.file_name()?.to_string_lossy().into_owned();
                let lang = lang_of(suffix(&name))?;
                keep(&fp).then_some((fp, lang))
            })
            .collect();
    }
    let mut all = Vec::new();
    walk(root, &mut all);
    let mut out = Vec::new();
    for ext in EXTENSIONS {
        for fp in &all {
            let name = fp.file_name().unwrap_or_default().to_string_lossy();
            if name.ends_with(ext) && keep(fp) {
                out.push((fp.clone(), lang_of(ext).unwrap_or("text")));
            }
        }
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<_> = entries.flatten().collect();
    paths.sort_by_key(|e| e.path());
    for e in paths {
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() && !skipped_dir(&e.path()) {
            walk(&e.path(), out);
        } else if !ft.is_dir() {
            out.push(e.path());
        }
    }
}

const QUERIES: [(&str, &str); 16] = [
    (
        "python",
        "(function_definition name: (identifier) @func.def)\n(class_definition name: (identifier) @class.def)",
    ),
    (
        "swift",
        "(function_declaration name: (simple_identifier) @func.def)\n(class_declaration name: (type_identifier) @class.def)\n(protocol_declaration name: (type_identifier) @protocol.def)",
    ),
    (
        "typescript",
        "(function_declaration name: (identifier) @func.def)\n(method_definition name: (property_identifier) @func.def)\n(class_declaration name: (type_identifier) @class.def)\n(variable_declarator name: (identifier) @func.def value: (arrow_function))",
    ),
    (
        "kotlin",
        "(function_declaration (identifier) @func.def)\n(class_declaration (identifier) @class.def)\n(object_declaration (identifier) @class.def)",
    ),
    (
        "rust",
        "(function_item name: (identifier) @func.def)\n(struct_item name: (type_identifier) @class.def)\n(enum_item name: (type_identifier) @class.def)\n(trait_item name: (type_identifier) @protocol.def)",
    ),
    (
        "go",
        "(function_declaration name: (identifier) @func.def)\n(method_declaration name: (field_identifier) @func.def)\n(type_declaration (type_spec name: (type_identifier) @class.def))",
    ),
    (
        "java",
        "(method_declaration name: (identifier) @func.def)\n(class_declaration name: (identifier) @class.def)\n(interface_declaration name: (identifier) @protocol.def)\n(enum_declaration name: (identifier) @class.def)",
    ),
    (
        "ruby",
        "(method name: (identifier) @func.def)\n(class name: (constant) @class.def)\n(module name: (constant) @class.def)",
    ),
    (
        "c",
        "(function_definition declarator: (function_declarator declarator: (identifier) @func.def))\n(struct_specifier name: (type_identifier) @class.def)\n(enum_specifier name: (type_identifier) @class.def)",
    ),
    (
        "cpp",
        "(function_definition declarator: (function_declarator declarator: (identifier) @func.def))\n(class_specifier name: (type_identifier) @class.def)\n(struct_specifier name: (type_identifier) @class.def)\n(enum_specifier name: (type_identifier) @class.def)",
    ),
    (
        "php",
        "(function_definition name: (name) @func.def)\n(method_declaration name: (name) @func.def)\n(class_declaration name: (name) @class.def)\n(interface_declaration name: (name) @class.def)\n(trait_declaration name: (name) @class.def)",
    ),
    ("hcl", "(block (string_lit) @class.def)"),
    ("bash", "(function_definition name: (word) @func.def)"),
    (
        "sql",
        "(create_table (object_reference name: (identifier) @class.def))\n(create_view (object_reference name: (identifier) @class.def))\n(create_function (object_reference name: (identifier) @func.def))",
    ),
    (
        "javascript",
        "(function_declaration name: (identifier) @func.def)\n(method_definition name: (property_identifier) @func.def)\n(class_declaration name: (identifier) @class.def)\n(variable_declarator name: (identifier) @func.def value: (arrow_function))",
    ),
    ("tsx", ""),
];

fn grammar(lang: &str) -> Option<Language> {
    Some(match lang {
        "python" => tree_sitter_python::LANGUAGE.into(),
        "swift" => tree_sitter_swift::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "kotlin" => tree_sitter_kotlin_ng::LANGUAGE.into(),
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "java" => tree_sitter_java::LANGUAGE.into(),
        "ruby" => tree_sitter_ruby::LANGUAGE.into(),
        "c" => tree_sitter_c::LANGUAGE.into(),
        "cpp" => tree_sitter_cpp::LANGUAGE.into(),
        // language_php parses whole .php files, text outside <?php ?> included.
        "php" => tree_sitter_php::LANGUAGE_PHP.into(),
        "hcl" => tree_sitter_hcl::LANGUAGE.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        "bash" => tree_sitter_bash::LANGUAGE.into(),
        "sql" => tree_sitter_sequel::LANGUAGE.into(),
        _ => return None,
    })
}

pub fn has_symbols(lang: &str) -> bool {
    QUERIES.iter().any(|(l, _)| *l == lang)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub kind: &'static str,
    pub line: i64,
}

/// Parsers and compiled queries, built once per scan rather than per file.
#[derive(Default)]
pub struct SymbolExtractor {
    cache: HashMap<&'static str, Option<(Parser, Query)>>,
}

impl SymbolExtractor {
    fn entry(&mut self, lang: &'static str) -> Option<&mut (Parser, Query)> {
        self.cache
            .entry(lang)
            .or_insert_with(|| {
                let query_lang = if lang == "tsx" { "typescript" } else { lang };
                let src = QUERIES.iter().find(|(l, _)| *l == query_lang)?.1;
                let language = grammar(lang)?;
                let mut parser = Parser::new();
                parser.set_language(&language).ok()?;
                let query = Query::new(&language, src).ok()?;
                Some((parser, query))
            })
            .as_mut()
    }

    /// Function/class/protocol definitions, grouped by capture name in the order
    /// a name first appears, each group in document order — as Python's
    /// `QueryCursor.captures()` dict comes out.
    pub fn extract(&mut self, path: &Path, lang: &'static str) -> Vec<Symbol> {
        let Ok(source) = std::fs::read(path) else {
            return Vec::new();
        };
        let Some((parser, query)) = self.entry(lang) else {
            return Vec::new();
        };
        let Some(tree) = parser.parse(&source, None) else {
            return Vec::new();
        };
        let names = query.capture_names();
        let mut groups: Vec<(u32, Vec<(usize, Symbol)>)> = Vec::new();
        let mut cursor = QueryCursor::new();
        let mut caps = cursor.captures(query, tree.root_node(), source.as_slice());
        while let Some((m, idx)) = caps.next() {
            let cap = m.captures[*idx];
            let cname = names[cap.index as usize];
            let kind = if cname.contains("protocol") {
                "protocol"
            } else if cname.contains("func") {
                "function"
            } else {
                "class"
            };
            let mut name = String::from_utf8_lossy(&source[cap.node.byte_range()]).into_owned();
            let quoted = name.len() > 1
                && name.starts_with(['"', '\''])
                && name.ends_with(name.chars().next().unwrap_or(' '));
            if lang == "hcl" && quoted {
                name = name[1..name.len() - 1].to_string();
            }
            let sym = Symbol {
                name,
                kind,
                line: cap.node.start_position().row as i64 + 1,
            };
            match groups.iter_mut().find(|(i, _)| *i == cap.index) {
                Some((_, g)) => g.push((cap.node.start_byte(), sym)),
                None => groups.push((cap.index, vec![(cap.node.start_byte(), sym)])),
            }
        }
        groups
            .into_iter()
            .flat_map(|(_, g)| g.into_iter().map(|(_, s)| s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_come_grouped_by_capture_in_first_appearance_order() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("order.py");
        std::fs::write(&p, "class First:\n    def method(self):\n        pass\n\ndef top():\n    pass\n\nclass Second:\n    pass\n").unwrap();
        let got: Vec<(String, &str, i64)> = SymbolExtractor::default()
            .extract(&p, "python")
            .into_iter()
            .map(|s| (s.name, s.kind, s.line))
            .collect();
        let want = [
            ("First", "class", 1),
            ("Second", "class", 8),
            ("method", "function", 2),
            ("top", "function", 5),
        ];
        assert_eq!(got, want.map(|(n, k, l)| (n.to_string(), k, l)));
    }

    #[test]
    fn every_grammar_compiles_its_query() {
        let mut ex = SymbolExtractor::default();
        for (lang, _) in QUERIES {
            assert!(ex.entry(lang).is_some(), "{lang}");
        }
    }

    #[test]
    fn suffix_is_pythons() {
        assert_eq!(suffix("a.min.js"), ".js");
        assert_eq!(suffix(".bashrc"), "");
        assert_eq!(suffix("Makefile"), "");
    }
}
