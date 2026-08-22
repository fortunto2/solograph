"""Scan source code files → File and Symbol nodes.

Uses tree-sitter for AST parsing to extract function/class definitions and imports.
"""

from pathlib import Path

from ..models import CallEdge, FileNode, ImportEdge, InheritsEdge, SymbolNode

# Language extensions → tree-sitter grammar module
LANG_MAP = {
    ".py": "python",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".kt": "kotlin",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

# Languages that share query definitions with another language
_QUERY_ALIASES = {"tsx": "typescript"}

# Directories to skip during scan
SKIP_DIRS = {
    # VCS / env
    ".git",
    # Agent worktrees: a full second copy of the repo, checked out under
    # .claude/worktrees/. Measured 22 Aug 2026 on video-analyzer — 566 of
    # 1138 scanned files came from one abandoned worktree, so every symbol
    # in the project existed twice and search returned paths inside a dead
    # branch as if they were the code. A copy of the repo is not the repo,
    # for the same reason .git is not.
    ".claude",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    # Build artifacts
    ".build",
    "target",  # Rust: absent here only because CARGO_TARGET_DIR moves it
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
    # Test / coverage
    "coverage",
    ".nyc_output",
    "htmlcov",
    ".pytest_cache",
    # Samples / examples / vendor (library code, not project code)
    "samples",
    "Samples",
    "examples",
    "Examples",
    "react-samples",
    "vendor",
    "third_party",
    "third-party",
    # Generated / cache
    "generated",
    ".cache",
    ".parcel-cache",
    ".swc",
    # IDE
    ".idea",
    ".vscode",
}

# File patterns to skip
SKIP_FILES = {".DS_Store", "package-lock.json", "yarn.lock", "uv.lock"}

# Directories whose markdown is generated output, not authored knowledge.
#
# Markdown earns its place in the index for a different reason than code does: a
# CLAUDE.md or a skill file records a decision, and that is worth retrieving. An
# LLM-written report is neither decision nor code, and there are usually hundreds of
# them. Measured 22 Aug 2026 on epiphan/sgr-chat-agent: 781 files under reports/, all
# generated, and they buried the code so thoroughly that a question about a parser
# returned five reports at 15-17% relevance and no source at all. The same query
# against a repo without them scored 84%.
#
# Separate from SKIP_DIRS because these hold real content — it is just content the
# project produced, not content the project is.
SKIP_DOC_DIRS = {
    "reports",
    "baselines",
    "snapshots",
    "__snapshots__",
    "fixtures",
    "golden",
}


def _get_ts_language(lang: str):
    """Get tree-sitter Language object, handling typescript API differences.

    tree-sitter-typescript v0.23+ uses language_typescript()/language_tsx()
    instead of language(). Other grammars use language().
    """
    import importlib

    from tree_sitter import Language

    grammar_map = {
        "python": ("tree_sitter_python", "language"),
        "swift": ("tree_sitter_swift", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"),
        "kotlin": ("tree_sitter_kotlin", "language"),
        "rust": ("tree_sitter_rust", "language"),
        "go": ("tree_sitter_go", "language"),
        "java": ("tree_sitter_java", "language"),
        "ruby": ("tree_sitter_ruby", "language"),
        "c": ("tree_sitter_c", "language"),
        "cpp": ("tree_sitter_cpp", "language"),
    }

    if lang not in grammar_map:
        return None

    module_name, func_name = grammar_map[lang]
    grammar_mod = importlib.import_module(module_name)
    lang_func = getattr(grammar_mod, func_name)
    return Language(lang_func())


def scan_files(project_path: Path, project_name: str) -> list[FileNode]:
    """Scan project directory for source code files."""
    files = []
    for ext in LANG_MAP:
        for fp in project_path.rglob(f"*{ext}"):
            # Skip excluded dirs
            if any(part in SKIP_DIRS for part in fp.parts):
                continue
            if fp.name in SKIP_FILES:
                continue
            try:
                lines = fp.read_text(encoding="utf-8", errors="ignore").count("\n") + 1
            except Exception:
                lines = 0

            rel = str(fp.relative_to(project_path))
            files.append(
                FileNode(
                    path=rel,
                    project=project_name,
                    lang=LANG_MAP[ext],
                    lines=lines,
                )
            )
    return files


def extract_symbols(file_path: Path, project_name: str, lang: str, rel_path: str = "") -> list[SymbolNode]:
    """Extract function/class definitions from a file using tree-sitter."""
    try:
        from tree_sitter import Parser, Query, QueryCursor

        ts_lang = _get_ts_language(lang)
        if ts_lang is None:
            return []
        parser = Parser(ts_lang)
    except (ImportError, Exception):
        return []

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except Exception:
        return []

    rel_path = rel_path or str(file_path.name)
    symbols = []

    # Language-specific queries
    queries_by_lang = {
        "python": """
            (function_definition name: (identifier) @func.def)
            (class_definition name: (identifier) @class.def)
        """,
        "swift": """
            (function_declaration name: (simple_identifier) @func.def)
            (class_declaration name: (type_identifier) @class.def)
            (protocol_declaration name: (type_identifier) @protocol.def)
        """,
        "typescript": """
            (function_declaration name: (identifier) @func.def)
            (class_declaration name: (type_identifier) @class.def)
        """,
        "kotlin": """
            (function_declaration (identifier) @func.def)
            (class_declaration (identifier) @class.def)
            (object_declaration (identifier) @class.def)
        """,
        "rust": """
            (function_item name: (identifier) @func.def)
            (struct_item name: (type_identifier) @class.def)
            (enum_item name: (type_identifier) @class.def)
            (trait_item name: (type_identifier) @protocol.def)
        """,
        "go": """
            (function_declaration name: (identifier) @func.def)
            (method_declaration name: (field_identifier) @func.def)
            (type_declaration (type_spec name: (type_identifier) @class.def))
        """,
        "java": """
            (method_declaration name: (identifier) @func.def)
            (class_declaration name: (identifier) @class.def)
            (interface_declaration name: (identifier) @protocol.def)
            (enum_declaration name: (identifier) @class.def)
        """,
        "ruby": """
            (method name: (identifier) @func.def)
            (class name: (constant) @class.def)
            (module name: (constant) @class.def)
        """,
        "c": """
            (function_definition declarator: (function_declarator declarator: (identifier) @func.def))
            (struct_specifier name: (type_identifier) @class.def)
            (enum_specifier name: (type_identifier) @class.def)
        """,
        "cpp": """
            (function_definition declarator: (function_declarator declarator: (identifier) @func.def))
            (class_specifier name: (type_identifier) @class.def)
            (struct_specifier name: (type_identifier) @class.def)
            (enum_specifier name: (type_identifier) @class.def)
        """,
    }

    query_lang = _QUERY_ALIASES.get(lang, lang)
    query_str = queries_by_lang.get(query_lang)
    if not query_str:
        return []

    try:
        query = Query(ts_lang, query_str)
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        for capture_name, nodes in captures.items():
            kind = "function" if "func" in capture_name else "class"
            if "protocol" in capture_name:
                kind = "protocol"
            for node in nodes:
                symbols.append(
                    SymbolNode(
                        name=node.text.decode("utf-8"),
                        kind=kind,
                        file=rel_path,
                        project=project_name,
                        line=node.start_point[0] + 1,
                    )
                )
    except Exception:
        pass

    return symbols


# How many rows go to the graph in one query.
#
# The scan used to send two queries per node, one node at a time: 43,674
# symbols in video-analyzer meant ~87,000 round-trips, and the scan took
# 3m43s at 9% CPU — almost all of it waiting on the socket rather than
# parsing. Batched, the same project is a fraction of that. 500 keeps any
# single query small enough to stay well inside the driver's limits.
BATCH = 500

# A callee name defined more often than this is not resolvable by name, so no
# CALLS edge is written for it. See ingest_calls for the measurement.
AMBIGUOUS_NAME_LIMIT = 8


def _batched(rows: list, size: int = BATCH):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def ingest_files(graph, files: list[FileNode]) -> int:
    """Create File nodes and HAS_FILE edges, a batch at a time."""
    count = 0
    for chunk in _batched(files):
        rows = [{"path": f.path, "project": f.project, "lang": f.lang, "lines": f.lines} for f in chunk]
        # Parameters, not interpolation: a path or a symbol name carrying a
        # quote used to need escaping by hand, and one backslash short of
        # correct is a broken query rather than a wrong answer.
        graph.query(
            "UNWIND $rows AS r "
            "MERGE (f:File {path: r.path, project: r.project}) "
            "SET f.lang = r.lang, f.lines = r.lines",
            {"rows": rows},
        )
        graph.query(
            "UNWIND $rows AS r "
            "MATCH (p:Project {name: r.project}), "
            "(f:File {path: r.path, project: r.project}) "
            "MERGE (p)-[:HAS_FILE]->(f)",
            {"rows": rows},
        )
        count += len(chunk)
    return count


def ingest_symbols(graph, symbols: list[SymbolNode]) -> int:
    """Create Symbol nodes and DEFINES edges, a batch at a time."""
    count = 0
    for chunk in _batched(symbols):
        rows = [
            {
                "name": sym.name,
                "project": sym.project,
                "file": sym.file,
                "kind": sym.kind,
                "line": sym.line,
            }
            for sym in chunk
        ]
        graph.query(
            "UNWIND $rows AS r "
            "MERGE (s:Symbol {name: r.name, project: r.project, file: r.file}) "
            "SET s.kind = r.kind, s.line = r.line",
            {"rows": rows},
        )
        graph.query(
            "UNWIND $rows AS r "
            "MATCH (f:File {path: r.file, project: r.project}), "
            "(s:Symbol {name: r.name, project: r.project, file: r.file}) "
            "MERGE (f)-[:DEFINES]->(s)",
            {"rows": rows},
        )
        count += len(chunk)
    return count


# ── Deep analysis (--deep) ────────────────────────────────────────

# Builtins / noise to skip in CALLS extraction
NOISE_CALLS: dict[str, set[str]] = {
    "python": {
        "print",
        "len",
        "range",
        "int",
        "str",
        "float",
        "list",
        "dict",
        "set",
        "isinstance",
        "hasattr",
        "getattr",
        "type",
        "super",
        "enumerate",
        "zip",
        "sorted",
        "open",
        "bool",
        "tuple",
        "map",
        "filter",
        "any",
        "all",
        "min",
        "max",
        "abs",
        "repr",
        "id",
        "vars",
        "next",
        "iter",
        "reversed",
        "round",
    },
    "typescript": {
        "log",
        "parseInt",
        "parseFloat",
        "String",
        "Number",
        "Boolean",
        "Array",
        "Object",
        "Promise",
        "setTimeout",
        "require",
        "console",
        "Error",
        "Map",
        "Set",
        "JSON",
        "Date",
        "Math",
        "RegExp",
    },
    "swift": {"print", "fatalError", "precondition", "debugPrint", "assert"},
    "kotlin": {
        "println",
        "print",
        "listOf",
        "mapOf",
        "setOf",
        "arrayOf",
        "emptyList",
        "emptyMap",
    },
    "rust": {
        "println",
        "eprintln",
        "format",
        "panic",
        "todo",
        "unimplemented",
        "vec",
        "assert",
        "assert_eq",
        "assert_ne",
        "dbg",
        "write",
        "writeln",
        "Some",
        "None",
        "Ok",
        "Err",
        "Box",
        "Arc",
        "Rc",
        "Vec",
        "String",
    },
    "go": {
        "Println",
        "Printf",
        "Sprintf",
        "Fprintf",
        "Errorf",
        "Fatal",
        "Fatalf",
        "Log",
        "Logf",
        "Panicf",
        "New",
        "Error",
        "make",
        "append",
        "len",
        "cap",
        "close",
        "delete",
        "copy",
        "panic",
        "recover",
    },
    "java": {
        "println",
        "printf",
        "format",
        "toString",
        "equals",
        "hashCode",
        "valueOf",
        "parseInt",
        "parseDouble",
        "getName",
        "getClass",
        "System",
        "String",
        "Integer",
        "Long",
        "Boolean",
        "List",
        "Map",
    },
    "ruby": {
        "puts",
        "print",
        "p",
        "pp",
        "raise",
        "require",
        "require_relative",
        "attr_reader",
        "attr_writer",
        "attr_accessor",
        "include",
        "extend",
        "new",
        "to_s",
        "to_i",
        "to_f",
        "nil?",
        "empty?",
        "each",
        "map",
        "select",
    },
    "c": {
        "printf",
        "fprintf",
        "sprintf",
        "snprintf",
        "scanf",
        "malloc",
        "calloc",
        "realloc",
        "free",
        "memcpy",
        "memset",
        "strlen",
        "strcmp",
        "strcpy",
        "assert",
        "exit",
        "abort",
        "sizeof",
    },
    "cpp": {
        "printf",
        "fprintf",
        "sprintf",
        "snprintf",
        "malloc",
        "calloc",
        "free",
        "memcpy",
        "memset",
        "strlen",
        "strcmp",
        "assert",
        "exit",
        "abort",
        "cout",
        "cerr",
        "endl",
        "move",
        "forward",
        "make_shared",
        "make_unique",
        "static_cast",
        "dynamic_cast",
        "reinterpret_cast",
    },
}

# Tree-sitter queries for deep analysis per language
_IMPORT_QUERIES: dict[str, str] = {
    "python": """
        (import_statement name: (dotted_name) @import.module)
        (import_from_statement module_name: (dotted_name) @import.from)
        (import_from_statement module_name: (relative_import) @import.relative)
    """,
    "typescript": """
        (import_statement source: (string) @import.source)
    """,
    "swift": """
        (import_declaration (identifier) @import.module)
    """,
    "kotlin": """
        (import_header (identifier) @import.module)
    """,
    "rust": """
        (use_declaration argument: (scoped_identifier) @import.module)
        (use_declaration argument: (identifier) @import.module)
    """,
    "go": """
        (import_spec path: (interpreted_string_literal) @import.module)
    """,
    "java": """
        (import_declaration (scoped_identifier) @import.module)
    """,
    "ruby": """
        (call method: (identifier) @_method arguments: (argument_list (string (string_content) @import.module))
            (#match? @_method "^require"))
    """,
    "c": """
        (preproc_include path: [(system_lib_string) (string_literal)] @import.module)
    """,
    "cpp": """
        (preproc_include path: [(system_lib_string) (string_literal)] @import.module)
    """,
}

_CALL_QUERIES: dict[str, str] = {
    "python": """
        (call function: (identifier) @call.func)
        (call function: (attribute attribute: (identifier) @call.method))
    """,
    "typescript": """
        (call_expression function: (identifier) @call.func)
        (call_expression function: (member_expression property: (property_identifier) @call.method))
    """,
    "swift": """
        (call_expression (simple_identifier) @call.func)
    """,
    "kotlin": """
        (call_expression (simple_identifier) @call.func)
    """,
    "rust": """
        (call_expression function: (identifier) @call.func)
        (call_expression function: (field_expression field: (field_identifier) @call.method))
        (call_expression function: (scoped_identifier name: (identifier) @call.func))
    """,
    "go": """
        (call_expression function: (identifier) @call.func)
        (call_expression function: (selector_expression field: (field_identifier) @call.method))
    """,
    "java": """
        (method_invocation name: (identifier) @call.func)
    """,
    "ruby": """
        (call method: (identifier) @call.func)
    """,
    "c": """
        (call_expression function: (identifier) @call.func)
    """,
    "cpp": """
        (call_expression function: (identifier) @call.func)
        (call_expression function: (field_expression field: (field_identifier) @call.method))
    """,
}

_INHERIT_QUERIES: dict[str, str] = {
    "python": """
        (class_definition
            name: (identifier) @inherit.child
            superclasses: (argument_list (identifier) @inherit.parent))
    """,
    "typescript": """
        (class_declaration
            name: (type_identifier) @inherit.child
            (class_heritage (extends_clause (identifier) @inherit.parent)))
    """,
    "swift": """
        (class_declaration
            name: (type_identifier) @inherit.child
            (type_inheritance_clause (user_type (type_identifier) @inherit.parent)))
    """,
    "kotlin": """
        (class_declaration
            (type_identifier) @inherit.child
            (delegation_specifiers
                (delegation_specifier
                    (user_type (type_identifier) @inherit.parent))))
    """,
    "java": """
        (class_declaration
            name: (identifier) @inherit.child
            (superclass (type_identifier) @inherit.parent))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @inherit.child
            (base_class_clause (type_identifier) @inherit.parent))
    """,
    # Rust: trait impl is tracked via impl_item in symbols, not inheritance
    # Go: no class inheritance (composition via embedding)
    # Ruby: class Foo < Bar
    "ruby": """
        (class
            name: (constant) @inherit.child
            superclass: (superclass (constant) @inherit.parent))
    """,
    # C: no inheritance
}


def _classify_import(module_text: str, lang: str) -> tuple[str, str]:
    """Classify an import as internal/external and return normalized module name.

    Returns (kind, module_name).
    """
    if lang == "python":
        if module_text.startswith("."):
            return "internal", module_text
        return "external", module_text.split(".")[0]
    elif lang == "typescript":
        clean = module_text.strip("'\"")
        if clean.startswith(".") or clean.startswith("/"):
            return "internal", clean
        parts = clean.split("/")
        if clean.startswith("@") and len(parts) >= 2:
            return "external", f"{parts[0]}/{parts[1]}"
        return "external", parts[0]
    elif lang == "rust":
        # crate:: = internal, std/external crate = external
        if module_text.startswith("crate::") or module_text.startswith("self::") or module_text.startswith("super::"):
            return "internal", module_text
        return "external", module_text.split("::")[0]
    elif lang == "go":
        clean = module_text.strip('"')
        # Internal: no dots in path (relative packages in same module)
        # In practice, Go imports are all absolute — classify by known stdlib
        # Simple heuristic: if contains "." it's external (github.com/...), else stdlib
        if "." in clean:
            return "external", clean
        return "external", clean
    elif lang == "java":
        # Top-level package: java.*, javax.*, org.*, com.*
        return "external", module_text.split(".")[0]
    elif lang == "ruby":
        if module_text.startswith("./") or module_text.startswith("../"):
            return "internal", module_text
        return "external", module_text
    elif lang in ("c", "cpp"):
        clean = module_text.strip('<>"')
        if module_text.startswith('"'):
            return "internal", clean
        return "external", clean
    else:
        return "external", module_text.split(".")[0]


def extract_deep(
    file_path: Path,
    project_name: str,
    lang: str,
    rel_path: str = "",
) -> tuple[list[ImportEdge], list[CallEdge], list[InheritsEdge]]:
    """Extract imports, calls, and inheritance from a file (single parse).

    Returns (imports, calls, inherits) lists.
    """
    try:
        from tree_sitter import Parser, Query, QueryCursor

        ts_lang = _get_ts_language(lang)
        if ts_lang is None:
            return [], [], []
        parser = Parser(ts_lang)
    except (ImportError, Exception):
        return [], [], []

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except Exception:
        return [], [], []

    rel_path = rel_path or str(file_path.name)
    imports: list[ImportEdge] = []
    calls: list[CallEdge] = []
    inherits: list[InheritsEdge] = []

    query_lang = _QUERY_ALIASES.get(lang, lang)

    # ── Imports ──
    import_query_str = _IMPORT_QUERIES.get(query_lang)
    if import_query_str:
        try:
            query = Query(ts_lang, import_query_str)
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            seen_modules: set[str] = set()
            for _capture_name, nodes in captures.items():
                for node in nodes:
                    module_text = node.text.decode("utf-8")
                    kind, module_name = _classify_import(module_text, query_lang)
                    if module_name not in seen_modules:
                        seen_modules.add(module_name)
                        imports.append(
                            ImportEdge(
                                source_file=rel_path,
                                project=project_name,
                                module=module_name,
                                kind=kind,
                            )
                        )
        except Exception:
            pass

    # ── Calls ──
    call_query_str = _CALL_QUERIES.get(query_lang)
    noise = NOISE_CALLS.get(query_lang, set())
    if call_query_str:
        try:
            query = Query(ts_lang, call_query_str)
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            seen_calls: set[str] = set()
            for _capture_name, nodes in captures.items():
                for node in nodes:
                    callee = node.text.decode("utf-8")
                    if callee not in noise and callee not in seen_calls:
                        seen_calls.add(callee)
                        calls.append(
                            CallEdge(
                                source_file=rel_path,
                                project=project_name,
                                callee_name=callee,
                            )
                        )
        except Exception:
            pass

    # ── Inheritance ──
    inherit_query_str = _INHERIT_QUERIES.get(query_lang)
    if inherit_query_str:
        try:
            query = Query(ts_lang, inherit_query_str)
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            # Group child/parent pairs from captures
            children = [n.text.decode("utf-8") for n in captures.get("inherit.child", [])]
            parents = [n.text.decode("utf-8") for n in captures.get("inherit.parent", [])]
            # tree-sitter returns alternating child/parent for each class
            # For multi-inheritance, one child may have multiple parents
            # We pair them positionally
            for i, parent in enumerate(parents):
                # Find the child: last child index <= i
                child_idx = min(i, len(children) - 1)
                if child_idx >= 0:
                    # For Python: one class can have multiple parents
                    # children list has fewer entries than parents when multi-inherit
                    # Find the correct child by checking which child this parent belongs to
                    child_name = children[child_idx]
                    inherits.append(
                        InheritsEdge(
                            child_name=child_name,
                            parent_name=parent,
                            child_file=rel_path,
                            project=project_name,
                        )
                    )
        except Exception:
            pass

    return imports, calls, inherits


def ingest_imports(graph, imports: list[ImportEdge], rust_index: dict | None = None) -> tuple[int, int]:
    """Create IMPORTS edges. Returns (internal_count, external_count).

    `rust_index` comes from `rust_modules.build_index` and is what lets a
    `use crate::a::b` find the file it names. Without it Rust resolved to
    nothing at all — a module path is not a path fragment, which is the
    assumption the generic branch below is built on.
    """
    from .rust_modules import resolve as resolve_rust

    internal = 0
    external = 0
    resolved_pairs: list[tuple[str, str, str]] = []

    if rust_index:
        remaining = []
        for imp in imports:
            if imp.source_file.endswith(".rs"):
                target = resolve_rust(imp.module, imp.source_file, rust_index)
                if target:
                    resolved_pairs.append((imp.source_file, target, imp.project))
                    continue
                # An unresolved Rust path is either an external crate or a
                # module this scan did not see; fall through so the external
                # branch can still match it against a Package.
            remaining.append(imp)
        imports = remaining

    for chunk_start in range(0, len(resolved_pairs), 500):
        chunk = resolved_pairs[chunk_start : chunk_start + 500]
        rows = [{"src": a, "tgt": b, "project": c} for a, b, c in chunk]
        try:
            result = graph.query(
                "UNWIND $rows AS r "
                "MATCH (src:File {path: r.src, project: r.project}), "
                "(tgt:File {path: r.tgt, project: r.project}) "
                "MERGE (src)-[:IMPORTS]->(tgt)",
                {"rows": rows},
            )
            internal += result.relationships_created
        except Exception:
            pass

    for imp in imports:
        src_escaped = imp.source_file.replace("'", "\\'")
        module_escaped = imp.module.replace("'", "\\'")
        if imp.kind == "internal":
            # Internal: File → File (target path contains module name)
            # Convert relative import to path fragment (e.g. ".utils" → "utils")
            path_fragment = imp.module.lstrip(".").replace(".", "/")
            if not path_fragment:
                continue
            try:
                result = graph.query(
                    f"MATCH (src:File {{path: '{src_escaped}', project: '{imp.project}'}}), "
                    f"(tgt:File {{project: '{imp.project}'}}) "
                    f"WHERE tgt.path CONTAINS '{path_fragment}' AND src <> tgt "
                    f"MERGE (src)-[:IMPORTS]->(tgt)"
                )
                if result.relationships_created > 0:
                    internal += result.relationships_created
            except Exception:
                pass
        else:
            # External: File → Package
            try:
                result = graph.query(
                    f"MATCH (src:File {{path: '{src_escaped}', project: '{imp.project}'}}), "
                    f"(pkg:Package {{name: '{module_escaped}'}}) "
                    f"MERGE (src)-[:IMPORTS]->(pkg)"
                )
                if result.relationships_created > 0:
                    external += result.relationships_created
            except Exception:
                pass
    return internal, external


def ingest_calls(graph, calls: list[CallEdge]) -> int:
    """Create CALLS edges (File → Symbol), a batch at a time.

    This is where a deep scan spent its time. One query per call edge, and
    video-analyzer has 45,114 of them: the scan sat at 6% CPU for four
    minutes, waiting on the socket. The work per edge is a two-node match,
    so batching turns 45,000 round-trips into ninety.
    """
    # One row per distinct (file, callee): a file calling `len` forty times
    # is one edge, and the graph has no place to put the other thirty-nine.
    # Sending them anyway is what made a batched scan slower than the
    # per-edge loop it replaced — the work is in the two-node match, and it
    # was being done once per call site rather than once per pair.
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for c in calls:
        key = (c.source_file, c.callee_name, c.project)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    # Drop names that too many files define. A CALLS edge is matched on the
    # callee's *name*, so `X::new()` links to every `new` in the project:
    # measured on video-analyzer, `new` came back with 2548 calling files in
    # a project that has 573, and the edge count for one project was 790,284.
    # An edge that points at every candidate answers no question — "who calls
    # new" is not a question anybody can act on — while costing a two-node
    # match each. Rust's `new`/`default`/`clone` and Swift's protocol
    # witnesses are the whole of this: names defined once still link exactly.
    if unique:
        project = unique[0].project
        ambiguous = {
            row[0]
            for row in graph.query(
                "MATCH (s:Symbol {project: $p}) WITH s.name AS n, count(*) AS c WHERE c > $lim RETURN n",
                {"p": project, "lim": AMBIGUOUS_NAME_LIMIT},
            ).result_set
        }
        unique = [c for c in unique if c.callee_name not in ambiguous]

    count = 0
    for chunk in _batched(unique):
        rows = [{"src": c.source_file, "project": c.project, "callee": c.callee_name} for c in chunk]
        try:
            result = graph.query(
                "UNWIND $rows AS r "
                "MATCH (src:File {path: r.src, project: r.project}), "
                "(sym:Symbol {name: r.callee, project: r.project}) "
                "WHERE src.path <> sym.file "
                "MERGE (src)-[:CALLS]->(sym)",
                {"rows": rows},
            )
            count += result.relationships_created
        except Exception:
            # A batch that fails takes its edges with it, where a per-edge
            # loop lost only one. Deliberate: an edge is an optimisation for
            # search, and a scan that dies on one bad name is worse.
            pass
    return count


def ingest_inherits(graph, inherits: list[InheritsEdge]) -> int:
    """Create INHERITS edges (Symbol → Symbol). Returns count of edges created."""
    count = 0
    for inh in inherits:
        child_escaped = inh.child_name.replace("'", "\\'")
        parent_escaped = inh.parent_name.replace("'", "\\'")
        try:
            result = graph.query(
                f"MATCH (child:Symbol {{name: '{child_escaped}', project: '{inh.project}'}}) "
                f"MATCH (parent:Symbol {{name: '{parent_escaped}', project: '{inh.project}'}}) "
                f"WHERE child <> parent "
                f"MERGE (child)-[:INHERITS]->(parent)"
            )
            if result.relationships_created > 0:
                count += result.relationships_created
        except Exception:
            pass
    return count
