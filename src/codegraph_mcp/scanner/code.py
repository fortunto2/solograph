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
}

# TSX uses same queries as TypeScript
_TS_FAMILY = {"typescript", "tsx"}

# Directories to skip during scan
SKIP_DIRS = {
    # VCS / env
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    # Build artifacts
    ".build", "DerivedData", "build", ".next", "dist", ".output",
    ".gradle", "Pods", ".eggs", ".tox", ".turbo", ".vercel", ".wrangler",
    # Test / coverage
    "coverage", ".nyc_output", "htmlcov", ".pytest_cache",
    # Samples / examples / vendor (library code, not project code)
    "samples", "Samples", "examples", "Examples", "react-samples",
    "vendor", "third_party", "third-party",
    # Generated / cache
    "generated", ".cache", ".parcel-cache", ".swc",
    # IDE
    ".idea", ".vscode",
}

# File patterns to skip
SKIP_FILES = {".DS_Store", "package-lock.json", "yarn.lock", "uv.lock"}


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
    }

    # tsx uses same queries as typescript
    query_lang = "typescript" if lang in _TS_FAMILY else lang
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


def ingest_files(graph, files: list[FileNode]) -> int:
    """Create File nodes and HAS_FILE edges."""
    count = 0
    for f in files:
        path_escaped = f.path.replace("'", "\\'")
        graph.query(
            f"MERGE (f:File {{path: '{path_escaped}', project: '{f.project}'}}) "
            f"SET f.lang = '{f.lang}', f.lines = {f.lines}"
        )
        graph.query(
            f"MATCH (p:Project {{name: '{f.project}'}}), "
            f"(f:File {{path: '{path_escaped}', project: '{f.project}'}}) "
            f"MERGE (p)-[:HAS_FILE]->(f)"
        )
        count += 1
    return count


def ingest_symbols(graph, symbols: list[SymbolNode]) -> int:
    """Create Symbol nodes and DEFINES edges."""
    count = 0
    for s in symbols:
        name_escaped = s.name.replace("'", "\\'")
        file_escaped = s.file.replace("'", "\\'")
        graph.query(
            f"MERGE (s:Symbol {{name: '{name_escaped}', project: '{s.project}', file: '{file_escaped}'}}) "
            f"SET s.kind = '{s.kind}', s.line = {s.line}"
        )
        graph.query(
            f"MATCH (f:File {{path: '{file_escaped}', project: '{s.project}'}}), "
            f"(s:Symbol {{name: '{name_escaped}', project: '{s.project}', file: '{file_escaped}'}}) "
            f"MERGE (f)-[:DEFINES]->(s)"
        )
        count += 1
    return count


# ── Deep analysis (--deep) ────────────────────────────────────────

# Builtins / noise to skip in CALLS extraction
NOISE_CALLS: dict[str, set[str]] = {
    "python": {
        "print", "len", "range", "int", "str", "float", "list", "dict", "set",
        "isinstance", "hasattr", "getattr", "type", "super", "enumerate", "zip",
        "sorted", "open", "bool", "tuple", "map", "filter", "any", "all", "min",
        "max", "abs", "repr", "id", "vars", "next", "iter", "reversed", "round",
    },
    "typescript": {
        "log", "parseInt", "parseFloat", "String", "Number", "Boolean",
        "Array", "Object", "Promise", "setTimeout", "require", "console",
        "Error", "Map", "Set", "JSON", "Date", "Math", "RegExp",
    },
    "swift": {"print", "fatalError", "precondition", "debugPrint", "assert"},
    "kotlin": {"println", "print", "listOf", "mapOf", "setOf", "arrayOf", "emptyList", "emptyMap"},
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
}


def _classify_import(module_text: str, lang: str) -> tuple[str, str]:
    """Classify an import as internal/external and return normalized module name.

    Returns (kind, module_name).
    """
    if lang == "python":
        if module_text.startswith("."):
            return "internal", module_text
        # Top-level package name for external
        return "external", module_text.split(".")[0]
    elif lang == "typescript":
        # Strip quotes from string captures
        clean = module_text.strip("'\"")
        if clean.startswith(".") or clean.startswith("/"):
            return "internal", clean
        # Package name: keep @scope/name
        parts = clean.split("/")
        if clean.startswith("@") and len(parts) >= 2:
            return "external", f"{parts[0]}/{parts[1]}"
        return "external", parts[0]
    else:
        # Swift, Kotlin: always external
        return "external", module_text.split(".")[0]


def extract_deep(
    file_path: Path, project_name: str, lang: str, rel_path: str = "",
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

    # tsx uses same queries/noise as typescript
    query_lang = "typescript" if lang in _TS_FAMILY else lang

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
                        imports.append(ImportEdge(
                            source_file=rel_path,
                            project=project_name,
                            module=module_name,
                            kind=kind,
                        ))
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
                        calls.append(CallEdge(
                            source_file=rel_path,
                            project=project_name,
                            callee_name=callee,
                        ))
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
                    inherits.append(InheritsEdge(
                        child_name=child_name,
                        parent_name=parent,
                        child_file=rel_path,
                        project=project_name,
                    ))
        except Exception:
            pass

    return imports, calls, inherits


def ingest_imports(graph, imports: list[ImportEdge]) -> tuple[int, int]:
    """Create IMPORTS edges. Returns (internal_count, external_count)."""
    internal = 0
    external = 0
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
    """Create CALLS edges (File → Symbol). Returns count of edges created."""
    count = 0
    for call in calls:
        src_escaped = call.source_file.replace("'", "\\'")
        callee_escaped = call.callee_name.replace("'", "\\'")
        try:
            result = graph.query(
                f"MATCH (src:File {{path: '{src_escaped}', project: '{call.project}'}}), "
                f"(sym:Symbol {{name: '{callee_escaped}', project: '{call.project}'}}) "
                f"WHERE src.path <> sym.file "
                f"MERGE (src)-[:CALLS]->(sym)"
            )
            if result.relationships_created > 0:
                count += result.relationships_created
        except Exception:
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
