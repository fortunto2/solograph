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
    "rust": {
        "println", "eprintln", "format", "panic", "todo", "unimplemented",
        "vec", "assert", "assert_eq", "assert_ne", "dbg", "write", "writeln",
        "Some", "None", "Ok", "Err", "Box", "Arc", "Rc", "Vec", "String",
    },
    "go": {
        "Println", "Printf", "Sprintf", "Fprintf", "Errorf", "Fatal", "Fatalf",
        "Log", "Logf", "Panicf", "New", "Error", "make", "append", "len", "cap",
        "close", "delete", "copy", "panic", "recover",
    },
    "java": {
        "println", "printf", "format", "toString", "equals", "hashCode",
        "valueOf", "parseInt", "parseDouble", "getName", "getClass",
        "System", "String", "Integer", "Long", "Boolean", "List", "Map",
    },
    "ruby": {
        "puts", "print", "p", "pp", "raise", "require", "require_relative",
        "attr_reader", "attr_writer", "attr_accessor", "include", "extend",
        "new", "to_s", "to_i", "to_f", "nil?", "empty?", "each", "map", "select",
    },
    "c": {
        "printf", "fprintf", "sprintf", "snprintf", "scanf", "malloc", "calloc",
        "realloc", "free", "memcpy", "memset", "strlen", "strcmp", "strcpy",
        "assert", "exit", "abort", "sizeof",
    },
    "cpp": {
        "printf", "fprintf", "sprintf", "snprintf", "malloc", "calloc", "free",
        "memcpy", "memset", "strlen", "strcmp", "assert", "exit", "abort",
        "cout", "cerr", "endl", "move", "forward", "make_shared", "make_unique",
        "static_cast", "dynamic_cast", "reinterpret_cast",
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
        clean = module_text.strip("<>\"")
        if module_text.startswith('"'):
            return "internal", clean
        return "external", clean
    else:
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
