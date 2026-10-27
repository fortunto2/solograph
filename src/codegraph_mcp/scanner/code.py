"""Find a project's source files and the definitions in them.

tree-sitter parses each file for function, class and protocol definitions. Calls,
imports and inheritance were extracted here too until 27 Oct 2026, for a graph
that got no structural questions in four months, and Grep answered those same
questions as well; portfolio.py keeps names and places only.
"""

from pathlib import Path

from ..models import SymbolNode

# Language extensions → tree-sitter grammar module
LANG_MAP = {
    ".py": "python",
    # Added 22 Aug 2026 after counting where commits actually land in the Epiphan set:
    # of the four repos with the most commits in 90 days, three were invisible here.
    # epiphan-marketing-infrastructure is 53 .tf + 12 .hcl and got 93 commits;
    # legacy_crm_php is 681 .php and got 89; .js/.mjs covers 105 files in legacy_crm_php
    # and the harness tooling. A language nobody can search is a repo nobody can search.
    ".php": "php",
    # .inc is PHP in the legacy CRM here — 18 tracked files of real includes that
    # looked like data because of the extension.
    ".inc": "php",
    ".tf": "hcl",
    ".hcl": "hcl",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # Added 22 Aug 2026 after counting every tracked extension across the set. These
    # are not exotic: 134 .html and 37 .mustache of CRM templates, 66 .yml/.yaml of CI
    # and Ansible, 31 .sql, 23 .sh. Config and templates are where half the answers to
    # an infra question live, and they were invisible.
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".sql": "sql",
    ".toml": "toml",
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

# Extensions with no tree-sitter grammar on PyPI that are still worth searching.
# They carry real content — 79 .astro pages and 37 .mustache templates in this set —
# and a file nobody can search is a file nobody knows exists. They get text chunking
# and no symbols, which is honest: we can find the words, not the structure.
TEXT_EXTENSIONS = {
    ".astro": "text",
    ".mustache": "text",
    ".hbs": "text",
    ".twig": "text",
    ".jinja": "text",
    ".j2": "text",
    ".xml": "text",
    ".tmpl": "text",
    ".tfvars": "text",
}

# Everything else git tracks that we have no grammar for. An allowlist was already
# stale the day after it was written — .tfvars, .tmpl and .xml were tracked and
# missing — and the next template language will do it again. Safe only because the
# file universe is git-tracked: a repo's own text, not node_modules.
TEXT_FALLBACK = "text"

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
SKIP_FILES = {".DS_Store", "package-lock.json"}

# Suffixes that mark a file as build output or a vendored library rather than source.
# Measured 22 Aug 2026 on epiphan/legacy_crm_php: canvasjs.min.js (488 KB), jquery and
# d3 sat in the index and a search for "how a license serial is validated" returned a
# jQuery slideshow plugin as its top hit. One minified line is one enormous chunk that
# embeds close to everything and matches nothing in particular.
SKIP_SUFFIXES = (".map", ".lock")

# `.min.` anywhere in the name, not just as a suffix: the first file this rule was
# written against is `slides.min.jquery.js`, where the minified marker sits in the
# middle. Same for bundles.
SKIP_MARKERS = (".min.", ".bundle.", "-min.", ".pack.", "-lock.")

# Never worth reading. The text fallback sends anything tracked with an extension to
# the chunker, which reads and UTF-8-decodes the whole file before a control-character
# heuristic rejects it — so a committed PDF or font is read in full to be thrown away.
BINARY_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".webp",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
    ".docx",
    ".xlsx",
    ".pptx",
    ".jar",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".pyc",
    ".pem",
)


def tracked_files(project_path: Path) -> list[Path] | None:
    """The files git tracks in this project, or None when it is not a checkout.

    This replaces guessing. SKIP_DIRS can only ever list the build and vendor
    directories somebody has already been bitten by, and the list is never finished —
    every new toolchain invents another one. What a project considers its own source
    is already written down, in .gitignore, and git will read it for us.

    Measured 22 Aug 2026 across a client set: sgr-chat-agent tracks 3,143 files and
    holds 113,207 on disk; epiphan-web tracks 822 and holds 83,725. Two orders of
    magnitude of node_modules, .next and vendored assets that no skip list caught.

    SKIP_DOC_DIRS and SKIP_MARKERS still apply on top, because generated reports and
    vendored minified libraries are frequently committed on purpose.
    """
    import subprocess

    try:
        r = subprocess.run(
            ["git", "-C", str(project_path), "ls-files", "-z"],
            capture_output=True,
            timeout=60,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return [project_path / name for name in r.stdout.decode("utf-8", "replace").split("\0") if name]


def project_files(project_path: Path, extensions, text_fallback: bool = False) -> list[tuple[Path, str]]:
    """(path, language) for every source file in a project, git-tracked when possible.

    `extensions` maps suffix to language name. Falls back to walking the tree with
    SKIP_DIRS when the project is not a git checkout, which is the only case where
    guessing is still necessary.
    """

    def keep(fp: Path, lang: str) -> bool:
        """One filter chain. Written twice it drifted within a day — the git branch
        checked is_file() and the walk branch did not."""
        # SKIP_DIRS applies to tracked files too, and that is not redundant with git:
        # the two answer different questions. Git says whether a file ships with the
        # project; SKIP_DIRS says whether it is the project's own code. A vendored Go
        # module is both — edge-mcp commits 675 files of golang.org/x, and a search for
        # a retry loop came back with zerrors_linux.go.
        if any(part in SKIP_DIRS for part in fp.parts):
            return False
        if is_skipped_file(fp):
            return False
        if lang == "markdown" and any(part in SKIP_DOC_DIRS for part in fp.parts):
            return False
        return fp.is_file()

    tracked = tracked_files(project_path)
    if tracked is not None:
        out = []
        for fp in tracked:
            lang = extensions.get(fp.suffix)
            if lang is None and text_fallback and fp.suffix:
                lang = TEXT_FALLBACK
            if lang and keep(fp, lang):
                out.append((fp, lang))
        return out

    return [
        (fp, lang)
        for ext, lang in extensions.items()
        for fp in project_path.rglob(f"*{ext}")
        if not any(part in SKIP_DIRS for part in fp.parts) and keep(fp, lang)
    ]


def is_skipped_file(path: Path) -> bool:
    """One predicate for both scanners, so the code graph and the vector index cannot
    disagree about what counts as source."""
    name = path.name
    return name in SKIP_FILES or name.endswith(SKIP_SUFFIXES) or any(m in name for m in SKIP_MARKERS)


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


# language name -> (python package, entry-point function). The entry point differs per
# package — tree-sitter-typescript has language_typescript and language_tsx, tree-sitter-php
# has language_php — which is exactly why this must exist once and be imported, not
# retyped somewhere else with `mod.language()` assumed.
GRAMMAR_MAP = {
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
    # language_php parses full .php files (text plus <?php ... ?>); language_php_only
    # would reject everything outside the tags, which is most of a legacy CRM file.
    "php": ("tree_sitter_php", "language_php"),
    "hcl": ("tree_sitter_hcl", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "bash": ("tree_sitter_bash", "language"),
    "sql": ("tree_sitter_sql", "language"),
}


def get_grammar(lang: str):
    """The raw grammar pointer from the tree-sitter package for a language.

    Separate from get_ts_language because the two consumers want different things:
    tree_sitter.Parser needs the wrapped Language.
    """
    import importlib

    if lang not in GRAMMAR_MAP:
        return None
    module_name, func_name = GRAMMAR_MAP[lang]
    try:
        return getattr(importlib.import_module(module_name), func_name)()
    except Exception:
        return None


def get_ts_language(lang: str):
    """The grammar wrapped for tree_sitter.Parser. See get_grammar for the raw pointer."""
    from tree_sitter import Language

    grammar = get_grammar(lang)
    return Language(grammar) if grammar is not None else None


# Which languages contribute symbols, and the tree-sitter query that finds them.
# Module level, not rebuilt per file: a full scan calls extract_symbols thousands
# of times and was reconstructing this dict every one of them.
#
# A grammar without an entry here is deliberate — html tags, yaml keys, json keys
# and css selectors are not symbols, and putting them in a code graph is noise.
# Those languages still get AST chunking; SYMBOL_LANGS is what says so.
SYMBOL_QUERIES = {
    "python": """
            (function_definition name: (identifier) @func.def)
            (class_definition name: (identifier) @class.def)
        """,
    "swift": """
            (function_declaration name: (simple_identifier) @func.def)
            (class_declaration name: (type_identifier) @class.def)
            (protocol_declaration name: (type_identifier) @protocol.def)
        """,
    # method_definition and arrow-function declarators were missing until 22 Aug
    # 2026, so a class method and `const handler = () => {}` — most of a modern TS
    # codebase — produced no symbol at all. The javascript block below captures
    # them; the two cannot be aliased because TS names classes with type_identifier
    # and JS with identifier.
    "typescript": """
            (function_declaration name: (identifier) @func.def)
            (method_definition name: (property_identifier) @func.def)
            (class_declaration name: (type_identifier) @class.def)
            (variable_declarator name: (identifier) @func.def value: (arrow_function))
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
    "php": """
            (function_definition name: (name) @func.def)
            (method_declaration name: (name) @func.def)
            (class_declaration name: (name) @class.def)
            (interface_declaration name: (name) @class.def)
            (trait_declaration name: (name) @class.def)
        """,
    # HCL has no functions. Its unit is the labelled block, and both labels are
    # worth a symbol: `resource "aws_s3_bucket" "telemetry"` should be findable by
    # the type and by the name, because people search for either.
    "hcl": """
            (block (string_lit) @class.def)
        """,
    "bash": """
            (function_definition name: (word) @func.def)
        """,
    "sql": """
            (create_table (object_reference name: (identifier) @class.def))
            (create_view (object_reference name: (identifier) @class.def))
            (create_function (object_reference name: (identifier) @func.def))
        """,
    "javascript": """
            (function_declaration name: (identifier) @func.def)
            (method_definition name: (property_identifier) @func.def)
            (class_declaration name: (identifier) @class.def)
            (variable_declarator name: (identifier) @func.def value: (arrow_function))
        """,
}

SYMBOL_LANGS = frozenset(SYMBOL_QUERIES) | set(_QUERY_ALIASES)


def extract_symbols(file_path: Path, project_name: str, lang: str, rel_path: str = "") -> list[SymbolNode]:
    """Extract function/class definitions from a file using tree-sitter."""
    # Before building a parser: html, css, yaml, toml and json have grammars for
    # chunking but contribute no symbols, and parsing ~250 files per scan only to
    # return [] is work nobody asked for.
    query_lang = _QUERY_ALIASES.get(lang, lang)
    query_str = SYMBOL_QUERIES.get(query_lang)
    if not query_str:
        return []

    try:
        from tree_sitter import Parser, Query, QueryCursor

        ts_lang = get_ts_language(lang)
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

    try:
        query = Query(ts_lang, query_str)
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        for capture_name, nodes in captures.items():
            kind = "function" if "func" in capture_name else "class"
            if "protocol" in capture_name:
                kind = "protocol"
            for node in nodes:
                name = node.text.decode("utf-8")
                # HCL block labels are string_lit nodes, so their text arrives quoted,
                # and "crm_prod_public_ip" does not match a search for
                # crm_prod_public_ip. Scoped to hcl: every other grammar captures bare
                # identifiers, and stripping there would silently eat a real character.
                if query_lang == "hcl" and len(name) > 1 and name[0] == name[-1] and name[0] in "\"'":
                    name = name[1:-1]
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=kind,
                        file=rel_path,
                        project=project_name,
                        line=node.start_point[0] + 1,
                    )
                )
    except Exception:
        pass

    return symbols
