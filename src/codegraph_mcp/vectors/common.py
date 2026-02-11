"""Shared utilities for vector indexes — embeddings, splitters, file scanning.

Embedding models:
  - Primary: MLX multilingual-e5-small-mlx (384 dim, Apple Silicon native, RU+EN)
  - Fallback: sentence-transformers all-MiniLM-L6-v2 (384 dim, any platform)

Both produce 384-dimensional vectors compatible with FalkorDB cosine similarity.
"""

from pathlib import Path

from ..scanner.code import LANG_MAP, SKIP_DIRS, SKIP_FILES

VECTORS_ROOT = Path.home() / ".solo" / "vectors"

# Extensions for markdown docs
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}

# All scannable extensions (code + docs)
ALL_EXTENSIONS = set(LANG_MAP.keys()) | DOC_EXTENSIONS

# tree-sitter grammar module mapping
TS_GRAMMAR_MAP = {
    "python": "tree_sitter_python",
    "swift": "tree_sitter_swift",
    "typescript": "tree_sitter_typescript",
    "kotlin": "tree_sitter_kotlin",
}

# Chunk capacity range (min, max) in characters
CHUNK_CAPACITY = (200, 1500)

# Embedding dimension (both MLX and ST models use 384)
EMBEDDING_DIM = 384


def init_embedding_function(backend: str | None = None):
    """Initialize embedding function. Returns callable: list[str] -> list[list[float]].

    Auto-detects Apple Silicon for MLX, falls back to sentence-transformers.
    Override with backend="mlx" or backend="st".
    """
    import platform

    use_mlx = False
    if backend == "mlx":
        use_mlx = True
    elif backend == "st":
        use_mlx = False
    elif backend is None:
        use_mlx = platform.machine() == "arm64" and platform.system() == "Darwin"

    if use_mlx:
        try:
            from mlx_embeddings.utils import load, generate

            model, tokenizer = load("mlx-community/multilingual-e5-small-mlx")

            def mlx_embed(texts: list[str]) -> list[list[float]]:
                embeddings = []
                for text in texts:
                    result = generate(model, tokenizer, text)
                    embeddings.append(result.text_embeds[0].tolist())
                return embeddings

            return mlx_embed
        except Exception:
            pass

    # Sentence Transformers fallback
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def st_embed(texts: list[str]) -> list[list[float]]:
        embeddings = st_model.encode(texts)
        return [[float(x) for x in emb] for emb in embeddings]

    return st_embed


def get_code_splitter(lang: str):
    """Create a CodeSplitter for the given language."""
    import importlib

    from semantic_text_splitter import CodeSplitter

    grammar_module = TS_GRAMMAR_MAP.get(lang)
    if not grammar_module:
        return None
    try:
        mod = importlib.import_module(grammar_module)
        return CodeSplitter(mod.language(), CHUNK_CAPACITY)
    except Exception:
        return None


def get_markdown_splitter():
    """Create a MarkdownSplitter."""
    from semantic_text_splitter import MarkdownSplitter

    return MarkdownSplitter(CHUNK_CAPACITY)


def scan_project_files(project_path: Path) -> list[tuple[Path, str]]:
    """Scan project directory for indexable files. Returns (abs_path, language)."""
    extended_lang_map = dict(LANG_MAP)
    for ext in DOC_EXTENSIONS:
        extended_lang_map[ext] = "markdown"

    files = []
    for ext, lang in extended_lang_map.items():
        for fp in project_path.rglob(f"*{ext}"):
            if any(part in SKIP_DIRS for part in fp.parts):
                continue
            if fp.name in SKIP_FILES:
                continue
            files.append((fp, lang))
    return files
