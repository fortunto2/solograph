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


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Pure Python cosine similarity. No numpy needed."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = sum(a * a for a in vec1) ** 0.5
    n2 = sum(b * b for b in vec2) ** 0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


# Default topic list for zero-shot tagging
DEFAULT_TOPICS = [
    "AI agents and automation",
    "revenue model and pricing strategy",
    "community building and audience growth",
    "security and privacy",
    "hardware setup and infrastructure",
    "productivity and workflows",
    "marketing and growth hacking",
    "product development and MVP",
    "fundraising and investment",
    "solo founder and bootstrapping",
    "open source and self-hosting",
    "e-commerce and conversion optimization",
    "content creation and YouTube",
    "developer tools and coding",
    "market research and validation",
]


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


def get_text_splitter(capacity: tuple[int, int] | None = None):
    """Create a TextSplitter for plain text (transcripts, descriptions)."""
    from semantic_text_splitter import TextSplitter

    return TextSplitter(capacity or CHUNK_CAPACITY)


def parse_chapters(description: str) -> list[dict]:
    """Extract chapter markers from a YouTube description.

    Looks for lines like "0:00 Intro", "5:30 Revenue Model", "1:05:20 Q&A".
    Returns list of {title, start_time, start_seconds}. Empty if < 2 chapters found.
    """
    import re

    pattern = re.compile(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(description))

    if len(matches) < 2:
        return []

    chapters = []
    for m in matches:
        timecode = m.group(1).strip()
        title = m.group(2).strip()
        # Parse timecode to seconds
        parts = timecode.split(":")
        if len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            seconds = int(parts[0]) * 60 + int(parts[1])
        chapters.append({
            "title": title,
            "start_time": timecode,
            "start_seconds": seconds,
        })

    return chapters


def chunk_transcript_by_chapters(
    transcript: str,
    chapters: list[dict],
    duration_seconds: int,
    capacity: tuple[int, int] | None = None,
) -> list[dict]:
    """Split transcript into chunks aligned to chapter boundaries.

    Maps chapter timecodes to proportional character positions in the transcript,
    then sub-chunks with TextSplitter if a section exceeds max capacity.

    Returns list of {text, chapter, start_time, chunk_index}.
    """
    if not chapters or not transcript or duration_seconds <= 0:
        return []

    splitter = get_text_splitter(capacity)
    total_chars = len(transcript)
    cap = capacity or CHUNK_CAPACITY
    max_cap = cap[1]

    result = []
    chunk_index = 0

    for i, ch in enumerate(chapters):
        # Proportional start/end positions in text
        start_pos = int((ch["start_seconds"] / duration_seconds) * total_chars)
        if i + 1 < len(chapters):
            end_pos = int((chapters[i + 1]["start_seconds"] / duration_seconds) * total_chars)
        else:
            end_pos = total_chars

        start_pos = max(0, min(start_pos, total_chars))
        end_pos = max(start_pos, min(end_pos, total_chars))

        section = transcript[start_pos:end_pos].strip()
        if not section:
            continue

        # Sub-chunk if section is too large
        if len(section) > max_cap:
            sub_chunks = splitter.chunks(section)
        else:
            sub_chunks = [section]

        for sub in sub_chunks:
            if sub.strip():
                result.append({
                    "text": sub.strip(),
                    "chapter": ch["title"],
                    "start_time": ch["start_time"],
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

    return result


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
