"""Shared utilities for vector indexes — embeddings, splitters, file scanning.

Embedding models:
  - Primary: MLX multilingual-e5-small-mlx (384 dim, Apple Silicon native, RU+EN)
  - Fallback: sentence-transformers all-MiniLM-L6-v2 (384 dim, any platform)

Both produce 384-dimensional vectors compatible with FalkorDB cosine similarity.
"""

from pathlib import Path

from ..scanner.code import LANG_MAP, TEXT_EXTENSIONS, project_files

VECTORS_ROOT = Path.home() / ".solo" / "vectors"

# Extensions for markdown docs
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}

# Chunk capacity range (min, max) in characters
CHUNK_CAPACITY = (200, 1500)

# Below this a chunk carries no retrievable meaning — see the note in
# ProjectGraphIndex._chunk_file for the measurement.
MIN_CHUNK_CHARS = 40

# Past this a file is data, not source, whatever its extension says. Measured
# 22 Aug 2026: a 2.9 MB tina-lock.json took 96.8s to split into 5,655 chunks, each of
# which then needs its own embedding, and one repo here commits a 48 MB index.json.
# Splitting is superlinear, so the cost is unbounded while the value is near zero —
# nobody searches a lockfile. Over the cap the file becomes one truncated chunk, which
# keeps it findable by name without paying for its contents.
MAX_CHUNKABLE_BYTES = 256_000

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
            from mlx_embeddings.utils import generate, load

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

    st_model = SentenceTransformer("intfloat/multilingual-e5-small")

    def st_embed(texts: list[str]) -> list[list[float]]:
        embeddings = st_model.encode(texts)
        return [[float(x) for x in emb] for emb in embeddings]

    return st_embed


def get_code_splitter(lang: str):
    """An AST-aware splitter for a language, or None if we have no grammar for it.

    This used to carry its own four-entry grammar map and call `mod.language()` on it.
    tree-sitter-typescript exposes `language_typescript`, not `language`, so the call
    raised and was swallowed — and every other language the symbol scanner knew about
    was simply missing from the map. Measured 22 Aug 2026: only python, swift and
    kotlin got AST chunks. TypeScript, Go, Rust, PHP, JavaScript and HCL fell through
    to `content[:1500]`, one chunk per file, everything past 1500 characters never
    indexed at all — 1011 .ts files in one repo here, 654 .go in another.

    So it now asks scanner.code, which already resolves each grammar's entry point for
    symbol extraction. One map, and a language cannot be searchable for symbols while
    silently unsearchable for text.
    """
    from semantic_text_splitter import CodeSplitter

    from ..scanner.code import get_grammar

    grammar = get_grammar(lang)
    if grammar is None:
        return None
    try:
        return CodeSplitter(grammar, CHUNK_CAPACITY)
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
        chapters.append(
            {
                "title": title,
                "start_time": timecode,
                "start_seconds": seconds,
            }
        )

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
                result.append(
                    {
                        "text": sub.strip(),
                        "chapter": ch["title"],
                        "start_time": ch["start_time"],
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

    return result


def chunk_segments_by_chapters(
    segments: list[dict],
    chapters: list[dict],
    duration_seconds: int,
    capacity: tuple[int, int] | None = None,
) -> list[dict]:
    """Split timestamped VTT segments into chunks aligned to chapter boundaries.

    Unlike chunk_transcript_by_chapters (which uses proportional text positions),
    this uses real per-segment timestamps from VTT for accurate start_seconds.

    Args:
        segments: [{start: float, text: str}, ...] from VTT parser
        chapters: [{title, start_time, start_seconds}, ...] from yt-dlp
        duration_seconds: total video duration
        capacity: (min_chars, max_chars) for chunk sizing

    Returns list of {text, chapter, start_time, start_seconds, chunk_index}.
    """
    if not segments:
        return []

    cap = capacity or CHUNK_CAPACITY
    max_cap = cap[1]
    splitter = get_text_splitter(capacity)

    # Build chapter boundaries as [(start_sec, end_sec, title, start_time)]
    ch_bounds = []
    if chapters:
        for i, ch in enumerate(chapters):
            ch_start = ch["start_seconds"]
            ch_end = chapters[i + 1]["start_seconds"] if i + 1 < len(chapters) else (duration_seconds or 99999)
            ch_bounds.append((ch_start, ch_end, ch["title"], ch.get("start_time", "")))
    else:
        # No chapters — single group
        ch_bounds.append((0, duration_seconds or 99999, "", "0:00"))

    # Group segments into chapters by their real timestamps
    chapter_groups = []
    for ch_start, ch_end, ch_title, ch_time in ch_bounds:
        group_segs = [s for s in segments if ch_start <= s["start"] < ch_end]
        if group_segs:
            chapter_groups.append((ch_title, ch_time, ch_start, group_segs))

    # Build chunks: merge segments within each chapter, sub-split if too large
    result = []
    chunk_index = 0

    for ch_title, ch_time, _ch_start_sec, group_segs in chapter_groups:
        # Merge all segment texts in this chapter group
        merged_text = " ".join(s["text"] for s in group_segs)

        if len(merged_text) <= max_cap:
            # Fits in one chunk — use first segment's timestamp
            result.append(
                {
                    "text": merged_text,
                    "chapter": ch_title,
                    "start_time": ch_time,
                    "start_seconds": group_segs[0]["start"],
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
        else:
            # Too large — split into sub-chunks, assign timestamps proportionally
            sub_texts = splitter.chunks(merged_text)
            # Map sub-chunk positions back to segment timestamps
            char_pos = 0
            for sub in sub_texts:
                sub = sub.strip()
                if not sub:
                    continue
                # Find the segment whose text starts at this position
                sub_start_sec = group_segs[0]["start"]  # default
                running = 0
                for s in group_segs:
                    seg_len = len(s["text"]) + 1  # +1 for join space
                    if running + seg_len > char_pos:
                        sub_start_sec = s["start"]
                        break
                    running += seg_len

                result.append(
                    {
                        "text": sub,
                        "chapter": ch_title,
                        "start_time": ch_time,
                        "start_seconds": sub_start_sec,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1
                char_pos += len(sub) + 1  # approximate

    return result


def scan_project_files(project_path: Path) -> list[tuple[Path, str]]:
    """Scan project directory for indexable files. Returns (abs_path, language)."""
    extended_lang_map = dict(LANG_MAP)
    for ext in DOC_EXTENSIONS:
        extended_lang_map[ext] = "markdown"
    # Only the vector lane takes these: the code graph indexes symbols, and there is
    # no grammar to get symbols from.
    extended_lang_map.update(TEXT_EXTENSIONS)
    # text_fallback: the vector lane can search anything with words in it. The
    # code graph cannot, so it does not pass this.
    return project_files(project_path, extended_lang_map, text_fallback=True)
