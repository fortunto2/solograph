"""Chunking YouTube transcripts along the video's own chapters.

Moved out of vectors/common.py when the embedding stack went: chunks are now
indexed for full-text search, and each still carries its chapter and start time
so a hit can open the video at that moment.
"""

# Chunk capacity range (min, max) in characters
CHUNK_CAPACITY = (200, 1500)


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
