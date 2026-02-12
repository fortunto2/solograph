"""Index YouTube video transcripts into per-source FalkorDB vectors.

Videos are stored as Video -> HAS_CHUNK -> VideoChunk graph structure.
Transcripts are split by chapter boundaries (from description) or by TextSplitter.
Descriptions are indexed as separate chunks for link/resource discovery.

Usage via CLI:
    solograph-cli index-youtube                            # All channels from channels.yaml
    solograph-cli index-youtube -c GregIsenberg             # One channel
    solograph-cli index-youtube -n 10                       # Last N videos per channel
    solograph-cli index-youtube --import-file FILE           # Import pre-processed data
    solograph-cli index-youtube --backend st                 # Force sentence-transformers

Requires: SearXNG tunnel active for transcript fetching, yt-dlp for metadata.
"""

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import httpx
import yaml

# Persistent VTT storage
VTT_DIR = Path.home() / ".solo" / "sources" / "youtube" / "vtt"

# Default channels.yaml location (symlinked from solopreneur)
DEFAULT_CHANNELS_PATH = Path.home() / ".solo" / "sources" / "youtube" / "channels.yaml"


# ---------------------------------------------------------------------------
# VTT parsing
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")
_TAG_RE = re.compile(r"<[^>]+>")


def ts_to_sec(ts: str) -> float:
    """Parse VTT timestamp (HH:MM:SS.mmm) to seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def vtt_to_text(vtt: str) -> str:
    """Convert WEBVTT subtitle content to plain text.

    Deduplicates repeated lines (YouTube auto-subs have rolling duplicates),
    strips timestamps, tags, and alignment markers.
    """
    lines = []
    seen: set[str] = set()
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        lines.append(clean)
    return " ".join(lines)


def parse_vtt_segments(vtt_text: str) -> list[dict]:
    """Parse VTT into timestamped segments: [{start: float, text: str}, ...].

    YouTube auto-subs have snapshot blocks (~10ms duration) with final text.
    We extract those for clean, non-overlapping segments.
    Falls back to all blocks if no snapshots found.
    """
    blocks = vtt_text.split("\n\n")
    segments: list[dict] = []
    prev_text = ""

    for block in blocks:
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _TS_RE.match(line.strip())
            if m:
                start = ts_to_sec(m.group(1))
                end = ts_to_sec(m.group(2))
                if (end - start) < 0.05:  # snapshot block
                    text_lines = [
                        _TAG_RE.sub("", l).strip()
                        for l in lines[i + 1:]
                        if l.strip()
                    ]
                    text = " ".join(text_lines)
                    if text and text != prev_text:
                        segments.append({"start": round(start, 1), "text": text})
                        prev_text = text
                break

    # Fallback: no snapshot blocks found
    if not segments:
        prev_text = ""
        for block in blocks:
            lines = block.strip().split("\n")
            for i, line in enumerate(lines):
                m = _TS_RE.match(line.strip())
                if m:
                    start = ts_to_sec(m.group(1))
                    text_lines = [
                        _TAG_RE.sub("", l).strip()
                        for l in lines[i + 1:]
                        if l.strip()
                    ]
                    text = " ".join(text_lines)
                    if text and text != prev_text:
                        segments.append({"start": round(start, 1), "text": text})
                        prev_text = text
                    break

    return segments


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# SearXNG helpers
# ---------------------------------------------------------------------------


def check_searxng(searxng_url: str) -> bool:
    """Check if SearXNG tunnel is active."""
    try:
        resp = httpx.get(f"{searxng_url}/health", timeout=3)
        return resp.is_success
    except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
        return False


def search_youtube_videos(searxng_url: str, channel_handle: str, limit: int = 10) -> list[dict]:
    """Discover videos via SearXNG youtube engine."""
    try:
        resp = httpx.post(
            f"{searxng_url}/search",
            json={
                "query": f"@{channel_handle}",
                "max_results": limit,
                "engines": "youtube",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except httpx.HTTPError as e:
        print(f"  SearXNG search error: {e}", file=sys.stderr)
        return []


def fetch_transcript(searxng_url: str, video_id: str) -> str | None:
    """Fetch video transcript via SearXNG /transcript endpoint."""
    try:
        resp = httpx.post(
            f"{searxng_url}/transcript",
            json={"video_id": video_id},
            timeout=30,
        )
        if resp.is_success:
            data = resp.json()
            return data.get("transcript", "")
    except httpx.HTTPError:
        pass
    return None


# ---------------------------------------------------------------------------
# yt-dlp metadata + VTT
# ---------------------------------------------------------------------------


def check_ytdlp() -> bool:
    """Check if yt-dlp is available on PATH."""
    return shutil.which("yt-dlp") is not None


def fetch_video_metadata(video_id: str) -> dict:
    """Fetch video metadata + transcript via yt-dlp.

    Two yt-dlp calls: -j for metadata, --skip-download --write-auto-sub for transcript.
    Returns dict with keys: title, channel, channel_handle, description, duration, chapters, tags, transcript, segments, upload_date.
    Falls back to empty values on failure.
    """
    import json
    import subprocess
    import tempfile

    url = f"https://www.youtube.com/watch?v={video_id}"
    empty = {"title": "", "channel": "", "channel_handle": "", "description": "", "duration": 0, "chapters": [], "tags": [], "transcript": "", "segments": [], "upload_date": ""}

    # 1. Metadata via --dump-json
    try:
        result = subprocess.run(
            ["yt-dlp", "-j", "--no-download", url],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return empty
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return empty

    title = data.get("title") or ""
    channel = data.get("channel") or ""
    channel_handle = (data.get("uploader_id") or "").lstrip("@") or channel
    description = (data.get("description") or "")[:5000]
    duration = int(data.get("duration") or 0)
    tags = data.get("tags") or []
    raw_date = data.get("upload_date") or ""
    upload_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else ""

    # Convert yt-dlp chapters to our format
    chapters = []
    for ch in data.get("chapters") or []:
        secs = int(ch.get("start_time", 0))
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        timecode = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        chapters.append({
            "title": ch.get("title", ""),
            "start_time": timecode,
            "start_seconds": secs,
        })

    # 2. Transcript via VTT — reuse cached file or download fresh
    transcript = ""
    vtt_raw = ""
    VTT_DIR.mkdir(parents=True, exist_ok=True)
    saved_vtt = VTT_DIR / f"{video_id}.vtt"

    if saved_vtt.exists():
        vtt_raw = saved_vtt.read_text(encoding="utf-8")
        transcript = vtt_to_text(vtt_raw)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = f"{tmpdir}/vid"
            try:
                subprocess.run(
                    ["yt-dlp", "--write-auto-sub", "--write-sub",
                     "--sub-lang", "en",
                     "--sub-format", "vtt", "--skip-download",
                     "--no-warnings", "-o", out_path, url],
                    capture_output=True, text=True, timeout=30,
                )
                for suffix in [".en.vtt", ".en-orig.vtt", ".en-US.vtt"]:
                    p = Path(f"{out_path}{suffix}")
                    if p.exists():
                        vtt_raw = p.read_text(encoding="utf-8")
                        break
                else:
                    for f in Path(tmpdir).glob("vid*.vtt"):
                        vtt_raw = f.read_text(encoding="utf-8")
                        break
            except (subprocess.TimeoutExpired, Exception):
                pass

        if vtt_raw:
            saved_vtt.write_text(vtt_raw, encoding="utf-8")
            transcript = vtt_to_text(vtt_raw)

    segments = parse_vtt_segments(vtt_raw) if vtt_raw else []

    return {
        "title": title,
        "channel": channel,
        "channel_handle": channel_handle,
        "description": description,
        "duration": duration,
        "chapters": chapters,
        "tags": tags,
        "transcript": transcript,
        "segments": segments,
        "upload_date": upload_date,
    }


# ---------------------------------------------------------------------------
# Channel loading
# ---------------------------------------------------------------------------


def load_channels(channels_path: Path | None = None) -> list[dict]:
    """Load channel handles from channels.yaml.

    Returns list of {"name": handle, "handle": handle}.
    """
    path = channels_path or DEFAULT_CHANNELS_PATH
    if not path.exists():
        print(f"channels.yaml not found at {path}", file=sys.stderr)
        print("Create it or symlink: ln -sf /path/to/channels.yaml ~/.solo/sources/youtube/channels.yaml", file=sys.stderr)
        return []

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    handles = data.get("channels", [])
    return [{"name": h, "handle": h} for h in handles]


# ---------------------------------------------------------------------------
# Import mode (flat SourceDoc — legacy, for pre-processed data without video IDs)
# ---------------------------------------------------------------------------


def import_file(filepath: str, idx, dry_run: bool = False) -> int:
    """Import pre-processed video summaries from a markdown file."""
    from ..models import SourceDoc

    path = Path(filepath).expanduser()
    if not path.exists():
        print(f"File not found: {filepath}", file=sys.stderr)
        return 0

    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n---\n", text)
    indexed = 0

    for section in sections:
        section = section.strip()
        if not section or len(section) < 200:
            continue

        title_match = re.search(r"^##\s*\d+\.\s*(.+)", section, re.MULTILINE)
        if not title_match:
            continue

        title = title_match.group(1).strip()
        doc_id = hashlib.md5(f"yt-import:{title}".encode()).hexdigest()

        if not dry_run and idx.exists("youtube", doc_id):
            print(f"  Exists: {title}")
            continue

        embed_text = f"{title}\n\n{section[:3000]}"
        embedding = idx.embed([embed_text])[0]

        if dry_run:
            print(f"  [DRY] {title}")
            indexed += 1
            continue

        doc = SourceDoc(
            doc_id=doc_id,
            source_type="youtube-transcript",
            source_name="ycombinator",
            title=title,
            content=section[:500],
            url=f"import:{path.name}#{title}",
            created="",
            tags="ycombinator,youtube",
            embed_text=embed_text,
        )
        idx.upsert_one("youtube", doc, embedding=embedding)
        indexed += 1
        print(f"  Indexed: {title}")

    return indexed


# ---------------------------------------------------------------------------
# YouTubeIndexer
# ---------------------------------------------------------------------------


class YouTubeIndexer:
    """Index YouTube video transcripts into FalkorDB source vectors."""

    def __init__(
        self,
        channels_path: Path | None = None,
        backend: str | None = None,
        searxng_url: str | None = None,
    ):
        self.channels_path = channels_path
        self.backend = backend
        self.searxng_url = searxng_url or os.environ.get("TAVILY_API_URL", "http://localhost:8013")

    def run(
        self,
        channels: list[str] | None = None,
        limit: int = 10,
        dry_run: bool = False,
    ) -> dict:
        """Main indexing loop. Returns stats dict."""
        from ..models import VideoChapter, VideoDoc
        from ..vectors.common import (
            chunk_segments_by_chapters,
            chunk_transcript_by_chapters,
            get_text_splitter,
            parse_chapters,
        )
        from ..vectors.source_index import SourceIndex

        print("Loading embedding model...")
        idx = SourceIndex(backend=self.backend)

        # Check SearXNG
        if not check_searxng(self.searxng_url):
            print(f"SearXNG not reachable at {self.searxng_url}", file=sys.stderr)
            print("Start tunnel or set TAVILY_API_URL env var.", file=sys.stderr)
            sys.exit(1)

        # Check yt-dlp
        if not check_ytdlp():
            print("yt-dlp not found. Install: brew install yt-dlp", file=sys.stderr)
            sys.exit(1)

        # Resolve channel list
        if channels:
            channel_list = [{"name": h, "handle": h} for h in channels]
        else:
            channel_list = load_channels(self.channels_path)

        if not channel_list:
            print("No channels found. Use -c handle or create channels.yaml")
            sys.exit(1)

        print(f"Channels: {', '.join(c['handle'] for c in channel_list)}")
        print(f"Limit: {limit} videos/channel")

        total_indexed = 0
        total_chunks = 0
        total_skipped_exists = 0
        total_skipped_no_transcript = 0
        total_with_chapters = 0

        splitter = get_text_splitter()

        for ch in channel_list:
            print(f"\n{'='*60}")
            print(f"Channel: @{ch['handle']} ({ch['name']})")
            print(f"{'='*60}")

            videos = search_youtube_videos(self.searxng_url, ch["handle"], limit=limit)
            print(f"  Found {len(videos)} videos")

            for video in videos:
                url = video.get("url", "")
                title = video.get("title", "Untitled")
                video_id = extract_video_id(url)

                if not video_id:
                    continue

                # Check if already indexed (Video node)
                if not dry_run and idx.video_exists("youtube", video_id):
                    total_skipped_exists += 1
                    continue

                print(f"  Processing: {title[:60]}...")

                # 1. Fetch all metadata + transcript via yt-dlp
                meta = fetch_video_metadata(video_id)
                transcript = meta["transcript"]
                segments = meta.get("segments", [])
                description = meta["description"]
                duration_seconds = meta["duration"]
                chapters_raw = meta["chapters"]
                yt_tags = meta["tags"]
                upload_date = meta.get("upload_date", "")

                if not transcript or len(transcript) < 200:
                    transcript = fetch_transcript(self.searxng_url, video_id) or ""

                if not transcript or len(transcript) < 200:
                    total_skipped_no_transcript += 1
                    print(f"    No transcript available")
                    continue

                print(f"    Transcript: {len(transcript)} chars, {len(segments)} VTT segments, Duration: {duration_seconds}s")

                # 2. If yt-dlp didn't return chapters, try parsing from description
                if not chapters_raw and description:
                    chapters_raw = parse_chapters(description)

                # 3. Build chunks
                chunks: list[dict] = []
                if segments and duration_seconds > 0:
                    chunks = chunk_segments_by_chapters(
                        segments, chapters_raw, duration_seconds
                    )
                    if chunks and chapters_raw:
                        total_with_chapters += 1
                        print(f"    Chapters: {len(chapters_raw)}, Chunks: {len(chunks)} (VTT timestamps)")
                    elif chunks:
                        print(f"    No chapters, Chunks: {len(chunks)} (VTT timestamps)")

                if not chunks and chapters_raw and duration_seconds > 0:
                    chunks = chunk_transcript_by_chapters(
                        transcript, chapters_raw, duration_seconds
                    )
                    if chunks:
                        total_with_chapters += 1
                        print(f"    Chapters: {len(chapters_raw)}, Chunks: {len(chunks)} (proportional)")

                if not chunks:
                    raw_chunks = splitter.chunks(transcript)
                    for i, text in enumerate(raw_chunks):
                        if text.strip():
                            chunks.append({
                                "text": text.strip(),
                                "chapter": "",
                                "start_time": "",
                                "start_seconds": 0.0,
                                "chunk_index": i,
                            })
                    print(f"    No chapters, TextSplitter: {len(chunks)} chunks")

                # 4. Add description as extra chunk if substantial
                if description and len(description) > 50:
                    chunks.append({
                        "text": description[:1500],
                        "chapter": "",
                        "start_time": "",
                        "start_seconds": 0.0,
                        "chunk_index": len(chunks),
                        "chunk_type": "description",
                    })

                # 5. Build VideoDoc
                video_chapters = [
                    VideoChapter(
                        title=c["title"],
                        start_time=c["start_time"],
                        start_seconds=c["start_seconds"],
                    )
                    for c in chapters_raw
                ]

                doc_id = hashlib.md5(f"yt:{video_id}".encode()).hexdigest()
                video_doc = VideoDoc(
                    video_id=video_id,
                    doc_id=doc_id,
                    source_name=ch["handle"],
                    title=title,
                    description=description,
                    url=f"https://youtube.com/watch?v={video_id}",
                    created=upload_date or video.get("publishedDate", "")[:10],
                    tags=",".join([ch["handle"], "youtube"] + yt_tags[:5]),
                    duration_seconds=duration_seconds,
                    chapters=video_chapters,
                    transcript=transcript,
                )

                if dry_run:
                    print(f"    [DRY] {len(chunks)} chunks")
                    total_indexed += 1
                    total_chunks += len(chunks)
                    continue

                # 6. Upsert into graph
                n_chunks = idx.upsert_video("youtube", video_doc, chunks,
                                             channel_name=ch["name"])
                total_indexed += 1
                total_chunks += n_chunks
                print(f"    Indexed: {n_chunks} chunks")

        # Summary
        stats = {
            "videos_indexed": total_indexed,
            "total_chunks": total_chunks,
            "with_chapters": total_with_chapters,
            "skipped_exists": total_skipped_exists,
            "skipped_no_transcript": total_skipped_no_transcript,
        }

        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  Videos indexed:          {total_indexed}")
        print(f"  Total chunks:            {total_chunks}")
        print(f"  With chapters:           {total_with_chapters}")
        print(f"  Skipped (exists):        {total_skipped_exists}")
        print(f"  Skipped (no transcript): {total_skipped_no_transcript}")
        print(f"{'='*60}")

        return stats

    def index_url(self, urls: list[str], dry_run: bool = False) -> dict:
        """Index specific videos by URL. No SearXNG needed — uses yt-dlp directly."""
        from ..models import VideoChapter, VideoDoc
        from ..vectors.common import (
            chunk_segments_by_chapters,
            chunk_transcript_by_chapters,
            get_text_splitter,
            parse_chapters,
        )
        from ..vectors.source_index import SourceIndex

        if not check_ytdlp():
            print("yt-dlp not found. Install: brew install yt-dlp", file=sys.stderr)
            sys.exit(1)

        print("Loading embedding model...")
        idx = SourceIndex(backend=self.backend)
        splitter = get_text_splitter()

        total_indexed = 0
        total_chunks = 0

        for url in urls:
            video_id = extract_video_id(url)
            if not video_id:
                print(f"  Cannot parse video ID from: {url}", file=sys.stderr)
                continue

            if not dry_run and idx.video_exists("youtube", video_id):
                print(f"  Already indexed: {video_id}")
                continue

            print(f"  Fetching: {video_id}...")
            meta = fetch_video_metadata(video_id)
            title = meta.get("title") or video_id
            channel_name = meta.get("channel", "")
            channel_handle = meta.get("channel_handle", "") or channel_name
            transcript = meta["transcript"]
            segments = meta.get("segments", [])
            description = meta["description"]
            duration_seconds = meta["duration"]
            chapters_raw = meta["chapters"]
            yt_tags = meta["tags"]
            upload_date = meta.get("upload_date", "")

            if not transcript or len(transcript) < 200:
                # Try SearXNG fallback
                if check_searxng(self.searxng_url):
                    transcript = fetch_transcript(self.searxng_url, video_id) or ""

            if not transcript or len(transcript) < 200:
                print(f"    No transcript for {video_id}")
                continue

            print(f"  Title: {title[:70]}")
            print(f"  Channel: {channel_handle}")
            print(f"  Transcript: {len(transcript)} chars, {len(segments)} segments, {duration_seconds}s")

            if not chapters_raw and description:
                chapters_raw = parse_chapters(description)

            chunks: list[dict] = []
            if segments and duration_seconds > 0:
                chunks = chunk_segments_by_chapters(segments, chapters_raw, duration_seconds)

            if not chunks and chapters_raw and duration_seconds > 0:
                chunks = chunk_transcript_by_chapters(transcript, chapters_raw, duration_seconds)

            if not chunks:
                raw_chunks = splitter.chunks(transcript)
                for i, text in enumerate(raw_chunks):
                    if text.strip():
                        chunks.append({
                            "text": text.strip(),
                            "chapter": "",
                            "start_time": "",
                            "start_seconds": 0.0,
                            "chunk_index": i,
                        })

            if description and len(description) > 50:
                chunks.append({
                    "text": description[:1500],
                    "chapter": "",
                    "start_time": "",
                    "start_seconds": 0.0,
                    "chunk_index": len(chunks),
                    "chunk_type": "description",
                })

            video_chapters = [
                VideoChapter(
                    title=c["title"],
                    start_time=c["start_time"],
                    start_seconds=c["start_seconds"],
                )
                for c in chapters_raw
            ]

            doc_id = hashlib.md5(f"yt:{video_id}".encode()).hexdigest()
            video_doc = VideoDoc(
                video_id=video_id,
                doc_id=doc_id,
                source_name=channel_handle,
                title=title,
                description=description,
                url=f"https://youtube.com/watch?v={video_id}",
                created=upload_date,
                tags=",".join([channel_handle, "youtube"] + yt_tags[:5]),
                duration_seconds=duration_seconds,
                chapters=video_chapters,
                transcript=transcript,
            )

            if dry_run:
                print(f"    [DRY] {len(chunks)} chunks")
                total_indexed += 1
                total_chunks += len(chunks)
                continue

            n_chunks = idx.upsert_video("youtube", video_doc, chunks,
                                         channel_name=channel_name or channel_handle)
            total_indexed += 1
            total_chunks += n_chunks
            print(f"    Indexed: {n_chunks} chunks")

        print(f"\nIndexed {total_indexed} videos, {total_chunks} chunks")
        return {"videos_indexed": total_indexed, "total_chunks": total_chunks}

    def import_file(self, filepath: str, dry_run: bool = False) -> int:
        """Import pre-processed video summaries from a markdown file."""
        from ..vectors.source_index import SourceIndex

        print("Loading embedding model...")
        idx = SourceIndex(backend=self.backend)

        print(f"Importing from {filepath}...")
        count = import_file(filepath, idx, dry_run=dry_run)
        print(f"\nImported: {count} entries")
        return count
