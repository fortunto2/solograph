#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["youtube-transcript-api>=1.2"]
# ///
"""Fetch new YouTube videos into a spool, on a machine YouTube does not refuse.

YouTube answers a laptop on hotel or office Wi-Fi with HTTP 429 on subtitles (a 45-channel
refresh on 28 Oct 2026 lost half its videos that way), and a datacenter IP fares worse. This
runs on an always-on box behind a home connection, slowly, and writes for each new video:

    <id>.info.json   the yt-dlp metadata the indexer reads (trimmed: no formats, no URLs)
    <id>.vtt         the transcript as WebVTT with real timestamps (youtube-transcript-api)

The indexing machine pulls the spool (scripts/yt-pull.sh) into ~/.solo/sources/youtube/vtt/,
where `solograph-cli index-youtube -u …` finds both files and asks YouTube nothing.

    yt-spool.py [--spool DIR] [--limit 5] [--max-videos 40] [--pause 20]

Inputs in the spool dir: channels.yaml (the indexer's list) and known.txt (video ids already
indexed, one per line). Both are pushed by yt-pull.sh. State: state.json in the spool.
Needs yt-dlp on PATH (discovery and metadata).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

INFO_KEYS = ("id", "title", "channel", "uploader_id", "description", "duration", "tags", "upload_date", "chapters")
MAX_TRIES = 3  # a video too fresh for captions is retried on the next runs, then given up


class Blocked(Exception):
    """YouTube refuses this IP: every further request extends the ban, so the run stops."""


def load_channels(path: Path) -> list[str]:
    # The indexer's channels.yaml is a flat list under `channels:`; no YAML library needed.
    return re.findall(r"^\s*-\s*([\w.-]+)\s*$", path.read_text(encoding="utf-8"), re.M)


def trim_info(data: dict) -> dict:
    """Keep what fetch_video_metadata reads; yt-dlp's full dump is ~8 MB of formats and URLs."""
    out = {k: data.get(k) for k in INFO_KEYS}
    out["description"] = (out["description"] or "")[:5000]
    out["chapters"] = [
        {"title": c.get("title", ""), "start_time": c.get("start_time", 0)} for c in data.get("chapters") or []
    ]
    return out


def vtt_time(sec: float) -> str:
    ms = round(sec * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def snippets_to_vtt(snippets: list[dict]) -> str:
    """[{text, start, duration}] -> WebVTT. Normal-length cues, which parse_vtt_segments reads."""
    cues = ["WEBVTT", ""]
    for s in snippets:
        text = " ".join((s.get("text") or "").split())
        if not text:
            continue
        start = float(s.get("start") or 0)
        end = start + max(float(s.get("duration") or 0), 0.5)
        cues += [f"{vtt_time(start)} --> {vtt_time(end)}", text, ""]
    return "\n".join(cues)


def ytdlp(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    proc = subprocess.run(["yt-dlp", "--no-warnings", *args], capture_output=True, text=True, timeout=timeout)
    if "HTTP Error 429" in (proc.stderr or "") or "Sign in to confirm you" in (proc.stderr or ""):
        raise Blocked(proc.stderr.strip().splitlines()[-1][:200])
    return proc


def discover(handle: str, limit: int) -> list[str]:
    proc = ytdlp(
        ["--flat-playlist", "--dump-json", "-I", f":{limit}", "--quiet", f"https://www.youtube.com/@{handle}/videos"]
    )
    ids = []
    for line in proc.stdout.splitlines():
        try:
            vid = json.loads(line).get("id", "")
        except json.JSONDecodeError:
            continue
        if len(vid) == 11:
            ids.append(vid)
    return ids


def fetch_transcript(video_id: str) -> list[dict] | None:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import IpBlocked, RequestBlocked

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US", "en-GB"])
    except (IpBlocked, RequestBlocked) as e:
        raise Blocked(type(e).__name__) from e
    except Exception as e:  # no captions yet, disabled, private, age-gated
        print(f"    no transcript: {type(e).__name__}")
        return None
    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched.snippets]


def fetch_one(video_id: str, spool: Path) -> bool:
    snippets = fetch_transcript(video_id)
    if not snippets:
        return False
    proc = ytdlp(["-j", "--no-download", f"https://www.youtube.com/watch?v={video_id}"])
    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"    no metadata: {(proc.stderr or '').strip()[-160:]}")
        return False
    info = trim_info(json.loads(proc.stdout))
    # The .vtt lands last: the puller treats a video as ready once both files exist.
    (spool / f"{video_id}.info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    (spool / f"{video_id}.vtt").write_text(snippets_to_vtt(snippets), encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spool", type=Path, default=Path.home() / ".solo" / "yt-spool")
    ap.add_argument("--limit", type=int, default=5, help="latest videos to look at per channel")
    ap.add_argument("--max-videos", type=int, default=40, help="videos to fetch in one run")
    ap.add_argument("--pause", type=float, default=20, help="seconds between videos (jittered)")
    a = ap.parse_args()

    spool: Path = a.spool
    spool.mkdir(parents=True, exist_ok=True)
    channels_file = spool / "channels.yaml"
    if not channels_file.exists():
        print(f"no {channels_file}: yt-pull.sh pushes it", file=sys.stderr)
        return 2
    known = set((spool / "known.txt").read_text().split()) if (spool / "known.txt").exists() else set()
    state_file = spool / "state.json"
    state: dict[str, dict] = json.loads(state_file.read_text()) if state_file.exists() else {}
    done = {v for v, s in state.items() if s.get("ok") or s.get("tries", 0) >= MAX_TRIES}

    started, fetched, missing = time.time(), 0, 0
    try:
        queue: list[str] = []
        for handle in load_channels(channels_file):
            ids = discover(handle, a.limit)
            queue += [v for v in ids if v not in known and v not in done and v not in queue]
            time.sleep(random.uniform(2, 5))
        print(f"{len(queue)} new videos to fetch, taking up to {a.max_videos}")
        for video_id in queue[: a.max_videos]:
            print(f"  {video_id}")
            ok = fetch_one(video_id, spool)
            s = state.setdefault(video_id, {"tries": 0})
            s["tries"] += 1
            s["ok"] = ok
            s["at"] = int(time.time())
            fetched += ok
            missing += not ok
            state_file.write_text(json.dumps(state))
            time.sleep(a.pause * random.uniform(0.75, 1.5))
    except Blocked as e:
        print(f"stopped: YouTube refuses this IP ({e}); the rest waits for the next run")
        state_file.write_text(json.dumps(state))
        print(f"done in {time.time() - started:.0f}s: {fetched} fetched, {missing} without transcript, blocked")
        return 3
    print(f"done in {time.time() - started:.0f}s: {fetched} fetched, {missing} without transcript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
