#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["google-api-python-client>=2.100", "google-auth>=2.20"]
# ///
"""Index your own YouTube channel through the official API, with the captions you uploaded.

The anonymous paths (yt-dlp, youtube-transcript-api) answer a channel owner the same way they
answer a stranger: HTTP 429 on hotel Wi-Fi, IpBlocked on a VPN or after a burst from a home IP.
The owner does not need them. With an OAuth token of the channel (scope youtube or
youtube.force-ssl) the Data API lists the uploads and hands over every caption track, the
uploaded ones included, and none of it depends on which IP asks.

For each public upload not yet in the index this writes, into the indexer's VTT cache:

    <id>.info.json   the metadata fields fetch_video_metadata reads (chapters come from the description)
    <id>.vtt         the best caption track: uploaded English, then auto English, then uploaded Russian

then runs `solograph-cli index-youtube -u …` on them, which finds both files and asks YouTube
nothing more.

    yt-own.py --token ~/.config/<tool>/youtube/token.json [--token …] [--max-units 6000] [--no-index]

Quota (10,000 units a day per Google project, shared with uploads): captions.list 50 and
captions.download 200 per video, listing ~2 per 50 videos. --max-units stops a run before it
eats the day's upload budget; the rest is picked up by the next run, newest videos first.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

VTT_DIR = Path.home() / ".solo" / "sources" / "youtube" / "vtt"
SEARCH_DB = Path.home() / ".solo" / "sources" / "search.db"
UNITS_PER_VIDEO = 250  # captions.list 50 + captions.download 200
# The index is English; a Russian-only video still beats no video.
TRACK_ORDER = [("en", "standard"), ("en", "asr"), ("ru", "standard"), ("ru", "asr")]


def iso_duration(s: str) -> int:
    """PT1H2M3S -> 3723. Days appear only on live streams that never ended; counted anyway."""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return 0
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


def pick_track(items: list[dict]) -> dict | None:
    """The caption track to index, by TRACK_ORDER; any serving track if none of those."""
    serving = [c for c in items if c["snippet"].get("status", "serving") == "serving"]

    def rank(c: dict) -> int:
        lang, kind = c["snippet"]["language"].split("-")[0], c["snippet"]["trackKind"].lower()
        return TRACK_ORDER.index((lang, kind)) if (lang, kind) in TRACK_ORDER else len(TRACK_ORDER)

    return min(serving, key=rank, default=None)


def to_info(video: dict, handle: str) -> dict:
    """A Data API videos#resource -> the fields of a yt-dlp dump the indexer reads."""
    sn = video["snippet"]
    return {
        "id": video["id"],
        "title": sn.get("title", ""),
        "channel": sn.get("channelTitle", ""),
        "uploader_id": handle,
        "description": (sn.get("description") or "")[:5000],
        "duration": iso_duration(video.get("contentDetails", {}).get("duration", "")),
        "tags": sn.get("tags") or [],
        "upload_date": (sn.get("publishedAt") or "")[:10].replace("-", ""),
        "chapters": [],  # the indexer parses them from the description
    }


def known_ids(db: Path) -> set[str]:
    if not db.exists():
        return set()
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        return {r[0] for r in con.execute("SELECT video_id FROM videos")}


def client(token: Path):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    # Refreshed in memory only: the token file belongs to the tool that minted it.
    return build("youtube", "v3", credentials=Credentials.from_authorized_user_file(str(token)), cache_discovery=False)


def public_uploads(yt) -> tuple[str, list[dict]]:
    """(@handle, public uploads newest first). About 2 units per 50 videos."""
    ch = yt.channels().list(part="snippet,contentDetails", mine=True).execute()["items"][0]
    handle = ch["snippet"].get("customUrl") or ch["snippet"]["title"]
    uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    ids, page = [], None
    while True:
        r = yt.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=50, pageToken=page).execute()
        # The uploads playlist can list one video twice (seen 8 Nov 2026); each repeat costs 250 units.
        ids += [v for i in r["items"] if (v := i["contentDetails"]["videoId"]) not in ids]
        page = r.get("nextPageToken")
        if not page:
            break
    videos = []
    for i in range(0, len(ids), 50):
        r = yt.videos().list(part="snippet,contentDetails,status", id=",".join(ids[i : i + 50])).execute()
        videos += [v for v in r["items"] if v["status"]["privacyStatus"] == "public"]
    videos.sort(key=lambda v: v["snippet"]["publishedAt"], reverse=True)
    return handle, videos


def export(yt, handle: str, video: dict, out: Path) -> bool:
    track = pick_track(yt.captions().list(part="snippet", videoId=video["id"]).execute()["items"])
    if not track:
        print(f"  {video['id']}  no caption track")
        return False
    body = yt.captions().download(id=track["id"], tfmt="vtt").execute()
    vtt = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    # .vtt last, as the spool does: a .vtt means its .info.json is already there.
    (out / f"{video['id']}.info.json").write_text(json.dumps(to_info(video, handle), ensure_ascii=False), "utf-8")
    (out / f"{video['id']}.vtt").write_text(vtt, "utf-8")
    sn = track["snippet"]
    print(f"  {video['id']}  {sn['language']}/{sn['trackKind']}  {video['snippet']['title'][:60]}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--token", type=Path, action="append", required=True, help="OAuth token of a channel (repeat)")
    ap.add_argument("--out", type=Path, default=VTT_DIR)
    ap.add_argument("--db", type=Path, default=SEARCH_DB, help="the sources index, to skip what is in it")
    ap.add_argument("--max-units", type=int, default=6000, help="Data API units to spend on captions in one run")
    ap.add_argument("--no-index", action="store_true", help="write the files, leave indexing to the caller")
    a = ap.parse_args()
    if isinstance(sys.stdout, io.TextIOWrapper):  # a log reads in order with the indexer's own output
        sys.stdout.reconfigure(line_buffering=True)

    a.out.mkdir(parents=True, exist_ok=True)
    known, budget, written, pending = known_ids(a.db), a.max_units, 0, []
    for token in a.token:
        token = token.expanduser()
        if not token.exists():
            print(f"{token}: no token, skipped (authorize that channel first)")
            continue
        yt = client(token)
        handle, videos = public_uploads(yt)
        new = [v["id"] for v in videos if v["id"] not in known]
        todo = [v for v in videos if v["id"] in new and not (a.out / f"{v['id']}.vtt").exists()]
        print(f"{handle}: {len(videos)} public, {len(new)} not indexed, {budget // UNITS_PER_VIDEO} fit the budget")
        for v in todo:
            if budget < UNITS_PER_VIDEO:
                break
            budget -= UNITS_PER_VIDEO
            written += export(yt, handle, v, a.out)
        # A pair cached by an earlier run that stopped before indexing is indexed now.
        pending += [i for i in new if (a.out / f"{i}.vtt").exists() and (a.out / f"{i}.info.json").exists()]
    print(f"{written} exported, {a.max_units - budget} units; {len(pending)} to index")
    if pending and not a.no_index:
        args = [x for vid in pending for x in ("-u", f"https://www.youtube.com/watch?v={vid}")]
        subprocess.run(["solograph-cli", "index-youtube", *args], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
