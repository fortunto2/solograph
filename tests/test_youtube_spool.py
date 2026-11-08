"""The spool contract: what scripts/yt-spool.py writes on a home box, the indexer reads offline.

YouTube refuses subtitles to a laptop on hotel Wi-Fi (HTTP 429), so an always-on machine fetches
<id>.info.json + <id>.vtt and the indexing machine must build the same document from those two
files without asking YouTube anything.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

from codegraph_mcp.indexers import youtube

_spec = importlib.util.spec_from_file_location("yt_spool", Path(__file__).parents[1] / "scripts" / "yt-spool.py")
yt_spool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yt_spool)

SNIPPETS = [
    {"text": "[music]", "start": 0.0, "duration": 2.5},
    {"text": "Some of the most successful companies", "start": 2.5, "duration": 3.1},
    {"text": "were created by founders with PhDs", "start": 5.6, "duration": 2.9},
    {"text": "  ", "start": 8.5, "duration": 1.0},
    {"text": "including Google and Anthropic.", "start": 3725.25, "duration": 0.0},
]
YTDLP_DUMP = {
    "id": "u-8KpXcPYCw",
    "title": "PhD founders",
    "channel": "Y Combinator",
    "uploader_id": "@ycombinator",
    "description": "x" * 9000,
    "duration": 3730,
    "tags": ["startups"],
    "upload_date": "20261106",
    "chapters": [{"title": "Intro", "start_time": 0.0, "end_time": 60.0}, {"title": "PhDs", "start_time": 60.0}],
    "formats": [{"url": "https://example.invalid/huge"}] * 3,
    "automatic_captions": {"en": [{"url": "https://example.invalid/timedtext"}]},
}


def _no_network(cmd, **kwargs):
    raise AssertionError(f"the indexer called out with a spooled pair present: {cmd}")


def test_a_spooled_pair_is_read_without_calling_youtube(monkeypatch, tmp_path):
    vid = "u-8KpXcPYCw"
    (tmp_path / f"{vid}.info.json").write_text(json.dumps(yt_spool.trim_info(YTDLP_DUMP)))
    (tmp_path / f"{vid}.vtt").write_text(yt_spool.snippets_to_vtt(SNIPPETS))
    monkeypatch.setattr(youtube, "VTT_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run", _no_network)

    meta = youtube.fetch_video_metadata(vid)

    assert meta["title"] == "PhD founders"
    assert meta["channel_handle"] == "ycombinator"
    assert meta["duration"] == 3730
    assert meta["upload_date"] == "2026-11-06"
    assert len(meta["description"]) == 5000
    assert [c["title"] for c in meta["chapters"]] == ["Intro", "PhDs"]
    assert meta["chapters"][1]["start_seconds"] == 60
    assert "founders with PhDs" in meta["transcript"]
    assert [s["start"] for s in meta["segments"]] == [0.0, 2.5, 5.6, 3725.2]
    assert meta["rate_limited"] is False


def test_the_spooled_metadata_drops_formats_and_caption_urls():
    info = yt_spool.trim_info(YTDLP_DUMP)
    assert "formats" not in info and "automatic_captions" not in info
    assert len(json.dumps(info)) < 6000


def test_channel_list_is_read_without_a_yaml_library(tmp_path):
    f = tmp_path / "channels.yaml"
    f.write_text("# comment\nchannels:\n  - GregIsenberg\n  - niche-pursuits\n  - t3dotgg\n")
    assert yt_spool.load_channels(f) == ["GregIsenberg", "niche-pursuits", "t3dotgg"]
