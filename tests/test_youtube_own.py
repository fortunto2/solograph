"""scripts/yt-own.py: an owner's Data API answer becomes the pair the indexer reads offline."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

from codegraph_mcp.indexers import youtube

_spec = importlib.util.spec_from_file_location("yt_own", Path(__file__).parents[1] / "scripts" / "yt-own.py")
yt_own = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yt_own)

VIDEO = {  # a videos#resource, trimmed to the parts the script reads
    "id": "I9EQRe1Ec5U",
    "snippet": {
        "title": "Aaron Levie on AI agents",
        "channelTitle": "Example Channel",
        "description": "0:00 Intro\n1:30 Agents\n35:40 Box and agents",
        "tags": ["AI agents"],
        "publishedAt": "2026-10-10T18:00:00Z",
    },
    "contentDetails": {"duration": "PT1H2M3S"},
}
VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:03.940\nAnd I think you can basically correlate that\n\n"


def _track(lang, kind, status="serving"):
    return {"id": f"{lang}-{kind}", "snippet": {"language": lang, "trackKind": kind, "status": status}}


def test_an_exported_pair_is_indexed_without_calling_youtube(monkeypatch, tmp_path):
    (tmp_path / "I9EQRe1Ec5U.info.json").write_text(json.dumps(yt_own.to_info(VIDEO, "@examplechannel")))
    (tmp_path / "I9EQRe1Ec5U.vtt").write_text(VTT)
    monkeypatch.setattr(youtube, "VTT_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError(a)))

    meta = youtube.fetch_video_metadata("I9EQRe1Ec5U")

    assert meta["channel_handle"] == "examplechannel"
    assert meta["duration"] == 3723
    assert meta["upload_date"] == "2026-10-10"
    assert "35:40 Box and agents" in meta["description"]  # chapters come from here
    assert "correlate that" in meta["transcript"]


def test_the_uploaded_english_track_wins_and_russian_beats_nothing():
    assert yt_own.pick_track([_track("ru", "standard"), _track("en", "asr"), _track("en", "standard")])["id"] == (
        "en-standard"
    )
    assert yt_own.pick_track([_track("ru", "asr"), _track("en", "standard", "failed")])["id"] == "ru-asr"
    assert yt_own.pick_track([_track("de", "standard")])["id"] == "de-standard"
    assert yt_own.pick_track([]) is None


def test_iso_durations():
    assert yt_own.iso_duration("PT32S") == 32
    assert yt_own.iso_duration("PT35M59S") == 2159
    assert yt_own.iso_duration("P1DT1S") == 86401
    assert yt_own.iso_duration("") == 0
