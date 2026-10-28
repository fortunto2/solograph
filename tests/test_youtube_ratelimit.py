"""A YouTube 429 is not "this video has no transcript".

Measured on 28 Oct 2026: a refresh of 45 channels found 225 videos and reported
111 as "No transcript available". A probe of one of them showed English
subtitles listed and the download refused with HTTP 429 Too Many Requests —
the burst had rate-limited the IP. Those videos are retried on the next run
(nothing is written for them), but the summary has to say which it was.
"""

from __future__ import annotations

import json
import subprocess

from codegraph_mcp.indexers import youtube


def _fake_run(stderr_for_subs: str):
    def run(cmd, **kwargs):
        if "-j" in cmd:
            meta = {"title": "t", "channel": "c", "uploader_id": "@c", "description": "", "duration": 60}
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(meta), stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr_for_subs)

    return run


def test_a_429_on_subtitles_is_reported_as_rate_limited(monkeypatch, tmp_path):
    monkeypatch.setattr(youtube, "VTT_DIR", tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run("ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests"),
    )
    meta = youtube.fetch_video_metadata("HKNpCwG3nlc")
    assert meta["transcript"] == ""
    assert meta["rate_limited"] is True


def test_a_429_inside_a_video_id_is_not_a_rate_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(youtube, "VTT_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_run("ERROR: [youtube] x429yQ: Video unavailable"))
    assert youtube.fetch_video_metadata("x429yQabcde")["rate_limited"] is False


def test_a_video_without_subtitles_is_not_rate_limited(monkeypatch, tmp_path):
    monkeypatch.setattr(youtube, "VTT_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_run("WARNING: There are no subtitles for the requested languages"))
    meta = youtube.fetch_video_metadata("abcdefghijk")
    assert meta["rate_limited"] is False


def test_subtitle_downloads_are_paced(monkeypatch, tmp_path):
    seen = []

    def run(cmd, **kwargs):
        seen.append(cmd)
        return _fake_run("")(cmd, **kwargs)

    monkeypatch.setattr(youtube, "VTT_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run", run)
    youtube.fetch_video_metadata("abcdefghijk")
    subs = [c for c in seen if "--skip-download" in c]
    assert subs and "--sleep-subtitles" in subs[0]
