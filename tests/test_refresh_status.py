"""scripts/refresh-status.sh against synthetic logs.

The check answers "is the nightly refresh refreshing?". Review of 28 Oct 2026: it
took the last "done" line of any scope, so a manual run over one project after a
failed nightly read as OK while 73 of 77 projects went unrefreshed.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "refresh-status.sh"


def stamp(hours_ago: float) -> str:
    return (dt.datetime.now() - dt.timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%S")


def whole(hours_ago: float, scanned: int = 3, unchanged: int = 70) -> list[str]:
    t = stamp(hours_ago)
    return [
        f"{t} done — scanned {scanned}, unchanged {unchanged}, failed 0 (scope: whole registry)",
        f"{t} sessions — 3 read, 900 indexed, 600 archived, 0 failed",
    ]


def status(tmp_path: Path, lines: list[str]) -> tuple[int, str]:
    log = tmp_path / "refresh.log"
    log.write_text("\n".join(lines) + "\n")
    r = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, "SOLOGRAPH_REFRESH_LOG": str(log)},
        check=False,
    )
    return r.returncode, r.stdout.strip()


def test_a_recent_whole_run_is_ok(tmp_path):
    code, out = status(tmp_path, whole(2))
    assert code == 0 and out.startswith("OK"), out


def test_a_failed_nightly_is_not_hidden_by_a_later_scoped_run(tmp_path):
    # As the real log of 28 Oct: nightly failed at 04:10, a scoped run at 09:31,
    # which syncs sessions too.
    lines = [
        *whole(26),
        f"{stamp(5)} FAILED reading /x/registry.yaml",
        f"{stamp(1)} done — scanned 1, unchanged 3, failed 0 (scope: /w/life2film/.solograph.yaml)",
        f"{stamp(1)} sessions — 0 read, 977 indexed, 638 archived, 0 failed",
    ]
    code, out = status(tmp_path, lines)
    assert code == 1 and out.startswith("FAIL"), out


def test_an_old_whole_run_with_a_recent_scoped_one_is_stale(tmp_path):
    lines = [
        *whole(240),
        f"{stamp(1)} done — scanned 1, unchanged 3, failed 0 (scope: SOLOGRAPH_SCOPE)",
        f"{stamp(1)} sessions — 0 read, 977 indexed, 638 archived, 0 failed",
    ]
    code, out = status(tmp_path, lines)
    assert code == 1 and out.startswith("STALE"), out


def test_a_run_that_looked_at_nothing_fails(tmp_path):
    code, out = status(tmp_path, whole(2, scanned=0, unchanged=0))
    assert code == 1 and "no project" in out, out


def test_a_missing_session_sync_fails(tmp_path):
    code, out = status(tmp_path, whole(2)[:1])
    assert code == 1 and "session" in out, out
