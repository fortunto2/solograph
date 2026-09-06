"""What a session search could not see.

The defect these cover was measured on a live machine, not imagined: the index
held 652 sessions while 747 transcripts sat on disk, and every search answered
with full confidence about the 87% it could reach. The auto-scan fires only when
the index is completely empty, so after the first run it never looks at the disk
again and nothing anywhere says the answer is partial.

Each test names the verdict that must NOT appear as well as the one that must.
A coverage field that can only ever say "complete" is worse than no field: it
converts an unknown into a reassurance.
"""

from __future__ import annotations

import pytest

from codegraph_mcp.scanner.sessions import count_session_files
from codegraph_mcp.server import _session_coverage


class FakeIndex:
    def __init__(self, n=0, raises=False):
        self._n, self._raises = n, raises

    def count(self):
        if self._raises:
            raise RuntimeError("store unavailable")
        return self._n


def _projects(tmp_path, n, sub="proj-a"):
    d = tmp_path / "projects" / sub
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i}.jsonl").write_text("{}\n")
    return tmp_path


def test_counts_transcripts_across_project_directories(tmp_path):
    _projects(tmp_path, 3, "proj-a")
    _projects(tmp_path, 2, "proj-b")
    assert count_session_files(tmp_path) == 5


def test_a_missing_directory_is_none_not_zero(tmp_path):
    # Zero is a measurement. "I could not look" is not, and must not become one.
    assert count_session_files(tmp_path / "nowhere") is None


def test_stale_index_says_so_and_names_the_gap(tmp_path, monkeypatch):
    _projects(tmp_path, 10)
    monkeypatch.setenv("CLAUDE_SESSIONS_DIR", str(tmp_path))
    cov = _session_coverage(FakeIndex(7))
    assert cov["status"] == "stale"
    assert cov["not_indexed"] == 3
    assert "3 session" in cov["why"]
    assert cov["status"] != "complete"


def test_complete_only_when_the_numbers_actually_agree(tmp_path, monkeypatch):
    _projects(tmp_path, 4)
    monkeypatch.setenv("CLAUDE_SESSIONS_DIR", str(tmp_path))
    assert _session_coverage(FakeIndex(4))["status"] == "complete"


def test_an_index_larger_than_the_disk_is_stale_not_complete(tmp_path, monkeypatch):
    # Built against another profile, or transcripts were deleted. Either way the
    # answer is not about what is on this disk, so it is never "complete".
    _projects(tmp_path, 2)
    monkeypatch.setenv("CLAUDE_SESSIONS_DIR", str(tmp_path))
    cov = _session_coverage(FakeIndex(9))
    assert cov["status"] == "stale"
    assert cov["not_indexed"] == 0


def test_unreadable_directory_is_unknown_never_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_SESSIONS_DIR", str(tmp_path / "gone"))
    cov = _session_coverage(FakeIndex(652))
    assert cov["status"] == "unknown"
    # The count it does have must still be reported; hiding it helps nobody.
    assert cov["indexed_sessions"] == 652
    assert "sessions_on_disk" not in cov


def test_an_uncountable_index_is_unknown_not_an_empty_one(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_SESSIONS_DIR", str(_projects(tmp_path, 5)))
    cov = _session_coverage(FakeIndex(raises=True))
    assert cov["status"] == "unknown"
    assert "could not be counted" in cov["why"]
    assert "not_indexed" not in cov


@pytest.mark.parametrize("indexed,on_disk", [(0, 0), (0, 5)])
def test_an_empty_index_is_never_silently_fine(tmp_path, monkeypatch, indexed, on_disk):
    monkeypatch.setenv("CLAUDE_SESSIONS_DIR", str(_projects(tmp_path, on_disk)))
    cov = _session_coverage(FakeIndex(indexed))
    # With nothing on disk there is genuinely nothing missing; with five there is.
    assert cov["status"] == ("complete" if on_disk == 0 else "stale")
