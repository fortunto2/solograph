"""What a session search could see.

The defect behind these was measured on a live machine: the vector index held
640 sessions and the disk 633, overlap zero, because Claude Code deletes
transcripts after 30 days and the index never looked at the disk again. Every
search answered with full confidence about sessions nobody could open.

Search now syncs before it answers, so "stale" cannot happen by construction.
What can still happen is a transcript that will not parse, or a directory that
cannot be read — and neither may come back as "complete".
"""

from __future__ import annotations

import json

from codegraph_mcp.session_search import SessionSearch, coverage


def _transcript(path, prompt, answer="Done."):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {"type": "user", "cwd": "/w/proj", "timestamp": "2026-10-01T10:00:00Z", "message": {"content": prompt}},
        {
            "type": "user",
            "isMeta": True,
            "message": {"content": [{"type": "text", "text": "Base directory for this skill: /skills/build"}]},
        },
        {"type": "assistant", "message": {"model": "m", "content": [{"type": "text", "text": answer}]}},
    ]
    # The format Claude Code writes: compact JSON, one object per line, each ended.
    path.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in lines))
    return path


def _search(tmp_path):
    return SessionSearch(db_path=tmp_path / "index.db", sessions_root=tmp_path / "claude")


def _proj(tmp_path):
    return tmp_path / "claude" / "projects" / "-w-proj"


def test_a_new_transcript_is_found_on_the_next_search(tmp_path):
    s = _search(tmp_path)
    _transcript(_proj(tmp_path) / "s1.jsonl", "настрой вебхук HubSpot для сделок")
    out = s.search("hubspot вебхуки")
    assert [r["session_id"] for r in out["results"]] == ["s1"]
    assert out["coverage"]["status"] == "complete"
    assert out["coverage"]["read_now"] == 1


def test_a_deleted_transcript_stays_searchable_and_says_so(tmp_path):
    s = _search(tmp_path)
    path = _transcript(_proj(tmp_path) / "s1.jsonl", "audio to midi model training")
    s.search("midi")
    path.unlink()

    out = s.search("midi")
    assert out["results"][0]["session_id"] == "s1"
    assert "only copy" in out["results"][0]["transcript"]
    assert out["coverage"]["archived"] == 1


def test_the_answer_is_searchable_not_just_the_prompt(tmp_path):
    s = _search(tmp_path)
    _transcript(_proj(tmp_path) / "s1.jsonl", "почему падает сборка", answer="Причина: ldflags потерял -lsqlite3")
    assert s.search("lsqlite3")["results"][0]["session_id"] == "s1"


def test_skill_expansions_are_not_the_session(tmp_path):
    s = _search(tmp_path)
    _transcript(_proj(tmp_path) / "s1.jsonl", "fix the parser")
    assert s.search("base directory skill")["results"] == []


def test_an_unreadable_directory_is_unknown_never_complete(tmp_path):
    s = SessionSearch(db_path=tmp_path / "index.db", sessions_root=tmp_path / "nowhere")
    cov = s.search("anything")["coverage"]
    assert cov["status"] == "unknown"
    assert cov["status"] != "complete"


def test_a_transcript_that_cannot_be_read_makes_the_answer_partial():
    cov = coverage({"on_disk": 3, "indexed": 2, "gone": 0, "read": 2, "failed": ["s3"]})
    assert cov["status"] == "partial"
    assert cov["failed"] == ["s3"]
    assert "1 transcript" in cov["why"]


def test_complete_only_with_nothing_failed():
    cov = coverage({"on_disk": 3, "indexed": 3, "gone": 0, "read": 0, "failed": []})
    assert cov["status"] == "complete"


def test_project_filter(tmp_path):
    s = _search(tmp_path)
    _transcript(_proj(tmp_path) / "s1.jsonl", "deploy the landing")
    _transcript(tmp_path / "claude" / "projects" / "-w-other" / "s2.jsonl", "deploy the landing")
    hits = s.search("deploy landing", project="other")["results"]
    assert [h["session_id"] for h in hits] == ["s2"]


def _line(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def test_a_growing_transcript_is_read_from_where_it_stopped(tmp_path):
    s = _search(tmp_path)
    path = _transcript(_proj(tmp_path) / "s1.jsonl", "start the export")
    s.search("export")
    size_before = path.stat().st_size
    # Proof that only the tail is read: the head changes in place, same length,
    # and a tail-only read cannot see it. A full re-read would.
    path.write_bytes(path.read_bytes().replace(b"export", b"zebras", 1))
    with path.open("a") as f:
        f.write(_line({"type": "assistant", "message": {"content": [{"type": "text", "text": "Saved as parquet"}]}}))

    out = s.search("parquet")
    assert out["results"][0]["session_id"] == "s1"
    assert out["results"][0]["title"] == "start the export"
    assert s.search("zebras")["results"] == []
    offset = s.store._db.execute("SELECT offset FROM docs WHERE doc_id = 's1'").fetchone()[0]
    assert offset == path.stat().st_size > size_before


def test_a_line_still_being_written_is_read_once_it_is_whole(tmp_path):
    s = _search(tmp_path)
    path = _transcript(_proj(tmp_path) / "s1.jsonl", "first prompt here")
    tail = _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "kubernetes rollout done"}]}})
    with path.open("a") as f:
        f.write(tail[:30])
    assert s.search("kubernetes")["results"] == []

    with path.open("a") as f:
        f.write(tail[30:])
    assert s.search("kubernetes")["results"][0]["session_id"] == "s1"


def test_claude_codes_own_title_wins_over_the_first_prompt(tmp_path):
    s = _search(tmp_path)
    path = _transcript(_proj(tmp_path) / "s1.jsonl", "ну что там с графом")
    assert s.search("графом")["results"][0]["title"] == "ну что там с графом"
    with path.open("a") as f:
        f.write(_line({"type": "ai-title", "aiTitle": "Solograph freshness fix", "sessionId": "s1"}))
    assert s.search("графом")["results"][0]["title"] == "Solograph freshness fix"


def test_batch_runs_from_temp_directories_are_not_conversations(tmp_path):
    s = _search(tmp_path)
    _transcript(_proj(tmp_path) / "s1.jsonl", "label these turns: deploy landing")
    _transcript(tmp_path / "claude" / "projects" / "-private-tmp" / "b1.jsonl", "label these turns: deploy landing")
    hits = s.search("deploy landing")["results"]
    assert [h["session_id"] for h in hits] == ["s1"]


def test_a_session_inside_a_private_folder_is_never_indexed(tmp_path):
    s = _search(tmp_path)
    private_cwd = tmp_path / "home" / "personal" / "family"
    private_cwd.mkdir(parents=True)
    path = _proj(tmp_path) / "s1.jsonl"
    _transcript(path, "семейная история, не для чужих глаз")
    path.write_text(path.read_text().replace('"cwd":"/w/proj"', f'"cwd":"{private_cwd}"'))
    assert s.search("семейная")["results"][0]["session_id"] == "s1"

    # The marker goes on a parent; the session is purged, archive copy included.
    (tmp_path / "home" / "personal" / ".solograph-private").touch()
    s2 = _search(tmp_path)
    out = s2.search("семейная")
    assert out["results"] == []
    assert out["coverage"]["indexed"] == 0
