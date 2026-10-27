"""The sources index as the scrapers write it and the Rust binary reads it.

Python writes (SourceIndex: upsert_one / upsert_video / upsert_maker), Rust reads
(`solograph sources ...`, the MCP tools, `solograph http`). The tables are the
contract, so the read tests go through the real binary on rows the real Python
code wrote — not through a second Python reader that could drift from it.
"""

from __future__ import annotations

import sqlite3

from codegraph_mcp.models import MakerProfile, SourceDoc, VideoDoc
from codegraph_mcp.sources import SourceIndex


def _idx(tmp_path):
    return SourceIndex(sources_root=str(tmp_path))


def _rows(tmp_path, sql, *params):
    with sqlite3.connect(tmp_path / "search.db") as db:
        return db.execute(sql, params).fetchall()


def _doc(doc_id, title, content, popularity=0, source="telegram"):
    return SourceDoc(
        doc_id=doc_id,
        source_type=f"{source}-post",
        source_name=source,
        title=title,
        content=content,
        url=f"https://t.me/x/{doc_id}",
        created="2026-03-01",
        popularity=popularity,
    )


def _video(video_id="abcdefghijk", title="Pricing a SaaS", description="How to set SaaS pricing tiers"):
    return VideoDoc(
        doc_id=f"yt:{video_id}",
        video_id=video_id,
        title=title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        source_name="@indiechannel",
        description=description,
        created="2026-02-01",
    )


def _chunks(*texts):
    return [
        {"text": t, "chunk_index": i, "chapter": f"ch{i}", "start_time": f"0:{i}0", "start_seconds": i * 10.0}
        for i, t in enumerate(texts)
    ]


# ── what the writers rely on (Python only) ────────────────────────


def test_a_document_is_written_once_and_known(tmp_path):
    idx = _idx(tmp_path)
    assert idx.upsert_one("telegram", _doc("p1", "Bootstrapped to 10k MRR", "no investors, just customers")) is True
    assert idx.upsert_one("telegram", _doc("p1", "Bootstrapped to 10k MRR", "edited")) is False
    assert idx.exists("telegram", "p1")
    assert not idx.exists("youtube", "p1")
    assert idx.get_all_urls("telegram") == {"https://t.me/x/p1"}


def test_auto_tags_come_from_keywords_not_embeddings(tmp_path):
    idx = _idx(tmp_path)
    tags = dict(idx.auto_tag_video("Building AI agents that automate support", "LLM agent workflows"))
    assert "AI agents and automation" in tags
    assert idx.auto_tag_video("Grandma's borscht recipe", "beets, cabbage, dill") == []


def test_reindexing_a_video_replaces_its_chunks(tmp_path):
    idx = _idx(tmp_path)
    idx.upsert_video("youtube", _video(), _chunks("old words here", "second"))
    idx.upsert_video("youtube", _video(), _chunks("fresh transcript"))
    assert idx.video_exists("youtube", "abcdefghijk")
    chunks = _rows(tmp_path, "SELECT count(*) FROM docs WHERE json_extract(meta, '$.type') = 'video_chunk'")
    assert chunks == [(1,)]


def test_a_maker_is_written_once(tmp_path):
    idx = _idx(tmp_path)
    maker = MakerProfile(username="jane", name="Jane Roe", headline="Building invoicing for freelancers")
    assert idx.upsert_maker("producthunt", maker) is True
    assert idx.upsert_maker("producthunt", maker) is False
    assert idx.maker_count("producthunt") == 1


# ── Python writes, the binary reads ───────────────────────────────


def test_a_written_document_is_found_by_the_binary(tmp_path, read):
    idx = _idx(tmp_path)
    idx.upsert_one("telegram", _doc("p1", "Churn", "reduce churn with annual plans"))
    idx.upsert_one("trustmrr", _doc("p2", "Churn", "reduce churn", source="trustmrr"))
    hits = read("sources", "search", "annual plans")
    assert hits[0]["doc_id"] == "p1" and hits[0]["source_name"] == "telegram"
    assert {"title", "url", "content", "score"} <= hits[0].keys()
    assert [h["doc_id"] for h in read("sources", "search", "churn", "-s", "trustmrr")] == ["p2"]


def test_popularity_written_here_breaks_a_tie_there(tmp_path, read):
    idx = _idx(tmp_path)
    idx.upsert_one("producthunt", _doc("a", "Invoice tool", "invoice generator", popularity=3, source="producthunt"))
    idx.upsert_one("producthunt", _doc("b", "Invoice tool", "invoice generator", popularity=900, source="producthunt"))
    for i in range(3):  # FTS5 scores a term found in half the corpus as 0
        idx.upsert_one("producthunt", _doc(f"f{i}", "Other", "unrelated", source="producthunt"))
    assert read("sources", "search", "invoice generator")[0]["doc_id"] == "b"


def test_video_chunks_come_back_once_per_video_with_a_timestamp(tmp_path, read):
    idx = _idx(tmp_path)
    assert idx.upsert_video("youtube", _video(), _chunks("intro talk", "tiered pricing anchors", "pricing pages")) == 3
    # 14,446 of the migrated chunks had "start_time" and no "start_seconds".
    idx.upsert_video(
        "youtube", _video("bbbbbbbbbbb", "Anchors"), [{"text": "anchoring", "chunk_index": 0, "start_time": "1:02:03"}]
    )
    hits = read("sources", "search", "pricing")
    assert len(hits) == 1
    assert hits[0]["url"].endswith(("&t=10s", "&t=20s")) and hits[0]["chapter"] in {"ch1", "ch2"}
    assert read("sources", "search", "anchoring")[0]["url"].endswith("&t=3723s")


def test_one_long_video_does_not_crowd_out_the_others(tmp_path, read):
    idx = _idx(tmp_path)
    idx.upsert_video("youtube", _video("aaaaaaaaaaa", "Agents course"), _chunks(*["ai agents ai agents"] * 30))
    idx.upsert_video("youtube", _video("bbbbbbbbbbb", "Short clip"), _chunks("ai agents in one line"))
    hits = read("sources", "search", "ai agents", "-s", "youtube", "-n", "2")
    assert {h["video_id"] for h in hits} == {"aaaaaaaaaaa", "bbbbbbbbbbb"}


def test_tags_and_related_videos(tmp_path, read):
    idx = _idx(tmp_path)
    idx.upsert_video("youtube", _video("aaaaaaaaaaa", "AI agents for pricing", "agent automation, SaaS pricing"), [])
    idx.upsert_video("youtube", _video("bbbbbbbbbbb", "Agents that automate ops", "AI agent automation"), [])
    idx.upsert_video("youtube", _video("ccccccccccc", "Borscht", "soup"), [])
    assert "AI agents and automation" in {t["name"] for t in read("sources", "tags")}
    assert [r["video_id"] for r in read("sources", "related", "aaaaaaaaaaa")] == ["bbbbbbbbbbb"]


def test_a_maker_is_found_with_a_profile_url(tmp_path, read):
    idx = _idx(tmp_path)
    idx.upsert_maker("producthunt", MakerProfile(username="jane", name="Jane", headline="Invoicing for freelancers"))
    assert read("sources", "search", "invoicing freelancers")[0]["url"] == "https://www.producthunt.com/@jane"


def test_list_then_delete(tmp_path, read):
    idx = _idx(tmp_path)
    idx.upsert_one("telegram", _doc("p1", "a", "alpha"))
    idx.upsert_video("youtube", _video(), _chunks("beta"))
    listed = {s["source"]: s for s in read("sources", "list")}
    assert listed["telegram"]["count"] == 1
    assert listed["youtube"]["videos"] == 1 and listed["youtube"]["video_chunks"] == 1
    assert idx.delete_source("telegram") is True
    assert read("sources", "search", "alpha") == []
    assert [s["source"] for s in read("sources", "list")] == ["youtube"]
