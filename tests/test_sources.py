"""External sources (YouTube, ProductHunt, TrustMRR, Telegram) on the FTS store.

Before: one FalkorDB graph per source with embeddings. On 26 Oct both
`source-list` and `source-search` hung past 120 s, and the YouTube graph alone
took 28 s to load. These pin the interface the indexers, the CLI and the MCP
tools use, now over SQLite FTS5.
"""

from __future__ import annotations

import asyncio

import codegraph_mcp.server as server
from codegraph_mcp.models import MakerProfile, SourceDoc, VideoDoc
from codegraph_mcp.sources import SourceIndex


def _idx(tmp_path):
    return SourceIndex(sources_root=str(tmp_path))


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


def test_a_document_is_found_and_known(tmp_path):
    idx = _idx(tmp_path)
    assert idx.upsert_one("telegram", _doc("p1", "Bootstrapped to 10k MRR", "no investors, just customers")) is True
    assert idx.upsert_one("telegram", _doc("p1", "Bootstrapped to 10k MRR", "edited")) is False
    assert idx.exists("telegram", "p1")
    assert not idx.exists("youtube", "p1")
    assert idx.get_all_urls("telegram") == {"https://t.me/x/p1"}
    hit = idx.search("investors bootstrapped")[0]
    assert hit["doc_id"] == "p1" and hit["source_name"] == "telegram"
    assert {"title", "url", "content", "score"} <= hit.keys()


def test_source_filter(tmp_path):
    idx = _idx(tmp_path)
    idx.upsert_one("telegram", _doc("p1", "Churn", "reduce churn"))
    idx.upsert_one("trustmrr", _doc("p2", "Churn", "reduce churn", source="trustmrr"))
    assert [h["doc_id"] for h in idx.search("churn", source="trustmrr")] == ["p2"]


def test_popularity_breaks_a_tie(tmp_path):
    idx = _idx(tmp_path)
    idx.upsert_one("producthunt", _doc("a", "Invoice tool", "invoice generator", popularity=3, source="producthunt"))
    idx.upsert_one("producthunt", _doc("b", "Invoice tool", "invoice generator", popularity=900, source="producthunt"))
    for i in range(3):  # FTS5 scores a term found in half the corpus as 0
        idx.upsert_one("producthunt", _doc(f"f{i}", "Other", "unrelated", source="producthunt"))
    assert idx.search("invoice generator")[0]["doc_id"] == "b"


def test_video_chunks_come_back_with_a_timestamp_once_per_video(tmp_path):
    idx = _idx(tmp_path)
    n = idx.upsert_video("youtube", _video(), _chunks("intro talk", "tiered pricing anchors", "pricing pages"))
    assert n == 3
    assert idx.video_exists("youtube", "abcdefghijk")
    assert idx.video_count("youtube") == {"videos": 1, "chunks": 3}

    hits = idx.search("pricing")
    assert len(hits) == 1
    assert hits[0]["url"].endswith("&t=10s") or hits[0]["url"].endswith("&t=20s")
    assert hits[0]["chapter"] in {"ch1", "ch2"}


def test_a_chunk_without_seconds_takes_them_from_its_start_time(tmp_path):
    # 14,446 of the migrated chunks had "start_time" and no "start_seconds".
    idx = _idx(tmp_path)
    idx.upsert_video("youtube", _video(), [{"text": "pricing anchors", "chunk_index": 0, "start_time": "1:02:03"}])
    assert idx.search("pricing")[0]["url"].endswith("&t=3723s")


def test_reindexing_a_video_replaces_its_chunks(tmp_path):
    idx = _idx(tmp_path)
    idx.upsert_video("youtube", _video(), _chunks("old words here"))
    idx.upsert_video("youtube", _video(), _chunks("fresh transcript"))
    assert idx.search("old words") == []
    assert idx.video_count("youtube") == {"videos": 1, "chunks": 1}


def test_auto_tags_come_from_keywords_not_embeddings(tmp_path):
    idx = _idx(tmp_path)
    tags = dict(idx.auto_tag_video("Building AI agents that automate support", "LLM agent workflows"))
    assert "AI agents and automation" in tags
    assert idx.auto_tag_video("Grandma's borscht recipe", "beets, cabbage, dill") == []


def test_tags_related_and_channel(tmp_path):
    idx = _idx(tmp_path)
    idx.upsert_video("youtube", _video("aaaaaaaaaaa", "AI agents for pricing", "agent automation, SaaS pricing"), [])
    idx.upsert_video("youtube", _video("bbbbbbbbbbb", "Agents that automate ops", "AI agent automation"), [])
    idx.upsert_video("youtube", _video("ccccccccccc", "Borscht", "soup"), [])
    names = {t["name"] for t in idx.list_tags("youtube")}
    assert "AI agents and automation" in names
    related = idx.related_videos("youtube", "aaaaaaaaaaa")
    assert [r["video_id"] for r in related] == ["bbbbbbbbbbb"]
    assert len(idx.channel_videos("youtube", "@indiechannel")) == 3
    assert [v["video_id"] for v in idx.search_by_tag("youtube", "AI agents and automation")] != []


def test_makers(tmp_path):
    idx = _idx(tmp_path)
    maker = MakerProfile(username="jane", name="Jane Roe", headline="Building invoicing for freelancers")
    assert idx.upsert_maker("producthunt", maker) is True
    assert idx.upsert_maker("producthunt", maker) is False
    assert idx.maker_count("producthunt") == 1
    assert idx.get_existing_maker_usernames("producthunt") == {"jane"}
    hit = idx.search("invoicing freelancers")[0]
    assert hit["url"] == "https://www.producthunt.com/@jane"


def test_list_count_delete(tmp_path):
    idx = _idx(tmp_path)
    idx.upsert_one("telegram", _doc("p1", "a", "alpha"))
    idx.upsert_video("youtube", _video(), _chunks("beta"))
    listed = {s["source"]: s for s in idx.list_sources()}
    assert listed["telegram"]["count"] == 1
    assert listed["youtube"]["videos"] == 1 and listed["youtube"]["video_chunks"] == 1
    assert idx.count() == 2
    assert idx.delete_source("telegram") is True
    assert idx.search("alpha") == []
    assert idx.count() == 1


def test_source_search_through_mcp(monkeypatch, tmp_path):
    idx = _idx(tmp_path)
    idx.upsert_one("telegram", _doc("p1", "Cold email", "cold email that converts"))
    monkeypatch.setattr(server, "_source_idx", idx)

    for name, args in [("source_search", {"query": "cold email"}), ("source_list", {})]:
        result = asyncio.run(server.mcp.call_tool(name, args))
        assert not result.is_error, result.content[0].text
        assert result.structured_content["result"]
