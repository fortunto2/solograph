"""The full-text store that replaced the vector indexes for sessions and the KB.

What it has to get right is what the vector stores got wrong on this machine:
the session index held 640 sessions and the disk 633 with an overlap of zero,
because Claude Code deletes transcripts after 30 days and nothing re-read the
disk. So: a file that changed is re-read on the next query, and a file that
vanished keeps its text, because the index is then the only copy left.
"""

from __future__ import annotations

import os

from codegraph_mcp.fts import FtsStore, match_expression


def _doc(path):
    text = path.read_text()
    return {"title": text.splitlines()[0], "body": text, "kind": "notes", "meta": {"name": path.name}}


def _store(tmp_path):
    return FtsStore(tmp_path / "index.db")


def _files(d):
    return {p.name: p for p in sorted(d.glob("*.txt"))}


def test_russian_inflections_meet_at_the_stem():
    assert match_expression("графами") == '"граф"*'
    assert match_expression("сессий") == '"сесс"*'


def test_english_stems_and_short_words_are_kept_whole():
    assert match_expression("indexing graphs") == '"index"* OR "graph"*'
    assert match_expression("O-1A visa") == '"o" OR "1a" OR "visa"*'


def test_a_stem_that_is_not_a_prefix_is_cut_back_to_one():
    assert match_expression("vocabulary") == '"vocabular"*'
    assert match_expression("studies") == '"studi"*'


def test_yo_is_folded_because_unicode61_does_not():
    assert match_expression("ёлка") == match_expression("елка")


def test_query_with_only_punctuation_matches_nothing():
    assert match_expression("*** ()") == ""


def test_a_new_file_is_found_and_an_unrelated_word_is_not(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.txt").write_text("Граф кода\nСтроим граф кода на tree-sitter")
    (d / "b.txt").write_text("Рецепт\nПлов с бараниной")
    store = _store(tmp_path)
    store.sync(_files(d), _doc)

    hits = store.search("графом")
    assert [h["doc_id"] for h in hits] == ["a.txt"]
    assert store.search("пицца") == []


def test_rare_term_outranks_common_one(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    for i in range(5):
        (d / f"c{i}.txt").write_text(f"Заметка {i}\nвидео монтаж видео")
    (d / "rare.txt").write_text("Азалия\nвидео про Азалию и машину времени")
    store = _store(tmp_path)
    store.sync(_files(d), _doc)

    assert store.search("видео Азалия")[0]["doc_id"] == "rare.txt"


def test_an_edited_file_is_reread_and_an_untouched_one_is_not(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.txt").write_text("first\nold words")
    (d / "b.txt").write_text("second\nstable words")
    store = _store(tmp_path)
    assert store.sync(_files(d), _doc)["read"] == 2

    (d / "a.txt").write_text("first\nfresh vocabulary here")
    report = store.sync(_files(d), _doc)

    assert report["read"] == 1
    assert store.search("vocabulary")[0]["doc_id"] == "a.txt"
    assert store.search("old") == []


def test_a_vanished_file_keeps_its_text_when_asked_to(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.txt").write_text("transcript\nhubspot webhook subscription")
    store = _store(tmp_path)
    store.sync(_files(d), _doc, keep_gone=True)

    os.remove(d / "a.txt")
    report = store.sync(_files(d), _doc, keep_gone=True)

    assert report["gone"] == 1
    hit = store.search("webhook")[0]
    assert hit["doc_id"] == "a.txt" and hit["gone"] is True


def test_a_vanished_file_leaves_by_default(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.txt").write_text("note\nkept nowhere")
    store = _store(tmp_path)
    store.sync(_files(d), _doc)
    os.remove(d / "a.txt")
    store.sync(_files(d), _doc)

    assert store.search("nowhere") == []
    assert store.count() == 0


def test_a_file_the_parser_rejects_is_counted_not_hidden(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.txt").write_text("ok\nfine")
    (d / "b.txt").write_text("bad\nbroken")

    def parse(p):
        if p.name == "b.txt":
            raise ValueError("cannot parse")
        return _doc(p)

    report = _store(tmp_path).sync(_files(d), parse)
    assert report["read"] == 1
    assert report["failed"] == ["b.txt"]


def test_kind_filter_is_a_substring_match(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.txt").write_text("one\nshared word")
    (d / "b.txt").write_text("two\nshared word")
    store = _store(tmp_path)
    store.sync(_files(d), lambda p: {**_doc(p), "kind": "life2film-ios" if p.name == "a.txt" else "other"})

    assert [h["doc_id"] for h in store.search("shared", kind="life2film")] == ["a.txt"]


def test_score_is_bm25_not_a_rank_dressed_as_relevance(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.txt").write_text("one\nalpha alpha alpha beta")
    (d / "b.txt").write_text("two\nalpha gamma delta epsilon zeta eta theta")
    # FTS5 scores a term found in half the corpus or more as 0.
    for name in ("c", "d", "e"):
        (d / f"{name}.txt").write_text(f"{name}\nomega")
    store = _store(tmp_path)
    store.sync(_files(d), _doc)

    scores = [h["score"] for h in store.search("alpha")]
    assert scores[0] > scores[1] > 0
    assert scores[0] != 1.0
