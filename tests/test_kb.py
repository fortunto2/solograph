"""KB search over markdown. The regressions these pin were measured on 26 Oct:
Playwright snapshots in a hidden directory outranked the actual notes, and every
top hit was "relevance 1.0" whatever it matched."""

from __future__ import annotations

from codegraph_mcp.kb import KnowledgeBase, markdown_files


def _kb(tmp_path):
    root = tmp_path / "kb"
    (root / "wiki").mkdir(parents=True)
    (root / "wiki" / "solograph.md").write_text(
        "---\ntitle: Solograph code graph\ntype: concept\ntags: [tree-sitter, graph]\n---\n"
        "Tree-sitter parses every project into a graph."
    )
    (root / "wiki" / "visa.md").write_text("---\ntype: project\n---\n# O-1A visa\n\nAttorney shortlist.")
    hidden = root / "crm" / ".playwright-mcp"
    hidden.mkdir(parents=True)
    (hidden / "amazon-nas.md").write_text("- generic graph code tree sitter button")
    return root


def test_hidden_directories_are_not_the_knowledge_base(tmp_path):
    assert set(markdown_files(_kb(tmp_path))) == {"wiki/solograph.md", "wiki/visa.md"}


def test_frontmatter_title_and_tags_are_searchable(tmp_path):
    kb = KnowledgeBase(_kb(tmp_path), db_path=tmp_path / "db")
    hits = kb.search("tree-sitter graph")
    assert hits[0]["file"] == "wiki/solograph.md"
    assert hits[0]["type"] == "concept"


def test_heading_is_the_title_when_frontmatter_has_none(tmp_path):
    kb = KnowledgeBase(_kb(tmp_path), db_path=tmp_path / "db")
    assert kb.search("attorney")[0]["title"] == "O-1A visa"


def test_type_filter(tmp_path):
    kb = KnowledgeBase(_kb(tmp_path), db_path=tmp_path / "db")
    assert kb.search("graph visa", doc_type="project")[0]["file"] == "wiki/visa.md"


def test_a_new_note_is_found_without_a_reindex_step(tmp_path):
    root = _kb(tmp_path)
    kb = KnowledgeBase(root, db_path=tmp_path / "db")
    kb.search("anything")
    (root / "wiki" / "new.md").write_text("# Caretta nests\n\nTemperature decides sex.")
    assert kb.search("caretta")[0]["file"] == "wiki/new.md"


def test_broken_frontmatter_is_still_a_document(tmp_path):
    root = _kb(tmp_path)
    (root / "wiki" / "bad.md").write_text("---\ntitle: [unclosed\n---\nmangosteen harvest")
    kb = KnowledgeBase(root, db_path=tmp_path / "db")
    assert kb.search("mangosteen")[0]["file"] == "wiki/bad.md"
