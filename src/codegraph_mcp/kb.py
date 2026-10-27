"""Knowledge base search: markdown with YAML frontmatter, ranked by BM25.

Was a FalkorDB vector index (multilingual-e5-small) fused with TF-IDF. Measured
on 26 Oct 2026 it took 14 s cold, scored results by rank position (the top hit
was always "relevance 1.0"), embedded without the "query:"/"passage:" prefixes
e5 is trained with, and indexed 36 Playwright page snapshots from a hidden
directory — the top answer to "tree-sitter code graph" was an Amazon NAS page.

Now: the same FtsStore as session search, synced against the disk before each
search, so there is no reindex step to forget. Hidden directories are skipped.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import frontmatter

from .fts import FtsStore

_SKIP_DIRS = {"node_modules", "__pycache__", "archive", "blog", "invoices", "solo-factory"}
_SKIP_FILES = {"README.md", "INDEX.md"}


def markdown_files(root: Path) -> dict[str, Path]:
    """{relative path: path} for every markdown file outside hidden and vendored dirs."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(".md") and name not in _SKIP_FILES:
                path = Path(dirpath) / name
                found[str(path.relative_to(root))] = path
    return found


def parse_markdown(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        post = frontmatter.loads(text)
        meta, body = post.metadata, post.content
    except Exception:  # broken frontmatter is still a document
        meta, body = {}, text
    title = str(meta.get("title") or "")
    if not title:
        heading = next((ln for ln in body.splitlines() if ln.startswith("# ")), "")
        title = heading[2:].strip() or path.stem
    tags = meta.get("tags") or []
    tags = ",".join(str(t) for t in tags) if isinstance(tags, list) else str(tags)
    doc_type = str(meta.get("type") or "unknown")
    return {
        "title": title,
        "body": f"{tags}\n{body}",
        "kind": doc_type,
        "meta": {"type": doc_type, "tags": tags},
    }


class KnowledgeBase:
    def __init__(self, kb_path: str | Path, db_path: str | Path | None = None):
        self.kb_path = Path(kb_path).expanduser()
        db_dir = Path(db_path).expanduser() if db_path else self.kb_path / ".solo" / "kb"
        self.store = FtsStore(db_dir / "search.db")

    def sync(self) -> dict:
        return self.store.sync(markdown_files(self.kb_path), parse_markdown)

    def search(self, query: str, n_results: int = 5, doc_type: str | None = None, expand_graph: bool = False):
        self.sync()
        results = [
            {
                "file": h["doc_id"],
                "title": h["title"],
                "type": h.get("type", h["kind"]),
                "tags": h.get("tags", ""),
                "score": h["score"],
                "snippet": h["snippet"],
            }
            for h in self.store.search(query, n_results=n_results, kind=doc_type)
        ]
        if expand_graph:
            results.extend(self._graph_neighbors(results))
        return results

    def _graph_neighbors(self, results: list[dict]) -> list[dict]:
        """1-hop neighbours of the top three hits from .metadata/knowledge_graph.json."""
        graph_path = self.kb_path / ".metadata" / "knowledge_graph.json"
        if not graph_path.exists():
            return []
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        adj = defaultdict(list)
        for e in graph["edges"]:
            if e["type"] in ("explicit", "shared_tags"):
                adj[e["source"]].append((e["target"], e["type"]))
                adj[e["target"]].append((e["source"], e["type"]))
        seen = {r["file"] for r in results}
        extra = []
        for r in results[:3]:
            for neighbor, edge_type in adj.get(r["file"], []):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                node = graph["nodes"].get(neighbor, {})
                extra.append(
                    {
                        "file": neighbor,
                        "title": node.get("title", neighbor),
                        "type": node.get("type", "unknown"),
                        "tags": ",".join(node.get("tags", [])),
                        "graph_source": f"via {edge_type}: {r['title']}",
                    }
                )
        return extra

    def stats(self) -> dict:
        report = self.sync()
        return {"documents": report["indexed"], "read_now": report["read"], "storage": str(self.store.path)}


def main():
    """`python -m codegraph_mcp.kb search <query>` against KB_PATH."""
    kb = KnowledgeBase(os.environ.get("KB_PATH", "."))
    if len(sys.argv) > 2 and sys.argv[1] == "search":
        for r in kb.search(" ".join(sys.argv[2:])):
            print(f"{r['score']:6.2f}  {r['file']}  — {r['title']}")
    else:
        print(kb.stats())


if __name__ == "__main__":
    main()
