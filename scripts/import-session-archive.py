"""Carry the old vector store's session summaries into the full-text index.

One-off, per store. Claude Code had already deleted every transcript the vector
store indexed (Dec 2025 – Aug 2026), so these summaries — five prompts, the
edited files, three answer fragments — are the only record of those sessions.
They go in marked gone; sessions already in the new index are left alone.

    uv run python scripts/import-session-archive.py [SESSIONS_DB_DIR]

SESSIONS_DB_DIR defaults to $CODEGRAPH_SESSIONS_DB or ~/.solo/sessions; the old
store is graph.db in it, the new index search.db next to it.
"""

import os
import re
import sys
from pathlib import Path

from redislite.falkordb_client import FalkorDB

from codegraph_mcp.fts import FtsStore

base = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CODEGRAPH_SESSIONS_DB", "~/.solo/sessions"))
base = base.expanduser()
old = FalkorDB(str(base / "graph.db")).select_graph("sessions")
rows = old.query("MATCH (s:SessionDoc) RETURN s.session_id, s.project_name, s.started_at, s.summary").result_set

store = FtsStore(base / "search.db")
known = {r[0] for r in store._db.execute("SELECT doc_id FROM docs")}
added = 0
with store._db:
    for session_id, project, started_at, summary in rows:
        if session_id in known or not summary or project == "tmp":  # batch runs, see session_search
            continue
        first = re.search(r"^1\. (.+)$", summary, re.M)
        doc = {
            "title": (first.group(1) if first else summary.splitlines()[0])[:200],
            "body": summary,
            "kind": project or "",
            "meta": {"session_id": session_id, "project": project, "started_at": started_at, "archived_summary": True},
        }
        store.put(session_id, doc, gone=True)
        added += 1
print(f"{base}: {len(rows)} in the old store, {added} imported, {len(rows) - added} already present or empty")
