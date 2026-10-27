"""The write side of solograph's full-text store, for the scrapers.

Since 0.9 every read — search, ranking, stemming, sync against the disk — lives in
the Rust binary (rust/solograph/src/fts.rs), the one the MCP server runs. Python
keeps only what the indexers need to write sources/search.db: insert or replace a
document, forget documents, run a transaction over the extra tables.

The schema below is the contract with fts.rs, byte for byte: a `docs` table with
each document's kind, file stamp and JSON meta, and an FTS5 table whose rowid is
docs.id. unicode61 does not fold ё into е, so stored text is folded here, the way
the reader folds its queries.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(
    id INTEGER PRIMARY KEY,
    doc_id TEXT UNIQUE NOT NULL,
    kind TEXT,
    mtime REAL,
    size INTEGER,
    offset INTEGER,
    gone INTEGER NOT NULL DEFAULT 0,
    meta TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    title, body, tokenize='unicode61 remove_diacritics 2'
);
"""


def _fold(text: str) -> str:
    return text.replace("ё", "е").replace("Ё", "Е")


class FtsStore:
    """Documents and their FTS5 rows, written; read by the Rust binary."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Re-entrant: a caller holding a transaction may call forget() or put().
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def put(self, doc_id: str, doc: dict | None, mtime=None, size=None, gone=False) -> None:
        """Insert or replace one document. `doc` None records it as seen but empty."""
        db = self._db
        row = db.execute("SELECT id FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
        kind = doc.get("kind", "") if doc else ""
        meta = json.dumps(doc.get("meta", {}), ensure_ascii=False) if doc else "{}"
        offset = doc.get("end") if doc else None
        if row:
            rowid = row[0]
            db.execute(
                "UPDATE docs SET kind = ?, mtime = ?, size = ?, offset = ?, gone = ?, meta = ? WHERE id = ?",
                (kind, mtime, size, offset, int(gone), meta, rowid),
            )
            db.execute("DELETE FROM fts WHERE rowid = ?", (rowid,))
        else:
            rowid = db.execute(
                "INSERT INTO docs(doc_id, kind, mtime, size, offset, gone, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, kind, mtime, size, offset, int(gone), meta),
            ).lastrowid
        if doc:
            db.execute(
                "INSERT INTO fts(rowid, title, body) VALUES (?, ?, ?)",
                (rowid, _fold(doc.get("title", "")), _fold(doc.get("body", ""))),
            )

    def forget(self, doc_ids: set[str]) -> None:
        """Remove documents outright, text included."""
        if not doc_ids:
            return
        with self._lock, self._db:
            for doc_id in doc_ids:
                self._db.execute("DELETE FROM fts WHERE rowid = (SELECT id FROM docs WHERE doc_id = ?)", (doc_id,))
                self._db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """The connection, locked, committed on success — for callers with their own tables."""
        with self._lock, self._db:
            yield self._db

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT count(*) FROM docs").fetchone()[0]
