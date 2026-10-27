"""Full-text search over files, kept in step with the disk at query time.

One SQLite file per corpus: a `docs` table that remembers each source file's
mtime and size, and an FTS5 table ranked by BM25. `sync()` is cheap enough to
run before every search — a stat per file, a parse only for files that moved —
so the index cannot drift the way the vector stores did. The session store held
640 sessions against 633 transcripts on disk with an overlap of zero: Claude
Code deletes transcripts after 30 days and nothing ever re-read the directory.

Borrowed rather than invented: SQLite + FTS5 + BM25 is where the code-graph
tools converged (colbymchenry/codegraph, DeusData/codebase-memory-mcp), and
re-syncing what changed before answering is codegraph's "never stale" rule.

Russian and English share one index. unicode61 lowercases but has no stemmer,
so queries are stemmed with Snowball and sent as prefix terms: "графами" asks
for "граф"*, which also matches "графа" and "графом". unicode61 does not fold
ё into е either, so both the stored text and the query are folded here.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

import snowballstemmer

_RU = snowballstemmer.stemmer("russian")
_EN = snowballstemmer.stemmer("english")
_CYRILLIC = re.compile(r"[а-я]")

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

# Title matches count three times a body match.
_RANK = "bm25(fts, 3.0, 1.0)"


def _fold(text: str) -> str:
    return text.replace("ё", "е").replace("Ё", "Е")


def match_expression(query: str) -> str:
    """Turn free text into an FTS5 OR-query of stemmed prefix terms.

    OR, not AND: BM25 already ranks documents that match more and rarer terms
    higher, and an AND over five words an agent guessed usually returns nothing.
    Short tokens and tokens with digits stay whole — a prefix of "o" or "1a"
    matches half the corpus.
    """
    terms: list[str] = []
    for token in re.findall(r"\w+", _fold(query.lower())):
        if len(token) < 3 or any(c.isdigit() for c in token):
            term = f'"{token}"'
        else:
            stem = (_RU if _CYRILLIC.search(token) else _EN).stemWord(token)
            # Snowball's stem is not always a prefix ("vocabulary" -> "vocabulari"),
            # and a prefix query needs one.
            n = 0
            while n < min(len(stem), len(token)) and stem[n] == token[n]:
                n += 1
            stem = stem[:n]
            term = f'"{stem}"*' if len(stem) >= 3 else f'"{token}"'
        if term not in terms:
            terms.append(term)
    return " OR ".join(terms)


class FtsStore:
    """A BM25 index over documents that each come from one file."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def put(self, doc_id: str, doc: dict | None, mtime=None, size=None, gone=False) -> None:
        """Insert or replace one document. `doc` None records the file as seen but empty.

        `doc["end"]`, when a parser gives it, is the byte offset it read up to —
        where `append` resumes next time.
        """
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

    def sync(
        self,
        files: dict[str, Path],
        parse: Callable[[Path], dict | None],
        keep_gone: bool = False,
        append: Callable[[Path, int], dict] | None = None,
    ) -> dict:
        """Bring the index in line with `files` ({doc_id: path}).

        A file is re-parsed only when its mtime or size changed. A document whose
        file is no longer listed is dropped, or with `keep_gone` kept and marked
        gone — for transcripts the disk deletes, where the index is the archive.

        `append(path, offset)` is for append-only files: when one only grew, it
        reads from where the last read ended and the text is added, so the cost
        follows the change rather than the file. Live transcripts reach 400 MB;
        re-reading one took 1.8 s, on every search, for every session running.
        """
        report = {"read": 0, "unchanged": 0, "failed": [], "removed": 0}
        with self._lock, self._db:
            known = {
                doc_id: (mtime, size, gone, offset)
                for doc_id, mtime, size, gone, offset in self._db.execute(
                    "SELECT doc_id, mtime, size, gone, offset FROM docs"
                )
            }
            for doc_id, path in files.items():
                try:
                    st = path.stat()
                except OSError:
                    report["failed"].append(doc_id)
                    continue
                prev = known.get(doc_id)
                if prev and prev[:3] == (st.st_mtime, st.st_size, 0):
                    report["unchanged"] += 1
                    continue
                try:
                    if append and prev and prev[3] and not prev[2] and st.st_size > prev[1]:
                        self._append(doc_id, append(path, prev[3]), st.st_mtime, st.st_size)
                    else:
                        self.put(doc_id, parse(path), st.st_mtime, st.st_size)
                except Exception:
                    report["failed"].append(doc_id)
                    continue
                report["read"] += 1
            for doc_id in known.keys() - files.keys():
                if keep_gone:
                    self._db.execute("UPDATE docs SET gone = 1 WHERE doc_id = ?", (doc_id,))
                else:
                    self._db.execute("DELETE FROM fts WHERE rowid = (SELECT id FROM docs WHERE doc_id = ?)", (doc_id,))
                    self._db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
                    report["removed"] += 1
            report["gone"] = self._db.execute("SELECT count(*) FROM docs WHERE gone = 1").fetchone()[0]
            report["indexed"] = self._db.execute("SELECT count(*) FROM docs").fetchone()[0]
        return report

    def forget(self, doc_ids: set[str]) -> None:
        """Remove documents outright — text included, archive or not."""
        if not doc_ids:
            return
        with self._lock, self._db:
            for doc_id in doc_ids:
                self._db.execute("DELETE FROM fts WHERE rowid = (SELECT id FROM docs WHERE doc_id = ?)", (doc_id,))
                self._db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))

    def _append(self, doc_id: str, delta: dict, mtime, size) -> None:
        """Add a parsed tail to an existing document: body appended, meta updated."""
        db = self._db
        rowid, meta = db.execute("SELECT id, meta FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
        merged = {**json.loads(meta or "{}"), **{k: v for k, v in delta.get("meta", {}).items() if v}}
        db.execute(
            "UPDATE docs SET mtime = ?, size = ?, offset = ?, meta = ? WHERE id = ?",
            (mtime, size, delta["end"], json.dumps(merged, ensure_ascii=False), rowid),
        )
        if delta.get("body") or delta.get("title"):
            title, body = db.execute("SELECT title, body FROM fts WHERE rowid = ?", (rowid,)).fetchone() or ("", "")
            db.execute("DELETE FROM fts WHERE rowid = ?", (rowid,))
            db.execute(
                "INSERT INTO fts(rowid, title, body) VALUES (?, ?, ?)",
                (rowid, _fold(delta["title"]) if delta.get("title") else title, body + "\n\n" + _fold(delta["body"])),
            )

    def search(self, query: str, n_results: int = 5, kind: str | None = None) -> list[dict]:
        expr = match_expression(query)
        if not expr:
            return []
        sql = (
            f"SELECT d.doc_id, d.kind, d.gone, d.meta, fts.title, "
            f"snippet(fts, 1, '«', '»', '…', 24), {_RANK} AS rank "
            "FROM fts JOIN docs d ON d.id = fts.rowid WHERE fts MATCH ?"
        )
        params: list = [expr]
        if kind:
            sql += " AND d.kind LIKE ?"
            params.append(f"%{kind}%")
        sql += " ORDER BY rank LIMIT ?"
        params.append(n_results)
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        hits = []
        for doc_id, k, gone, meta, title, snippet, rank in rows:
            hit = {"doc_id": doc_id, "kind": k, "title": title, "snippet": snippet}
            hit.update(json.loads(meta or "{}"))
            hit["score"] = round(-rank, 3)
            hit["gone"] = bool(gone)
            hits.append(hit)
        return hits

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT count(*) FROM docs").fetchone()[0]
