"""The portfolio: every project in the registry, its packages, files and symbol
definitions, in one SQLite file.

This replaced a FalkorDB code graph. Measured 27 Oct 2026: its structural layer
(781k CALLS edges, imports, inheritance, session→file and git edges) got zero
structural questions in four months — five calls in total, every one of them
"list the projects" or stats. Grep answers the per-repo questions as well: on
video-analyzer a dedicated code-graph MCP (colbymchenry/codegraph) against plain
Grep/Read gave the same accuracy in all 8 runs, -16% cost on a call-chain
question and none on an impact question. This keeps what grep cannot do: one
index across every project, so "where else did I write a montage profile?" has
an answer.

Symbol names are indexed by their words: `resolve_montage_profile` and
`MontageProfile` both carry "montage profile" (the camel/snake split
codebase-memory-mcp's tokenizer does). A file is re-read only when its mtime or
size changed, so a rescan of an untouched project is a stat per file.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .fts import match_expression

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects(
    name TEXT PRIMARY KEY, path TEXT, stacks TEXT, status TEXT,
    last_commit TEXT, description TEXT, scanned_at TEXT
);
CREATE TABLE IF NOT EXISTS packages(
    project TEXT NOT NULL, name TEXT NOT NULL, version TEXT, source TEXT NOT NULL,
    PRIMARY KEY (project, source, name)
);
CREATE INDEX IF NOT EXISTS packages_name ON packages(name);
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY, project TEXT NOT NULL, path TEXT NOT NULL,
    lang TEXT, lines INTEGER, mtime REAL, size INTEGER, UNIQUE (project, path)
);
CREATE TABLE IF NOT EXISTS symbols(
    id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, name TEXT NOT NULL, kind TEXT, line INTEGER
);
CREATE INDEX IF NOT EXISTS symbols_file ON symbols(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    words, tokenize='unicode61 remove_diacritics 2'
);
"""

_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_identifier(name: str) -> str:
    """`resolve_montage_profile`, `MontageProfile`, `HTTPServer` → their lowercase words."""
    return " ".join(w.lower() for w in _WORD.findall(name))


def default_db_path() -> Path:
    """$SOLOGRAPH_PORTFOLIO_DB, else portfolio.db next to $CODEGRAPH_DB_PATH.

    Next to it so every profile that already sets CODEGRAPH_DB_PATH (the Epiphan
    one points into its own .solograph/) gets its own store without new config.
    """
    explicit = os.environ.get("SOLOGRAPH_PORTFOLIO_DB")
    if explicit:
        return Path(explicit).expanduser()
    graph_db = os.environ.get("CODEGRAPH_DB_PATH", str(Path.home() / ".solo" / "codegraph.db"))
    return Path(graph_db).expanduser().parent / "portfolio.db"


def _dir_key(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}/"
    return f"{parts[0]}/" if len(parts) == 2 else "(root)"


class Portfolio:
    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path).expanduser() if db_path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    # ── writing ──────────────────────────────────────────────────────

    def sync_projects(self, entries: list[dict], prune: bool = True) -> list[str]:
        """Upsert registry entries; with `prune`, drop projects no longer listed. Returns the dropped names."""
        db = self._db
        with db:
            for e in entries:
                stacks = e.get("stacks") or []
                db.execute(
                    "INSERT INTO projects(name, path, stacks, status, last_commit, description) "
                    "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET path = excluded.path, "
                    "stacks = excluded.stacks, status = excluded.status, "
                    "last_commit = excluded.last_commit, description = excluded.description",
                    (
                        e["name"],
                        e["path"],
                        ",".join(stacks) if isinstance(stacks, list) else str(stacks),
                        e.get("status", "active"),
                        str(e.get("last_commit") or ""),
                        (e.get("description") or "")[:500],
                    ),
                )
            dropped = []
            if prune:
                listed = {e["name"] for e in entries}
                dropped = [n for (n,) in db.execute("SELECT name FROM projects") if n not in listed]
                for name in dropped:
                    self._forget_project(name)
        return dropped

    def _forget_project(self, name: str) -> None:
        db = self._db
        ids = [i for (i,) in db.execute("SELECT id FROM files WHERE project = ?", (name,))]
        self._forget_files(ids)
        db.execute("DELETE FROM packages WHERE project = ?", (name,))
        db.execute("DELETE FROM projects WHERE name = ?", (name,))

    def _forget_files(self, file_ids: list[int]) -> None:
        db = self._db
        for fid in file_ids:
            db.execute("DELETE FROM symbols_fts WHERE rowid IN (SELECT id FROM symbols WHERE file_id = ?)", (fid,))
            db.execute("DELETE FROM symbols WHERE file_id = ?", (fid,))
            db.execute("DELETE FROM files WHERE id = ?", (fid,))

    def scan_project(self, name: str, path: str | Path) -> dict:
        """Bring one project's files, symbols and packages in line with the disk."""
        from .scanner.code import LANG_MAP, SYMBOL_LANGS, extract_symbols, project_files
        from .scanner.deps import scan_deps

        root = Path(path)
        report = {"read": 0, "unchanged": 0, "removed": 0, "symbols": 0}
        db = self._db
        known = {
            p: (fid, mtime, size)
            for fid, p, mtime, size in db.execute("SELECT id, path, mtime, size FROM files WHERE project = ?", (name,))
        }
        seen: set[str] = set()
        with db:
            for fp, lang in project_files(root, LANG_MAP):
                rel = str(fp.relative_to(root))
                seen.add(rel)
                try:
                    st = fp.stat()
                except OSError:
                    continue
                prev = known.get(rel)
                if prev and prev[1:] == (st.st_mtime, st.st_size):
                    report["unchanged"] += 1
                    continue
                if prev:
                    self._forget_files([prev[0]])
                try:
                    lines = fp.read_bytes().count(b"\n") + 1
                except OSError:
                    lines = 0
                fid = db.execute(
                    "INSERT INTO files(project, path, lang, lines, mtime, size) VALUES (?, ?, ?, ?, ?, ?)",
                    (name, rel, lang, lines, st.st_mtime, st.st_size),
                ).lastrowid
                symbols = extract_symbols(fp, name, lang, rel_path=rel) if lang in SYMBOL_LANGS else []
                for sym in symbols:
                    sid = db.execute(
                        "INSERT INTO symbols(file_id, name, kind, line) VALUES (?, ?, ?, ?)",
                        (fid, sym.name, sym.kind, sym.line),
                    ).lastrowid
                    db.execute(
                        "INSERT INTO symbols_fts(rowid, words) VALUES (?, ?)",
                        (sid, f"{split_identifier(sym.name)} {sym.name.lower()}"),
                    )
                report["read"] += 1
                report["symbols"] += len(symbols)
            gone = [v[0] for k, v in known.items() if k not in seen]
            self._forget_files(gone)
            report["removed"] = len(gone)

            db.execute("DELETE FROM packages WHERE project = ?", (name,))
            for pkg in scan_deps(root):
                db.execute(
                    "INSERT OR REPLACE INTO packages(project, name, version, source) VALUES (?, ?, ?, ?)",
                    (name, pkg.name, pkg.version or "", pkg.source),
                )
            db.execute(
                "UPDATE projects SET scanned_at = ? WHERE name = ?",
                (datetime.now(UTC).isoformat(timespec="seconds"), name),
            )
        return report

    # ── reading ──────────────────────────────────────────────────────

    def search_symbols(
        self, query: str, n_results: int = 20, project: str | None = None, kind: str | None = None
    ) -> list[dict]:
        """Definitions whose name contains the query's words, best first; an exact name wins."""
        words = " ".join(split_identifier(w) or w for w in query.split())
        expr = match_expression(words)
        if not expr:
            return []
        sql = (
            "SELECT s.id, s.name, s.kind, s.line, f.path, f.project, bm25(symbols_fts) AS rank "
            "FROM symbols_fts JOIN symbols s ON s.id = symbols_fts.rowid JOIN files f ON f.id = s.file_id "
            "WHERE symbols_fts MATCH ?"
        )
        params: list = [expr]
        if project:
            sql += " AND f.project = ?"
            params.append(project)
        if kind:
            sql += " AND s.kind = ?"
            params.append(kind)
        sql += " ORDER BY rank LIMIT ?"
        exact = re.sub(r"[\s_]", "", query).lower()
        limit = max(n_results * 5, 50)
        # All the words first, then any of them: "montage profile" should mean
        # both, but "authentication middleware" should still find AuthMiddleware.
        hits: dict[int, dict] = {}
        for match in (" AND ".join(expr.split(" OR ")), expr):
            for sid, name, k, line, path, proj, rank in self._db.execute(sql, [match, *params[1:], limit]):
                if sid in hits:
                    continue
                boost = 3 if name.replace("_", "").lower() == exact else 1
                tier = 0 if match != expr or " OR " not in expr else 1
                hits[sid] = {
                    "name": name,
                    "kind": k,
                    "project": proj,
                    "file": path,
                    "line": line,
                    "score": round(-rank * boost, 3),
                    "_tier": tier,
                }
            if len(hits) >= n_results:
                break
        ranked = sorted(hits.values(), key=lambda h: (h.pop("_tier"), -h["score"]))
        return ranked[:n_results]

    def shared_packages(self, package: str | None = None) -> list[dict]:
        """Packages used by more than one project — or, given a name, every project using it."""
        if package:
            rows = self._db.execute(
                "SELECT name, source, group_concat(project, '|') FROM packages WHERE lower(name) LIKE ? "
                "GROUP BY name, source ORDER BY count(*) DESC, name",
                (f"%{package.lower()}%",),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT name, source, group_concat(project, '|') FROM packages GROUP BY name, source "
                "HAVING count(DISTINCT project) > 1 ORDER BY count(*) DESC, name"
            ).fetchall()
        return [{"package": n, "source": s, "projects": sorted(set(p.split("|")))} for n, s, p in rows]

    def stats(self) -> dict:
        db = self._db
        out = {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("projects", "files", "symbols")}
        out["packages"] = db.execute("SELECT count(DISTINCT name) FROM packages").fetchone()[0]
        out["scanned_projects"] = db.execute("SELECT count(*) FROM projects WHERE scanned_at IS NOT NULL").fetchone()[0]
        out["symbols_by_kind"] = dict(db.execute("SELECT kind, count(*) FROM symbols GROUP BY kind ORDER BY 2 DESC"))
        return out

    def dependencies(self, project: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT name, version, source FROM packages WHERE project = ? ORDER BY source, name", (project,)
        )
        return [{"name": n, "version": v, "source": s} for n, v, s in rows]

    def explain(self, project: str) -> dict | str:
        """What a project is made of: stack, languages, directories, patterns, dependencies, biggest files."""
        db = self._db
        row = db.execute("SELECT stacks, description FROM projects WHERE name = ?", (project,)).fetchone()
        if not row:
            return f"Project '{project}' not found."
        files = db.execute("SELECT id, path, lang, lines FROM files WHERE project = ?", (project,)).fetchall()
        symbols = db.execute(
            "SELECT s.name, s.kind, f.path FROM symbols s JOIN files f ON f.id = s.file_id WHERE f.project = ?",
            (project,),
        ).fetchall()
        deps = db.execute(
            "SELECT name, source FROM packages WHERE project = ? ORDER BY source, name", (project,)
        ).fetchall()
        data: dict = {
            "name": project,
            "stack": row[0] or "(no stack)",
            "description": row[1] or "",
            "counts": {"files": len(files), "symbols": len(symbols), "packages": len(deps)},
        }
        langs: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        dirs: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for _fid, path, lang, lines in files:
            if lang:
                langs[lang][0] += 1
                langs[lang][1] += lines or 0
            dirs[_dir_key(path)][0] += 1
        per_file: dict[str, int] = defaultdict(int)
        for _name, _kind, path in symbols:
            dirs[_dir_key(path)][1] += 1
            per_file[path] += 1
        if langs:
            data["languages"] = sorted(((k, v[0], v[1]) for k, v in langs.items()), key=lambda x: -x[1])
        if dirs:
            data["layers"] = sorted(((k, v[0], v[1]) for k, v in dirs.items()), key=lambda x: -x[1])[:12]
        patterns = []
        mixins = sorted(n for n, k, _ in symbols if k == "class" and "Mixin" in n)
        if mixins:
            patterns.append(f"Mixins: {', '.join(mixins[:8])}")
        crud = sum(1 for n, k, _ in symbols if k == "class" and n.endswith(("Create", "Read", "Update", "Schema")))
        if crud > 3:
            patterns.append(f"CRUD schemas: {crud} Create/Read/Update/Schema classes")
        protocols = sum(1 for _, k, _ in symbols if k in ("protocol", "interface"))
        if protocols:
            patterns.append(f"Protocols/Interfaces: {protocols}")
        if patterns:
            data["patterns"] = patterns
        if deps:
            data["dependencies"] = [(n, s) for n, s in deps][:15]
        biggest = sorted(per_file.items(), key=lambda x: -x[1])[:8]
        if biggest:
            data["largest_files"] = [(p, n) for p, n in biggest if n >= 3]
        return data
