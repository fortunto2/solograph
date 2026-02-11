"""FalkorDBLite connection and schema management."""

from pathlib import Path

from redislite.falkordb_client import FalkorDB

DEFAULT_DB_PATH = Path.home() / ".solo" / "codegraph.db"


def get_db(db_path: Path | None = None) -> FalkorDB:
    """Get or create FalkorDBLite instance."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return FalkorDB(str(path))


def get_graph(db: FalkorDB, name: str = "codegraph"):
    """Select the codegraph graph."""
    return db.select_graph(name)


def init_schema(graph) -> None:
    """Create indexes for faster lookups.

    FalkorDB doesn't require explicit schema creation —
    nodes and edges are created on the fly. But indexes
    speed up MATCH queries significantly.
    """
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.name)",
        "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.path)",
        "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.project)",
        "CREATE INDEX IF NOT EXISTS FOR (s:Symbol) ON (s.name)",
        "CREATE INDEX IF NOT EXISTS FOR (pkg:Package) ON (pkg.name)",
        "CREATE INDEX IF NOT EXISTS FOR (sess:Session) ON (sess.session_id)",
        "CREATE INDEX IF NOT EXISTS FOR (sess:Session) ON (sess.project_name)",
    ]
    for idx in indexes:
        try:
            graph.query(idx)
        except Exception:
            pass  # Index may already exist


def clear_project(graph, project_name: str) -> None:
    """Remove all nodes and edges for a project before re-scan."""
    queries = [
        f"MATCH (s:Symbol {{project: '{project_name}'}}) DETACH DELETE s",
        f"MATCH (f:File {{project: '{project_name}'}}) DETACH DELETE f",
        f"MATCH (p:Project {{name: '{project_name}'}}) DETACH DELETE p",
    ]
    for q in queries:
        try:
            graph.query(q)
        except Exception:
            pass


def graph_stats(graph) -> dict:
    """Get graph statistics."""
    stats = {}
    for label in ["Project", "File", "Symbol", "Package", "Session"]:
        result = graph.query(f"MATCH (n:{label}) RETURN COUNT(n) AS cnt")
        stats[label] = result.result_set[0][0] if result.result_set else 0

    result = graph.query("MATCH ()-[r]->() RETURN TYPE(r) AS t, COUNT(r) AS cnt")
    edge_stats = {}
    for row in result.result_set:
        edge_stats[row[0]] = row[1]
    stats["edges"] = edge_stats

    return stats
