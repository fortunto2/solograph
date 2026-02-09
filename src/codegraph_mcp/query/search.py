"""Pre-built Cypher queries for common operations."""


def files_by_project(graph, project_name: str) -> list[dict]:
    """List all files in a project."""
    result = graph.query(
        f"MATCH (p:Project {{name: '{project_name}'}})-[:HAS_FILE]->(f:File) "
        f"RETURN f.path AS path, f.lang AS lang, f.lines AS lines "
        f"ORDER BY f.lang, f.path"
    )
    return [{"path": r[0], "lang": r[1], "lines": r[2]} for r in result.result_set]


def deps_by_project(graph, project_name: str) -> list[dict]:
    """List dependencies of a project."""
    result = graph.query(
        f"MATCH (p:Project {{name: '{project_name}'}})-[:DEPENDS_ON]->(pkg:Package) "
        f"RETURN pkg.name AS name, pkg.version AS version, pkg.source AS source "
        f"ORDER BY pkg.source, pkg.name"
    )
    return [{"name": r[0], "version": r[1], "source": r[2]} for r in result.result_set]


def shared_packages(graph) -> list[dict]:
    """Find packages used by multiple projects."""
    result = graph.query(
        "MATCH (p1:Project)-[:DEPENDS_ON]->(pkg:Package)<-[:DEPENDS_ON]-(p2:Project) "
        "WHERE p1.name <> p2.name "
        "RETURN pkg.name, COLLECT(DISTINCT p1.name) AS projects "
        "ORDER BY SIZE(projects) DESC"
    )
    return [{"package": r[0], "projects": r[1]} for r in result.result_set]


def hotfiles(graph, project_name: str, limit: int = 10) -> list[dict]:
    """Find most frequently modified files."""
    result = graph.query(
        f"MATCH (f:File {{project: '{project_name}'}})<-[m:MODIFIED]-(f) "
        f"RETURN f.path, COUNT(m) AS changes, SUM(m.lines_added) AS added "
        f"ORDER BY changes DESC LIMIT {limit}"
    )
    return [{"path": r[0], "changes": r[1], "added": r[2]} for r in result.result_set]


def hub_files(graph, limit: int = 10) -> list[dict]:
    """Find files with the most symbols defined."""
    result = graph.query(
        "MATCH (f:File)-[:DEFINES]->(s:Symbol) "
        f"RETURN f.path, f.project, COUNT(s) AS symbols "
        f"ORDER BY symbols DESC LIMIT {limit}"
    )
    return [{"path": r[0], "project": r[1], "symbols": r[2]} for r in result.result_set]


def symbols_in_file(graph, file_path: str) -> list[dict]:
    """List symbols defined in a file."""
    path_escaped = file_path.replace("'", "\\'")
    result = graph.query(
        f"MATCH (f:File {{path: '{path_escaped}'}})-[:DEFINES]->(s:Symbol) "
        f"RETURN s.name, s.kind, s.line ORDER BY s.line"
    )
    return [{"name": r[0], "kind": r[1], "line": r[2]} for r in result.result_set]


# ── Session queries ──────────────────────────────────────────────


def sessions_by_project(graph, project_name: str, limit: int = 20) -> list[dict]:
    """List sessions for a project, newest first."""
    result = graph.query(
        f"MATCH (s:Session {{project_name: '{project_name}'}}) "
        f"RETURN s.session_id, s.started_at, s.slug, s.user_prompts, "
        f"       s.tool_uses, s.model, s.duration_ms "
        f"ORDER BY s.started_at DESC LIMIT {limit}"
    )
    return [
        {
            "session_id": r[0],
            "started_at": r[1],
            "slug": r[2],
            "user_prompts": r[3],
            "tool_uses": r[4],
            "model": r[5],
            "duration_ms": r[6],
        }
        for r in result.result_set
    ]


def files_in_session(graph, session_id: str) -> list[dict]:
    """List files touched/edited/created in a session."""
    result = graph.query(
        f"MATCH (s:Session {{session_id: '{session_id}'}})-[r]->(f:File) "
        f"WHERE TYPE(r) IN ['TOUCHED', 'EDITED', 'CREATED'] "
        f"RETURN f.path, f.project, TYPE(r) AS action, r.count "
        f"ORDER BY action, f.path"
    )
    return [
        {"path": r[0], "project": r[1], "action": r[2], "count": r[3]}
        for r in result.result_set
    ]


def sessions_for_file(graph, file_path: str, limit: int = 20) -> list[dict]:
    """Find sessions that touched a specific file."""
    path_escaped = file_path.replace("'", "\\'")
    result = graph.query(
        f"MATCH (s:Session)-[r]->(f:File) "
        f"WHERE f.path CONTAINS '{path_escaped}' "
        f"AND TYPE(r) IN ['TOUCHED', 'EDITED', 'CREATED'] "
        f"RETURN s.session_id, s.project_name, s.started_at, s.slug, "
        f"       TYPE(r) AS action, r.count "
        f"ORDER BY s.started_at DESC LIMIT {limit}"
    )
    return [
        {
            "session_id": r[0],
            "project_name": r[1],
            "started_at": r[2],
            "slug": r[3],
            "action": r[4],
            "count": r[5],
        }
        for r in result.result_set
    ]


def hotfiles_by_sessions(graph, project_name: str, limit: int = 10) -> list[dict]:
    """Files most frequently edited across sessions in a project."""
    result = graph.query(
        f"MATCH (s:Session {{project_name: '{project_name}'}})-[r:EDITED]->(f:File) "
        f"RETURN f.path, COUNT(r) AS sessions, SUM(r.count) AS edits "
        f"ORDER BY sessions DESC LIMIT {limit}"
    )
    return [
        {"path": r[0], "sessions": r[1], "edits": r[2]}
        for r in result.result_set
    ]


def import_graph(graph, project_name: str) -> list[dict]:
    """File import relationships for a project."""
    result = graph.query(
        f"MATCH (src:File {{project: '{project_name}'}})-[:IMPORTS]->(tgt) "
        f"RETURN src.path AS source, labels(tgt)[0] AS target_type, "
        f"CASE WHEN tgt.path IS NOT NULL THEN tgt.path ELSE tgt.name END AS target "
        f"ORDER BY src.path, target"
    )
    return [
        {"source": r[0], "target_type": r[1], "target": r[2]}
        for r in result.result_set
    ]


def callers_of(graph, symbol_name: str, project: str | None = None) -> list[dict]:
    """Find files that call a given symbol."""
    name_escaped = symbol_name.replace("'", "\\'")
    project_filter = f" AND s.project = '{project}'" if project else ""
    result = graph.query(
        f"MATCH (f:File)-[:CALLS]->(s:Symbol {{name: '{name_escaped}'}}) "
        f"WHERE f.path <> s.file{project_filter} "
        f"RETURN f.path AS caller, f.project AS project, s.file AS defined_in "
        f"ORDER BY f.project, f.path"
    )
    return [
        {"caller": r[0], "project": r[1], "defined_in": r[2]}
        for r in result.result_set
    ]


def class_hierarchy(graph, class_name: str, project: str | None = None) -> dict:
    """Get parents and children for a class."""
    name_escaped = class_name.replace("'", "\\'")
    project_filter = f" AND c.project = '{project}'" if project else ""

    parents_result = graph.query(
        f"MATCH (c:Symbol {{name: '{name_escaped}'}})-[:INHERITS]->(p:Symbol) "
        f"WHERE true{project_filter} "
        f"RETURN p.name AS name, p.file AS file, p.project AS project"
    )
    parents = [
        {"name": r[0], "file": r[1], "project": r[2]}
        for r in parents_result.result_set
    ]

    children_result = graph.query(
        f"MATCH (child:Symbol)-[:INHERITS]->(c:Symbol {{name: '{name_escaped}'}}) "
        f"WHERE true{project_filter} "
        f"RETURN child.name AS name, child.file AS file, child.project AS project"
    )
    children = [
        {"name": r[0], "file": r[1], "project": r[2]}
        for r in children_result.result_set
    ]

    return {"class": class_name, "parents": parents, "children": children}


def external_imports_by_project(graph, project_name: str) -> list[dict]:
    """Package import counts for a project (via IMPORTS edges)."""
    result = graph.query(
        f"MATCH (f:File {{project: '{project_name}'}})-[:IMPORTS]->(pkg:Package) "
        f"RETURN pkg.name AS package, COUNT(f) AS importers "
        f"ORDER BY importers DESC"
    )
    return [{"package": r[0], "importers": r[1]} for r in result.result_set]


def session_stats(graph) -> dict:
    """Aggregate session statistics."""
    total = graph.query("MATCH (s:Session) RETURN COUNT(s)")
    total_count = total.result_set[0][0] if total.result_set else 0

    projects = graph.query(
        "MATCH (s:Session) RETURN s.project_name, COUNT(s) AS cnt "
        "ORDER BY cnt DESC"
    )
    by_project = {r[0]: r[1] for r in projects.result_set}

    edges = graph.query(
        "MATCH (s:Session)-[r]->() "
        "WHERE TYPE(r) IN ['TOUCHED', 'EDITED', 'CREATED', 'IN_PROJECT'] "
        "RETURN TYPE(r), COUNT(r)"
    )
    edge_counts = {r[0]: r[1] for r in edges.result_set}

    return {
        "total_sessions": total_count,
        "by_project": by_project,
        "edges": edge_counts,
    }
