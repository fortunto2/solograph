"""Architecture explainer — human-readable project overview from graph data."""

from collections import defaultdict


def explain_project(graph, project_name: str) -> dict | str:
    """Generate structured architecture overview from graph data.

    Returns a dict with structured data for Rich rendering,
    or a string error message if project not found.
    """

    # ── Header: project info ──
    proj_result = graph.query(
        f"MATCH (p:Project {{name: '{project_name}'}}) "
        f"RETURN p.stack, p.description"
    )
    if not proj_result.result_set:
        return f"Project '{project_name}' not found in graph."

    stack = proj_result.result_set[0][0] or "(no stack)"
    description = proj_result.result_set[0][1] or ""

    # Counts
    file_count = _count(graph, f"MATCH (f:File {{project: '{project_name}'}}) RETURN COUNT(f)")
    sym_count = _count(graph, f"MATCH (s:Symbol {{project: '{project_name}'}}) RETURN COUNT(s)")
    pkg_count = _count(
        graph,
        f"MATCH (p:Project {{name: '{project_name}'}})-[:DEPENDS_ON]->(pkg:Package) RETURN COUNT(pkg)",
    )

    data: dict = {
        "name": project_name,
        "stack": stack,
        "description": description,
        "counts": {"files": file_count, "symbols": sym_count, "packages": pkg_count},
    }

    # ── Language breakdown ──
    lang_result = graph.query(
        f"MATCH (f:File {{project: '{project_name}'}}) "
        f"WHERE f.lang IS NOT NULL AND f.lang <> '' "
        f"RETURN f.lang, COUNT(f) AS cnt, SUM(f.lines) AS total_lines "
        f"ORDER BY cnt DESC"
    )
    if lang_result.result_set:
        data["languages"] = [
            (row[0], row[1], row[2] or 0) for row in lang_result.result_set
        ]

    # ── Directory layers ──
    dir_result = graph.query(
        f"MATCH (f:File {{project: '{project_name}'}}) "
        f"RETURN f.path "
        f"ORDER BY f.path"
    )
    if dir_result.result_set:
        dir_stats: dict[str, dict] = defaultdict(lambda: {"files": 0, "symbols": 0})
        for row in dir_result.result_set:
            path = row[0]
            parts = path.split("/")
            if len(parts) >= 3:
                dir_key = f"{parts[0]}/{parts[1]}/"
            elif len(parts) >= 2:
                dir_key = f"{parts[0]}/"
            else:
                dir_key = "(root)"
            dir_stats[dir_key]["files"] += 1

        # Count symbols per directory
        sym_dir_result = graph.query(
            f"MATCH (s:Symbol {{project: '{project_name}'}}) "
            f"RETURN s.file"
        )
        for row in sym_dir_result.result_set:
            path = row[0]
            parts = path.split("/")
            if len(parts) >= 3:
                dir_key = f"{parts[0]}/{parts[1]}/"
            elif len(parts) >= 2:
                dir_key = f"{parts[0]}/"
            else:
                dir_key = "(root)"
            dir_stats[dir_key]["symbols"] += 1

        sorted_dirs = sorted(dir_stats.items(), key=lambda x: -x[1]["files"])
        data["layers"] = [
            (dir_key, stats["files"], stats["symbols"])
            for dir_key, stats in sorted_dirs[:12]
        ]

    # ── Key patterns (heuristic) ──
    patterns: list[str] = []

    # Mixins
    mixin_result = graph.query(
        f"MATCH (s:Symbol {{project: '{project_name}'}}) "
        f"WHERE s.name CONTAINS 'Mixin' AND s.kind = 'class' "
        f"RETURN s.name ORDER BY s.name"
    )
    if mixin_result.result_set:
        mixin_names = [r[0] for r in mixin_result.result_set]
        patterns.append(f"Mixins: {', '.join(mixin_names[:8])}")

    # Abstract/Base classes with children
    base_result = graph.query(
        f"MATCH (child:Symbol {{project: '{project_name}'}})-[:INHERITS]->(parent:Symbol {{project: '{project_name}'}}) "
        f"RETURN parent.name, COUNT(child) AS children "
        f"ORDER BY children DESC LIMIT 5"
    )
    if base_result.result_set:
        for row in base_result.result_set:
            parent, children_count = row[0], row[1]
            if children_count >= 2:
                patterns.append(f"Base class: {parent} -> {children_count} children")

    # CRUD schemas
    crud_result = graph.query(
        f"MATCH (s:Symbol {{project: '{project_name}'}}) "
        f"WHERE s.kind = 'class' AND "
        f"(s.name ENDS WITH 'Create' OR s.name ENDS WITH 'Read' OR "
        f" s.name ENDS WITH 'Update' OR s.name ENDS WITH 'Schema') "
        f"RETURN COUNT(s)"
    )
    if crud_result.result_set and crud_result.result_set[0][0] > 3:
        count = crud_result.result_set[0][0]
        patterns.append(f"CRUD schemas: {count} Create/Read/Update/Schema classes")

    # Protocol/Interface count
    proto_result = graph.query(
        f"MATCH (s:Symbol {{project: '{project_name}'}}) "
        f"WHERE s.kind IN ['protocol', 'interface'] "
        f"RETURN COUNT(s)"
    )
    if proto_result.result_set and proto_result.result_set[0][0] > 0:
        count = proto_result.result_set[0][0]
        patterns.append(f"Protocols/Interfaces: {count}")

    if patterns:
        data["patterns"] = patterns

    # ── Top dependencies ──
    deps_result = graph.query(
        f"MATCH (p:Project {{name: '{project_name}'}})-[:DEPENDS_ON]->(pkg:Package) "
        f"RETURN pkg.name, pkg.source "
        f"ORDER BY pkg.source, pkg.name LIMIT 15"
    )
    if deps_result.result_set:
        data["dependencies"] = [(r[0], r[1]) for r in deps_result.result_set]

    # ── Hub files (most connections) ──
    hub_result = graph.query(
        f"MATCH (f:File {{project: '{project_name}'}})-[r]-() "
        f"RETURN f.path, COUNT(r) AS connections "
        f"ORDER BY connections DESC LIMIT 8"
    )
    if hub_result.result_set:
        hubs = [(row[0], row[1]) for row in hub_result.result_set if row[1] >= 3]
        if hubs:
            data["hub_files"] = hubs

    return data


def _count(graph, query: str) -> int:
    """Run a COUNT query and return the integer result."""
    result = graph.query(query)
    return result.result_set[0][0] if result.result_set else 0
