"""Mermaid diagram generators from Solograph data."""

import re


def _safe_id(text: str) -> str:
    """Convert a path or name to a valid mermaid node ID."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text)


def _safe_label(text: str) -> str:
    """Escape label text for mermaid."""
    return text.replace('"', "'")


def inheritance_diagram(graph, project: str, max_nodes: int = 50) -> str:
    """Generate mermaid flowchart of class inheritance."""
    result = graph.query(
        f"MATCH (child:Symbol)-[:INHERITS]->(parent:Symbol) "
        f"WHERE child.project = '{project}' "
        f"RETURN child.name, parent.name "
        f"ORDER BY parent.name, child.name"
    )
    if not result.result_set:
        return ""

    lines = ["graph TD"]
    seen_nodes: set[str] = set()
    edge_count = 0

    for row in result.result_set:
        if edge_count >= max_nodes:
            break
        child, parent = row[0], row[1]
        child_id = _safe_id(child)
        parent_id = _safe_id(parent)

        if child_id not in seen_nodes:
            lines.append(f'    {child_id}["{_safe_label(child)}"]')
            seen_nodes.add(child_id)
        if parent_id not in seen_nodes:
            lines.append(f'    {parent_id}["{_safe_label(parent)}"]')
            seen_nodes.add(parent_id)

        lines.append(f"    {child_id} --> {parent_id}")
        edge_count += 1

    return "\n".join(lines)


def imports_diagram(graph, project: str, max_nodes: int = 50) -> str:
    """Generate mermaid flowchart of internal file imports."""
    result = graph.query(
        f"MATCH (src:File {{project: '{project}'}})-[:IMPORTS]->(tgt:File {{project: '{project}'}}) "
        f"RETURN src.path, tgt.path "
        f"ORDER BY src.path, tgt.path"
    )
    if not result.result_set:
        return ""

    lines = ["graph LR"]
    seen_nodes: set[str] = set()
    edge_count = 0

    for row in result.result_set:
        if edge_count >= max_nodes:
            break
        src, tgt = row[0], row[1]
        src_id = "f_" + _safe_id(src)
        tgt_id = "f_" + _safe_id(tgt)

        if src_id not in seen_nodes:
            lines.append(f'    {src_id}["{_safe_label(src)}"]')
            seen_nodes.add(src_id)
        if tgt_id not in seen_nodes:
            lines.append(f'    {tgt_id}["{_safe_label(tgt)}"]')
            seen_nodes.add(tgt_id)

        lines.append(f"    {src_id} --> {tgt_id}")
        edge_count += 1

    return "\n".join(lines)


def calls_diagram(
    graph,
    project: str,
    symbol: str | None = None,
    max_nodes: int = 50,
) -> str:
    """Generate mermaid flowchart of call relationships (File -> Symbol)."""
    if symbol:
        sym_escaped = symbol.replace("'", "\\'")
        query = (
            f"MATCH (f:File)-[:CALLS]->(s:Symbol {{name: '{sym_escaped}', project: '{project}'}}) "
            f"WHERE f.path <> s.file "
            f"RETURN f.path, s.name, s.file "
            f"ORDER BY f.path"
        )
    else:
        query = (
            f"MATCH (f:File {{project: '{project}'}})-[:CALLS]->(s:Symbol {{project: '{project}'}}) "
            f"WHERE f.path <> s.file "
            f"RETURN f.path, s.name, s.file "
            f"ORDER BY f.path, s.name"
        )

    result = graph.query(query)
    if not result.result_set:
        return ""

    lines = ["graph LR"]
    seen_nodes: set[str] = set()
    edge_count = 0

    for row in result.result_set:
        if edge_count >= max_nodes:
            break
        caller_path, callee_name, callee_file = row[0], row[1], row[2]
        caller_id = "f_" + _safe_id(caller_path)
        callee_id = "s_" + _safe_id(callee_name + "_" + callee_file)

        if caller_id not in seen_nodes:
            lines.append(f'    {caller_id}["{_safe_label(caller_path)}"]')
            seen_nodes.add(caller_id)
        if callee_id not in seen_nodes:
            lines.append(f'    {callee_id}(["{_safe_label(callee_name)}"])')
            seen_nodes.add(callee_id)

        lines.append(f"    {caller_id} --> {callee_id}")
        edge_count += 1

    return "\n".join(lines)


def deps_diagram(graph, project: str) -> str:
    """Generate mermaid flowchart of package dependencies."""
    result = graph.query(
        f"MATCH (p:Project {{name: '{project}'}})-[:DEPENDS_ON]->(pkg:Package) "
        f"RETURN pkg.name, pkg.source "
        f"ORDER BY pkg.source, pkg.name"
    )
    if not result.result_set:
        return ""

    lines = ["graph LR"]
    project_id = _safe_id(project)
    lines.append(f'    {project_id}[["{_safe_label(project)}"]]')

    # Group by source
    by_source: dict[str, list[str]] = {}
    for row in result.result_set:
        name, source = row[0], row[1] or "unknown"
        by_source.setdefault(source, []).append(name)

    for source, pkgs in sorted(by_source.items()):
        source_id = "src_" + _safe_id(source)
        lines.append(f'    {source_id}{{"{_safe_label(source)}"}}')
        lines.append(f"    {project_id} --> {source_id}")
        for pkg in pkgs:
            pkg_id = "pkg_" + _safe_id(pkg)
            lines.append(f'    {pkg_id}("{_safe_label(pkg)}")')
            lines.append(f"    {source_id} --> {pkg_id}")

    return "\n".join(lines)
