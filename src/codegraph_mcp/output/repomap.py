"""Generate a RepoMap (like Aider) from graph data."""

from pathlib import Path

import yaml


def _get_line_from_file(project_path: str, filepath: str, line_num: int) -> str:
    """Read a specific line from a file to use as the signature."""
    if not project_path:
        return ""

    target_path = Path(project_path) / filepath
    if not target_path.exists():
        return ""

    try:
        lines = target_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        # line_num is 1-indexed from tree-sitter start_point[0] + 1
        idx = line_num - 1
        if 0 <= idx < len(lines):
            line_str = lines[idx].strip()

            # If it's a decorator (like @property or @mcp.tool()), get the next non-decorator line too
            if line_str.startswith("@") and idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                if not next_line.startswith("@"):
                    line_str = f"{line_str} {next_line}"

            return line_str
    except Exception:
        pass

    return ""


def generate_repomap(graph, project_name: str, max_files: int = 20) -> str:
    """Generate a YAML-formatted RepoMap for the most important files.

    Ranks files by degree (incoming/outgoing edges), takes the top N files,
    and lists their defined symbols WITH signatures.
    """
    # 1. Check if project exists and get its path
    proj_result = graph.query(f"MATCH (p:Project {{name: '{project_name}'}}) RETURN p.path")
    if not proj_result.result_set:
        return f"Error: Project '{project_name}' not found in graph."

    project_path = proj_result.result_set[0][0]

    # 2. Get top "hub" files by total connections (IMPORTS + CALLS etc)
    hub_result = graph.query(
        f"MATCH (f:File {{project: '{project_name}'}})-[r]-() "
        f"RETURN f.path, COUNT(r) AS connections "
        f"ORDER BY connections DESC LIMIT {max_files}"
    )

    if not hub_result.result_set:
        return f"No files found for project '{project_name}'."

    top_files = [row[0] for row in hub_result.result_set]

    # 3. Fetch symbols for these top files
    repomap_dict = {}

    for filepath in top_files:
        sym_result = graph.query(
            f"MATCH (f:File {{path: '{filepath}', project: '{project_name}'}})-[:DEFINES]->(s:Symbol) "
            f"RETURN s.name, s.kind, s.line "
            f"ORDER BY s.line"
        )

        symbols = []
        if sym_result.result_set:
            for row in sym_result.result_set:
                name, kind, line_num = row[0], row[1], row[2]

                # Fetch signature from file
                signature = _get_line_from_file(project_path, filepath, line_num)

                sym_data = {"name": name, "kind": kind, "line": line_num}
                if signature:
                    sym_data["signature"] = signature

                symbols.append(sym_data)

        # Only include files that actually export symbols (or at least note they exist)
        if symbols:
            repomap_dict[filepath] = {"symbols": symbols}
        else:
            repomap_dict[filepath] = {"symbols": []}

    # Format as YAML
    yaml_str = yaml.dump(repomap_dict, default_flow_style=False, sort_keys=False)

    header = f"RepoMap for {project_name} (Top {len(top_files)} files by importance):\n"
    header += "-" * 50 + "\n"

    return header + yaml_str
