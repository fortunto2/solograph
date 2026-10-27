"""Scan registry.yaml → Project nodes in graph."""

from pathlib import Path

import yaml

from ..db import cypher_str
from ..models import ProjectNode


def scan_registry(registry_path: Path) -> list[ProjectNode]:
    """Parse registry.yaml and return ProjectNode list."""
    with open(registry_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    projects = []
    for entry in data.get("projects", []):
        projects.append(
            ProjectNode(
                name=entry["name"],
                stack=",".join(entry.get("stacks", [])),
                path=entry["path"],
                status=entry.get("status", "active"),
                has_claude_md=entry.get("has_claude_md", False),
                description=entry.get("description", ""),
            )
        )
    return projects


def ingest_projects(graph, projects: list[ProjectNode]) -> int:
    """Create Project nodes in the graph."""
    count = 0
    for p in projects:
        desc = p.description.replace("'", "\\'").replace('"', '\\"')[:200]
        graph.query(
            f"MERGE (p:Project {{name: {cypher_str(p.name)}}}) "
            f"SET p.stack = {cypher_str(p.stack)}, p.path = {cypher_str(p.path)}, "
            f"p.status = {cypher_str(p.status)}, p.has_claude_md = {str(p.has_claude_md).lower()}, "
            f'p.description = "{desc}"'
        )
        count += 1
    return count
