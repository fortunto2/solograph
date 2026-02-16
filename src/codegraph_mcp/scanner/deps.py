"""Scan dependency files → Package nodes and DEPENDS_ON edges."""

import json
from pathlib import Path

from ..models import PackageNode


def scan_deps(project_path: Path) -> list[PackageNode]:
    """Detect and parse dependency files."""
    packages = []

    # package.json (npm/pnpm) — root + one level deep (monorepos)
    for pkg_json in _find_dep_files(project_path, "package.json"):
        packages.extend(_parse_package_json(pkg_json))

    # pyproject.toml (uv/pip/poetry)
    pyproject = project_path / "pyproject.toml"
    if pyproject.exists():
        packages.extend(_parse_pyproject(pyproject))

    # requirements.txt fallback
    req_txt = project_path / "requirements.txt"
    if req_txt.exists() and not pyproject.exists():
        packages.extend(_parse_requirements_txt(req_txt))

    # Package.swift (SPM) — just detect, no deep parse
    pkg_swift = project_path / "Package.swift"
    if pkg_swift.exists():
        packages.extend(_parse_package_swift(pkg_swift))

    # build.gradle.kts (Kotlin)
    gradle = project_path / "app" / "build.gradle.kts"
    if not gradle.exists():
        gradle = project_path / "build.gradle.kts"
    if gradle.exists():
        packages.extend(_parse_gradle(gradle))

    return packages


def _find_dep_files(project_path: Path, filename: str) -> list[Path]:
    """Find dep file in root and one level deep (for monorepos)."""
    found = []
    root_file = project_path / filename
    if root_file.exists():
        found.append(root_file)
    for child in project_path.iterdir():
        if (
            child.is_dir()
            and not child.name.startswith(".")
            and child.name not in ("node_modules", ".next", "dist", "build", "__pycache__")
        ):
            sub = child / filename
            if sub.exists():
                found.append(sub)
    return found


def _parse_package_json(path: Path) -> list[PackageNode]:
    """Parse package.json for dependencies."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    packages = []
    for section in ("dependencies", "devDependencies"):
        for name, version in data.get(section, {}).items():
            packages.append(PackageNode(name=name, version=version.lstrip("^~>="), source="npm"))
    return packages


def _parse_pyproject(path: Path) -> list[PackageNode]:
    """Parse pyproject.toml for dependencies."""
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    packages = []

    # PEP 621 format: [project].dependencies = ["package>=1.0"]
    for dep in data.get("project", {}).get("dependencies", []):
        name = dep.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
        version = dep.replace(name, "").strip().lstrip(">=<~!")
        packages.append(PackageNode(name=name, version=version or None, source="pip"))

    # Poetry format: [tool.poetry.dependencies] / [tool.poetry.dev-dependencies]
    poetry = data.get("tool", {}).get("poetry", {})
    for section in ("dependencies", "dev-dependencies"):
        for name, ver in poetry.get(section, {}).items():
            if name == "python":
                continue
            if isinstance(ver, dict):
                ver = ver.get("version", "")
            version = str(ver).lstrip("^~>=<!")
            packages.append(PackageNode(name=name, version=version or None, source="pip"))

    return packages


def _parse_requirements_txt(path: Path) -> list[PackageNode]:
    """Parse requirements.txt."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    packages = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = line.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
        version = line.replace(name, "").strip().lstrip(">=<~!")
        if name:
            packages.append(PackageNode(name=name, version=version or None, source="pip"))
    return packages


def _parse_package_swift(path: Path) -> list[PackageNode]:
    """Extract package names from Package.swift (basic regex)."""
    import re

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return []

    packages = []
    # Match .package(url: "https://github.com/org/name", ...)
    for match in re.finditer(r'\.package\(url:\s*"https?://[^/]+/[^/]+/([^"]+)"', content):
        name = match.group(1).rstrip(".git")
        packages.append(PackageNode(name=name, source="spm"))
    return packages


def _parse_gradle(path: Path) -> list[PackageNode]:
    """Extract dependencies from build.gradle.kts (basic regex)."""
    import re

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return []

    packages = []
    # Match implementation("group:artifact:version")
    for match in re.finditer(r'implementation\("([^:]+):([^:]+):([^"]+)"\)', content):
        name = f"{match.group(1)}:{match.group(2)}"
        version = match.group(3)
        packages.append(PackageNode(name=name, version=version, source="gradle"))
    return packages


def ingest_packages(graph, packages: list[PackageNode], project_name: str) -> int:
    """Create Package nodes and DEPENDS_ON edges."""
    count = 0
    for pkg in packages:
        name_escaped = pkg.name.replace("'", "\\'")
        version = pkg.version or ""
        graph.query(
            f"MERGE (pkg:Package {{name: '{name_escaped}'}}) SET pkg.version = '{version}', pkg.source = '{pkg.source}'"
        )
        graph.query(
            f"MATCH (p:Project {{name: '{project_name}'}}), "
            f"(pkg:Package {{name: '{name_escaped}'}}) "
            f"MERGE (p)-[:DEPENDS_ON]->(pkg)"
        )
        count += 1
    return count
