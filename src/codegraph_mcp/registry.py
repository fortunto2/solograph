#!/usr/bin/env python3
"""
Project Registry — scans startups/active/ and builds a registry with detected stacks.

Usage:
    python scripts/project_registry.py scan          # Scan and update registry
    python scripts/project_registry.py list          # List projects
    python scripts/project_registry.py info <name>   # Show project details
    python scripts/project_registry.py stacks        # List available stack templates
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Configurable via env vars
STACKS_DIR = Path(os.environ.get("CODEGRAPH_STACKS_DIR", str(Path.home() / ".solo" / "stacks"))).expanduser()
REGISTRY_PATH = Path(os.environ.get("CODEGRAPH_REGISTRY", str(Path.home() / ".solo" / "registry.yaml"))).expanduser()
_scan_raw = os.environ.get("CODEGRAPH_SCAN_PATH", str(Path.home() / "startups" / "active"))
SCAN_PATHS = [Path(p.strip()).expanduser() for p in _scan_raw.split(",") if p.strip()]
SCAN_PATH = SCAN_PATHS[0]  # backward compat: primary path
OLD_PATH = Path(os.environ.get("CODEGRAPH_OLD_PATH", str(Path.home() / "projects" / "archive"))).expanduser()


class ProjectInfo(BaseModel):
    name: str
    path: str
    status: Literal["active", "paused", "idea", "archived"] = "active"
    stacks: list[str] = Field(default_factory=list)
    has_git: bool = False
    has_claude_md: bool = False
    last_commit: str | None = None
    description: str = ""
    prd_path: str | None = None


class ProjectRegistry(BaseModel):
    projects: list[ProjectInfo] = Field(default_factory=list)
    scan_path: str = ""
    last_scan: str = ""


# Stack detection rules: (indicator_file_or_dir, max_depth, stack_name)
STACK_DETECTORS: list[tuple[str, int, str]] = [
    ("*.xcodeproj", 3, "ios-swift"),
    ("project.yml", 2, "ios-swift"),  # XcodeGen projects
    ("build.gradle.kts", 3, "kotlin-android"),
    ("astro.config.*", 2, "astro-static"),
    ("wrangler.toml", 2, "cloudflare-workers"),
    ("wrangler.jsonc", 2, "cloudflare-workers"),
    ("pyproject.toml", 2, "python-ml"),
]


_VENDORED = {"node_modules", ".next", "dist", "build"}


def _shallow_entries(root: Path, max_depth: int = 3) -> list[tuple[int, Path]]:
    """Every file and directory up to max_depth below root, with its depth.

    Walks only as deep as the deepest marker, and never enters hidden or vendored
    directories. The first version ran `rglob` per marker and filtered depth
    afterwards, so it walked every node_modules in full: 74 s for the 87
    directories in active/ and epiphan/ (balalar-app alone 14 s), most of it in
    the kernel.
    """
    out: list[tuple[int, Path]] = []

    def walk(d: Path, depth: int) -> None:
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        for e in entries:
            out.append((depth, Path(e.path)))
            if (
                depth < max_depth
                and e.is_dir(follow_symlinks=False)
                and not e.name.startswith(".")
                and e.name not in _VENDORED
            ):
                walk(Path(e.path), depth + 1)

    walk(root, 1)
    return out


def _package_json_deps(entries: list[tuple[int, Path]]) -> dict[str, str]:
    """Merged dependencies of the shallowest package.json within two levels."""
    for _, pkg_json in sorted((d, p) for d, p in entries if d <= 2 and p.name == "package.json"):
        try:
            data = json.loads(pkg_json.read_text())
            return {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        except Exception:
            continue
    return {}


def _mentions_fastapi(entries: list[tuple[int, Path]]) -> bool:
    for depth, pyproject in entries:
        if depth <= 2 and pyproject.name == "pyproject.toml":
            try:
                if "fastapi" in pyproject.read_text().lower():
                    return True
            except Exception:
                continue
    return False


def detect_stacks(project_path: Path) -> list[str]:
    """Detect stacks used in a project."""
    entries = _shallow_entries(project_path, max(d for _, d, _ in STACK_DETECTORS))
    detected = set()

    for pattern, max_depth, stack in STACK_DETECTORS:
        if any(d <= max_depth and fnmatch(p.name, pattern) for d, p in entries):
            detected.add(stack)

    # Next.js needs special detection (via package.json deps)
    # AI agents stack = Next.js + Vercel AI SDK ('ai' package), always includes supabase
    deps = _package_json_deps(entries)
    if "next" in deps:
        detected.add("nextjs-supabase")
        if "ai" in deps:
            detected.add("nextjs-ai-agents")

    # FastAPI: upgrade python-ml to python-api
    if "python-ml" in detected and _mentions_fastapi(entries):
        detected.discard("python-ml")
        detected.add("python-api")

    return sorted(detected)


def get_last_commit(project_path: Path) -> str | None:
    """Get last commit date from git."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_description(project_path: Path) -> str:
    """Try to get project description from README or CLAUDE.md."""
    for fname in ("CLAUDE.md", "README.md"):
        fpath = project_path / fname
        if fpath.exists():
            try:
                text = fpath.read_text(encoding="utf-8")
                # Get first non-empty, non-heading line
                for line in text.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("---"):
                        return line[:120]
            except Exception:
                pass
    return ""


def is_project_dir(path: Path) -> bool:
    """A real git checkout, and nothing else.

    A worktree (`.git` is a file) is a second copy of a repo that is already
    listed: epiphan/ holds four of sgr-chat-agent. A directory without git has
    nothing for `scan --incremental` to diff against, and the nightly refresh
    skips it anyway. Zip files and loose notes are not directories at all.
    """
    return path.is_dir() and not path.name.startswith(".") and (path / ".git").is_dir()


def _project_info(project_dir: Path, status: str = "active") -> ProjectInfo:
    return ProjectInfo(
        name=project_dir.name,
        path=str(project_dir),
        status=status,
        stacks=detect_stacks(project_dir),
        has_git=(project_dir / ".git").is_dir(),
        has_claude_md=(project_dir / "CLAUDE.md").is_file(),
        last_commit=get_last_commit(project_dir),
        description=get_description(project_dir),
    )


def _scan_dir(scan_dir: Path, status: str = "active") -> list[ProjectInfo]:
    """Scan a directory for projects."""
    if not scan_dir.exists():
        return []
    return [_project_info(d, status) for d in sorted(scan_dir.iterdir()) if is_project_dir(d)]


def scan_projects(include_old: bool = False) -> ProjectRegistry:
    """Scan all CODEGRAPH_SCAN_PATH directories and build registry."""
    projects: list[ProjectInfo] = []

    for scan_dir in SCAN_PATHS:
        if not scan_dir.exists():
            print(f"Scan path not found: {scan_dir}")
            continue
        projects.extend(_scan_dir(scan_dir, status="active"))

    if not projects and not include_old:
        print(f"No projects found in: {', '.join(str(p) for p in SCAN_PATHS)}")
        sys.exit(1)

    if include_old:
        if OLD_PATH.exists():
            projects.extend(_scan_dir(OLD_PATH, status="archived"))
        else:
            print(f"Old path not found: {OLD_PATH}")

    registry = ProjectRegistry(
        projects=projects,
        scan_path=",".join(str(p) for p in SCAN_PATHS),
        last_scan=datetime.now().isoformat(),
    )

    return registry


def save_registry(registry: ProjectRegistry, path: Path | None = None) -> None:
    """Save registry to YAML."""
    path = path or REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    data = registry.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Registry saved: {path}")


def load_registry(path: Path | None = None) -> ProjectRegistry | None:
    """Load registry from YAML."""
    path = path or REGISTRY_PATH
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ProjectRegistry(**data)


_KEEP_BACKUPS = 5


@dataclass
class RefreshReport:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)
    backup: Path | None = None
    total: int = 0


def refresh_registry(scan_paths: list[Path] | None = None, path: Path | None = None) -> RefreshReport:
    """Bring the registry up to date with the disk without losing what was put there.

    Entries already listed stay, in their order, with metadata re-read: some live
    outside every scan path on purpose (a nested repo, a shared package). An entry
    whose directory is gone leaves and is named. A new checkout in a scan path is
    appended, unless its name is taken: the graph keys projects by name, so a
    second `api` would silently overwrite the first.

    `scan_paths` defaults to the env var when set, else to the paths the registry
    recorded last time — the nightly job runs with neither configured.
    """
    path = path or REGISTRY_PATH
    existing = load_registry(path) or ProjectRegistry()
    if scan_paths is None:
        recorded = existing.scan_path or _scan_raw
        raw = os.environ.get("CODEGRAPH_SCAN_PATH") or recorded
        scan_paths = [Path(p.strip()).expanduser() for p in raw.split(",") if p.strip()]

    report = RefreshReport()
    kept: list[ProjectInfo] = []
    for p in existing.projects:
        d = Path(p.path)
        if not d.exists():
            report.removed.append(p.name)
            continue
        fresh = _project_info(d, p.status)
        kept.append(p.model_copy(update=fresh.model_dump(exclude={"name", "status", "prd_path"})))

    known_paths = {str(Path(p.path)) for p in kept}
    known_names = {p.name for p in kept}
    for scan_dir in scan_paths:
        if not scan_dir.exists():
            print(f"Scan path not found: {scan_dir}")
            continue
        for d in sorted(scan_dir.iterdir()):
            if str(d) in known_paths or not is_project_dir(d):
                continue
            if d.name in known_names:
                report.collisions.append(str(d))
                continue
            kept.append(_project_info(d))
            known_paths.add(str(d))
            known_names.add(d.name)
            report.added.append(d.name)

    if path.exists():
        report.backup = path.with_name(f"{path.name}.{datetime.now():%Y%m%d-%H%M%S-%f}.bak")
        shutil.copy2(path, report.backup)
        # The nightly job refreshes every night; keep the last few, not a year of them.
        for old in sorted(path.parent.glob(f"{path.name}.*.bak"))[:-_KEEP_BACKUPS]:
            old.unlink()

    registry = ProjectRegistry(
        projects=kept,
        scan_path=",".join(str(p) for p in scan_paths),
        last_scan=datetime.now().isoformat(),
    )
    save_registry(registry, path)
    report.total = len(kept)
    return report


def list_available_stacks() -> None:
    """List available stack templates."""
    if not STACKS_DIR.exists():
        print("No stacks directory found.")
        return

    print("\nAvailable Stack Templates:\n")

    for yaml_file in sorted(STACKS_DIR.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            stack = yaml.safe_load(f)

        name = stack.get("name", yaml_file.stem)
        platform = stack.get("platform", "?")
        lang = stack.get("language", "?")
        deploy = stack.get("deploy", "?")
        packages = stack.get("key_packages", [])

        print(f"  {yaml_file.stem}")
        print(f"    Name: {name}")
        print(f"    Platform: {platform} | Language: {lang} | Deploy: {deploy}")
        print(f"    Packages: {', '.join(packages[:5])}")
        print()


def print_projects(registry: ProjectRegistry) -> None:
    """Print project list."""
    print(f"\nActive Projects ({len(registry.projects)}):\n")

    for p in registry.projects:
        stacks_str = ", ".join(p.stacks) if p.stacks else "unknown"
        git_icon = "git" if p.has_git else "no-git"
        claude_icon = "CLAUDE.md" if p.has_claude_md else ""

        commit_date = ""
        if p.last_commit:
            commit_date = p.last_commit.split(" ")[0]

        print(f"  {p.name}")
        print(f"    Stacks: {stacks_str}")
        print(f"    Last commit: {commit_date} | {git_icon} {claude_icon}")
        if p.description:
            print(f"    {p.description[:80]}")
        print()

    print(f"Scan path: {registry.scan_path}")
    print(f"Last scan: {registry.last_scan}")


def print_project_info(registry: ProjectRegistry, name: str) -> None:
    """Print detailed info for a single project."""
    for p in registry.projects:
        if p.name.lower() == name.lower():
            print(f"\nProject: {p.name}")
            print(f"  Path: {p.path}")
            print(f"  Status: {p.status}")
            print(f"  Stacks: {', '.join(p.stacks) if p.stacks else 'unknown'}")
            print(f"  Git: {'yes' if p.has_git else 'no'}")
            print(f"  CLAUDE.md: {'yes' if p.has_claude_md else 'no'}")
            print(f"  Last commit: {p.last_commit or 'N/A'}")
            print(f"  Description: {p.description or 'N/A'}")
            print(f"  PRD: {p.prd_path or 'none'}")

            # Show stack details
            if p.stacks:
                print("\n  Stack Details:")
                for stack_name in p.stacks:
                    stack_file = STACKS_DIR / f"{stack_name}.yaml"
                    if stack_file.exists():
                        with open(stack_file, encoding="utf-8") as f:
                            stack = yaml.safe_load(f)
                        packages = stack.get("key_packages", [])
                        print(f"    {stack_name}: {', '.join(packages[:6])}")
            return

    print(f"Project not found: {name}")
    print(f"Available: {', '.join(p.name for p in registry.projects)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/project_registry.py <command> [args]")
        print("\nCommands:")
        print("  scan           Scan startups/active/ and update registry")
        print("  list           List all projects")
        print("  info <name>    Show project details")
        print("  stacks         List available stack templates")
        return 1

    command = sys.argv[1]

    if command == "scan":
        paths = SCAN_PATHS + ([OLD_PATH] if "--old" in sys.argv else [])
        print(f"Scanning {', '.join(str(p) for p in paths)}...")
        report = refresh_registry(paths)
        print(f"added {report.added}, removed {report.removed}, name taken {report.collisions}")
        print_projects(load_registry() or ProjectRegistry())

    elif command == "list":
        registry = load_registry()
        if not registry:
            print("No registry found. Run 'scan' first.")
            return 1
        print_projects(registry)

    elif command == "info":
        if len(sys.argv) < 3:
            print("Usage: python scripts/project_registry.py info <project-name>")
            return 1
        registry = load_registry()
        if not registry:
            print("No registry found. Run 'scan' first.")
            return 1
        print_project_info(registry, sys.argv[2])

    elif command == "stacks":
        list_available_stacks()

    else:
        print(f"Unknown command: {command}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
