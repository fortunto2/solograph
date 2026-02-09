#!/usr/bin/env python3
"""
Project Registry — scans startups/active/ and builds a registry with detected stacks.

Usage:
    python scripts/project_registry.py scan          # Scan and update registry
    python scripts/project_registry.py list          # List projects
    python scripts/project_registry.py info <name>   # Show project details
    python scripts/project_registry.py stacks        # List available stack templates
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

import os

# Configurable via env vars
STACKS_DIR = Path(os.environ.get("CODEGRAPH_STACKS_DIR", str(Path.home() / ".codegraph" / "stacks"))).expanduser()
REGISTRY_PATH = Path(os.environ.get("CODEGRAPH_REGISTRY", str(Path.home() / ".codegraph" / "registry.yaml"))).expanduser()
SCAN_PATH = Path(os.environ.get("CODEGRAPH_SCAN_PATH", str(Path.home() / "projects"))).expanduser()
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
    ("build.gradle.kts", 3, "kotlin-android"),
    ("astro.config.*", 2, "astro-static"),
    ("wrangler.toml", 2, "cloudflare-workers"),
    ("wrangler.jsonc", 2, "cloudflare-workers"),
    ("pyproject.toml", 2, "python-ml"),
]


def _find_files(root: Path, pattern: str, max_depth: int) -> list[Path]:
    """Find files matching glob pattern up to max_depth."""
    results = []
    for p in root.rglob(pattern):
        # Calculate depth relative to root
        try:
            rel = p.relative_to(root)
            if len(rel.parts) <= max_depth:
                # Skip node_modules, .venv, etc.
                if not any(
                    part.startswith(".")
                    or part in ("node_modules", ".next", "dist", "build")
                    for part in rel.parts[:-1]
                ):
                    results.append(p)
        except ValueError:
            continue
    return results


def _read_package_json_deps(project_path: Path) -> dict[str, str]:
    """Read merged dependencies from package.json files in project root."""
    for pkg_json in project_path.rglob("package.json"):
        try:
            rel = pkg_json.relative_to(project_path)
            if len(rel.parts) > 2:
                continue
            if any(
                part in ("node_modules", ".next", "dist")
                for part in rel.parts[:-1]
            ):
                continue
            import json

            data = json.loads(pkg_json.read_text())
            return {
                **data.get("dependencies", {}),
                **data.get("devDependencies", {}),
            }
        except Exception:
            continue
    return {}


def _detect_nextjs(project_path: Path) -> bool:
    """Detect Next.js by checking package.json for 'next' dependency."""
    deps = _read_package_json_deps(project_path)
    return "next" in deps


def _detect_ai_sdk(project_path: Path) -> bool:
    """Detect Vercel AI SDK by checking package.json for 'ai' dependency."""
    deps = _read_package_json_deps(project_path)
    return "ai" in deps


def _detect_fastapi(project_path: Path) -> bool:
    """Detect FastAPI by checking pyproject.toml for 'fastapi' dependency."""
    for pyproject in project_path.rglob("pyproject.toml"):
        try:
            rel = pyproject.relative_to(project_path)
            if len(rel.parts) > 2:
                continue
            content = pyproject.read_text()
            if "fastapi" in content.lower():
                return True
        except Exception:
            continue
    return False


def detect_stacks(project_path: Path) -> list[str]:
    """Detect stacks used in a project."""
    detected = set()

    for pattern, max_depth, stack in STACK_DETECTORS:
        if _find_files(project_path, pattern, max_depth):
            detected.add(stack)

    # Next.js needs special detection (via package.json deps)
    # AI agents stack = Next.js + Vercel AI SDK ('ai' package), always includes supabase
    if _detect_nextjs(project_path):
        detected.add("nextjs-supabase")
        if _detect_ai_sdk(project_path):
            detected.add("nextjs-ai-agents")

    # FastAPI: upgrade python-ml to python-api
    if "python-ml" in detected and _detect_fastapi(project_path):
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


def _scan_dir(scan_dir: Path, status: str = "active") -> list[ProjectInfo]:
    """Scan a directory for projects."""
    projects = []
    if not scan_dir.exists():
        return projects

    for project_dir in sorted(scan_dir.iterdir()):
        if not project_dir.is_dir() or project_dir.name.startswith("."):
            continue

        info = ProjectInfo(
            name=project_dir.name,
            path=str(project_dir),
            status=status,
            stacks=detect_stacks(project_dir),
            has_git=(project_dir / ".git").is_dir(),
            has_claude_md=(project_dir / "CLAUDE.md").is_file(),
            last_commit=get_last_commit(project_dir),
            description=get_description(project_dir),
        )

        projects.append(info)

    return projects


def scan_projects(include_old: bool = False) -> ProjectRegistry:
    """Scan startups/active/ and build registry."""
    if not SCAN_PATH.exists():
        print(f"Scan path not found: {SCAN_PATH}")
        sys.exit(1)

    projects = _scan_dir(SCAN_PATH, status="active")

    if include_old:
        if OLD_PATH.exists():
            projects.extend(_scan_dir(OLD_PATH, status="archived"))
        else:
            print(f"Old path not found: {OLD_PATH}")

    registry = ProjectRegistry(
        projects=projects,
        scan_path=str(SCAN_PATH),
        last_scan=datetime.now().isoformat(),
    )

    return registry


def save_registry(registry: ProjectRegistry) -> None:
    """Save registry to YAML."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = registry.model_dump()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Registry saved: {REGISTRY_PATH}")


def load_registry() -> ProjectRegistry | None:
    """Load registry from YAML."""
    if not REGISTRY_PATH.exists():
        return None

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ProjectRegistry(**data)


def list_available_stacks() -> None:
    """List available stack templates."""
    if not STACKS_DIR.exists():
        print("No stacks directory found.")
        return

    print("\nAvailable Stack Templates:\n")

    for yaml_file in sorted(STACKS_DIR.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
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
                print(f"\n  Stack Details:")
                for stack_name in p.stacks:
                    stack_file = STACKS_DIR / f"{stack_name}.yaml"
                    if stack_file.exists():
                        with open(stack_file, "r", encoding="utf-8") as f:
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
        include_old = "--old" in sys.argv
        print(f"Scanning {SCAN_PATH}..." + (" + old" if include_old else ""))
        registry = scan_projects(include_old=include_old)
        save_registry(registry)
        print_projects(registry)

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
