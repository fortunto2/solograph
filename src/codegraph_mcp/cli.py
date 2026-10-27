"""Click CLI for Solograph."""

import os
from pathlib import Path

import click

from .output.console import (
    console,
    print_explain,
    print_scan_progress,
    print_scan_summary,
    print_shared_packages,
    print_stats,
)

# Registry path: env var > ~/.solo/registry.yaml
_REGISTRY_ENV = os.environ.get("CODEGRAPH_REGISTRY", "")
DEFAULT_REGISTRY = Path(_REGISTRY_ENV).expanduser() if _REGISTRY_ENV else Path.home() / ".solo" / "registry.yaml"


@click.group()
@click.option("--db-path", type=click.Path(), default=None, help="Portfolio store (default: portfolio.db, see README)")
@click.pass_context
def cli(ctx, db_path):
    """Solograph — portfolio index, session and knowledge search."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = Path(db_path) if db_path else None


def _portfolio(ctx):
    from .portfolio import Portfolio

    return Portfolio(ctx.obj.get("db_path"))


@cli.command()
@click.argument("projects_dir", required=False, type=click.Path())
@click.option("--deep", is_flag=True, hidden=True, help="Ignored; kept so old scripts keep working")
@click.pass_context
def init(ctx, projects_dir, deep):
    """Initialize solograph — create ~/.solo/, find projects, index them.

    \b
    Examples:
      solograph-cli init ~/startups/active    # scan this directory
      solograph-cli init                      # interactive prompt
    """
    from .registry import REGISTRY_PATH, SCAN_PATHS, load_registry, refresh_registry

    if not projects_dir:
        default = ",".join(str(p) for p in SCAN_PATHS)
        projects_dir = click.prompt("Where are your projects?", default=default)

    projects_path = Path(projects_dir).expanduser().resolve()
    if not projects_path.exists():
        console.print(f"[red]Directory not found:[/red] {projects_path}")
        raise SystemExit(1)

    (Path.home() / ".solo").mkdir(parents=True, exist_ok=True)

    # Merged into what is already there: a second `init` on another directory
    # used to replace the whole file.
    report = refresh_registry([projects_path])
    registry = load_registry()
    console.print(f"[green]✓[/green] {report.total} projects found → {REGISTRY_PATH}")
    for p in registry.projects if registry else []:
        stacks = ", ".join(p.stacks) if p.stacks else "?"
        console.print(f"  {p.name} [dim]({stacks})[/dim]")

    console.print("\nIndexing projects...")
    ctx.invoke(scan, project=None, registry=str(REGISTRY_PATH))
    console.print("\n[bold green]Done![/bold green] Run [bold]solograph-cli stats[/bold] to verify.")


@cli.command()
@click.option("--project", "-p", default=None, help="Scan only this project")
@click.option("--registry", "-r", type=click.Path(exists=True), default=None, help="Registry YAML path")
@click.option("--deep", is_flag=True, hidden=True, help="Ignored; kept so old scripts keep working")
@click.option("--incremental", "-i", is_flag=True, hidden=True, help="Ignored; every scan is incremental now")
@click.pass_context
def scan(ctx, project, registry, deep=False, incremental=False):
    """Index projects: files, symbol definitions, packages.

    Every scan is incremental: a file is re-read only when its mtime or size
    changed, and one that is gone takes its symbols with it.
    """
    from .registry import load_registry

    registry_path = Path(registry) if registry else DEFAULT_REGISTRY
    reg = load_registry(registry_path) if registry_path.exists() else None
    if reg is None:
        console.print(f"[red]Registry not found:[/red] {registry_path}")
        raise SystemExit(1)

    pf = _portfolio(ctx)
    entries = [p.model_dump() for p in reg.projects]
    # The registry is the whole truth even when scanning one project, so a
    # project that left it leaves the store on the next scan of any project.
    dropped = pf.sync_projects(entries)
    for name in dropped:
        console.print(f"  [yellow]{name}: left the registry, removed[/yellow]")

    targets = [p for p in reg.projects if not project or p.name == project]
    if project and not targets:
        console.print(f"[red]Project not found:[/red] {project}")
        raise SystemExit(1)

    console.print(f"Scanning [bold]{len(targets)}[/bold] project(s)\n")
    for proj in targets:
        if not Path(proj.path).exists():
            console.print(f"  [yellow]Skipping {proj.name}[/yellow] (path not found)")
            continue
        report = pf.scan_project(proj.name, proj.path)
        if not report["read"] and not report["removed"]:
            # refresh-changed.sh counts these lines; keep the wording.
            console.print(f"  [dim]{proj.name}: unchanged[/dim]")
            continue
        print_scan_progress(proj.name, ",".join(proj.stacks), report)
    print_scan_summary(pf.stats())


@cli.command("registry")
def registry_cmd():
    """Add new checkouts to the registry, drop vanished ones, keep the rest.

    Scan paths: CODEGRAPH_SCAN_PATH if set, else the ones the registry recorded.
    The previous file is kept next to it as registry.yaml.<stamp>.bak.
    """
    from .registry import REGISTRY_PATH, refresh_registry

    report = refresh_registry()
    console.print(f"[green]✓[/green] {report.total} projects in {REGISTRY_PATH}")
    if report.added:
        console.print(f"  added ({len(report.added)}): {', '.join(report.added)}")
    if report.removed:
        console.print(
            f"  [yellow]removed, directory gone ({len(report.removed)}):[/yellow] {', '.join(report.removed)}"
        )
    for path in report.collisions:
        console.print(f"  [red]not added, name already taken:[/red] {path}")


@cli.command()
@click.pass_context
def stats(ctx):
    """Counts across the portfolio."""
    print_stats(_portfolio(ctx).stats())


@cli.command("shared")
@click.argument("package", required=False)
@click.pass_context
def shared_cmd(ctx, package):
    """Packages used by more than one project, or every project using PACKAGE."""
    results = _portfolio(ctx).shared_packages(package)
    if not results:
        console.print("[yellow]No shared packages found[/yellow]")
        return
    print_shared_packages(results)


@cli.command()
@click.argument("project_name")
@click.pass_context
def explain(ctx, project_name):
    """What a project is made of."""
    result = _portfolio(ctx).explain(project_name)
    if isinstance(result, str):
        console.print(f"[red]{result}[/red]")
        return
    print_explain(result)


@cli.command()
@click.argument("project_name")
@click.pass_context
def deps(ctx, project_name):
    """A project's dependencies."""
    results = _portfolio(ctx).dependencies(project_name)
    if not results:
        click.echo(f"No dependencies found for {project_name}")
        return
    click.echo(f"\nDependencies of {project_name} ({len(results)}):\n")
    for d in results:
        ver = f" v{d['version']}" if d["version"] else ""
        click.echo(f"  [{d['source']}] {d['name']}{ver}")


@cli.command("symbols")
@click.argument("query")
@click.option("--project", "-p", default=None, help="Limit to one project")
@click.option("--kind", "-k", default=None, help="function, class or protocol")
@click.option("--limit", "-n", default=20, help="Number of results")
@click.pass_context
def symbols_cmd(ctx, query, project, kind, limit):
    """Where something is defined, across every project ("montage profile")."""
    hits = _portfolio(ctx).search_symbols(query, n_results=limit, project=project, kind=kind)
    if not hits:
        click.echo("No definitions found.")
        return
    for h in hits:
        click.echo(f"  {h['name']:40} {h['kind']:9} {h['project']}  {h['file']}:{h['line']}")


@cli.command("scan-sessions")
@click.option(
    "--claude-dir",
    type=click.Path(),
    default=None,
    help="Claude config dir holding projects/ (default ~/.claude).",
)
def scan_sessions(claude_dir):
    """Read new and grown transcripts into the session index (searches do this too)."""
    from .session_search import SessionSearch

    root = Path(claude_dir).expanduser() if claude_dir else None
    report = SessionSearch(sessions_root=root).sync()
    click.echo(
        f"Session index: {report.get('indexed', 0)} sessions, {report.get('read', 0)} read now, "
        f"{report.get('gone', 0)} archived, {len(report.get('failed', []))} failed"
    )


@cli.command("session-search")
@click.argument("query")
@click.option("--project", "-p", default=None, help="Filter by project")
@click.option("--limit", "-n", default=5, help="Number of results")
def session_search_cmd(query, project, limit):
    """Full-text search across Claude Code sessions, deleted transcripts included."""
    from .session_search import SessionSearch

    out = SessionSearch().search(query, n_results=limit, project=project)
    if not out["results"]:
        click.echo(f"No sessions found. ({out['coverage']['status']})")
        return
    for i, r in enumerate(out["results"], 1):
        click.echo(f"{i}. [{r['project']}] {r['started_at'][:10] or '?'}  score {r['score']}  {r['title'][:80]}")
        click.echo(f"   {r.get('resume') or r.get('transcript', '')}")
        click.echo(f"   {r['snippet'][:160]}")
        click.echo()


@cli.command("web-search")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results")
@click.option(
    "--engines",
    "-e",
    default=None,
    help="Override engines (e.g. 'reddit', 'arxiv,google scholar')",
)
@click.option("--raw", is_flag=True, help="Include raw page content (up to 5000 chars)")
def web_search_cmd(query, limit, engines, raw):
    """Search the web via SearXNG / Tavily API."""
    import httpx

    url = os.environ.get("TAVILY_API_URL", "http://localhost:8013")
    key = os.environ.get("TAVILY_API_KEY", "")

    payload = {"query": query, "max_results": limit, "include_raw_content": raw}
    if engines:
        payload["engines"] = engines

    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = httpx.post(f"{url}/search", json=payload, headers=headers, timeout=30)
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {url}[/red]")
        console.print("[dim]Start SearXNG or set TAVILY_API_URL[/dim]")
        raise SystemExit(1)

    if resp.status_code != 200:
        console.print(f"[red]Search error {resp.status_code}:[/red] {resp.text[:300]}")
        raise SystemExit(1)

    data = resp.json()
    results = data.get("results", [])

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold]Web search:[/bold] '{query}' ({len(results)} results)\n")
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        link = r.get("url", "")
        snippet = r.get("content", "")[:200]
        console.print(f"[bold]{i}.[/bold] {title}")
        console.print(f"   [dim]{link}[/dim]")
        if snippet:
            console.print(f"   {snippet}")
        if raw and r.get("raw_content"):
            console.print(f"   [dim]--- raw ({len(r['raw_content'])} chars) ---[/dim]")
            console.print(f"   {r['raw_content'][:500]}")
        console.print()


@cli.command("compact")
@click.option("--min-chars", default=2000, help="Minimum content length to suggest compaction (default: 2000)")
@click.option("--days", default=7, help="Minimum age in days (default: 7)")
@click.option("--dir", "scan_dir", default=None, help="Directory to scan (default: 3-inbox)")
@click.option("--all", "show_all", is_flag=True, help="Show all files, not just long/old ones")
def compact_cmd(min_chars, days, scan_dir, show_all):
    """List documents ready for LLM compaction (summarization).

    Finds long or old capture/research documents that can be compressed
    into short summaries. Works well with the distill workflow.

    \b
    Examples:
      solograph-cli compact                        # inbox items > 2000 chars, > 7 days
      solograph-cli compact --min-chars 1000       # lower threshold
      solograph-cli compact --days 3               # more recent files too
      solograph-cli compact --dir 4-opportunities  # scan different dir
      solograph-cli compact --all                  # show everything
    """
    from datetime import datetime, timedelta

    import frontmatter as fm

    kb_path = os.environ.get("KB_PATH", "")
    if not kb_path:
        console.print("[red]Set KB_PATH env var[/red]")
        raise SystemExit(1)

    target = Path(kb_path) / (scan_dir or "3-inbox")
    if not target.exists():
        console.print(f"[red]Directory not found:[/red] {target}")
        raise SystemExit(1)

    cutoff = datetime.now() - timedelta(days=days)
    candidates = []

    for md_file in sorted(target.rglob("*.md")):
        if md_file.name == "README.md":
            continue
        try:
            post = fm.load(str(md_file))
        except Exception:
            continue

        content = post.content or ""
        char_count = len(content)
        title = post.metadata.get("title", md_file.stem)
        created_str = str(post.metadata.get("created", ""))
        compacted = post.metadata.get("compacted", False)
        distilled = post.metadata.get("distilled", False)
        status = post.metadata.get("status", "draft")

        if compacted:
            continue

        # Parse created date
        created_date = None
        if created_str:
            for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
                try:
                    created_date = datetime.strptime(created_str, fmt)
                    break
                except ValueError:
                    pass

        is_old = created_date and created_date < cutoff
        is_long = char_count >= min_chars

        if not show_all and not (is_old or is_long):
            continue

        candidates.append(
            {
                "file": md_file,
                "title": title,
                "chars": char_count,
                "created": created_str,
                "is_old": is_old,
                "is_long": is_long,
                "distilled": distilled,
                "status": status,
            }
        )

    if not candidates:
        console.print("[green]Nothing to compact.[/green] All inbox items are short and recent.")
        return

    console.print(f"[bold]Documents ready for compaction[/bold] ({len(candidates)} found)\n")

    for c in candidates:
        rel = c["file"].relative_to(Path(kb_path))
        flags = []
        if c["is_long"]:
            flags.append(f"[yellow]{c['chars']} chars[/yellow]")
        else:
            flags.append(f"{c['chars']} chars")
        if c["is_old"]:
            flags.append(f"[yellow]{c['created']}[/yellow]")
        else:
            flags.append(c["created"] or "?")
        if c["distilled"]:
            flags.append("[green]distilled[/green]")
        flag_str = " | ".join(flags)

        console.print(f"  {c['title']}")
        console.print(f"    [dim]{rel}[/dim]  ({flag_str})")
        console.print()

    console.print("[dim]To compact: ask Claude 'compact 3-inbox/filename.md'[/dim]")
    console.print("[dim]Sets compacted: true, replaces content with LLM summary[/dim]")


@cli.command("index-kb")
@click.option(
    "--kb-path",
    type=click.Path(exists=True),
    default=None,
    help="Knowledge base path (default: KB_PATH env)",
)
def index_kb_cmd(kb_path):
    """Sync the knowledge base search index now (searches also sync on their own)."""
    from .kb import KnowledgeBase

    path = kb_path or os.environ.get("KB_PATH", "")
    if not path:
        console.print("[red]Set KB_PATH env var or pass --kb-path[/red]")
        raise SystemExit(1)
    stats = KnowledgeBase(path).stats()
    console.print(f"[bold green]KB index:[/bold green] {stats['documents']} documents, {stats['read_now']} read now")


from . import cli_sources  # noqa: E402, F401 — registers the source commands on `cli`

if __name__ == "__main__":
    cli()
