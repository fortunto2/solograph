"""Click CLI for Solograph."""

import os
from pathlib import Path

import click

from . import db as cg_db
from .output.console import (
    console,
    print_explain,
    print_scan_progress,
    print_scan_summary,
    print_shared_packages,
    print_stats,
    print_xray_table,
)

# Registry path: env var > ~/.solo/registry.yaml
_REGISTRY_ENV = os.environ.get("CODEGRAPH_REGISTRY", "")
DEFAULT_REGISTRY = Path(_REGISTRY_ENV).expanduser() if _REGISTRY_ENV else Path.home() / ".solo" / "registry.yaml"


@click.group()
@click.option("--db-path", type=click.Path(), default=None, help="Path to FalkorDB file")
@click.pass_context
def cli(ctx, db_path):
    """Solograph — multi-project code intelligence graph."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = Path(db_path) if db_path else None


@cli.command()
@click.argument("projects_dir", required=False, type=click.Path())
@click.option("--deep", is_flag=True, help="Also run deep analysis (imports, calls, inheritance)")
@click.pass_context
def init(ctx, projects_dir, deep):
    """Initialize solograph — create ~/.solo/, scan projects, build graph.

    \b
    Examples:
      solograph-cli init ~/startups/active    # scan this directory
      solograph-cli init                      # interactive prompt
      solograph-cli init ~/projects --deep    # with deep analysis
    """
    from .registry import REGISTRY_PATH, SCAN_PATHS, load_registry, refresh_registry

    codegraph_dir = Path.home() / ".solo"

    # 1. Ask for projects dir if not provided
    if not projects_dir:
        default = ",".join(str(p) for p in SCAN_PATHS)
        projects_dir = click.prompt("Where are your projects?", default=default)

    projects_path = Path(projects_dir).expanduser().resolve()
    if not projects_path.exists():
        console.print(f"[red]Directory not found:[/red] {projects_path}")
        raise SystemExit(1)

    # 2. Create ~/.solo/
    codegraph_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] {codegraph_dir}")

    # 3. Scan projects → registry.yaml, merged into what is already there:
    # a second `init` on another directory used to replace the whole file.
    report = refresh_registry([projects_path])
    registry = load_registry()
    n = report.total
    console.print(f"[green]✓[/green] {n} projects found → {REGISTRY_PATH}")

    for p in registry.projects if registry else []:
        stacks = ", ".join(p.stacks) if p.stacks else "?"
        console.print(f"  {p.name} [dim]({stacks})[/dim]")

    # 4. Build code graph
    console.print("\nBuilding code graph...")
    ctx.invoke(scan, project=None, registry=str(REGISTRY_PATH), deep=deep)

    console.print("\n[bold green]Done![/bold green] Run [bold]solograph-cli stats[/bold] to verify.")
    if not deep:
        console.print(f"[dim]Tip: solograph-cli init {projects_path} --deep  for imports/calls/inheritance[/dim]")


@cli.command()
@click.option("--project", "-p", default=None, help="Scan only this project")
@click.option(
    "--registry",
    "-r",
    type=click.Path(exists=True),
    default=None,
    help="Registry YAML path",
)
@click.option(
    "--deep",
    is_flag=True,
    help="Extract imports, calls, inheritance (tree-sitter deep analysis)",
)
@click.option(
    "--incremental",
    "-i",
    is_flag=True,
    help="Only rescan files git says changed since the last scan of this project",
)
@click.pass_context
def scan(ctx, project, registry, deep, incremental):
    """Scan projects and build the code graph."""
    from .scanner.code import extract_symbols, ingest_files, ingest_symbols, scan_files
    from .scanner.deps import ingest_packages, scan_deps
    from .scanner.git import ingest_modifications, scan_git_log
    from .scanner.incremental import (
        delta_since_last_scan,
        forget_files,
        record_scan,
        should_full_rescan,
    )
    from .scanner.registry import ingest_projects, scan_registry

    registry_path = Path(registry) if registry else DEFAULT_REGISTRY
    if not registry_path.exists():
        console.print(f"[red]Registry not found:[/red] {registry_path}")
        raise SystemExit(1)

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)
    cg_db.init_schema(graph)

    projects = scan_registry(registry_path)
    if project:
        projects = [p for p in projects if p.name == project]
        if not projects:
            console.print(f"[red]Project not found:[/red] {project}")
            raise SystemExit(1)

    mode = "deep" if deep else "standard"
    console.print(f"Scanning [bold]{len(projects)}[/bold] project(s) [dim][{mode}][/dim]\n")

    # Deep analysis imports (lazy)
    if deep:
        from .scanner.code import (
            extract_deep,
            ingest_calls,
            ingest_imports,
            ingest_inherits,
            tsconfig_aliases,
        )

    for proj in projects:
        proj_path = Path(proj.path)
        if not proj_path.exists():
            console.print(f"  [yellow]Skipping {proj.name}[/yellow] (path not found)")
            continue

        # Full or partial? Partial is asked for, possible only when this
        # project has been scanned before, and forced back to full every
        # FULL_RESCAN_EVERY passes so cross-file drift cannot accumulate.
        delta = None
        partial = False
        if incremental:
            delta = delta_since_last_scan(graph, proj.name, proj_path)
            partial = delta.from_commit is not None and not should_full_rescan(graph, proj.name)
            if partial and delta.is_noop:
                console.print(f"  [dim]{proj.name}: unchanged[/dim]")
                continue

        if partial:
            # Only the files that moved leave the graph.
            forget_files(graph, proj.name, delta.changed + delta.deleted)
        else:
            cg_db.clear_project(graph, proj.name)

        # Projects
        ingest_projects(graph, [proj])

        # Files
        files = scan_files(proj_path, proj.name)
        if partial:
            touched = set(delta.changed)
            files = [f for f in files if f.path in touched]
        ingest_files(graph, files)

        # Symbols (tree-sitter)
        all_symbols = []
        for f in files:
            syms = extract_symbols(proj_path / f.path, proj.name, f.lang, rel_path=f.path)
            all_symbols.extend(syms)
        if all_symbols:
            ingest_symbols(graph, all_symbols)

        # Dependencies and git history describe the project, not a file, so
        # a partial pass leaves them alone — they were not invalidated by the
        # three files that changed, and re-reading them is most of the cost
        # this pass exists to avoid.
        packages = []
        if not partial:
            packages = scan_deps(proj_path)
            if packages:
                ingest_packages(graph, packages, proj.name)

            mods = scan_git_log(proj_path, proj.name)
            if mods:
                ingest_modifications(graph, mods)

        print_scan_progress(proj.name, proj.stack, len(files), len(all_symbols), len(packages))

        # Deep analysis (optional)
        if deep:
            all_imports = []
            all_calls = []
            all_inherits = []
            # Once per project: a path alias is a per-project fact, and reading it is
            # what stops `@/lib/db` being filed as an npm scope that owns no package.
            aliases = tsconfig_aliases(proj_path)
            for f in files:
                if not f.lang:
                    continue
                imp, cal, inh = extract_deep(proj_path / f.path, proj.name, f.lang, rel_path=f.path, aliases=aliases)
                all_imports.extend(imp)
                all_calls.extend(cal)
                all_inherits.extend(inh)

            # The Rust module map is built from the files this pass saw, so
            # a `use crate::a::b` can find the file it names. On a partial
            # pass that is only the changed files, which is why an
            # incremental run resolves fewer internal imports than a full
            # one — and why every twentieth pass is full.
            from .scanner.rust_modules import build_index as build_rust_index

            rust_index = build_rust_index(files)
            int_imp, ext_imp = ingest_imports(graph, all_imports, rust_index)
            calls_created = ingest_calls(graph, all_calls)
            inherits_created = ingest_inherits(graph, all_inherits)

            console.print(
                f"    [dim]Deep: {len(all_imports)} imports "
                f"({int_imp} int + {ext_imp} ext), "
                f"{len(all_calls)} calls ({calls_created} edges), "
                f"{len(all_inherits)} inherits ({inherits_created} edges)[/dim]"
            )

        # Stamp what we scanned, so the next pass knows where to start.
        if delta is None:
            from .scanner.incremental import delta_since_last_scan as _d

            head = _d(graph, proj.name, proj_path).head
        else:
            head = delta.head
        if head:
            record_scan(graph, proj.name, head, incremental=partial)

    # Summary
    stats = cg_db.graph_stats(graph)
    print_scan_summary(stats)


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
@click.argument("cypher_query")
@click.pass_context
def query(ctx, cypher_query):
    """Execute a raw Cypher query."""
    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    result = graph.query(cypher_query)
    if not result.result_set:
        click.echo("(no results)")
        return

    # Print header from column names if available
    if result.header:
        headers = [h[1] for h in result.header]
        click.echo(" | ".join(str(h) for h in headers))
        click.echo("-" * 60)

    for row in result.result_set:
        click.echo(" | ".join(str(v) for v in row))


@cli.command()
@click.argument("project_name")
@click.pass_context
def files(ctx, project_name):
    """List files in a project."""
    from .query.search import files_by_project

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    results = files_by_project(graph, project_name)
    if not results:
        click.echo(f"No files found for {project_name}")
        return

    click.echo(f"\nFiles in {project_name} ({len(results)}):\n")
    for f in results:
        click.echo(f"  {f['path']} ({f['lang']}, {f['lines']} lines)")


@cli.command()
@click.argument("project_name")
@click.pass_context
def deps(ctx, project_name):
    """Show project dependencies."""
    from .query.search import deps_by_project

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    results = deps_by_project(graph, project_name)
    if not results:
        click.echo(f"No dependencies found for {project_name}")
        return

    click.echo(f"\nDependencies of {project_name} ({len(results)}):\n")
    for d in results:
        ver = f" v{d['version']}" if d["version"] else ""
        click.echo(f"  [{d['source']}] {d['name']}{ver}")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show graph statistics."""
    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    s = cg_db.graph_stats(graph)
    print_stats(s)


@cli.command("shared")
@click.pass_context
def shared_cmd(ctx):
    """Show packages shared across projects."""
    from .query.search import shared_packages

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    results = shared_packages(graph)
    if not results:
        console.print("[yellow]No shared packages found[/yellow]")
        return

    print_shared_packages(results)


@cli.command("scan-sessions")
@click.option("--project", "-p", default=None, help="Filter by project name")
@click.option(
    "--claude-dir",
    type=click.Path(),
    default=None,
    help="Claude config dir holding projects/ (default ~/.claude). "
    "A second workspace keeps its history elsewhere — point this at it.",
)
@click.pass_context
def scan_sessions(ctx, project, claude_dir):
    """Scan Claude Code chat history into the graph and the search index."""
    from .scanner.sessions import (
        ingest_session_files,
        ingest_sessions,
        link_sessions_to_projects,
        scan_all_sessions,
    )
    from .session_search import SessionSearch

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)
    cg_db.init_schema(graph)

    click.echo("Scanning Claude Code sessions...\n")

    all_sessions = []
    all_edges = []
    project_counts: dict[str, int] = {}

    for session, edges in scan_all_sessions(
        claude_dir=Path(claude_dir).expanduser() if claude_dir else None,
        project_filter=project,
    ):
        all_sessions.append(session)
        all_edges.extend(edges)
        project_counts[session.project_name] = project_counts.get(session.project_name, 0) + 1

    if not all_sessions:
        click.echo("No sessions found.")
        return

    # Ingest into graph
    click.echo(f"Sessions: {len(all_sessions)}")
    ingest_sessions(graph, all_sessions)

    click.echo(f"File edges: {len(all_edges)}")
    linked = ingest_session_files(graph, all_edges)
    click.echo(f"  Linked to existing File nodes: {linked}")

    project_links = link_sessions_to_projects(graph)
    click.echo(f"  Linked to Project nodes: {project_links}")

    # Search index: new and grown transcripts only
    root = Path(claude_dir).expanduser() if claude_dir else None
    report = SessionSearch(sessions_root=root).sync()
    click.echo(f"\nSearch index: {report.get('indexed', 0)} sessions, {report.get('read', 0)} read now")

    # Summary
    click.echo(f"\nProjects ({len(project_counts)}):")
    for pname, cnt in sorted(project_counts.items(), key=lambda x: -x[1]):
        click.echo(f"  {pname}: {cnt} sessions")


@cli.command("sessions")
@click.argument("project_name")
@click.pass_context
def sessions_cmd(ctx, project_name):
    """List sessions for a project."""
    from .query.search import sessions_by_project

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    results = sessions_by_project(graph, project_name)
    if not results:
        click.echo(f"No sessions found for {project_name}")
        return

    click.echo(f"\nSessions for {project_name} ({len(results)}):\n")
    for r in results:
        date = r["started_at"][:10] if r["started_at"] else "?"
        slug = r["slug"] or ""
        dur = ""
        if r["duration_ms"]:
            dur = f" ({r['duration_ms'] // 1000}s)"
        click.echo(f"  {date} {r['session_id'][:8]}.. [{r['user_prompts']}p/{r['tool_uses']}t]{dur} {slug}")


@cli.command("session-files")
@click.argument("session_id")
@click.pass_context
def session_files_cmd(ctx, session_id):
    """Show files touched in a session."""
    from .query.search import files_in_session

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    results = files_in_session(graph, session_id)
    if not results:
        click.echo(f"No files found for session {session_id[:8]}..")
        return

    click.echo(f"\nFiles in session {session_id[:8]}.. ({len(results)}):\n")
    for r in results:
        click.echo(f"  [{r['action']}] {r['path']} (x{r['count']})")


@cli.command("file-sessions")
@click.argument("file_path")
@click.pass_context
def file_sessions_cmd(ctx, file_path):
    """Find sessions that touched a file."""
    from .query.search import sessions_for_file

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    results = sessions_for_file(graph, file_path)
    if not results:
        click.echo(f"No sessions found for {file_path}")
        return

    click.echo(f"\nSessions for {file_path} ({len(results)}):\n")
    for r in results:
        date = r["started_at"][:10] if r["started_at"] else "?"
        click.echo(
            f"  {date} {r['session_id'][:8]}.. [{r['action']} x{r['count']}] {r['project_name']} {r.get('slug', '')}"
        )


@cli.command("session-search")
@click.argument("query")
@click.option("--project", "-p", default=None, help="Filter by project")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.pass_context
def session_search_cmd(ctx, query, project, limit):
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


# ── Wow Features ─────────────────────────────────────────────────


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--deep", is_flag=True, help="Extract imports, calls, inheritance")
@click.pass_context
def xray(ctx, directory, deep):
    """Portfolio X-Ray — zero-config scan of a directory of projects."""

    from .models import ProjectNode
    from .registry import detect_stacks
    from .scanner.code import (
        extract_symbols,
        ingest_files,
        ingest_symbols,
        scan_files,
    )
    from .scanner.deps import ingest_packages, scan_deps
    from .scanner.registry import ingest_projects

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)
    cg_db.init_schema(graph)

    directory_path = Path(directory).resolve()
    subdirs = sorted(
        [d for d in directory_path.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )

    if not subdirs:
        console.print(f"[yellow]No subdirectories found in {directory_path}[/yellow]")
        return

    if deep:
        from .scanner.code import (
            extract_deep,
            ingest_calls,
            ingest_imports,
            ingest_inherits,
        )

    mode = "deep" if deep else "standard"
    console.print(f"X-Ray scanning [bold]{directory_path}[/bold] [dim][{mode}][/dim]\n")

    xray_results = []
    total_files = 0
    total_symbols = 0
    total_packages = 0

    for subdir in subdirs:
        stacks = detect_stacks(subdir)
        stack_str = ",".join(stacks) if stacks else ""
        proj = ProjectNode(name=subdir.name, path=str(subdir), stack=stack_str)

        # Clear old data and ingest project
        cg_db.clear_project(graph, proj.name)
        ingest_projects(graph, [proj])

        # Files
        files = scan_files(subdir, proj.name)
        if files:
            ingest_files(graph, files)

        # Symbols
        all_symbols = []
        for f in files:
            syms = extract_symbols(subdir / f.path, proj.name, f.lang, rel_path=f.path)
            all_symbols.extend(syms)
        if all_symbols:
            ingest_symbols(graph, all_symbols)

        # Dependencies
        packages = scan_deps(subdir)
        if packages:
            ingest_packages(graph, packages, proj.name)

        # Deep analysis
        if deep:
            all_imports = []
            all_calls = []
            all_inherits = []
            for f in files:
                if not f.lang:
                    continue
                imp, cal, inh = extract_deep(subdir / f.path, proj.name, f.lang, rel_path=f.path)
                all_imports.extend(imp)
                all_calls.extend(cal)
                all_inherits.extend(inh)
            ingest_imports(graph, all_imports)
            ingest_calls(graph, all_calls)
            ingest_inherits(graph, all_inherits)

        proj_files = len(files)
        proj_symbols = len(all_symbols)
        proj_packages = len(packages)
        total_files += proj_files
        total_symbols += proj_symbols
        total_packages += proj_packages

        xray_results.append(
            {
                "name": proj.name,
                "stack": stack_str or None,
                "files": proj_files,
                "symbols": proj_symbols,
                "packages": proj_packages,
            }
        )

    # Shared packages
    from .query.search import shared_packages

    shared = shared_packages(graph)

    print_xray_table(xray_results, total_files, total_symbols, total_packages, shared)

    if not deep:
        console.print(f"\n[dim]Run 'solograph-cli xray {directory} --deep' for imports/calls/inheritance[/dim]")


@cli.command()
@click.argument("project_name")
@click.option(
    "--type",
    "diagram_type",
    type=click.Choice(["inheritance", "imports", "calls", "deps"]),
    default="inheritance",
    help="Diagram type",
)
@click.option("--symbol", "-s", default=None, help="Filter calls diagram by symbol name")
@click.option("--max-nodes", default=50, help="Maximum edges in diagram")
@click.pass_context
def diagram(ctx, project_name, diagram_type, symbol, max_nodes):
    """Generate mermaid diagram from code graph."""
    from .output.mermaid import (
        calls_diagram,
        deps_diagram,
        imports_diagram,
        inheritance_diagram,
    )

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    generators = {
        "inheritance": lambda: inheritance_diagram(graph, project_name, max_nodes),
        "imports": lambda: imports_diagram(graph, project_name, max_nodes),
        "calls": lambda: calls_diagram(graph, project_name, symbol, max_nodes),
        "deps": lambda: deps_diagram(graph, project_name),
    }

    result = generators[diagram_type]()
    if not result:
        hint = " Run: solograph-cli scan --deep" if diagram_type != "deps" else ""
        click.echo(f"No {diagram_type} data found for {project_name}.{hint}")
        return

    click.echo(f"```mermaid\n{result}\n```")


@cli.command()
@click.argument("project_name")
@click.pass_context
def explain(ctx, project_name):
    """Architecture overview of a project."""
    from .output.explain import explain_project

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    result = explain_project(graph, project_name)
    if isinstance(result, str):
        console.print(f"[red]{result}[/red]")
        return
    print_explain(result)


# ── Deep Analysis Commands ────────────────────────────────────────


@cli.command("imports")
@click.argument("project_name")
@click.pass_context
def imports_cmd(ctx, project_name):
    """Show import graph for a project."""
    from .query.search import import_graph

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    results = import_graph(graph, project_name)
    if not results:
        click.echo(f"No import edges found for {project_name}. Run: solograph-cli scan --deep")
        return

    click.echo(f"\nImport graph for {project_name} ({len(results)} edges):\n")
    for r in results:
        arrow = "→" if r["target_type"] == "File" else "⇒"
        click.echo(f"  {r['source']} {arrow} {r['target']}")


@cli.command("callers")
@click.argument("symbol_name")
@click.option("--project", "-p", default=None, help="Filter by project")
@click.pass_context
def callers_cmd(ctx, symbol_name, project):
    """Show files that call a symbol."""
    from .query.search import callers_of

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    results = callers_of(graph, symbol_name, project=project)
    if not results:
        click.echo(f"No callers found for '{symbol_name}'. Run: solograph-cli scan --deep")
        return

    click.echo(f"\nCallers of '{symbol_name}' ({len(results)}):\n")
    for r in results:
        click.echo(f"  [{r['project']}] {r['caller']}  (defined in {r['defined_in']})")


@cli.command("hierarchy")
@click.argument("class_name")
@click.option("--project", "-p", default=None, help="Filter by project")
@click.pass_context
def hierarchy_cmd(ctx, class_name, project):
    """Show class hierarchy (parents and children)."""
    from .query.search import class_hierarchy

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    result = class_hierarchy(graph, class_name, project=project)
    parents = result["parents"]
    children = result["children"]

    if not parents and not children:
        click.echo(f"No hierarchy found for '{class_name}'. Run: solograph-cli scan --deep")
        return

    click.echo(f"\nHierarchy for '{class_name}':\n")
    if parents:
        click.echo("  Parents:")
        for p in parents:
            click.echo(f"    ↑ {p['name']}  ({p['file']}, {p['project']})")
    if children:
        click.echo("  Children:")
        for c in children:
            click.echo(f"    ↓ {c['name']}  ({c['file']}, {c['project']})")


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


@cli.command()
@click.argument("project_name")
@click.option("--limit", "-n", default=20, help="Max files to include in map (default 20)")
@click.pass_context
def repomap(ctx, project_name, limit):
    """Generate a YAML-formatted repository map (like Aider)."""
    from .output.repomap import generate_repomap

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)

    result = generate_repomap(graph, project_name, max_files=limit)
    click.echo(result)


from . import cli_sources  # noqa: E402, F401 — registers the source commands on `cli`

if __name__ == "__main__":
    cli()
