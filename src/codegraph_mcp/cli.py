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
    from .registry import REGISTRY_PATH, SCAN_PATH, save_registry, scan_projects

    codegraph_dir = Path.home() / ".solo"

    # 1. Ask for projects dir if not provided
    if not projects_dir:
        default = str(SCAN_PATH)
        projects_dir = click.prompt("Where are your projects?", default=default)

    projects_path = Path(projects_dir).expanduser().resolve()
    if not projects_path.exists():
        console.print(f"[red]Directory not found:[/red] {projects_path}")
        raise SystemExit(1)

    # 2. Create ~/.solo/
    codegraph_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] {codegraph_dir}")

    # 3. Scan projects → registry.yaml
    os.environ["CODEGRAPH_SCAN_PATH"] = str(projects_path)
    # Re-import to pick up new env
    from . import registry as reg

    reg.SCAN_PATH = projects_path

    registry = scan_projects()
    save_registry(registry)
    n = len(registry.projects)
    console.print(f"[green]✓[/green] {n} projects found → {REGISTRY_PATH}")

    for p in registry.projects:
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
@click.pass_context
def scan(ctx, project, registry, deep):
    """Scan projects and build the code graph."""
    from .scanner.code import extract_symbols, ingest_files, ingest_symbols, scan_files
    from .scanner.deps import ingest_packages, scan_deps
    from .scanner.git import ingest_modifications, scan_git_log
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
        )

    for proj in projects:
        proj_path = Path(proj.path)
        if not proj_path.exists():
            console.print(f"  [yellow]Skipping {proj.name}[/yellow] (path not found)")
            continue

        # Clear old data for this project
        cg_db.clear_project(graph, proj.name)

        # Projects
        ingest_projects(graph, [proj])

        # Files
        files = scan_files(proj_path, proj.name)
        ingest_files(graph, files)

        # Symbols (tree-sitter)
        all_symbols = []
        for f in files:
            syms = extract_symbols(proj_path / f.path, proj.name, f.lang, rel_path=f.path)
            all_symbols.extend(syms)
        if all_symbols:
            ingest_symbols(graph, all_symbols)

        # Dependencies
        packages = scan_deps(proj_path)
        if packages:
            ingest_packages(graph, packages, proj.name)

        # Git modifications
        mods = scan_git_log(proj_path, proj.name)
        if mods:
            ingest_modifications(graph, mods)

        print_scan_progress(proj.name, proj.stack, len(files), len(all_symbols), len(packages))

        # Deep analysis (optional)
        if deep:
            all_imports = []
            all_calls = []
            all_inherits = []
            for f in files:
                if not f.lang:
                    continue
                imp, cal, inh = extract_deep(proj_path / f.path, proj.name, f.lang, rel_path=f.path)
                all_imports.extend(imp)
                all_calls.extend(cal)
                all_inherits.extend(inh)

            int_imp, ext_imp = ingest_imports(graph, all_imports)
            calls_created = ingest_calls(graph, all_calls)
            inherits_created = ingest_inherits(graph, all_inherits)

            console.print(
                f"    [dim]Deep: {len(all_imports)} imports "
                f"({int_imp} int + {ext_imp} ext), "
                f"{len(all_calls)} calls ({calls_created} edges), "
                f"{len(all_inherits)} inherits ({inherits_created} edges)[/dim]"
            )

    # Summary
    stats = cg_db.graph_stats(graph)
    print_scan_summary(stats)


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
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
@click.pass_context
def scan_sessions(ctx, project, backend):
    """Scan Claude Code chat history into graph + vectors."""
    from .scanner.sessions import (
        ingest_session_files,
        ingest_sessions,
        link_sessions_to_projects,
        scan_all_sessions,
    )
    from .vectors.session_index import SessionIndex

    fdb = cg_db.get_db(ctx.obj["db_path"])
    graph = cg_db.get_graph(fdb)
    cg_db.init_schema(graph)

    click.echo("Scanning Claude Code sessions...\n")

    all_sessions = []
    all_edges = []
    all_summaries = []
    project_counts: dict[str, int] = {}

    for session, edges, summary in scan_all_sessions(project_filter=project):
        all_sessions.append(session)
        all_edges.extend(edges)
        all_summaries.append(summary)
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

    # Index into FalkorDB vectors
    click.echo(f"\nIndexing {len(all_summaries)} summaries into FalkorDB...")
    idx = SessionIndex(backend=backend)
    indexed = idx.upsert(all_summaries)
    click.echo(f"  Indexed: {indexed}")

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
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
@click.pass_context
def session_search_cmd(ctx, query, project, limit, backend):
    """Semantic search across session summaries."""
    from .vectors.session_index import SessionIndex

    idx = SessionIndex(backend=backend)
    results = idx.search(query, n_results=limit, project=project)

    if not results:
        click.echo("No sessions found.")
        return

    click.echo(f"\nSearch: '{query}' ({len(results)} results):\n")
    for i, r in enumerate(results, 1):
        date = r["started_at"][:10] if r["started_at"] else "?"
        click.echo(f"{i}. [{r['project_name']}] {date} (relevance: {r['relevance']:.0%})")
        click.echo(f"   {r['session_id'][:8]}..")
        # Show first line of summary
        summary_line = r["summary"].split("\n")[0] if r["summary"] else ""
        click.echo(f"   {summary_line}")
        click.echo()


# ── FalkorDB Graph Vector Commands ──────────────────────────────


@cli.command("index-projects")
@click.option("--project", "-p", default=None, help="Index only this project")
@click.option(
    "--registry",
    "-r",
    type=click.Path(exists=True),
    default=None,
    help="Registry YAML path",
)
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
@click.pass_context
def index_projects_cmd(ctx, project, registry, backend):
    """Index project source code and docs into per-project FalkorDB vector DBs."""
    from .scanner.registry import scan_registry
    from .vectors.project_graph_index import ProjectGraphIndex

    registry_path = Path(registry) if registry else DEFAULT_REGISTRY
    if not registry_path.exists():
        click.echo(f"Registry not found: {registry_path}")
        raise SystemExit(1)

    projects = scan_registry(registry_path)
    if project:
        projects = [p for p in projects if p.name == project]
        if not projects:
            click.echo(f"Project not found: {project}")
            raise SystemExit(1)

    idx = ProjectGraphIndex(backend=backend)
    click.echo(f"Indexing {len(projects)} project(s) into FalkorDB...\n")

    total_chunks = 0
    for proj in projects:
        proj_path = Path(proj.path)
        if not proj_path.exists():
            click.echo(f"  Skipping {proj.name} (path not found)")
            continue

        click.echo(f"  {proj.name}...", nl=False)
        stats = idx.index_project(proj_path, proj.name)
        click.echo(
            f" {stats['files']} files, {stats['chunks']} chunks "
            f"({stats['code_chunks']} code + {stats['doc_chunks']} doc)"
        )
        total_chunks += stats["chunks"]

    click.echo(f"\nTotal: {total_chunks} chunks indexed (FalkorDB)")


@cli.command("project-search")
@click.argument("query")
@click.option("--project", "-p", default=None, help="Search in one project")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option(
    "--type",
    "chunk_type",
    type=click.Choice(["code", "doc"]),
    default=None,
    help="Filter by chunk type",
)
@click.option("--hybrid", is_flag=True, default=False, help="Hybrid search (show sibling chunks)")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
@click.pass_context
def project_search_cmd(ctx, query, project, limit, chunk_type, hybrid, backend):
    """Semantic search over project source code and docs (FalkorDB)."""
    from .vectors.project_graph_index import ProjectGraphIndex

    idx = ProjectGraphIndex(backend=backend)

    if hybrid and project:
        results = idx.search_hybrid(query, project=project, n_results=limit)
    else:
        results = idx.search(query, project=project, n_results=limit, chunk_type=chunk_type)

    if not results:
        click.echo("No results found.")
        return

    engine = "FalkorDB"
    scope = f"[{project}]" if project else "[all projects]"
    click.echo(f"\nSearch: '{query}' {scope} ({len(results)} results, {engine}):\n")

    for i, r in enumerate(results, 1):
        click.echo(
            f"{i}. [{r['project']}] {r['file']} ({r['language']}, {r['chunk_type']}) relevance: {r['relevance']:.0%}"
        )
        if r.get("sibling_chunks"):
            click.echo(f"   siblings: chunks {r['sibling_chunks']}")
        lines = r["snippet"].strip().split("\n")[:2]
        for line in lines:
            click.echo(f"   {line[:120]}")
        click.echo()


@cli.command("project-collections")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
@click.pass_context
def project_collections_cmd(ctx, backend):
    """List indexed project vector databases (FalkorDB)."""
    from .vectors.project_graph_index import ProjectGraphIndex

    idx = ProjectGraphIndex(backend=backend)
    projects = idx.list_projects()

    if not projects:
        click.echo("No projects indexed (FalkorDB). Run: solograph-cli index-projects")
        return

    click.echo(f"\nFalkorDB-indexed projects ({len(projects)}):\n")
    total_chunks = 0
    total_size = 0.0
    for p in projects:
        click.echo(f"  {p['name']}: {p['chunks']} chunks ({p['size_mb']} MB)")
        total_chunks += p["chunks"]
        total_size += p["size_mb"]

    click.echo(f"\nTotal: {total_chunks} chunks, {total_size:.2f} MB")


@cli.command("project-delete")
@click.argument("project_name")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
@click.pass_context
def project_delete_cmd(ctx, project_name, backend):
    """Delete a project's FalkorDB vector database."""
    from .vectors.project_graph_index import ProjectGraphIndex

    idx = ProjectGraphIndex(backend=backend)
    if idx.delete_project(project_name):
        click.echo(f"Deleted FalkorDB vectors for {project_name}")
    else:
        click.echo(f"No FalkorDB vectors found for {project_name}")


# ── Deep Analysis Commands ────────────────────────────────────────


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


@cli.command("source-search")
@click.argument("query")
@click.option("--source", "-s", default=None, help="Filter by source name (telegram, youtube)")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def source_search_cmd(query, source, limit, backend):
    """Search indexed external sources (Telegram, YouTube, etc.)."""
    from .vectors.source_index import SourceIndex

    idx = SourceIndex(backend=backend)
    results = idx.search(query, source=source, n_results=limit)

    if not results:
        click.echo("No results found.")
        return

    scope = f"[{source}]" if source else "[all sources]"
    click.echo(f"\nSearch: '{query}' {scope} ({len(results)} results):\n")
    for i, r in enumerate(results, 1):
        src = r.get("source_type", "?")
        # Show chapter + timecode for video chunks
        chapter_info = ""
        if r.get("chapter"):
            chapter_info = f" [{r['chapter']}"
            if r.get("start_time"):
                chapter_info += f" @ {r['start_time']}"
            chapter_info += "]"
        click.echo(f"{i}. [{src}] {r['title'][:80]}{chapter_info}  (relevance: {r['relevance']:.0%})")
        if r.get("url"):
            click.echo(f"   {r['url']}")
        if r.get("tags"):
            click.echo(f"   Tags: {r['tags']}")
        if r.get("content"):
            click.echo(f"   {r['content'][:120]}")
        if r.get("context"):
            click.echo(f"   Context: {r['context'][:200]}...")
        click.echo()


@cli.command("source-list")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def source_list_cmd(backend):
    """List indexed external sources with document counts."""
    from .vectors.source_index import SourceIndex

    idx = SourceIndex(backend=backend)
    sources = idx.list_sources()

    if not sources:
        click.echo("No sources indexed. Run: index-telegram or index-youtube")
        return

    click.echo(f"\nIndexed sources ({len(sources)}):\n")
    total = 0
    for s in sources:
        extra = ""
        if s.get("videos"):
            extra = f"  ({s['videos']} videos, {s.get('video_chunks', 0)} chunks)"
        click.echo(f"  {s['source']:12s} {s['count']:4d} docs{extra}  {s['path']}")
        total += s["count"]
    click.echo(f"\n  Total: {total} documents")


@cli.command("source-tags")
@click.option("--source", "-s", default="youtube", help="Source name (default: youtube)")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def source_tags_cmd(source, backend):
    """List all auto-detected topics with video counts."""
    from .vectors.source_index import SourceIndex

    idx = SourceIndex(backend=backend)
    tags = idx.list_tags(source)

    if not tags:
        click.echo("No tags found. Re-index videos to generate auto-tags.")
        return

    click.echo(f"\nTopics in [{source}] ({len(tags)} tags):\n")
    for t in tags:
        conf = f"  (avg {t['avg_confidence']:.0%})" if t.get("avg_confidence") else ""
        click.echo(f"  {t['count']:3d}  {t['name']}{conf}")


@cli.command("source-related")
@click.argument("video_id")
@click.option("--source", "-s", default="youtube", help="Source name (default: youtube)")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def source_related_cmd(video_id, source, backend):
    """Find related videos by shared tags."""
    import re

    from .vectors.source_index import SourceIndex

    # Extract video ID from URL if needed
    m = re.search(r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})", video_id)
    if m:
        video_id = m.group(1)

    idx = SourceIndex(backend=backend)
    results = idx.related_videos(source, video_id)

    if not results:
        click.echo(f"No related videos found for {video_id}")
        return

    click.echo(f"\nRelated videos for {video_id} ({len(results)}):\n")
    for r in results:
        tags = ", ".join(r["shared_tags"]) if r["shared_tags"] else ""
        rel = f" relevance: {r['relevance']:.2f}" if r.get("relevance") else ""
        click.echo(f"  [{r['overlap']} shared{rel}] {r['title'][:70]}")
        if tags:
            click.echo(f"    Tags: {tags}")
        if r.get("url"):
            click.echo(f"    {r['url']}")
        click.echo()


@cli.command("index-youtube")
@click.option("--channels", "-c", multiple=True, help="Channel handles to index")
@click.option("--channels-file", type=click.Path(exists=True), help="Path to channels.yaml")
@click.option("--limit", "-n", type=int, default=10, help="Max videos per channel (default: 10)")
@click.option("--url", "-u", multiple=True, help="Index specific video URLs (no SearXNG needed)")
@click.option(
    "--import-file",
    "import_path",
    type=click.Path(exists=True),
    help="Import pre-processed data file",
)
@click.option("--dry-run", is_flag=True, help="Parse only, don't insert into DB")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def index_youtube_cmd(channels, channels_file, limit, url, import_path, dry_run, backend):
    """Index YouTube transcripts into FalkorDB source graph."""
    from .indexers.youtube import YouTubeIndexer

    channels_path = Path(channels_file) if channels_file else None
    searxng_url = os.environ.get("TAVILY_API_URL", "http://localhost:8013")

    indexer = YouTubeIndexer(
        channels_path=channels_path,
        backend=backend,
        searxng_url=searxng_url,
    )

    if import_path:
        indexer.import_file(import_path, dry_run=dry_run)
    elif url:
        indexer.index_url(list(url), dry_run=dry_run)
    else:
        indexer.run(
            channels=list(channels) if channels else None,
            limit=limit,
            dry_run=dry_run,
        )


@cli.command("index-trustmrr")
@click.option("--categories", "-c", multiple=True, help="Category slugs to scrape (default: all)")
@click.option("--limit", "-n", type=int, default=None, help="Max startups to scrape")
@click.option("--dry-run", is_flag=True, help="Scrape only, don't insert into DB")
@click.option("--force", is_flag=True, help="Re-scrape all, ignore already-indexed")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def index_trustmrr_cmd(categories, limit, dry_run, force, backend):
    """Scrape TrustMRR verified startup revenues into FalkorDB source graph."""
    from .indexers.trustmrr import TrustMRRIndexer

    indexer = TrustMRRIndexer(backend=backend)
    indexer.run(
        categories=list(categories) if categories else None,
        limit=limit,
        dry_run=dry_run,
        force=force,
    )


@cli.command("index-producthunt")
@click.option("--days", "-d", type=int, default=30, help="Days back to scrape (default: 30)")
@click.option("--limit", "-n", type=int, default=None, help="Max products to scrape")
@click.option("--dry-run", is_flag=True, help="Scrape only, don't insert into DB")
@click.option("--force", is_flag=True, help="Re-scrape all, ignore already-indexed")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def index_producthunt_cmd(days, limit, dry_run, force, backend):
    """Scrape ProductHunt leaderboard into FalkorDB source graph.

    \b
    Uses PH GraphQL API v2 (no browser needed).
    Set PH_TOKEN env var (developer token) or PH_CLIENT_ID + PH_CLIENT_SECRET.

    \b
    Examples:
      solograph-cli index-producthunt                  # Last 30 days
      solograph-cli index-producthunt -d 7             # Last 7 days
      solograph-cli index-producthunt -n 50            # Limit 50 products
      solograph-cli index-producthunt --dry-run        # Preview only
    """
    from .indexers.producthunt import ProductHuntIndexer

    indexer = ProductHuntIndexer(backend=backend)
    indexer.run(
        days=days,
        limit=limit,
        dry_run=dry_run,
        force=force,
    )


@cli.command("source-delete")
@click.argument("source_name")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def source_delete_cmd(source_name, backend):
    """Delete a source's FalkorDB vector database."""
    from .vectors.source_index import SourceIndex

    idx = SourceIndex(backend=backend)
    if idx.delete_source(source_name):
        click.echo(f"Deleted source: {source_name}")
    else:
        click.echo(f"Source not found: {source_name}")


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


@cli.command("watch")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--debounce-ms",
    default=1500,
    help="Debounce delay in milliseconds (default: 1500)",
)
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def watch_cmd(paths, debounce_ms, backend):
    """Watch KB directories for markdown changes and auto-reindex.

    \b
    Examples:
      solograph-cli watch .                        # watch current dir
      solograph-cli watch ~/kb ~/notes             # watch multiple dirs
      solograph-cli watch . --debounce-ms 3000     # slower debounce
    """
    import signal
    import time

    from .kb import KnowledgeEmbeddings
    from .watcher import KBWatcher

    kb_path = os.environ.get("KB_PATH", "")
    if not kb_path:
        console.print("[red]Set KB_PATH env var[/red]")
        raise SystemExit(1)

    watch_paths = list(paths) if paths else [kb_path]

    kb = KnowledgeEmbeddings(kb_path, backend=backend)
    change_count = {"n": 0}

    def on_change(event_type: str, file_path):
        rel = file_path.name
        change_count["n"] += 1
        console.print(f"  [dim]{event_type}:[/dim] {rel}")
        if event_type == "deleted":
            console.print("    [yellow]Deleted (stale index entry remains until reindex)[/yellow]")
        else:
            kb.index_all_markdown()

    watcher = KBWatcher(watch_paths, on_change, debounce_ms=debounce_ms)

    console.print(f"[bold]Watching {len(watch_paths)} path(s)[/bold] (debounce: {debounce_ms}ms)")
    for p in watch_paths:
        console.print(f"  {p}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    watcher.start()

    def _shutdown(sig, frame):
        console.print(f"\n[bold]Stopped.[/bold] {change_count['n']} changes processed.")
        watcher.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(1)


@cli.command("index-kb")
@click.option(
    "--kb-path",
    type=click.Path(exists=True),
    default=None,
    help="Knowledge base path (default: KB_PATH env)",
)
@click.option("--force", is_flag=True, help="Force re-indexing of all documents")
@click.option(
    "--backend",
    type=click.Choice(["mlx", "st"]),
    default=None,
    help="Embedding backend",
)
def index_kb_cmd(kb_path, force, backend):
    """Index knowledge base markdown files into FalkorDB vectors."""
    from .kb import KnowledgeEmbeddings

    path = kb_path or os.environ.get("KB_PATH", "")
    if not path:
        console.print("[red]Set KB_PATH env var or pass --kb-path[/red]")
        raise SystemExit(1)

    kb = KnowledgeEmbeddings(path, backend=backend)
    kb.index_all_markdown(force=force)
    stats = kb.get_stats()
    console.print(
        f"\n[bold green]KB indexed:[/bold green] {stats['total_documents']} documents ({stats['unique_tags']} unique tags)"
    )


if __name__ == "__main__":
    cli()
