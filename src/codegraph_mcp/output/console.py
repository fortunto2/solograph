"""Rich console output for Solograph CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


def print_xray_table(
    results: list[dict],
    total_files: int,
    total_symbols: int,
    total_packages: int,
    shared: list[dict] | None = None,
) -> None:
    """Render xray results as a Rich table."""
    table = Table(title="Portfolio X-Ray", show_lines=False)
    table.add_column("Project", style="cyan", no_wrap=True)
    table.add_column("Stack", style="green")
    table.add_column("Files", justify="right", style="white")
    table.add_column("Symbols", justify="right", style="white")
    table.add_column("Packages", justify="right", style="white")

    for r in results:
        table.add_row(
            r["name"],
            r["stack"] or "[dim](no stack)[/dim]",
            str(r["files"]),
            str(r["symbols"]),
            str(r["packages"]),
        )

    console.print(table)

    # Totals panel
    lines = [f"[bold]{len(results)}[/bold] projects"]
    lines.append(
        f"[bold]{total_files:,}[/bold] files  |  "
        f"[bold]{total_symbols:,}[/bold] symbols  |  "
        f"[bold]{total_packages:,}[/bold] packages"
    )
    if shared:
        top = shared[:10]
        parts = [f"{s['package']}({len(s['projects'])})" for s in top]
        lines.append(f"[dim]Shared:[/dim] {', '.join(parts)}")

    console.print(Panel("\n".join(lines), title="Totals", border_style="blue"))


def print_explain(data: dict) -> None:
    """Render explain output with Rich panels, tables, and trees."""
    # Header
    title = f"[bold cyan]{data['name']}[/bold cyan] — {data['stack']}"
    if data.get("description"):
        title += f"\n{data['description']}"

    counts = data["counts"]
    title += (
        f"\n[dim]{counts['files']} files | "
        f"{counts['symbols']} symbols | "
        f"{counts['packages']} packages[/dim]"
    )
    console.print(Panel(title, border_style="cyan"))

    # Languages
    if data.get("languages"):
        lang_table = Table(show_header=True, show_lines=False, title="Languages")
        lang_table.add_column("Language", style="green")
        lang_table.add_column("Files", justify="right")
        lang_table.add_column("Lines", justify="right")
        for lang, cnt, lines in data["languages"]:
            lang_table.add_row(lang, str(cnt), f"{int(lines):,}")
        console.print(lang_table)

    # Layers
    if data.get("layers"):
        layer_table = Table(show_header=True, show_lines=False, title="Layers")
        layer_table.add_column("Directory", style="cyan", no_wrap=True)
        layer_table.add_column("Files", justify="right")
        layer_table.add_column("Symbols", justify="right")
        for dir_key, files, symbols in data["layers"]:
            layer_table.add_row(dir_key, str(files), str(symbols))
        console.print(layer_table)

    # Patterns
    if data.get("patterns"):
        tree = Tree("[bold]Key patterns[/bold]")
        for p in data["patterns"]:
            tree.add(p)
        console.print(tree)

    # Dependencies
    if data.get("dependencies"):
        dep_parts = [f"{name} [dim]({src})[/dim]" for name, src in data["dependencies"]]
        console.print(Panel(", ".join(dep_parts), title="Top dependencies", border_style="dim"))

    # Hub files
    if data.get("hub_files"):
        tree = Tree("[bold]Hub files[/bold]")
        for path, conns in data["hub_files"]:
            tree.add(f"[cyan]{path}[/cyan] — {conns} connections")
        console.print(tree)


def print_stats(stats: dict) -> None:
    """Render graph statistics as Rich tables."""
    # Node counts
    node_table = Table(title="Solograph Statistics", show_lines=False)
    node_table.add_column("Node type", style="cyan")
    node_table.add_column("Count", justify="right", style="bold")

    for label, count in stats.items():
        if label != "edges":
            node_table.add_row(label, f"{count:,}")

    console.print(node_table)

    # Edge counts
    if "edges" in stats:
        edge_table = Table(title="Edges", show_lines=False)
        edge_table.add_column("Type", style="green")
        edge_table.add_column("Count", justify="right", style="bold")
        for etype, count in stats["edges"].items():
            edge_table.add_row(etype, f"{count:,}")
        console.print(edge_table)


def print_scan_progress(name: str, stack: str | None, files: int, symbols: int, packages: int) -> None:
    """Print single project scan result."""
    stack_display = stack or "[dim](no stack)[/dim]"
    console.print(
        f"  [cyan]{name}[/cyan] [green]{stack_display}[/green]  "
        f"{files} files  {symbols} symbols  {packages} packages"
    )


def print_scan_summary(stats: dict) -> None:
    """Render scan summary with Rich."""
    console.print("\n[bold]Graph summary:[/bold]")
    node_table = Table(show_header=False, show_lines=False, padding=(0, 1))
    node_table.add_column("Label", style="cyan")
    node_table.add_column("Count", justify="right")

    for label, count in stats.items():
        if label != "edges":
            node_table.add_row(label, f"{count:,}")
    console.print(node_table)

    if "edges" in stats:
        edge_table = Table(show_header=False, show_lines=False, padding=(0, 1), title="Edges")
        edge_table.add_column("Type", style="green")
        edge_table.add_column("Count", justify="right")
        for etype, count in stats["edges"].items():
            edge_table.add_row(etype, f"{count:,}")
        console.print(edge_table)


def print_shared_packages(shared: list[dict]) -> None:
    """Render shared packages as a Rich table."""
    table = Table(title="Shared Packages", show_lines=False)
    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Projects", style="green")

    for r in shared:
        table.add_row(r["package"], ", ".join(r["projects"]))

    console.print(table)
