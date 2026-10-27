"""Rich console output for Solograph CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


def print_explain(data: dict) -> None:
    """Render explain output with Rich panels, tables, and trees."""
    # Header
    title = f"[bold cyan]{data['name']}[/bold cyan] — {data['stack']}"
    if data.get("description"):
        title += f"\n{data['description']}"

    counts = data["counts"]
    title += f"\n[dim]{counts['files']} files | {counts['symbols']} symbols | {counts['packages']} packages[/dim]"
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

    # Files with the most definitions
    if data.get("largest_files"):
        tree = Tree("[bold]Most definitions[/bold]")
        for path, n in data["largest_files"]:
            tree.add(f"[cyan]{path}[/cyan] — {n} symbols")
        console.print(tree)


def _counts_table(stats: dict, title: str | None = None) -> Table:
    table = Table(title=title, show_lines=False)
    table.add_column("What", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    for label, value in stats.items():
        if isinstance(value, dict):
            for k, v in value.items():
                table.add_row(f"  {label}: {k}", f"{v:,}")
        else:
            table.add_row(label, f"{value:,}")
    return table


def print_stats(stats: dict) -> None:
    """Portfolio counts."""
    console.print(_counts_table(stats, "Solograph portfolio"))


def print_scan_progress(name: str, stack: str | None, report: dict) -> None:
    """One project's scan: what was read, kept and removed."""
    stack_display = stack or "[dim](no stack)[/dim]"
    console.print(
        f"  [cyan]{name}[/cyan] [green]{stack_display}[/green]  {report['read']} files read  "
        f"{report['unchanged']} unchanged  {report['removed']} removed  {report['symbols']} symbols"
    )


def print_scan_summary(stats: dict) -> None:
    console.print()
    console.print(_counts_table(stats, "Portfolio"))


def print_shared_packages(shared: list[dict]) -> None:
    """Render shared packages as a Rich table."""
    table = Table(title="Shared Packages", show_lines=False)
    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Source", style="dim")
    table.add_column("Projects", style="green")

    for r in shared:
        table.add_row(r["package"], r.get("source", ""), ", ".join(r["projects"]))

    console.print(table)
