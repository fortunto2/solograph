"""solograph-cli: the scrapers and indexers that write the sources index, and `compact`.

Every read — symbols, sessions, KB, sources search, explain, stats — and the index
itself (scan, registry, session sync) are the Rust binary since 0.9 (`solograph
--help`), the same code the MCP server runs. What stays here needs Python's
ecosystem: yt-dlp, Scrapy, Playwright.
"""

import os
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
def cli():
    """Solograph scrapers: write YouTube, ProductHunt, TrustMRR into the sources index."""


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


from . import cli_sources  # noqa: E402, F401 — registers the source commands on `cli`

if __name__ == "__main__":
    cli()
