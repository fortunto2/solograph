"""CLI commands for external sources: YouTube, ProductHunt, TrustMRR, Telegram.

Split out of cli.py (1674 lines, over the 1000-line module limit). The commands
register on the same `cli` group; cli.py imports this module last.
"""

import os
from pathlib import Path

import click

from .cli import cli, console


@cli.command("source-search")
@click.argument("query")
@click.option("--source", "-s", default=None, help="Filter by source name (telegram, youtube)")
@click.option("--limit", "-n", default=5, help="Number of results")
def source_search_cmd(query, source, limit):
    """Search indexed external sources (Telegram, YouTube, etc.)."""
    from .sources import SourceIndex

    idx = SourceIndex()
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
        click.echo(f"{i}. [{src}] {r['title'][:80]}{chapter_info}  (score {r['score']})")
        if r.get("url"):
            click.echo(f"   {r['url']}")
        if r.get("tags"):
            click.echo(f"   Tags: {r['tags']}")
        if r.get("content"):
            click.echo(f"   {r['content'][:120]}")
        click.echo()


@cli.command("source-list")
def source_list_cmd():
    """List indexed external sources with document counts."""
    from .sources import SourceIndex

    idx = SourceIndex()
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
        if s.get("makers"):
            extra += f"  ({s['makers']} makers)"
        click.echo(f"  {s['source']:12s} {s['count']:4d} docs{extra}  {s['path']}")
        total += s["count"]
    click.echo(f"\n  Total: {total} documents")


@cli.command("source-tags")
@click.option("--source", "-s", default="youtube", help="Source name (default: youtube)")
def source_tags_cmd(source):
    """List all auto-detected topics with video counts."""
    from .sources import SourceIndex

    idx = SourceIndex()
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
def source_related_cmd(video_id, source):
    """Find related videos by shared tags."""
    import re

    from .sources import SourceIndex

    # Extract video ID from URL if needed
    m = re.search(r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})", video_id)
    if m:
        video_id = m.group(1)

    idx = SourceIndex()
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
def index_youtube_cmd(channels, channels_file, limit, url, import_path, dry_run):
    """Index YouTube transcripts into the sources index."""
    from .indexers.youtube import YouTubeIndexer

    channels_path = Path(channels_file) if channels_file else None
    searxng_url = os.environ.get("TAVILY_API_URL", "http://localhost:8013")

    indexer = YouTubeIndexer(
        channels_path=channels_path,
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
def index_trustmrr_cmd(categories, limit, dry_run, force):
    """Scrape TrustMRR verified startup revenues into the sources index."""
    from .indexers.trustmrr import TrustMRRIndexer

    indexer = TrustMRRIndexer()
    indexer.run(
        categories=list(categories) if categories else None,
        limit=limit,
        dry_run=dry_run,
        force=force,
    )


@cli.command("index-producthunt")
@click.option("--days", "-d", type=int, default=30, help="Days back to scrape (default: 30)")
@click.option("--limit", "-n", type=int, default=None, help="Max products to scrape")
@click.option("--all", "all_time", is_flag=True, help="Scrape ALL products since 2013 (~13k featured)")
@click.option("--resume", is_flag=True, help="Enable JSONL resume file for long scrapes")
@click.option("--include-all", "include_all", is_flag=True, help="Include non-featured posts too")
@click.option("--dry-run", is_flag=True, help="Scrape only, don't insert into DB")
@click.option("--force", is_flag=True, help="Re-scrape all, ignore already-indexed")
def index_producthunt_cmd(days, limit, all_time, resume, include_all, dry_run, force):
    """Scrape ProductHunt leaderboard into the sources index.

    \b
    Uses PH GraphQL API v2 (no browser needed).
    Set PH_TOKEN env var (developer token) or PH_CLIENT_ID + PH_CLIENT_SECRET.

    \b
    Examples:
      solograph-cli index-producthunt                  # Last 30 days
      solograph-cli index-producthunt -d 7             # Last 7 days
      solograph-cli index-producthunt -n 50            # Limit 50 products
      solograph-cli index-producthunt --all --resume   # Full dump (~13k, ~2h)
      solograph-cli index-producthunt --dry-run        # Preview only
    """
    from .indexers.producthunt import ProductHuntIndexer

    if all_time:
        days = 4500  # Since PH launch (~2013)
        resume = True  # Always resume for full dumps

    indexer = ProductHuntIndexer()
    indexer.run(
        days=days,
        limit=limit,
        dry_run=dry_run,
        force=force,
        resume=resume,
        featured_only=not include_all,
    )


@cli.command("import-producthunt")
@click.argument("jsonl_path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Preview only, don't insert into DB")
def import_producthunt_cmd(jsonl_path, dry_run):
    """Import ProductHunt products from a JSONL file into the sources index.

    \b
    Use this to re-index from a previously scraped dump file.

    \b
    Examples:
      solograph-cli import-producthunt ~/.solo/sources/producthunt_scrape.jsonl
      solograph-cli import-producthunt dump.jsonl --dry-run
    """
    import json

    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    click.echo(f"Loaded {len(items)} products from {jsonl_path}")

    from .indexers.producthunt import ProductHuntIndexer

    indexer = ProductHuntIndexer()
    indexer.import_items(items, dry_run=dry_run)


@cli.command("index-producthunt-makers")
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True),
    default=None,
    help="JSONL file with scraped posts (to extract usernames)",
)
@click.option("--limit", "-n", type=int, default=None, help="Max profiles to scrape")
@click.option("--delay", "-d", type=float, default=3.0, help="Seconds between requests (default: 3)")
@click.option(
    "--resume",
    type=click.Path(),
    default=None,
    help="Resume file path (default: ~/.solo/sources/producthunt_makers.jsonl)",
)
@click.option("--dry-run", is_flag=True, help="Extract usernames only, don't scrape")
def index_producthunt_makers_cmd(source, limit, delay, resume, dry_run):
    """Scrape ProductHunt maker profiles via Playwright and index into the sources index.

    \b
    Two-phase pipeline:
      1. Extract unique usernames from scraped posts JSONL
      2. Playwright visits /@username pages, extracts Apollo cache data

    \b
    Examples:
      solograph-cli index-producthunt-makers --dry-run --limit 5
      solograph-cli index-producthunt-makers --limit 10 --delay 3
      solograph-cli index-producthunt-makers -s ~/dump.jsonl -n 20
      solograph-cli index-producthunt-makers --resume ~/.solo/sources/makers.jsonl
    """
    from .indexers.producthunt import DEFAULT_RESUME_PATH, ProductHuntIndexer
    from .scrapers.producthunt_makers import extract_makers_from_posts, run_maker_scraper

    # Default resume path for makers
    default_makers_resume = str(Path.home() / ".solo" / "sources" / "producthunt_makers.jsonl")
    resume_path = resume or default_makers_resume

    # Phase 1: extract usernames
    jsonl_path = source or DEFAULT_RESUME_PATH
    if not Path(jsonl_path).exists():
        console.print(f"[red]Posts JSONL not found:[/red] {jsonl_path}")
        console.print("[dim]Run: solograph-cli index-producthunt --resume  first[/dim]")
        raise SystemExit(1)

    usernames = extract_makers_from_posts(jsonl_path)
    console.print(f"Extracted [green]{len(usernames)}[/green] unique maker usernames from {jsonl_path}")

    if not usernames:
        console.print("[yellow]No usernames found in JSONL[/yellow]")
        return

    if dry_run:
        show = usernames[:20] if limit is None else usernames[:limit]
        for u in show:
            console.print(f"  @{u}")
        if len(usernames) > len(show):
            console.print(f"  ... and {len(usernames) - len(show)} more")
        return

    # Phase 2: Playwright scrape
    Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
    console.print(f"Resume file: [dim]{resume_path}[/dim]")

    profiles = run_maker_scraper(
        usernames=usernames,
        limit=limit,
        resume_path=resume_path,
        delay=delay,
    )

    console.print(f"Scraped [green]{len(profiles)}[/green] maker profiles")

    if not profiles:
        return

    # Phase 3: index into the sources index
    indexer = ProductHuntIndexer()
    indexer.import_makers(profiles)


@cli.command("scrape-ph-browser")
@click.option(
    "--mode",
    type=click.Choice(["enrich", "discover"]),
    default="enrich",
    help="Mode: enrich existing JSONL or discover new products from leaderboard",
)
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True),
    default=None,
    help="Source JSONL file with existing products (enrich mode)",
)
@click.option("--year", type=int, default=2026, help="Year for discover mode (default: 2026)")
@click.option("--from-date", default=None, help="Start date YYYY-MM-DD (discover mode)")
@click.option("--to-date", default=None, help="End date YYYY-MM-DD (discover mode, default: today)")
@click.option("--limit", "-n", type=int, default=None, help="Max products to scrape")
@click.option("--resume", is_flag=True, help="Skip already-enriched slugs (check output file)")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output JSONL path (default: ~/.solo/sources/ph_2026_enriched.jsonl)",
)
@click.option("--dry-run", is_flag=True, help="Scrape a few items and print, don't save")
@click.option("--delay", type=float, default=2.0, help="Seconds between requests (default: 2)")
@click.option("--concurrency", "-j", type=int, default=1, help="Parallel browser tabs (default: 1, try 3-5)")
@click.option("--index", is_flag=True, help="Also import enriched items into the sources index")
def scrape_ph_browser_cmd(
    mode, source, year, from_date, to_date, limit, resume, output, dry_run, delay, concurrency, index
):
    """Scrape ProductHunt product pages via HTTP (SSR HTML) to get non-redacted maker data.

    \b
    PH GraphQL API redacts maker info (~93% of 2026 products). Browser spider
    extracts full data from SSR HTML: maker usernames, JSON-LD, topics, upvotes.

    \b
    Enrich mode — update existing JSONL with browser data:
      solograph-cli scrape-ph-browser --mode enrich -s ph_2026_all.jsonl --limit 5 --dry-run
      solograph-cli scrape-ph-browser --mode enrich -s ph_2026_all.jsonl --resume

    \b
    Discover mode — crawl daily leaderboard for new products:
      solograph-cli scrape-ph-browser --mode discover --from-date 2026-01-10 --dry-run
      solograph-cli scrape-ph-browser --mode discover --from-date 2026-01-10 --limit 500
    """
    import json as json_mod

    from .indexers.producthunt import ProductHuntIndexer
    from .scrapers.producthunt_browser import run_ph_browser_scraper

    default_output = str(Path.home() / ".solo" / "sources" / "ph_2026_enriched.jsonl")
    output_path = output or default_output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing items for enrich mode
    original_items: list[dict] = []
    slugs: list[str] = []

    if mode == "enrich":
        if not source:
            default_source = str(Path.home() / ".solo" / "sources" / "ph_2026_all.jsonl")
            if Path(default_source).exists():
                source = default_source
            else:
                console.print("[red]Enrich mode requires --source JSONL file[/red]")
                raise SystemExit(1)

        for line in Path(source).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    original_items.append(json_mod.loads(line))
                except json_mod.JSONDecodeError:
                    pass

        slugs = [item.get("slug", "") for item in original_items if item.get("slug")]
        console.print(f"Loaded [green]{len(original_items)}[/green] products from {source}")

    # Resume: skip slugs already in output file
    skip_slugs: list[str] = []
    if resume and Path(output_path).exists():
        for line in Path(output_path).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    item = json_mod.loads(line)
                    s = item.get("slug", "")
                    if s and item.get("_source") == "browser":
                        skip_slugs.append(s)
                except json_mod.JSONDecodeError:
                    pass
        if skip_slugs:
            console.print(f"Resuming: skipping [yellow]{len(skip_slugs)}[/yellow] already-enriched slugs")

    # Build scraper kwargs
    scraper_kwargs: dict = {
        "limit": limit,
        "skip_slugs": skip_slugs,
        "resume_path": output_path if resume else None,
        "delay": delay,
        "concurrency": concurrency,
    }

    if mode == "enrich":
        scraper_kwargs["slugs"] = slugs
    else:
        if from_date:
            scraper_kwargs["from_date"] = from_date
        if to_date:
            scraper_kwargs["to_date"] = to_date

    console.print(f"[bold]Scraping PH products (Playwright, {mode} mode)...[/bold]")
    browser_items = run_ph_browser_scraper(**scraper_kwargs)
    console.print(f"Scraped [green]{len(browser_items)}[/green] products")

    if dry_run:
        for item in browser_items[:10]:
            makers = item.get("makers", [])
            maker_str = ", ".join(f"@{m.get('username', '?')}" for m in makers[:3])
            console.print(
                f"  {item.get('name', '?')} | {item.get('upvotes', 0)}upvotes | "
                f"makers: {maker_str or 'none'} | topics: {item.get('topics', '')[:40]}"
            )
        if len(browser_items) > 10:
            console.print(f"  ... and {len(browser_items) - 10} more")
        return

    if not browser_items:
        console.print("[yellow]No items scraped[/yellow]")
        return

    # Merge and save
    indexer = ProductHuntIndexer()

    if mode == "enrich" and original_items:
        merged = indexer.enrich_items(original_items, browser_items)
    else:
        merged = browser_items

    # Save JSONL
    _save_jsonl_cli(merged, output_path)
    console.print(f"Saved [green]{len(merged)}[/green] products to {output_path}")

    # Optionally index into the sources index
    if index:
        console.print("\nIndexing into the sources index...")
        indexer.import_items(merged)


def _save_jsonl_cli(items: list[dict], path: str):
    """Save items to JSONL file (atomic write)."""
    import json as json_mod

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for item in items:
            f.write(json_mod.dumps(item, ensure_ascii=False) + "\n")
    Path(tmp).rename(path)


@cli.command("ph-makers-csv")
@click.option(
    "--products",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Products JSONL file (default: ~/.solo/sources/ph_2026_enriched.jsonl)",
)
@click.option(
    "--profiles",
    type=click.Path(exists=True),
    default=None,
    help="Profiles JSONL file (default: ~/.solo/sources/ph_2026_profiles.jsonl)",
)
@click.option("--min-points", type=int, default=5, help="Minimum points (default: 5)")
@click.option("--max-points", type=int, default=1000, help="Maximum points (default: 1000)")
@click.option("--min-upvotes", type=int, default=5, help="Minimum product upvotes (default: 5)")
@click.option("--max-upvotes", type=int, default=700, help="Maximum product upvotes (default: 700)")
@click.option("--linkedin-only", is_flag=True, help="Only include profiles with LinkedIn")
@click.option("--min-streak", type=int, default=0, help="Minimum streak (default: 0)")
@click.option("--max-streak", type=int, default=None, help="Maximum streak (default: no limit)")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output CSV path")
def ph_makers_csv_cmd(
    products,
    profiles,
    min_points,
    max_points,
    min_upvotes,
    max_upvotes,
    linkedin_only,
    min_streak,
    max_streak,
    output,
):
    """Generate CSV of PH makers/commenters with profile_type from product data.

    \b
    Cross-references products (with maker roles) and profile data to produce
    a CSV with profile_type (maker/commenter), points, streak, LinkedIn, etc.

    \b
    Examples:
      solograph-cli ph-makers-csv --min-points 5 --max-points 1000
      solograph-cli ph-makers-csv --linkedin-only --min-streak 5 --max-streak 400
      solograph-cli ph-makers-csv --min-upvotes 10 --max-upvotes 500 -o makers.csv
    """
    import csv
    import json as json_mod

    default_products = str(Path.home() / ".solo" / "sources" / "ph_2026_enriched.jsonl")
    default_profiles = str(Path.home() / ".solo" / "sources" / "ph_2026_profiles.jsonl")
    products_path = products or default_products
    profiles_path = profiles or default_profiles

    if not Path(products_path).exists():
        console.print(f"[red]Products file not found: {products_path}[/red]")
        raise SystemExit(1)

    # Step 1: Build username → role map + per-role product lists
    user_roles: dict[str, str] = {}  # username → "maker" if maker in ANY product, else "commenter"
    user_maker_of: dict[str, list[str]] = {}  # products where user is maker
    user_commenter_on: dict[str, list[str]] = {}  # products where user is commenter
    user_upvotes: dict[str, int] = {}  # username → max upvotes of their products
    user_featured: dict[str, int] = {}  # username → count of featured products

    for line in Path(products_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json_mod.loads(line)
        except json_mod.JSONDecodeError:
            continue
        product_name = item.get("name", item.get("slug", "?"))
        slug = item.get("slug", "")
        product_url = f"https://www.producthunt.com/posts/{slug}" if slug else ""
        featured = bool(item.get("featured") or item.get("featured_at"))
        feat_tag = " [F]" if featured else ""
        product_label = f"{product_name}{feat_tag} ({product_url})" if product_url else f"{product_name}{feat_tag}"
        upvotes = item.get("upvotes", 0)
        if upvotes < min_upvotes or upvotes > max_upvotes:
            continue
        for m in item.get("makers", []):
            username = m.get("username", "")
            if not username:
                continue
            role = m.get("role", "maker")  # old data without role = assume maker
            if role == "maker":
                user_roles[username] = "maker"
                user_maker_of.setdefault(username, []).append(product_label)
            else:
                if user_roles.get(username) != "maker":
                    user_roles[username] = "commenter"
                user_commenter_on.setdefault(username, []).append(product_label)
            user_upvotes[username] = max(user_upvotes.get(username, 0), upvotes)
            if featured:
                user_featured.setdefault(username, 0)
                user_featured[username] += 1

    console.print(
        f"Products: [green]{len(user_roles)}[/green] unique people from "
        f"products with {min_upvotes}-{max_upvotes} upvotes"
    )

    # Step 2: Load profiles
    profile_map: dict[str, dict] = {}
    if Path(profiles_path).exists():
        for line in Path(profiles_path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                prof = json_mod.loads(line)
                username = prof.get("username", "")
                if username:
                    profile_map[username] = prof
            except json_mod.JSONDecodeError:
                pass
        console.print(f"Profiles: [green]{len(profile_map)}[/green] loaded from {profiles_path}")
    else:
        console.print(f"[yellow]No profiles file: {profiles_path}[/yellow]")

    # Step 3: Build merged rows with filters
    rows = []
    for username in sorted(user_roles.keys()):
        profile = profile_map.get(username, {})
        points = profile.get("points", 0)
        streak = profile.get("streak", 0)
        linkedin = profile.get("linkedin", "")

        # Filters
        if points < min_points or points > max_points:
            continue
        if linkedin_only and not linkedin:
            continue
        if streak < min_streak:
            continue
        if max_streak is not None and streak > max_streak:
            continue

        profile_type = user_roles.get(username, "commenter")
        maker_of = user_maker_of.get(username, [])
        commenter_on = user_commenter_on.get(username, [])

        rows.append(
            {
                "username": username,
                "profile_type": profile_type,
                "name": profile.get("headline", "") or "",
                "points": points,
                "streak": streak,
                "linkedin": linkedin,
                "twitter": profile.get("twitter", ""),
                "github": profile.get("github", ""),
                "website": profile.get("website", ""),
                "is_maker_ph": profile.get("is_maker"),
                "followers": profile.get("followers", 0),
                "max_upvotes": user_upvotes.get(username, 0),
                "featured_count": user_featured.get(username, 0),
                "maker_of_count": len(maker_of),
                "commenter_on_count": len(commenter_on),
                "maker_of": "; ".join(maker_of[:5]),
                "commenter_on": "; ".join(commenter_on[:5]),
                "profile_url": f"https://www.producthunt.com/@{username}",
            }
        )

    if not rows:
        console.print("[yellow]No matching profiles found[/yellow]")
        return

    # Step 4: Save CSV
    if not output:
        suffix = ""
        if linkedin_only:
            suffix += "_linkedin"
        if min_streak > 0:
            suffix += f"_streak{min_streak}"
        output = str(Path.home() / ".solo" / "sources" / f"ph_2026_makers{suffix}.csv")

    fieldnames = list(rows[0].keys())
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    makers_count = sum(1 for r in rows if r["profile_type"] == "maker")
    commenters_count = sum(1 for r in rows if r["profile_type"] == "commenter")
    console.print(
        f"\nSaved [green]{len(rows)}[/green] profiles to {output}\n"
        f"  Makers: [green]{makers_count}[/green], Commenters: [cyan]{commenters_count}[/cyan]"
    )


@cli.command("source-delete")
@click.argument("source_name")
def source_delete_cmd(source_name):
    """Delete one source from the sources index."""
    from .sources import SourceIndex

    idx = SourceIndex()
    if idx.delete_source(source_name):
        click.echo(f"Deleted source: {source_name}")
    else:
        click.echo(f"Source not found: {source_name}")
