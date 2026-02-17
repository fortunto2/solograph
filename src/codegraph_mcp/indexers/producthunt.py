"""ProductHunt indexer — scrape leaderboard and index into FalkorDB vectors.

Runs PH GraphQL API scraper → maps items to SourceDoc → embeds and upserts.
After indexing, products are searchable via `source_search("AI productivity tool")`.

Usage:
    solograph-cli index-producthunt                  # Last 30 days
    solograph-cli index-producthunt -d 7             # Last 7 days
    solograph-cli index-producthunt -n 50            # Limit to 50 products
    solograph-cli index-producthunt --all --resume   # Full dump with resume
    solograph-cli index-producthunt --dry-run        # Parse only, no DB
"""

import hashlib
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from ..models import MakerProfile, SourceDoc
from ..vectors.source_index import SourceIndex

console = Console()

SOURCE_NAME = "producthunt"
DEFAULT_RESUME_PATH = str(Path.home() / ".solo" / "sources" / "producthunt_scrape.jsonl")


class ProductHuntIndexer:
    """Scrape ProductHunt leaderboard and index into FalkorDB source vectors."""

    def __init__(self, backend: str | None = None):
        self.backend = backend

    def run(
        self,
        days: int = 30,
        limit: int | None = None,
        dry_run: bool = False,
        force: bool = False,
        resume: bool = False,
        featured_only: bool = True,
    ):
        """Run full pipeline: scrape → map → embed → store."""
        from ..scrapers.producthunt import run_ph_scraper

        console.print(f"[bold]Scraping ProductHunt leaderboard...[/bold] days={days}, limit={limit}")

        # Skip already-indexed slugs (unless --force)
        skip_slugs: list[str] = []
        if not force and not dry_run:
            skip_slugs = self._get_existing_slugs()
            if skip_slugs:
                console.print(
                    f"Skipping [yellow]{len(skip_slugs)}[/yellow] already-indexed products (use --force to re-scrape)"
                )

        # Resume path for long scrapes
        resume_path = DEFAULT_RESUME_PATH if resume else None
        if resume_path:
            Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
            console.print(f"Resume file: [dim]{resume_path}[/dim]")

        items = run_ph_scraper(
            days=days,
            limit=limit,
            skip_slugs=skip_slugs,
            resume_path=resume_path,
            featured_only=featured_only,
        )

        console.print(f"Scraped [green]{len(items)}[/green] products")

        if dry_run:
            for item in items[:15]:
                up = item.get("upvotes", 0)
                topics = item.get("topics", "")
                console.print(f"  {item.get('name', '?')} | {up}↑ | {topics}")
            if len(items) > 15:
                console.print(f"  ... and {len(items) - 15} more")
            return items

        return self._index_items(items)

    def import_items(self, items: list[dict], dry_run: bool = False) -> list[dict]:
        """Index pre-scraped items (e.g. from JSON file)."""
        if dry_run:
            for item in items[:5]:
                console.print(f"  {item.get('name', '?')} — {item.get('upvotes', 0)}↑")
            return items
        return self._index_items(items)

    def _index_items(self, items: list[dict]) -> list[dict]:
        """Map scraped items to SourceDoc and upsert into FalkorDB."""
        idx = SourceIndex(backend=self.backend)
        new_count = 0
        skip_count = 0

        with Progress(console=console) as progress:
            task = progress.add_task("Indexing...", total=len(items))
            for item in items:
                doc = self._to_source_doc(item)
                if not doc:
                    skip_count += 1
                    progress.advance(task)
                    continue

                was_new = idx.upsert_one(SOURCE_NAME, doc)
                if was_new:
                    new_count += 1
                progress.advance(task)

        console.print(
            f"Done: [green]{new_count} new[/green], {len(items) - new_count - skip_count} updated, {skip_count} skipped"
        )
        return items

    def import_makers(
        self,
        makers: list[MakerProfile],
        dry_run: bool = False,
    ) -> list[MakerProfile]:
        """Index maker profiles into FalkorDB source graph.

        Creates Maker nodes with embeddings and links them to existing
        SourceDoc products via CREATED edges.
        """
        if dry_run:
            for m in makers[:10]:
                console.print(
                    f"  @{m.username} ({m.name}) — {m.points}pts, "
                    f"{m.streak_days}d streak, {m.followers_count} followers"
                )
            if len(makers) > 10:
                console.print(f"  ... and {len(makers) - 10} more")
            return makers

        idx = SourceIndex(backend=self.backend)
        new_count = 0

        with Progress(console=console) as progress:
            task = progress.add_task("Indexing makers...", total=len(makers))
            for maker in makers:
                was_new = idx.upsert_maker(SOURCE_NAME, maker)
                if was_new:
                    new_count += 1
                progress.advance(task)

        total_makers = idx.maker_count(SOURCE_NAME)
        console.print(
            f"Done: [green]{new_count} new[/green], "
            f"{len(makers) - new_count} updated "
            f"(total: {total_makers} makers in graph)"
        )
        return makers

    @staticmethod
    def _get_existing_slugs() -> list[str]:
        """Query FalkorDB for already-indexed ProductHunt slugs."""
        try:
            idx = SourceIndex()
            urls = idx.get_all_urls(SOURCE_NAME)
            slugs = []
            for url in urls:
                # https://www.producthunt.com/posts/{slug}
                parts = url.rstrip("/").split("/")
                if len(parts) >= 2 and parts[-2] == "posts":
                    slugs.append(parts[-1])
            return slugs
        except Exception:
            return []

    @staticmethod
    def _to_source_doc(item: dict) -> SourceDoc | None:
        """Map a scraped PH product to SourceDoc for vector storage."""
        slug = item.get("slug", "")
        name = item.get("name", "")
        if not slug or not name:
            return None

        doc_id = hashlib.md5(f"ph:{slug}".encode()).hexdigest()

        # Build rich text for embedding (semantic search)
        parts = [name]
        if item.get("tagline"):
            parts.append(item["tagline"])
        if item.get("description"):
            parts.append(item["description"][:500])
        if item.get("topics"):
            parts.append(f"Topics: {item['topics']}")
        if item.get("upvotes"):
            parts.append(f"Upvotes: {item['upvotes']}")
        # Makers: support both list[dict] (new) and string (legacy JSONL)
        makers = item.get("makers", [])
        if isinstance(makers, list) and makers:
            maker_names = [m.get("name", m.get("username", "")) for m in makers if isinstance(m, dict)]
            if maker_names:
                parts.append(f"Makers: {', '.join(maker_names)}")
        elif isinstance(makers, str) and makers:
            parts.append(f"Makers: {makers}")
        if item.get("launch_date"):
            parts.append(f"Launched: {item['launch_date']}")
        embed_text = ". ".join(parts)

        # Build content preview
        content_parts = []
        if item.get("tagline"):
            content_parts.append(item["tagline"])
        if item.get("upvotes"):
            content_parts.append(f"{item['upvotes']}↑")
        if item.get("comments"):
            content_parts.append(f"{item['comments']} comments")
        if item.get("rating"):
            content_parts.append(f"★{item['rating']}")
        if item.get("topics"):
            content_parts.append(item["topics"])
        if item.get("website"):
            content_parts.append(item["website"])
        content = " | ".join(content_parts) if content_parts else ""

        # Tags
        tags = ["producthunt"]
        if item.get("topics"):
            for topic in item["topics"].split(", ")[:4]:
                tags.append(topic.lower().replace(" ", "-"))
        if item.get("launch_date"):
            tags.append(item["launch_date"][:7])  # YYYY-MM

        return SourceDoc(
            doc_id=doc_id,
            source_type="producthunt-product",
            source_name=SOURCE_NAME,
            title=f"{name} — {item.get('upvotes', 0)}↑",
            content=content,
            url=item.get("url", f"https://www.producthunt.com/posts/{slug}"),
            created=item.get("launch_date", ""),
            tags=",".join(tags),
            embed_text=embed_text[:3000],
            popularity=item.get("upvotes", 0),
        )
