"""TrustMRR indexer — scrape startups and index into FalkorDB vectors.

Runs TrustMRRSpider → maps items to SourceDoc → embeds and upserts.
After indexing, startups are searchable via `source_search("AI SaaS revenue")`.

Usage:
    solograph-cli index-trustmrr                     # All categories
    solograph-cli index-trustmrr -c ai -c saas       # Specific categories
    solograph-cli index-trustmrr -n 20               # Limit per category
    solograph-cli index-trustmrr --dry-run            # Parse only, no DB
"""

import hashlib

from rich.console import Console
from rich.progress import Progress

from ..models import SourceDoc
from ..vectors.source_index import SourceIndex

console = Console()

SOURCE_NAME = "trustmrr"


class TrustMRRIndexer:
    """Scrape TrustMRR and index into FalkorDB source vectors."""

    def __init__(self, backend: str | None = None):
        self.backend = backend

    def run(
        self,
        categories: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        force: bool = False,
    ):
        """Run full pipeline: scrape → map → embed → store."""
        from ..scrapers.base import run_spider
        from ..scrapers.trustmrr import TrustMRRSpider

        console.print(f"[bold]Scraping TrustMRR...[/bold] categories={categories or 'all'}, limit={limit}")

        # Skip already-indexed slugs (unless --force)
        skip_slugs: list[str] = []
        if not force and not dry_run:
            skip_slugs = self._get_existing_slugs()
            if skip_slugs:
                console.print(
                    f"Skipping [yellow]{len(skip_slugs)}[/yellow] already-indexed startups (use --force to re-scrape)"
                )

        # Full site ~4000 startups at 1.5s/req = ~100min → need large timeout
        timeout = 7200 if not limit else 600
        items = run_spider(TrustMRRSpider, timeout=timeout, categories=categories, limit=limit, skip_slugs=skip_slugs)

        console.print(f"Scraped [green]{len(items)}[/green] startups")

        if dry_run:
            for item in items[:10]:
                rev = item.get("revenue_30d", "") or "n/a"
                cat = item.get("category", "") or "?"
                stack = item.get("tech_stack", "") or ""
                console.print(f"  {item.get('name', '?')} | {rev}/30d | {cat} | {stack}")
            if len(items) > 10:
                console.print(f"  ... and {len(items) - 10} more")
            return items

        return self._index_items(items)

    def import_items(self, items: list[dict], dry_run: bool = False) -> list[dict]:
        """Index pre-scraped items (e.g. from JSON file or API)."""
        if dry_run:
            for item in items[:5]:
                console.print(f"  {item.get('name', '?')} — MRR: {item.get('mrr', '?')}")
            return items
        return self._index_items(items)

    def _index_items(self, items: list[dict]) -> list[dict]:
        """Map spider items to SourceDoc and upsert into FalkorDB."""
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

    @staticmethod
    def _get_existing_slugs() -> list[str]:
        """Query FalkorDB for already-indexed TrustMRR slugs."""
        try:
            idx = SourceIndex()
            urls = idx.get_all_urls(SOURCE_NAME)
            slugs = []
            for url in urls:
                # https://trustmrr.com/startup/{slug}
                parts = url.rstrip("/").split("/")
                if len(parts) >= 2 and parts[-2] == "startup":
                    slugs.append(parts[-1])
            return slugs
        except Exception:
            return []

    @staticmethod
    def _to_source_doc(item: dict) -> SourceDoc | None:
        """Map a spider item to SourceDoc for vector storage."""
        slug = item.get("slug", "")
        name = item.get("name", "")
        if not slug or not name:
            return None

        doc_id = hashlib.md5(f"trustmrr:{slug}".encode()).hexdigest()

        # Build rich text for embedding (semantic search)
        parts = [name]
        if item.get("description"):
            parts.append(item["description"])
        if item.get("revenue_30d"):
            parts.append(f"Revenue: {item['revenue_30d']}/month")
        if item.get("category"):
            parts.append(f"Category: {item['category']}")
        if item.get("tech_stack"):
            parts.append(f"Stack: {item['tech_stack']}")
        if item.get("country"):
            parts.append(f"Country: {item['country']}")
        embed_text = ". ".join(parts)

        # Build content preview
        content_parts = []
        if item.get("revenue_30d"):
            content_parts.append(f"Revenue: {item['revenue_30d']}/30d")
        if item.get("tech_stack"):
            content_parts.append(f"Stack: {item['tech_stack']}")
        if item.get("founder_x"):
            content_parts.append(f"Founder: @{item['founder_x']}")
        if item.get("country"):
            content_parts.append(f"Country: {item['country']}")
        content = " | ".join(content_parts) if content_parts else item.get("description", "")[:500]

        # Tags
        tags = ["trustmrr"]
        if item.get("category"):
            tags.append(item["category"])
        if item.get("tech_stack"):
            for tech in item["tech_stack"].split(", ")[:3]:
                tags.append(tech.lower().replace(".", ""))

        return SourceDoc(
            doc_id=doc_id,
            source_type="trustmrr-startup",
            source_name=SOURCE_NAME,
            title=f"{name} — {item.get('revenue_30d', 'N/A')}/mo",
            content=content,
            url=item.get("url", f"https://trustmrr.com/startup/{slug}"),
            created="",
            tags=",".join(tags),
            embed_text=embed_text[:3000],
        )
