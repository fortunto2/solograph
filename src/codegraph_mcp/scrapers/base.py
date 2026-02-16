"""Base spider for source scraping.

Provides common patterns: rate limiting, deduplication, error handling,
and item collection via Scrapy FEEDS export.

Scrapers only collect publicly available data from sites that allow
crawling via robots.txt. No authentication bypass, no PII extraction.
For research and educational purposes.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

import scrapy


class BaseSourceSpider(scrapy.Spider):
    """Base spider with common patterns for source scraping.

    Subclasses must set:
      - name: str — unique spider name
      - start_urls: list[str] — entry points
      - source_type: str — e.g. "trustmrr-startup", "producthunt-profile"

    Subclasses must implement:
      - parse(response) — pagination / link discovery
      - parse_item(response) — extract a single item dict
    """

    source_type: ClassVar[str] = ""

    custom_settings: ClassVar[dict] = {
        "DOWNLOAD_DELAY": 2,
        "CONCURRENT_REQUESTS": 1,
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "WARNING",
        "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
        "USER_AGENT": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self, limit: int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.collected_items: list[dict] = []
        self._count = 0

    def _should_stop(self) -> bool:
        return bool(self.limit and self._count >= self.limit)

    def _collect(self, item: dict):
        """Add item to collected list and increment counter."""
        self._count += 1
        self.collected_items.append(item)

    @staticmethod
    def make_id(prefix: str, key: str) -> str:
        """Generate deterministic doc_id from prefix:key."""
        return hashlib.md5(f"{prefix}:{key}".encode()).hexdigest()

    def parse_item(self, response) -> dict | None:
        """Extract a single item from a detail page. Override in subclass."""
        raise NotImplementedError


def run_spider(spider_cls, timeout: int = 600, **kwargs) -> list[dict]:
    """Run a spider in a subprocess, return collected items as dicts.

    Uses subprocess + FEEDS JSON export to avoid Twisted reactor issues.
    Each call spawns a fresh Python process.
    kwargs are passed via temp file (safe for large payloads like skip_slugs).
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        output_path = f.name

    # Write kwargs to temp file (avoids shell escaping issues with large data)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(kwargs, f)
        kwargs_path = f.name

    spider_module = spider_cls.__module__
    spider_name = spider_cls.__name__

    script = f"""
import json
from scrapy.crawler import CrawlerProcess
from {spider_module} import {spider_name}

with open("{kwargs_path}") as f:
    kwargs = json.load(f)

settings = dict({spider_name}.custom_settings)
settings["FEEDS"] = {{"{output_path}": {{"format": "jsonlines"}}}}
settings["LOG_LEVEL"] = "WARNING"

process = CrawlerProcess(settings)
process.crawl({spider_name}, **kwargs)
process.start()
"""

    # Run in subprocess with same Python
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Clean up kwargs file
    Path(kwargs_path).unlink(missing_ok=True)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            lines = stderr.splitlines()[-5:]
            for line in lines:
                print(f"  Spider error: {line}", file=sys.stderr)

    # Read collected items from JSONL
    items = []
    output = Path(output_path)
    if output.exists():
        for line in output.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        output.unlink(missing_ok=True)

    return items
