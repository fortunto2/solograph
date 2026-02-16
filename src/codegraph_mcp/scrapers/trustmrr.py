"""TrustMRR spider — scrape verified startup revenue data.

Discovery via sitemap.xml (4000+ startups), data from OG tags + RSC payload.
Extracts: name, revenue (last 30 days), description, tech stack, founder, country.

Usage:
    from codegraph_mcp.scrapers.trustmrr import TrustMRRSpider
    from codegraph_mcp.scrapers.base import run_spider
    items = run_spider(TrustMRRSpider, limit=50)
"""

import re
from typing import ClassVar

import scrapy

from .base import BaseSourceSpider

BASE_URL = "https://trustmrr.com"

# Actual categories from sitemap (32 total)
CATEGORIES = [
    "ai",
    "analytics",
    "community",
    "content-creation",
    "crypto-web3",
    "customer-support",
    "design-tools",
    "developer-tools",
    "ecommerce",
    "education",
    "entertainment",
    "fintech",
    "games",
    "green-tech",
    "health-fitness",
    "iot-hardware",
    "legal",
    "marketing",
    "marketplace",
    "mobile-apps",
    "news-magazines",
    "no-code",
    "productivity",
    "real-estate",
    "recruiting",
    "saas",
    "sales",
    "security",
    "social-media",
    "travel",
    "utilities",
]


class TrustMRRSpider(BaseSourceSpider):
    name = "trustmrr"
    source_type = "trustmrr-startup"
    allowed_domains: ClassVar[list[str]] = ["trustmrr.com"]
    start_urls: ClassVar[list[str]] = [f"{BASE_URL}/sitemap-0.xml"]

    custom_settings: ClassVar[dict] = {
        **BaseSourceSpider.custom_settings,
        "DOWNLOAD_DELAY": 1.5,
        "CONCURRENT_REQUESTS": 2,
    }

    def __init__(
        self,
        categories: list[str] | None = None,
        limit: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, limit=limit, **kwargs)
        self.filter_categories = set(categories) if categories else None

    def parse(self, response):
        """Parse sitemap XML to discover all startup URLs."""
        urls = re.findall(r"<loc>(https://trustmrr\.com/startup/[^<]+)</loc>", response.text)
        queued = 0
        for url in urls:
            if self.limit and queued >= self.limit:
                return
            slug = url.rstrip("/").split("/")[-1]
            yield scrapy.Request(url, callback=self.parse_item, meta={"slug": slug})
            queued += 1

    def parse_item(self, response):
        """Extract startup data from OG tags + page content."""
        if self._should_stop():
            return

        slug = response.meta.get("slug", response.url.rstrip("/").split("/")[-1])

        # --- OG tags (most reliable source) ---
        og_title = response.css('meta[property="og:title"]::attr(content)').get("")
        og_desc = response.css('meta[property="og:description"]::attr(content)').get("")
        meta_desc = response.css('meta[name="description"]::attr(content)').get("")

        # Name and revenue from og:title: "ChatDash, LLC - $43,629 last 30 days | TrustMRR"
        name = ""
        revenue_30d = ""
        m = re.match(r"^(.+?)\s*-\s*\$([\d,]+)\s*last\s*\d+\s*days", og_title)
        if m:
            name = m.group(1).strip()
            revenue_30d = f"${m.group(2)}"
        else:
            # Fallback: "Name - Verified revenue | TrustMRR" (no revenue shown)
            name = og_title.split(" - ")[0].strip() if " - " in og_title else ""

        if not name:
            name = response.css("h1::text").get("").strip()

        # Description: prefer meta description over og:description
        description = meta_desc or og_desc

        # --- Category from /category/ links ---
        category = ""
        cat_links = response.css('a[href*="/category/"]::attr(href)').getall()
        for href in cat_links:
            cat = href.rstrip("/").split("/")[-1]
            if cat and cat != "category":
                category = cat
                break

        # Filter by category if specified
        if self.filter_categories and category and category not in self.filter_categories:
            return

        # --- Country from /country/ links ---
        country = ""
        country_links = response.css('a[href*="/country/"]::attr(href)').getall()
        if country_links:
            country = country_links[0].rstrip("/").split("/")[-1].upper()

        # --- Founder X handle ---
        founder_x = ""
        x_links = response.css('a[href*="x.com/"]::attr(href)').getall()
        x_links += response.css('a[href*="twitter.com/"]::attr(href)').getall()
        for href in x_links:
            handle = href.rstrip("/").split("/")[-1]
            if handle and handle not in ("x.com", "twitter.com", ""):
                founder_x = handle
                break

        # --- Tech stack from /tech/ links ---
        tech_links = response.css('a[href*="/tech/"]::attr(href)').getall()
        techs = []
        for href in tech_links:
            tech = href.rstrip("/").split("/")[-1]
            if tech and tech != "tech":
                techs.append(tech)
        stack = ", ".join(techs[:10])

        # --- Website ---
        website = ""
        for href in response.css('a[rel="noopener noreferrer"]::attr(href)').getall():
            if "trustmrr.com" not in href and href.startswith("http"):
                website = href
                break

        item = {
            "slug": slug,
            "name": name,
            "description": description[:500] if description else "",
            "revenue_30d": revenue_30d,
            "tech_stack": stack,
            "founder_x": founder_x,
            "website": website,
            "country": country,
            "category": category,
            "url": response.url,
        }

        self._collect(item)
        yield item
