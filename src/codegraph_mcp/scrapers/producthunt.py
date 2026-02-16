"""ProductHunt scraper — Playwright-based leaderboard scraper.

PH is behind Cloudflare so Scrapy/HTTP approaches fail (403).
Uses headless Chrome via Playwright to extract products from leaderboard pages.

Data sources:
  - /leaderboard/daily/YYYY/M/D — ~15 products per day (featured)
  - Archive goes back to 2013

Usage:
    from codegraph_mcp.scrapers.producthunt import run_ph_scraper
    items = run_ph_scraper(days=30, limit=100)

Requires: playwright (pip install playwright && playwright install chromium)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# JS to extract products from a leaderboard page.
# Structure: main > SECTION[cursor=pointer] per product card.
# Each card: IMG, DIV(SPAN[name], SPAN[tagline], DIV[topics]), BUTTON[comments], BUTTON[upvotes]
EXTRACT_JS = """() => {
  const products = [];
  const main = document.querySelector('main');
  if (!main) return [];

  const productLinks = main.querySelectorAll('a[href^="/products/"]');
  const seen = new Set();

  for (const link of productLinks) {
    const href = link.getAttribute('href');
    if (!href || href.includes('/reviews') || href.includes('?comment')) continue;
    const slug = href.replace('/products/', '').split('/')[0].split('?')[0];
    if (!slug || seen.has(slug)) continue;

    // Card = SECTION = link -> SPAN -> DIV -> SECTION
    const nameSpan = link.parentElement;
    const infoDiv = nameSpan?.parentElement;
    const card = infoDiv?.parentElement;
    if (!card || card.tagName !== 'SECTION') continue;

    seen.add(slug);
    const name = link.textContent.trim();

    // Tagline = next sibling SPAN after the name SPAN
    const tagline = nameSpan.nextElementSibling?.textContent?.trim() || '';

    // Topics from /topics/ links
    const topicLinks = card.querySelectorAll('a[href^="/topics/"]');
    const topics = [...topicLinks].map(t => t.textContent.trim()).filter(Boolean);

    // Buttons: first = comments, second = upvotes
    const buttons = [...card.querySelectorAll(':scope > button')];
    let comments = 0, upvotes = 0;
    if (buttons.length >= 2) {
      comments = parseInt(buttons[0].textContent.replace(/,/g, '').trim()) || 0;
      upvotes = parseInt(buttons[1].textContent.replace(/,/g, '').trim()) || 0;
    }

    const promoted = card.querySelector('a[href="/sponsor"]') !== null;

    products.push({ slug, name, tagline, topics: topics.join(', '), upvotes, comments, promoted });
  }
  return products;
}"""


def run_ph_scraper(
    days: int = 30,
    limit: int | None = None,
    skip_slugs: list[str] | None = None,
    timeout: int = 600,
) -> list[dict]:
    """Scrape ProductHunt leaderboard via Playwright subprocess.

    Args:
        days: Number of days back from today to scrape (default: 30)
        limit: Max total products to collect
        skip_slugs: Slugs to skip (already indexed)
        timeout: Subprocess timeout in seconds

    Returns:
        List of product dicts with keys:
        slug, name, tagline, topics, upvotes, comments, promoted, launch_date, url
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        output_path = f.name

    config = {
        "days": days,
        "limit": limit,
        "skip_slugs": skip_slugs or [],
        "output_path": output_path,
        "extract_js": EXTRACT_JS,
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(config, f)
        config_path = f.name

    script = f"""
import json
import sys
import time
from datetime import datetime, timedelta

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)
    sys.exit(1)

with open("{config_path}") as f:
    config = json.load(f)

days = config["days"]
limit = config.get("limit")
skip_slugs = set(config.get("skip_slugs", []))
output_path = config["output_path"]
extract_js = config["extract_js"]

items = []
today = datetime.now()

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        channel="chrome",
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        ),
        viewport={{"width": 1440, "height": 900}},
        locale="en-US",
        timezone_id="America/New_York",
    )
    # Remove webdriver flag
    page = context.new_page()
    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {{get: () => undefined}})")
    collected = 0
    consecutive_errors = 0

    for day_offset in range(days):
        if limit and collected >= limit:
            break
        if consecutive_errors >= 5:
            print("Too many consecutive errors, stopping", file=sys.stderr)
            break

        date = today - timedelta(days=day_offset)
        url = f"https://www.producthunt.com/leaderboard/daily/{{date.year}}/{{date.month}}/{{date.day}}"

        success = False
        for attempt in range(3):
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
                time.sleep(2)

                products = page.evaluate(extract_js)
                day_count = 0

                for product in products:
                    slug = product.get("slug", "")
                    if not slug or slug in skip_slugs:
                        continue
                    if product.get("promoted"):
                        continue

                    product["launch_date"] = date.strftime("%Y-%m-%d")
                    product["url"] = f"https://www.producthunt.com/posts/{{slug}}"
                    items.append(product)
                    collected += 1
                    day_count += 1

                    if limit and collected >= limit:
                        break

                print(f"  {{date.strftime('%Y-%m-%d')}}: {{day_count}} products", file=sys.stderr)
                success = True
                consecutive_errors = 0
                break

            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                print(f"  {{date.strftime('%Y-%m-%d')}}: error — {{e}}", file=sys.stderr)

        if not success:
            consecutive_errors += 1

        time.sleep(1.5)

    browser.close()

with open(output_path, "w") as f:
    for item in items:
        f.write(json.dumps(item) + "\\n")

print(f"Total: {{len(items)}} products scraped", file=sys.stderr)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    Path(config_path).unlink(missing_ok=True)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            for line in stderr.splitlines()[-5:]:
                print(f"  PH scraper: {line}", file=sys.stderr)

    if result.stderr:
        for line in result.stderr.strip().splitlines():
            if line.strip().startswith(("2", "T")):  # date lines + Total
                print(f"  {line.strip()}", file=sys.stderr)

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
