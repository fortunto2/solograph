"""ProductHunt browser scraper — Playwright with CF bypass.

PH GraphQL API redacts maker data (~93% of 2026 products return [REDACTED]).
PH is behind Cloudflare JS challenge. We use Playwright with stealth settings
to solve the challenge once, then reuse the browser session for all pages.

Two modes:
  A) Enrich — read slugs from existing JSONL, visit /posts/{slug}, extract non-redacted data
  B) Discover — crawl daily leaderboard pages to find new products

Usage:
    from codegraph_mcp.scrapers.producthunt_browser import run_ph_browser_scraper
    items = run_ph_browser_scraper(slugs=["bolt-new"], limit=5)
    items = run_ph_browser_scraper(from_date="2026-01-10", limit=50)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

BASE_URL = "https://www.producthunt.com"

# JS: extract product data from rendered PH page
EXTRACT_PRODUCT_JS = """
() => {
    const result = {};

    // JSON-LD
    const ldScripts = document.querySelectorAll('script[type="application/ld+json"]');
    for (const s of ldScripts) {
        try {
            const data = JSON.parse(s.textContent);
            const items = Array.isArray(data) ? data : [data];
            for (const item of items) {
                const t = item['@type'];
                const types = Array.isArray(t) ? t : [t];
                if (types.some(x => ['WebApplication', 'SoftwareApplication', 'Product'].includes(x))
                    && item.author) {
                    result.jsonLd = item;
                    break;
                }
            }
        } catch(e) {}
    }

    // Collect all /@username links with role detection
    const allUserLinks = document.querySelectorAll('a[href^="/@"]');
    const people = [];
    const seenUsernames = new Set();
    const skipPaths = new Set(['', 'topics', 'posts', 'search', 'newsletter', 'login', 'signup']);

    // JSON-LD authors = confirmed makers
    const makerUsernames = new Set();
    if (result.jsonLd) {
        let authors = result.jsonLd.author || [];
        if (!Array.isArray(authors)) authors = [authors];
        for (const a of authors) {
            if (a && a.url && a.url.includes('/@')) {
                const u = a.url.split('/@').pop().replace(/\\/$/, '');
                if (u) makerUsernames.add(u);
            }
        }
    }

    // Hunter detection: look for "Hunter" label near a /@username link
    const hunterUsernames = new Set();
    const bodyText = document.body.innerText || '';
    // PH shows "Hunted by @username" or has a Hunter badge section
    const hunterMatch = bodyText.match(/(?:Hunted by|Hunter)\\s+([A-Za-z0-9_]+)/i);
    if (hunterMatch) hunterUsernames.add(hunterMatch[1]);

    for (const a of allUserLinks) {
        const href = a.getAttribute('href');
        // Skip /reviews links like /@user/reviews
        if (href.includes('/reviews')) continue;
        const username = href.replace(/^\\//, '').replace(/^@/, '');
        if (!username || seenUsernames.has(username) || skipPaths.has(username)) continue;
        seenUsernames.add(username);

        let role = 'commenter';
        if (makerUsernames.has(username)) role = 'maker';
        else if (hunterUsernames.has(username)) role = 'hunter';

        people.push({
            username: username,
            name: a.textContent.trim() || username,
            url: `https://www.producthunt.com/@${username}`,
            role: role,
        });
    }
    // Sort: makers first, then hunters, then commenters
    people.sort((a, b) => {
        const order = {maker: 0, hunter: 1, commenter: 2};
        return (order[a.role] || 9) - (order[b.role] || 9);
    });
    result.makers = people;

    // Topics
    const topicLinks = document.querySelectorAll('a[href*="/topics/"]');
    const topics = [];
    const seenTopics = new Set();
    for (const a of topicLinks) {
        const slug = a.getAttribute('href').split('/').pop();
        if (slug && slug !== 'topics' && !seenTopics.has(slug)) {
            seenTopics.add(slug);
            topics.push(a.textContent.trim() || slug.replace(/-/g, ' '));
        }
    }
    result.topics = topics;

    // OG tags
    result.ogTitle = document.querySelector('meta[property="og:title"]')?.content || '';
    result.ogDesc = document.querySelector('meta[property="og:description"]')?.content || '';
    result.metaDesc = document.querySelector('meta[name="description"]')?.content || '';
    result.title = document.title || '';

    // votesCount, featuredAt, ranks, counts from inline scripts
    const scripts = document.querySelectorAll('script');
    for (const s of scripts) {
        const text = s.textContent || '';
        if (!result.votesCount) {
            const m = text.match(/"votesCount"\\s*:\\s*(\\d+)/);
            if (m) result.votesCount = parseInt(m[1]);
        }
        if (!result.featuredAt) {
            const m = text.match(/"featuredAt"\\s*:\\s*"([^"]+)"/);
            if (m) result.featuredAt = m[1];
        }
        if (!result.createdAt) {
            const m = text.match(/"createdAt"\\s*:\\s*"([^"]+)"/);
            if (m) result.createdAt = m[1];
        }
        if (result.isFeatured === undefined) {
            const m = text.match(/"isFeatured"\\s*:\\s*(true|false)/);
            if (m) result.isFeatured = m[1] === 'true';
        }
        if (!result.dailyRank) {
            const m = text.match(/"dailyRank"\\s*:\\s*(\\d+)/);
            if (m) result.dailyRank = parseInt(m[1]);
        }
        if (!result.weeklyRank) {
            const m = text.match(/"weeklyRank"\\s*:\\s*(\\d+)/);
            if (m) result.weeklyRank = parseInt(m[1]);
        }
        if (!result.commentsCount) {
            const m = text.match(/"commentsCount"\\s*:\\s*(\\d+)/);
            if (m) result.commentsCount = parseInt(m[1]);
        }
        if (!result.reviewsCount) {
            const m = text.match(/"reviewsCount"\\s*:\\s*(\\d+)/);
            if (m) result.reviewsCount = parseInt(m[1]);
        }
        if (!result.reviewsRating) {
            const m = text.match(/"reviewsRating"\\s*:\\s*(\\d+)/);
            if (m) result.reviewsRating = parseInt(m[1]);
        }
    }

    return result;
}
"""

# JS: extract product slugs from leaderboard page
EXTRACT_LEADERBOARD_JS = """
() => {
    const slugs = [];
    const seen = new Set();
    // PH leaderboard uses /products/{slug} links (not /posts/)
    const links = document.querySelectorAll('a[href*="/products/"]');
    for (const a of links) {
        const href = a.getAttribute('href');
        // Skip footer/review links like /products/lovable/reviews?ref=footer
        if (href.includes('/reviews') || href.includes('?ref=footer')) continue;
        const parts = href.split('/');
        const slug = parts[parts.indexOf('products') + 1];
        if (slug && !seen.has(slug)) {
            seen.add(slug);
            slugs.push(slug);
        }
    }
    return slugs;
}
"""


def _parse_product(slug: str, data: dict) -> dict | None:
    """Parse extracted JS data into a product dict."""
    json_ld = data.get("jsonLd") or {}
    makers = data.get("makers", [])
    topics = data.get("topics", [])

    # Enrich makers from JSON-LD authors (confirmed makers)
    if json_ld:
        seen = {m["username"] for m in makers}
        authors = json_ld.get("author", [])
        if isinstance(authors, dict):
            authors = [authors]
        for author in authors:
            if not isinstance(author, dict):
                continue
            url = author.get("url", "")
            name = author.get("name", "")
            if "producthunt.com/@" in url:
                username = url.split("/@")[-1].rstrip("/")
                if username and username not in seen:
                    seen.add(username)
                    makers.append({"username": username, "name": name or username, "url": url, "role": "maker"})
                elif username in seen:
                    for m in makers:
                        if m["username"] == username:
                            m["role"] = "maker"
                            if name and m["name"] == username:
                                m["name"] = name
                            break

    # Name
    name = json_ld.get("name", "") if json_ld else ""
    if not name:
        og = data.get("ogTitle", "")
        name = og.split(" | Product Hunt")[0].strip() if " | Product Hunt" in og else og
    if not name:
        t = data.get("title", "")
        name = t.split(" | Product Hunt")[0].strip() if " | Product Hunt" in t else t
        if " - " in name:
            name = name.split(" - ", 1)[0].strip()

    if not name or name.lower() == "just a moment...":
        return None

    description = json_ld.get("description", "") if json_ld else data.get("ogDesc", "")
    tagline = data.get("metaDesc", "")
    launch_date = json_ld.get("datePublished", "")[:10] if json_ld else ""
    featured_at = data.get("featuredAt", "")
    if not launch_date and featured_at:
        launch_date = featured_at[:10]

    website = ""
    if json_ld and json_ld.get("offers", {}).get("url"):
        website = json_ld["offers"]["url"]

    upvotes = data.get("votesCount", 0)
    if not upvotes and json_ld:
        for stat in json_ld.get("interactionStatistic", []):
            if isinstance(stat, dict) and "UserLikes" in str(stat.get("interactionType", "")):
                try:
                    upvotes = int(stat.get("userInteractionCount", 0))
                except (ValueError, TypeError):
                    pass

    # Determine product status: featured > launched > created
    is_featured = data.get("isFeatured", False)
    created_at = data.get("createdAt", "")
    product_status = "featured" if (is_featured or featured_at) else "created"

    return {
        "slug": slug,
        "name": name,
        "tagline": tagline[:500] if tagline else "",
        "description": description[:1000] if description else "",
        "topics": ", ".join(topics),
        "upvotes": upvotes,
        "url": f"{BASE_URL}/posts/{slug}",
        "launch_date": launch_date,
        "featured_at": featured_at,
        "created_at": created_at,
        "product_status": product_status,
        "is_featured": is_featured,
        "daily_rank": data.get("dailyRank", 0),
        "weekly_rank": data.get("weeklyRank", 0),
        "comments_count": data.get("commentsCount", 0),
        "reviews_count": data.get("reviewsCount", 0),
        "reviews_rating": data.get("reviewsRating", 0),
        "makers": makers,
        "website": website,
        "_source": "browser",
    }


async def _wait_for_cf(page, timeout: int = 30) -> bool:
    """Wait for Cloudflare challenge to resolve. Returns True if resolved."""
    for _ in range(timeout):
        title = await page.title()
        if "just a moment" not in title.lower():
            return True
        await page.wait_for_timeout(1000)
    return False


EXTRACT_MAKERS_PAGE_JS = r"""
() => {
    const makers = [];
    const seen = new Set();
    for (const a of document.querySelectorAll('a[href^="/@"]')) {
        const href = a.getAttribute('href');
        if (href.includes('/reviews')) continue;
        const username = href.replace(/^\//, '').replace(/^@/, '');
        if (!username || seen.has(username)) continue;
        seen.add(username);
        makers.push({
            username: username,
            name: a.textContent.trim() || username,
            url: `https://www.producthunt.com/@${username}`,
            role: 'maker',
        });
    }
    return makers;
}
"""


async def _scrape_product(page, slug: str) -> dict | None:
    """Navigate to /posts/{slug} for product data, then /products/{slug}/makers for real makers."""
    url = f"{BASE_URL}/posts/{slug}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Wait for CF if needed (usually instant after first solve)
        if not await _wait_for_cf(page, timeout=10):
            print(f"  [{slug}] Stuck on CF challenge", file=sys.stderr)
            return None

        # Wait for content to render
        try:
            await page.wait_for_selector(
                'script[type="application/ld+json"], meta[property="og:title"]',
                timeout=10000,
            )
        except Exception:
            await page.wait_for_timeout(3000)

        data = await page.evaluate(EXTRACT_PRODUCT_JS)
        if not data:
            return None

        result = _parse_product(slug, data)
        if not result:
            return None

        # Step 2: Visit /products/{slug}/makers to identify real makers
        # Cross-reference: anyone on makers page → role="maker", others keep role
        try:
            makers_url = f"{BASE_URL}/products/{slug}/makers"
            await page.goto(makers_url, wait_until="domcontentloaded", timeout=15000)
            if await _wait_for_cf(page, timeout=5):
                await page.wait_for_timeout(2000)
                real_makers = await page.evaluate(EXTRACT_MAKERS_PAGE_JS)
                if real_makers:
                    real_maker_usernames = {m["username"] for m in real_makers}
                    existing_usernames = {m["username"] for m in result["makers"]}
                    # Update roles: if on makers page → maker, else keep original
                    for m in result["makers"]:
                        if m["username"] in real_maker_usernames:
                            m["role"] = "maker"
                        elif m["role"] == "maker":
                            # Was tagged maker from JSON-LD but NOT on makers page
                            m["role"] = "commenter"
                    # Add any makers from the page not already in the list
                    for m in real_makers:
                        if m["username"] not in existing_usernames:
                            result["makers"].append(m)
                    # Re-sort: makers first, hunters, commenters
                    order = {"maker": 0, "hunter": 1, "commenter": 2}
                    result["makers"].sort(key=lambda x: order.get(x.get("role", "commenter"), 9))
        except Exception:
            pass  # Keep makers from product page as fallback

        return result
    except Exception as e:
        print(f"  [{slug}] Error: {e}", file=sys.stderr)
        return None


async def _scrape_leaderboard(page, date_str: str) -> list[str]:
    """Navigate to daily leaderboard and extract product slugs."""
    from datetime import date

    d = date.fromisoformat(date_str)
    url = f"{BASE_URL}/leaderboard/daily/{d.year}/{d.month}/{d.day}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        if not await _wait_for_cf(page, timeout=10):
            return []

        try:
            await page.wait_for_selector('a[href*="/posts/"]', timeout=10000)
        except Exception:
            await page.wait_for_timeout(3000)

        return await page.evaluate(EXTRACT_LEADERBOARD_JS) or []
    except Exception as e:
        print(f"  [leaderboard {date_str}] Error: {e}", file=sys.stderr)
        return []


async def _run_browser_scraper(
    slugs: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    skip_slugs: list[str] | None = None,
    limit: int | None = None,
    delay: float = 2.0,
    resume_path: str | None = None,
    headless: bool | None = None,
    concurrency: int = 1,
) -> list[dict]:
    """Async Playwright scraper for PH product pages.

    Uses headed mode on macOS to bypass Cloudflare JS challenge.
    concurrency > 1 opens multiple tabs in the same browser session.
    """
    from playwright.async_api import async_playwright

    skip_set = set(skip_slugs) if skip_slugs else set()
    items: list[dict] = []

    # Resume support
    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            for line in rp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                    s = item.get("slug", "")
                    if s:
                        skip_set.add(s)
                except json.JSONDecodeError:
                    pass
            if items:
                print(f"  Resumed: {len(items)} products from {resume_path}", file=sys.stderr)

    # Determine headed/headless mode
    use_headless = headless if headless is not None else (sys.platform != "darwin")
    mode_str = "headless" if use_headless else "headed"
    conc_str = f", {concurrency} tabs" if concurrency > 1 else ""
    print(f"  Launching Playwright ({mode_str}{conc_str}) for CF bypass...", file=sys.stderr)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=use_headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )
        # Anti-detection
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        page = await context.new_page()

        # Step 1: Solve CF challenge on homepage
        print("  Solving Cloudflare challenge...", file=sys.stderr)
        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
        cf_ok = await _wait_for_cf(page, timeout=30)
        if not cf_ok:
            print("  ERROR: CF challenge not solved. Try non-headless mode.", file=sys.stderr)
            await browser.close()
            return items
        print("  CF challenge passed!", file=sys.stderr)

        # Step 2: Build target slug list
        target_slugs: list[str] = []

        if slugs:
            target_slugs = [s for s in slugs if s not in skip_set]
        else:
            # Discover mode
            from datetime import date, timedelta

            start = date.fromisoformat(from_date) if from_date else date(2026, 1, 1)
            end = date.fromisoformat(to_date) if to_date else date.today()

            current = start
            while current <= end:
                lb_slugs = await _scrape_leaderboard(page, current.isoformat())
                new_slugs = [s for s in lb_slugs if s not in skip_set]
                target_slugs.extend(new_slugs)
                print(
                    f"  Leaderboard {current}: {len(lb_slugs)} products, {len(new_slugs)} new",
                    file=sys.stderr,
                )
                current += timedelta(days=1)
                await asyncio.sleep(delay)

        if limit:
            target_slugs = target_slugs[:limit]

        total = len(target_slugs)
        print(f"  Scraping {total} product pages...", file=sys.stderr)

        # Step 3: Scrape product pages
        failed = 0
        done = 0

        if concurrency <= 1:
            # Sequential mode (original)
            for i, slug in enumerate(target_slugs):
                item = await _scrape_product(page, slug)
                if item:
                    items.append(item)
                    all_people = item.get("makers", [])
                    n_makers = sum(1 for m in all_people if m.get("role") == "maker")
                    n_others = len(all_people) - n_makers
                    makers_str = ", ".join(f"@{m.get('username', '?')}" for m in all_people if m.get("role") == "maker")
                    role_str = f"{n_makers}m"
                    if n_others:
                        role_str += f"+{n_others}c"
                    print(
                        f"  [{i + 1}/{total}] {item['name']}: "
                        f"{item.get('upvotes', 0)}↑, [{role_str}] {makers_str or 'none'}",
                        file=sys.stderr,
                    )
                else:
                    failed += 1
                    print(f"  [{i + 1}/{total}] {slug}: FAILED", file=sys.stderr)

                if resume_path and item:
                    _save_items_jsonl(items, resume_path)

                if i < total - 1:
                    await asyncio.sleep(delay)
        else:
            # Parallel mode: N tabs in same browser context
            pages = [page]  # reuse the CF-solved page
            for _ in range(concurrency - 1):
                pages.append(await context.new_page())

            sem = asyncio.Semaphore(concurrency)

            async def _scrape_one(slug: str, idx: int) -> dict | None:
                async with sem:
                    # Pick a free page by semaphore order
                    pg = pages[idx % concurrency]
                    return await _scrape_product(pg, slug)

            # Process in batches to enable resume saves
            batch_size = concurrency
            for batch_start in range(0, total, batch_size):
                batch = target_slugs[batch_start : batch_start + batch_size]
                tasks = [_scrape_one(slug, batch_start + j) for j, slug in enumerate(batch)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for j, result in enumerate(results):
                    slug = batch[j]
                    done += 1
                    if isinstance(result, Exception):
                        failed += 1
                        print(f"  [{done}/{total}] {slug}: ERROR {result}", file=sys.stderr)
                    elif result:
                        items.append(result)
                        makers_str = ", ".join(f"@{m.get('username', '?')}" for m in result.get("makers", [])[:3])
                        print(
                            f"  [{done}/{total}] {result['name']}: "
                            f"{result.get('upvotes', 0)}↑, makers: {makers_str or 'none'}",
                            file=sys.stderr,
                        )
                    else:
                        failed += 1
                        print(f"  [{done}/{total}] {slug}: FAILED", file=sys.stderr)

                if resume_path:
                    _save_items_jsonl(items, resume_path)

                if batch_start + batch_size < total:
                    await asyncio.sleep(delay)

            # Close extra pages
            for pg in pages[1:]:
                await pg.close()

        await browser.close()

    if resume_path:
        _save_items_jsonl(items, resume_path)

    print(f"  Done: {len(items)} products, {failed} failed", file=sys.stderr)
    return items


def run_ph_browser_scraper(
    slugs: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    skip_slugs: list[str] | None = None,
    limit: int | None = None,
    delay: float = 2.0,
    resume_path: str | None = None,
    headless: bool | None = None,
    concurrency: int = 1,
) -> list[dict]:
    """Scrape ProductHunt product pages via Playwright.

    Uses headed browser on macOS to bypass Cloudflare.

    Args:
        slugs: Product slugs to visit (enrich mode). If None, discover mode.
        from_date: Start date for discover mode (YYYY-MM-DD)
        to_date: End date for discover mode (YYYY-MM-DD, default: today)
        skip_slugs: Slugs to skip (already enriched)
        limit: Max products to scrape
        delay: Seconds between requests (default: 2)
        resume_path: JSONL file for incremental saves / resume
        headless: Force headless (True) or headed (False). Default: auto.
        concurrency: Number of parallel browser tabs (default: 1).

    Returns:
        List of product dicts with makers, topics, upvotes, etc.
    """
    return asyncio.run(
        _run_browser_scraper(
            slugs=slugs,
            from_date=from_date,
            to_date=to_date,
            skip_slugs=skip_slugs,
            limit=limit,
            delay=delay,
            resume_path=resume_path,
            headless=headless,
            concurrency=concurrency,
        )
    )


def _save_items_jsonl(items: list[dict], path: str):
    """Save items to JSONL file (atomic write)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    Path(tmp).rename(path)


# ---------------------------------------------------------------------------
# Maker profile scraper
# ---------------------------------------------------------------------------

EXTRACT_PROFILE_JS = r"""
() => {
    const result = {};

    // Social links
    const socials = {};
    for (const a of document.querySelectorAll("a")) {
        const href = a.getAttribute("href") || "";
        if (href.includes("linkedin.com/in/")) socials.linkedin = href;
        else if ((href.includes("twitter.com/") || href.includes("x.com/"))
                 && !href.includes("producthunt") && !href.includes("ProductHunt")) socials.twitter = href;
        else if (href.includes("github.com/") && !href.includes("producthunt") && !href.includes("ProductHunt")) socials.github = href;
        else if (href.includes("instagram.com/")) socials.instagram = href;
        else if (href.includes("youtube.com/")) socials.youtube = href;
    }
    result.socials = socials;

    // Inline script data
    const scripts = document.querySelectorAll("script");
    for (const s of scripts) {
        const t = s.textContent || "";
        if (t.includes("followersCount") && !result.followersCount) {
            const m = t.match(/"followersCount"\s*:\s*(\d+)/);
            if (m) result.followersCount = parseInt(m[1]);
        }
        if (t.includes("isMaker") && result.isMaker === undefined) {
            const m = t.match(/"isMaker"\s*:\s*(true|false)/);
            if (m) result.isMaker = m[1] === "true";
        }
        if (t.includes("headline") && !result.headline) {
            const m = t.match(/"headline"\s*:\s*"([^"]+)"/);
            if (m) result.headline = m[1];
        }
        if (t.includes("websiteUrl") && !result.websiteUrl) {
            const m = t.match(/"websiteUrl"\s*:\s*"([^"]+)"/);
            if (m) result.websiteUrl = m[1];
        }
        if (t.includes("twitterUsername") && !result.twitterUsername) {
            const m = t.match(/"twitterUsername"\s*:\s*"([^"]+)"/);
            if (m) result.twitterUsername = m[1];
        }
    }

    // Streak + points from body text
    const body = document.body.innerText || "";
    const streakM = body.match(/(\d+)\s*day streak/);
    if (streakM) result.streak = parseInt(streakM[1]);
    const pointsM = body.match(/([\d,]+)\s*points/);
    if (pointsM) result.points = parseInt(pointsM[1].replace(/,/g, ""));

    result.ogTitle = document.querySelector('meta[property="og:title"]')?.content || "";
    result.ogDesc = document.querySelector('meta[property="og:description"]')?.content || "";

    return result;
}
"""


async def _scrape_profile(page, username: str) -> dict | None:
    """Navigate to /@username and extract profile data."""
    url = f"{BASE_URL}/@{username}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        if not await _wait_for_cf(page, timeout=10):
            return None
        try:
            await page.wait_for_selector('meta[property="og:title"]', timeout=10000)
        except Exception:
            await page.wait_for_timeout(3000)

        data = await page.evaluate(EXTRACT_PROFILE_JS)
        if not data:
            return None

        return {
            "username": username,
            "profile_url": url,
            "headline": data.get("headline", ""),
            "linkedin": data.get("socials", {}).get("linkedin", ""),
            "twitter": (f"https://x.com/{data['twitterUsername']}" if data.get("twitterUsername") else "")
            or data.get("socials", {}).get("twitter", ""),
            "github": data.get("socials", {}).get("github", ""),
            "website": data.get("websiteUrl", ""),
            "is_maker": data.get("isMaker"),
            "streak": data.get("streak", 0),
            "points": data.get("points", 0),
            "followers": data.get("followersCount", 0),
        }
    except Exception as e:
        print(f"  [@{username}] Error: {e}", file=sys.stderr)
        return None


async def _run_profile_scraper(
    usernames: list[str],
    skip_usernames: list[str] | None = None,
    limit: int | None = None,
    delay: float = 1.0,
    resume_path: str | None = None,
    headless: bool | None = None,
    concurrency: int = 1,
) -> list[dict]:
    """Scrape PH maker profile pages."""
    from playwright.async_api import async_playwright

    skip_set = set(skip_usernames) if skip_usernames else set()
    items: list[dict] = []

    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            for line in rp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                    u = item.get("username", "")
                    if u:
                        skip_set.add(u)
                except json.JSONDecodeError:
                    pass
            if items:
                print(f"  Resumed: {len(items)} profiles from {resume_path}", file=sys.stderr)

    targets = [u for u in usernames if u not in skip_set]
    if limit:
        targets = targets[:limit]

    use_headless = headless if headless is not None else (sys.platform != "darwin")
    mode_str = "headless" if use_headless else "headed"
    conc_str = f", {concurrency} tabs" if concurrency > 1 else ""
    print(f"  Launching Playwright ({mode_str}{conc_str}) for profiles...", file=sys.stderr)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=use_headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        page = await context.new_page()

        print("  Solving Cloudflare challenge...", file=sys.stderr)
        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
        if not await _wait_for_cf(page, timeout=30):
            print("  ERROR: CF challenge not solved.", file=sys.stderr)
            await browser.close()
            return items
        print("  CF challenge passed!", file=sys.stderr)

        total = len(targets)
        print(f"  Scraping {total} maker profiles...", file=sys.stderr)
        failed = 0
        done = 0

        if concurrency <= 1:
            for i, username in enumerate(targets):
                prof = await _scrape_profile(page, username)
                if prof:
                    items.append(prof)
                    li = "LI" if prof.get("linkedin") else ""
                    print(
                        f"  [{i + 1}/{total}] @{username}: {prof.get('streak', 0)}🔥 "
                        f"{prof.get('points', 0)}pts {li} {'maker' if prof.get('is_maker') else ''}",
                        file=sys.stderr,
                    )
                else:
                    failed += 1
                    print(f"  [{i + 1}/{total}] @{username}: FAILED", file=sys.stderr)
                if resume_path and prof:
                    _save_items_jsonl(items, resume_path)
                if i < total - 1:
                    await asyncio.sleep(delay)
        else:
            pages = [page]
            for _ in range(concurrency - 1):
                pages.append(await context.new_page())
            sem = asyncio.Semaphore(concurrency)

            async def _scrape_one(uname: str, idx: int) -> dict | None:
                async with sem:
                    pg = pages[idx % concurrency]
                    return await _scrape_profile(pg, uname)

            batch_size = concurrency
            for batch_start in range(0, total, batch_size):
                batch = targets[batch_start : batch_start + batch_size]
                tasks = [_scrape_one(u, batch_start + j) for j, u in enumerate(batch)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for j, result in enumerate(results):
                    uname = batch[j]
                    done += 1
                    if isinstance(result, Exception):
                        failed += 1
                        print(f"  [{done}/{total}] @{uname}: ERROR {result}", file=sys.stderr)
                    elif result:
                        items.append(result)
                        li = "LI" if result.get("linkedin") else ""
                        print(
                            f"  [{done}/{total}] @{uname}: {result.get('streak', 0)}🔥 "
                            f"{result.get('points', 0)}pts {li} {'maker' if result.get('is_maker') else ''}",
                            file=sys.stderr,
                        )
                    else:
                        failed += 1
                        print(f"  [{done}/{total}] @{uname}: FAILED", file=sys.stderr)
                if resume_path:
                    _save_items_jsonl(items, resume_path)
                if batch_start + batch_size < total:
                    await asyncio.sleep(delay)
            for pg in pages[1:]:
                await pg.close()

        await browser.close()

    if resume_path:
        _save_items_jsonl(items, resume_path)

    print(f"  Done: {len(items)} profiles, {failed} failed", file=sys.stderr)
    return items


def run_ph_profile_scraper(
    usernames: list[str],
    skip_usernames: list[str] | None = None,
    limit: int | None = None,
    delay: float = 1.0,
    resume_path: str | None = None,
    headless: bool | None = None,
    concurrency: int = 1,
) -> list[dict]:
    """Scrape PH maker profiles for LinkedIn, streak, etc."""
    return asyncio.run(
        _run_profile_scraper(
            usernames=usernames,
            skip_usernames=skip_usernames,
            limit=limit,
            delay=delay,
            resume_path=resume_path,
            headless=headless,
            concurrency=concurrency,
        )
    )
