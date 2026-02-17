"""ProductHunt maker profile scraper — Playwright based.

Profiles (/@username) and streaks (/visit-streaks) are behind Cloudflare,
so we use Playwright (headless Chromium) to render pages and extract data
from the Apollo Client cache via JS evaluate.

Usage:
    from codegraph_mcp.scrapers.producthunt_makers import run_maker_scraper
    profiles = run_maker_scraper(usernames=["levelsio", "marckohlbrugge"], limit=10)

    # Extract usernames from existing JSONL dump:
    from codegraph_mcp.scrapers.producthunt_makers import extract_makers_from_posts
    usernames = extract_makers_from_posts("~/.solo/sources/producthunt_scrape.jsonl")
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from ..models import MakerProfile

# JS snippet to extract user data from Apollo Client cache.
# PH uses Next.js + Apollo, so all structured data is in window.__APOLLO_CLIENT__.
APOLLO_EXTRACT_JS = """
() => {
    try {
        const cache = window.__APOLLO_CLIENT__.cache.extract();
        const userKey = Object.keys(cache).find(k => /^User:\\d+$/.test(k));
        if (!userKey) return null;
        const user = cache[userKey];

        // Decode social links from UserLink objects
        const links = {};
        for (const [k, v] of Object.entries(cache)) {
            if (k.startsWith('UserLink:') && v && v.url) {
                try {
                    const decoded = atob(v.url);
                    const type = (v.type || '').toLowerCase();
                    if (type) links[type] = decoded;
                } catch(e) {
                    // url might not be base64
                    const type = (v.type || '').toLowerCase();
                    if (type) links[type] = v.url;
                }
            }
        }

        // Extract karma/points from karmaBadge
        let points = 0;
        for (const [k, v] of Object.entries(cache)) {
            if (k.startsWith('KarmaBadge:') && v && v.score != null) {
                points = v.score;
                break;
            }
        }

        // Extract streak from visitStreak
        let streak = 0;
        for (const [k, v] of Object.entries(cache)) {
            if (k.startsWith('VisitStreak:') && v && v.count != null) {
                streak = v.count;
                break;
            }
        }

        return {
            id: user.id || '',
            username: user.username || '',
            name: user.name || '',
            headline: user.headline || '',
            bio: user.about || '',
            twitterUsername: user.twitterUsername || '',
            websiteUrl: user.websiteUrl || '',
            profileImage: user.profileImage || '',
            followersCount: user.followersCount || 0,
            followingCount: user.followingCount || 0,
            madePosts: user.madePosts ? user.madePosts.totalCount || 0 : 0,
            submittedPosts: user.submittedPosts ? user.submittedPosts.totalCount || 0 : 0,
            isMaker: user.isMaker || false,
            createdAt: user.createdAt || '',
            links: links,
            points: points,
            streak: streak,
        };
    } catch(e) {
        return null;
    }
}
"""


def extract_makers_from_posts(jsonl_path: str) -> list[str]:
    """Extract unique maker usernames from a scraped posts JSONL file.

    Supports both formats:
      - New: makers is list[dict] with "username" key
      - Legacy: makers is a comma-separated string of names (skipped)
    """
    path = Path(jsonl_path).expanduser()
    if not path.exists():
        return []

    usernames: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        makers = item.get("makers", [])
        if isinstance(makers, list):
            for m in makers:
                if isinstance(m, dict):
                    username = m.get("username", "").strip()
                    if username:
                        usernames.add(username)

    return sorted(usernames)


def extract_makers_from_graph(source_name: str = "producthunt") -> list[str]:
    """Extract unique maker usernames from already-indexed SourceDoc nodes."""
    try:
        from ..vectors.source_index import SourceIndex

        idx = SourceIndex()
        graph = idx._get_graph(source_name)
        graph.query("MATCH (d:SourceDoc) WHERE d.source_type = 'producthunt-product' RETURN d.url")
        # Parse slugs — makers are stored in the embed text, not easy to extract
        # Better to use JSONL directly
        return []
    except Exception:
        return []


async def _scrape_profile(page, username: str) -> MakerProfile | None:
    """Navigate to /@username and extract profile via Apollo cache."""
    url = f"https://www.producthunt.com/@{username}"
    try:
        resp = await page.goto(url, wait_until="networkidle", timeout=30000)
        if not resp or resp.status >= 400:
            print(f"  [{username}] HTTP {resp.status if resp else '?'}", file=sys.stderr)
            return None

        # Wait for Apollo to hydrate
        await page.wait_for_timeout(2000)

        # Extract from Apollo cache
        data = await page.evaluate(APOLLO_EXTRACT_JS)
        if not data:
            # Fallback: try waiting longer
            await page.wait_for_timeout(3000)
            data = await page.evaluate(APOLLO_EXTRACT_JS)

        if not data:
            print(f"  [{username}] No Apollo data found", file=sys.stderr)
            return None

        # Build MakerProfile
        links = data.get("links", {})
        linkedin = links.get("linkedin", "")

        return MakerProfile(
            username=data.get("username", username),
            name=data.get("name", ""),
            headline=data.get("headline", ""),
            bio=data.get("bio", ""),
            twitter_username=data.get("twitterUsername", ""),
            linkedin_url=linkedin,
            website_url=data.get("websiteUrl", "") or links.get("website", ""),
            points=data.get("points", 0),
            streak_days=data.get("streak", 0),
            followers_count=data.get("followersCount", 0),
            following_count=data.get("followingCount", 0),
            products_count=data.get("madePosts", 0),
            hunted_count=data.get("submittedPosts", 0),
            is_maker=data.get("isMaker", False),
            avatar_url=data.get("profileImage", ""),
            created_at=data.get("createdAt", ""),
            ph_user_id=str(data.get("id", "")),
        )
    except Exception as e:
        print(f"  [{username}] Error: {e}", file=sys.stderr)
        return None


async def _run_scraper_async(
    usernames: list[str],
    limit: int | None = None,
    resume_path: str | None = None,
    delay: float = 3.0,
) -> list[MakerProfile]:
    """Async Playwright scraper for maker profiles."""
    from playwright.async_api import async_playwright

    # Resume support: load already-scraped usernames
    done_usernames: set[str] = set()
    profiles: list[MakerProfile] = []

    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            for line in rp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    done_usernames.add(data.get("username", ""))
                    profiles.append(MakerProfile(**data))
                except (json.JSONDecodeError, Exception):
                    pass
            if profiles:
                print(f"  Resumed: {len(profiles)} profiles from {resume_path}", file=sys.stderr)

    # Filter out already-scraped
    remaining = [u for u in usernames if u not in done_usernames]
    if limit:
        remaining = remaining[:limit]

    if not remaining:
        print("  No new usernames to scrape", file=sys.stderr)
        return profiles

    print(f"  Scraping {len(remaining)} maker profiles...", file=sys.stderr)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )
        page = await context.new_page()

        for i, username in enumerate(remaining):
            profile = await _scrape_profile(page, username)
            if profile:
                profiles.append(profile)
                print(
                    f"  [{i + 1}/{len(remaining)}] {username}: "
                    f"{profile.points}pts, {profile.streak_days}d streak, "
                    f"{profile.followers_count} followers",
                    file=sys.stderr,
                )
            else:
                print(f"  [{i + 1}/{len(remaining)}] {username}: FAILED", file=sys.stderr)

            # Save after each profile (resume support)
            if resume_path and profile:
                _save_profiles_jsonl(profiles, resume_path)

            # Rate limiting
            if i < len(remaining) - 1:
                await asyncio.sleep(delay)

        await browser.close()

    # Final save
    if resume_path:
        _save_profiles_jsonl(profiles, resume_path)

    return profiles


def run_maker_scraper(
    usernames: list[str],
    limit: int | None = None,
    resume_path: str | None = None,
    delay: float = 3.0,
) -> list[MakerProfile]:
    """Scrape ProductHunt maker profiles via Playwright.

    Args:
        usernames: List of PH usernames to scrape
        limit: Max profiles to scrape (None = all)
        resume_path: Path to JSONL file for incremental saves / resume
        delay: Seconds between requests (default: 3)

    Returns:
        List of MakerProfile objects
    """
    return asyncio.run(
        _run_scraper_async(
            usernames=usernames,
            limit=limit,
            resume_path=resume_path,
            delay=delay,
        )
    )


def _save_profiles_jsonl(profiles: list[MakerProfile], path: str):
    """Save profiles to JSONL file (atomic write)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for p in profiles:
            f.write(p.model_dump_json() + "\n")
    Path(tmp).rename(path)
