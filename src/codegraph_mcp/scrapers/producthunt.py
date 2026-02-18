"""ProductHunt scraper — GraphQL API based.

Uses PH GraphQL API v2 (api.producthunt.com) with OAuth client credentials.
No Playwright/Chrome needed — pure HTTP requests.

Rate limits: 450 requests / 15 min. We use 1 req/2 sec (safe).
Each request fetches up to 20 posts.

Data sources:
  - posts query with postedAfter/Before date filtering
  - Cursor-based pagination within each date range

Usage:
    from codegraph_mcp.scrapers.producthunt import run_ph_scraper
    items = run_ph_scraper(days=30, limit=100)

    # Full dump (all posts since 2013):
    items = run_ph_scraper(days=4000, limit=None)

Env vars:
    PH_TOKEN — Developer token (takes priority, never expires)
    PH_CLIENT_ID / PH_CLIENT_SECRET — OAuth client credentials (fallback)
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = None

GRAPHQL_URL = "https://api.producthunt.com/v2/api/graphql"
OAUTH_TOKEN_URL = "https://api.producthunt.com/v2/oauth/token"

# Credentials — set via env vars PH_TOKEN, PH_CLIENT_ID, PH_CLIENT_SECRET
# PH_TOKEN (developer token) takes priority over OAuth client credentials.

POSTS_QUERY = """
query($first: Int!, $after: String, $postedAfter: DateTime, $postedBefore: DateTime, $featured: Boolean) {
  posts(first: $first, after: $after, postedAfter: $postedAfter, postedBefore: $postedBefore, featured: $featured, order: VOTES) {
    totalCount
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id slug name tagline description
        votesCount commentsCount reviewsCount reviewsRating
        website url
        createdAt launchedAt featuredAt
        dailyRank weeklyRank
        makers { username name url followersCount }
        productLinks { type url }
        topics(first: 5) { edges { node { name slug } } }
        thumbnail { url }
      }
    }
  }
}
"""

# Token cache
_token: str | None = None
_token_expires: float = 0


def _get_token(
    client_id: str | None = None,
    client_secret: str | None = None,
    developer_token: str | None = None,
) -> str:
    """Get access token. Developer token takes priority, then OAuth client credentials."""
    global _token, _token_expires

    # Developer token (never expires)
    dev_token = developer_token or os.environ.get("PH_TOKEN", "")
    if dev_token:
        return dev_token

    # Cached OAuth token
    if _token and time.time() < _token_expires:
        return _token

    cid = client_id or os.environ.get("PH_CLIENT_ID", "")
    csecret = client_secret or os.environ.get("PH_CLIENT_SECRET", "")
    if not cid or not csecret:
        raise RuntimeError(
            "PH credentials required. Set PH_TOKEN (developer token) or both PH_CLIENT_ID + PH_CLIENT_SECRET env vars."
        )

    body = json.dumps(
        {
            "client_id": cid,
            "client_secret": csecret,
            "grant_type": "client_credentials",
        }
    ).encode()

    req = Request(
        OAUTH_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=15, context=_SSL_CTX) as resp:
        data = json.loads(resp.read())

    _token = data["access_token"]
    # PH tokens don't have explicit expiry, refresh every 12 hours
    _token_expires = time.time() + 43200
    return _token


def _graphql(query: str, variables: dict, token: str) -> dict:
    """Execute a GraphQL query against PH API."""
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "SoloGraph/1.0",
        },
        method="POST",
    )
    with urlopen(req, timeout=30, context=_SSL_CTX) as resp:
        data = json.loads(resp.read())

    if "errors" in data:
        raise RuntimeError(f"PH API error: {data['errors']}")
    return data["data"]


def run_ph_scraper(
    days: int = 30,
    limit: int | None = None,
    skip_slugs: list[str] | None = None,
    timeout: int = 600,
    client_id: str | None = None,
    client_secret: str | None = None,
    developer_token: str | None = None,
    resume_path: str | None = None,
    featured_only: bool = True,
) -> list[dict]:
    """Scrape ProductHunt posts via GraphQL API.

    Args:
        days: Number of days back from today to scrape (default: 30)
        limit: Max total products to collect (None = all)
        skip_slugs: Slugs to skip (already indexed)
        timeout: Not used (kept for API compat with old Playwright scraper)
        client_id: PH OAuth client ID (or PH_CLIENT_ID env var)
        client_secret: PH OAuth client secret (or PH_CLIENT_SECRET env var)
        developer_token: PH developer token (or PH_TOKEN env var)
        resume_path: Path to JSONL file for incremental saves / resume
        featured_only: Only fetch featured posts (default: True)

    Returns:
        List of product dicts with keys:
        slug, name, tagline, description, topics, upvotes, comments,
        rating, website, launch_date, url, makers, thumbnail
    """
    token = _get_token(client_id, client_secret, developer_token)
    skip_set = set(skip_slugs or [])

    # Resume support: load existing items
    items: list[dict] = []
    if resume_path:
        resume_file = Path(resume_path)
        if resume_file.exists():
            for line in resume_file.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        items.append(item)
                        skip_set.add(item.get("slug", ""))
                    except json.JSONDecodeError:
                        pass
            if items:
                print(f"  Resumed: {len(items)} products from {resume_path}", file=sys.stderr)

    today = datetime.now(UTC)
    start_date = today - timedelta(days=days)

    # Chunk size: 1 day for all posts, 7 days for featured-only
    chunk_days = 1 if not featured_only else 7
    collected = len(items)
    request_count = 0

    current_start = start_date
    while current_start < today:
        if limit and collected >= limit:
            break

        current_end = min(current_start + timedelta(days=chunk_days), today)
        cursor = None
        chunk_count = 0

        while True:
            if limit and collected >= limit:
                break

            variables = {
                "first": 20,
                "postedAfter": current_start.strftime("%Y-%m-%dT00:00:00Z"),
                "postedBefore": current_end.strftime("%Y-%m-%dT23:59:59Z"),
                "featured": True if featured_only else None,
            }
            if cursor:
                variables["after"] = cursor

            try:
                data = _graphql(POSTS_QUERY, variables, token)
                request_count += 1
            except HTTPError as e:
                if e.code == 429:
                    print("  Rate limited, sleeping 60s...", file=sys.stderr)
                    time.sleep(60)
                    continue
                print(f"  HTTP {e.code} error, skipping chunk", file=sys.stderr)
                break
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                break

            posts_data = data.get("posts", {})
            edges = posts_data.get("edges", [])
            page_info = posts_data.get("pageInfo", {})

            for edge in edges:
                node = edge["node"]
                slug = node.get("slug", "")
                if not slug or slug in skip_set:
                    continue

                # Extract topics
                topic_names = []
                for t in node.get("topics", {}).get("edges", []):
                    topic_names.append(t["node"]["name"])

                # Extract makers (structured)
                makers_list = []
                for m in node.get("makers", []):
                    makers_list.append({
                        "username": m.get("username", ""),
                        "name": m.get("name", ""),
                        "url": m.get("url", ""),
                        "followers": m.get("followersCount", 0),
                    })

                # Extract product links (LinkedIn, Twitter, Website, etc.)
                links = {}
                for pl in node.get("productLinks", []):
                    link_type = pl.get("type", "").lower()
                    if link_type and pl.get("url"):
                        links[link_type] = pl["url"]

                # Parse dates
                created = node.get("createdAt", "")
                launched = node.get("launchedAt", "")
                featured_at = node.get("featuredAt", "")
                launch_date = (launched or created or "")[:10]
                created_at = created[:10] if created else ""

                # Determine product status: featured > launched > created
                daily_rank = node.get("dailyRank") or 0
                if featured_at:
                    product_status = "featured"
                elif daily_rank > 0:
                    product_status = "launched"
                else:
                    product_status = "created"

                item = {
                    "slug": slug,
                    "name": node.get("name", ""),
                    "tagline": node.get("tagline", ""),
                    "description": node.get("description", ""),
                    "topics": ", ".join(topic_names),
                    "upvotes": node.get("votesCount", 0),
                    "comments": node.get("commentsCount", 0),
                    "reviews_count": node.get("reviewsCount", 0),
                    "rating": node.get("reviewsRating", 0),
                    "daily_rank": daily_rank,
                    "weekly_rank": node.get("weeklyRank"),
                    "featured": bool(featured_at),
                    "featured_at": featured_at[:10] if featured_at else "",
                    "created_at": created_at,
                    "website": node.get("website", ""),
                    "url": node.get("url", f"https://www.producthunt.com/posts/{slug}"),
                    "launch_date": launch_date,
                    "product_status": product_status,
                    "makers": makers_list,
                    "product_links": links,
                    "thumbnail": (node.get("thumbnail") or {}).get("url", ""),
                }

                items.append(item)
                skip_set.add(slug)
                collected += 1
                chunk_count += 1

                if limit and collected >= limit:
                    break

            # Pagination
            if page_info.get("hasNextPage") and page_info.get("endCursor"):
                cursor = page_info["endCursor"]
            else:
                break

            # Rate limit: 1 request per 2 seconds
            time.sleep(2)

        period = f"{current_start.strftime('%Y-%m-%d')} → {current_end.strftime('%Y-%m-%d')}"
        print(f"  {period}: {chunk_count} products (total: {collected})", file=sys.stderr)

        # Save after each weekly chunk (reliable for long runs)
        if resume_path and chunk_count > 0:
            _save_jsonl(items, resume_path)

        current_start = current_end

    # Final save
    if resume_path:
        _save_jsonl(items, resume_path)

    total = posts_data.get("totalCount", "?") if "posts_data" in dir() else "?"
    print(f"  Total: {collected} products scraped ({request_count} API calls, PH has ~{total})", file=sys.stderr)

    return items


def _save_jsonl(items: list[dict], path: str):
    """Save items to JSONL file (atomic write)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    Path(tmp).rename(path)
