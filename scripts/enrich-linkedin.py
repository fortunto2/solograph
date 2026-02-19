#!/usr/bin/env python3
"""Enrich PH profiles with LinkedIn URLs via SearXNG web search.

Reads profiles without LinkedIn, searches for their LinkedIn page,
and appends found URLs to an output JSONL file.

Usage:
    uv run python scripts/enrich-linkedin.py [--limit N] [--delay 2.0]
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

# --- Config ---
SEARXNG_URL = "http://localhost:8013"
SOLO_CONFIG = Path.home() / ".solo/config.yaml"


def get_data_root():
    if SOLO_CONFIG.exists():
        for line in SOLO_CONFIG.read_text().splitlines():
            if line.startswith("data_root:"):
                path = line.split(":", 1)[1].strip().replace("~", str(Path.home()))
                return Path(path)
    return Path.home() / "data"


def search_linkedin(query: str, max_results: int = 5) -> list[dict]:
    """Search via SearXNG Tavily-compatible API (POST /search)."""
    payload = json.dumps(
        {
            "query": query,
            "max_results": max_results,
            "include_raw_content": False,
        }
    ).encode()
    try:
        req = urllib.request.Request(
            f"{SEARXNG_URL}/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("results", [])
    except Exception as e:
        print(f"    Search error: {e}", file=sys.stderr)
        return []


def extract_name_parts(username: str) -> list[str]:
    """Extract likely name parts from a PH username.

    Examples: niklas_fischer → [niklas, fischer]
              matthiasrossini → [matthias, rossini] (if camelCase-ish)
              adrianmontoya → [adrian, montoya]
    """
    # Remove trailing /reviews etc
    username = username.split("/")[0]
    # Split on _ - .
    parts = re.split(r"[_\-.]", username)
    # Filter out short/numeric parts
    parts = [p.lower() for p in parts if len(p) >= 3 and p.isalpha()]
    if len(parts) >= 2:
        return parts
    # Try CamelCase split: "matthiasRossini" or just long lowercase
    # For long single words, not much we can do
    if len(parts) == 1 and len(parts[0]) >= 6:
        # Try camelCase
        camel = re.findall(r"[A-Z]?[a-z]+", username.split("/")[0])
        camel = [c.lower() for c in camel if len(c) >= 3]
        if len(camel) >= 2:
            return camel
    return parts


def extract_linkedin_url(results: list[dict], username: str = "", headline: str = "") -> str | None:
    """Extract LinkedIn profile URL from search results with verification."""
    # Normalize username for matching
    norm_user = username.lower().replace("_", "").replace("-", "").replace(".", "").replace("/", "")
    name_parts = extract_name_parts(username)

    for r in results:
        url = r.get("url", "")
        title = r.get("title", "").lower()
        # Match linkedin.com/in/username patterns
        if not re.match(r"https?://([\w]+\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?$", url):
            continue
        li_slug = url.rstrip("/").split("/")[-1].lower().replace("-", "").replace("_", "")

        # High confidence: username matches LinkedIn slug
        if norm_user and len(norm_user) >= 4 and (norm_user in li_slug or li_slug in norm_user):
            return url

        # Medium confidence: name parts from username appear in LinkedIn title
        # e.g. username=niklas_fischer, title="niklas fischer - freelancer"
        if len(name_parts) >= 2:
            matches = sum(1 for part in name_parts if part in title or part in li_slug)
            if matches >= 2:
                return url

    return None


def main():
    parser = argparse.ArgumentParser(description="Enrich PH profiles with LinkedIn via search")
    parser.add_argument("--limit", type=int, default=0, help="Max profiles to process (0=all)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between searches (seconds)")
    parser.add_argument("--min-points", type=int, default=5, help="Min points filter")
    parser.add_argument("--max-points", type=int, default=1000, help="Max points filter")
    args = parser.parse_args()

    data_root = get_data_root()
    ph = data_root / "ph"
    profiles_path = ph / "profiles/ph_2026_profiles.jsonl"
    output_path = ph / "profiles/ph_2026_linkedin_enriched.jsonl"

    # Load existing profiles
    profiles = {}
    for line in profiles_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            p = json.loads(line)
            profiles[p["username"]] = p
        except (json.JSONDecodeError, KeyError):
            pass

    # Load already enriched (for resume)
    already_done = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                already_done.add(item["username"])
            except (json.JSONDecodeError, KeyError):
                pass
        if already_done:
            print(f"Resumed: {len(already_done)} already enriched", file=sys.stderr)

    # Filter: points range, no LinkedIn, has headline
    targets = []
    for username, p in sorted(profiles.items()):
        if username in already_done:
            continue
        pts = p.get("points", 0)
        if not (args.min_points <= pts <= args.max_points):
            continue
        if p.get("linkedin"):
            continue  # already has LinkedIn
        headline = p.get("headline", "").strip()
        if not headline or len(headline) < 3:
            continue
        targets.append(p)

    if args.limit:
        targets = targets[: args.limit]

    print(f"Targets: {len(targets)} profiles without LinkedIn (pts {args.min_points}-{args.max_points})")
    print(f"Output: {output_path}")

    found = 0
    not_found = 0

    with open(output_path, "a") as f:
        for i, p in enumerate(targets):
            username = p["username"]
            headline = p.get("headline", "")

            # Build search query with site: for LinkedIn-only results
            name_parts = extract_name_parts(username)
            name_str = " ".join(name_parts) if len(name_parts) >= 2 else ""

            # Strategy: site:linkedin.com/in + name/headline
            if name_str:
                query = f"site:linkedin.com/in {name_str} {headline[:50]}"
            else:
                query = f"site:linkedin.com/in {headline[:60]}"

            results = search_linkedin(query)
            li_url = extract_linkedin_url(results, username=username, headline=headline)

            if li_url:
                found += 1
                tag = "✓ LI"
            else:
                not_found += 1
                tag = "  --"

            print(f"  [{i + 1}/{len(targets)}] @{username}: {tag}  ({headline[:40]})")

            # Save result (both found and not found for resume)
            record = {
                "username": username,
                "linkedin_found": li_url or "",
                "query": query,
                "headline": headline,
                "points": p.get("points", 0),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if i < len(targets) - 1:
                time.sleep(args.delay)

    print(f"\nDone: {found} found, {not_found} not found, {found + not_found} total")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
