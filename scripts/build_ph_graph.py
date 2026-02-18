#!/usr/bin/env python3
"""Build full ProductHunt graph from merged JSONL.

Creates separate FalkorDB source with:
  - SourceDoc nodes (products) with embeddings, popularity, status, rank
  - Maker nodes (users) — lightweight, no embedding
  - MADE edges: Maker -> SourceDoc (role=maker)
  - COMMENTED edges: Maker -> SourceDoc (role=commenter)
  - PARTICIPATED edges: Maker -> SourceDoc (role=unknown)

Data lives in ~/data/ph/ (centralized, matches ph-pipeline.sh):
  raw/       — API scrapes (ph_2026_all.jsonl)
  enriched/  — browser-enriched (ph_2026_enriched.jsonl)
  final/     — merged ready for graph (ph_2026_merged.jsonl)

Usage:
    # Build from final merged JSONL (default: ~/data/ph/final/ph_2026_merged.jsonl)
    python scripts/build_ph_graph.py

    # Custom source name and file
    python scripts/build_ph_graph.py --source producthunt-2026 --file /path/to/data.jsonl

    # Specific year
    python scripts/build_ph_graph.py --year 2025

    # Dry run (no DB writes)
    python scripts/build_ph_graph.py --dry-run

    # Skip products, only rebuild edges (fast)
    python scripts/build_ph_graph.py --edges-only

    # Wipe and rebuild from scratch
    python scripts/build_ph_graph.py --clean
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for local dev
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codegraph_mcp.indexers.producthunt import ProductHuntIndexer
from codegraph_mcp.vectors.source_index import SourceIndex


def load_items(path: str) -> list[dict]:
    """Load items from JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_graph(
    items: list[dict],
    source: str,
    edges_only: bool = False,
    dry_run: bool = False,
    clean: bool = False,
):
    """Build full graph: products + users + edges."""

    if dry_run:
        print(f"DRY RUN: would index {len(items)} products")
        makers_set = set()
        for item in items:
            for m in item.get("makers", []):
                if m.get("username"):
                    makers_set.add(m["username"])
        print(f"  Unique users: {len(makers_set)}")
        return

    idx = SourceIndex()

    # Clean: remove old graph
    if clean:
        db_path = Path.home() / ".solo" / "sources" / source
        if db_path.exists():
            print(f"Removing old graph: {db_path}")
            shutil.rmtree(db_path)

    graph = idx._get_graph(source)

    # --- Phase 1: Index SourceDoc nodes ---
    if not edges_only:
        print(f"\n=== Phase 1: Indexing {len(items)} products ===")
        new_count = 0
        skip_count = 0

        for i, item in enumerate(items):
            doc = ProductHuntIndexer._to_source_doc(item)
            if not doc:
                skip_count += 1
                continue
            doc.source_name = source

            was_new = idx.upsert_one(source, doc)
            if was_new:
                new_count += 1

            # Set extra fields (product_status, daily_rank, weekly_rank)
            doc_id = hashlib.md5(f"ph:{item.get('slug', '')}".encode()).hexdigest()
            status = str(item.get("product_status", "")) or "launched"
            daily_rank = item.get("daily_rank") or 0
            weekly_rank = item.get("weekly_rank") or 0
            comments_count = item.get("comments_count") or item.get("comments") or 0

            graph.query(
                "MATCH (d:SourceDoc {doc_id: $did}) "
                "SET d.product_status = $status, "
                "d.daily_rank = $dr, d.weekly_rank = $wr, "
                "d.comments_count = $cc",
                {
                    "did": doc_id,
                    "status": status,
                    "dr": daily_rank,
                    "wr": weekly_rank,
                    "cc": comments_count,
                },
            )

            if (i + 1) % 1000 == 0:
                print(f"  Products: {i+1}/{len(items)} ({new_count} new)")

        print(
            f"Products done: {new_count} new, "
            f"{len(items) - new_count - skip_count} updated, "
            f"{skip_count} skipped"
        )

    # --- Phase 2: Create Maker nodes + edges ---
    print(f"\n=== Phase 2: Creating user nodes and edges ===")

    # Collect user data
    user_products: dict[str, list[tuple[str, str]]] = {}  # username -> [(slug, role)]
    user_info: dict[str, dict] = {}

    for item in items:
        slug = item.get("slug", "")
        if not slug:
            continue
        for m in item.get("makers", []):
            username = m.get("username", "")
            if not username:
                continue
            if username not in user_products:
                user_products[username] = []
                user_info[username] = {
                    "name": m.get("name", username),
                    "url": m.get("url", ""),
                }
            user_products[username].append((slug, m.get("role", "?")))

    print(f"Unique users: {len(user_products)}")

    created_count = 0
    edge_count = 0

    for i, (username, products) in enumerate(user_products.items()):
        info = user_info[username]
        roles = set(r for _, r in products)
        role_str = ",".join(sorted(roles))

        # Upsert Maker node (lightweight — no embedding)
        graph.query(
            "MERGE (m:Maker {username: $username}) "
            "SET m.name = $name, m.url = $url, m.roles = $roles, "
            "m.product_count = $pc",
            {
                "username": username,
                "name": info["name"],
                "url": info["url"],
                "roles": role_str,
                "pc": len(products),
            },
        )
        created_count += 1

        # Create edges to products
        for slug, role in products:
            doc_id = hashlib.md5(f"ph:{slug}".encode()).hexdigest()
            if role == "maker":
                edge_type = "MADE"
            elif role == "commenter":
                edge_type = "COMMENTED"
            else:
                edge_type = "PARTICIPATED"

            try:
                graph.query(
                    f"MATCH (m:Maker {{username: $u}}) "
                    f"MATCH (d:SourceDoc {{doc_id: $did}}) "
                    f"MERGE (m)-[:{edge_type}]->(d)",
                    {"u": username, "did": doc_id},
                )
                edge_count += 1
            except Exception:
                pass

        if (i + 1) % 1000 == 0:
            print(f"  Users: {i+1}/{len(user_products)} ({edge_count} edges)")

    # --- Stats ---
    print(f"\n{'=' * 50}")
    print("DONE:")
    print(f"  Products indexed: {len(items)}")
    print(f"  Users: {created_count} Maker nodes")
    print(f"  Edges: {edge_count} (MADE/COMMENTED/PARTICIPATED)")

    sd = graph.query("MATCH (d:SourceDoc) RETURN count(d)")
    mk = graph.query("MATCH (m:Maker) RETURN count(m)")
    print(f"\nGraph totals:")
    print(f"  SourceDoc: {sd.result_set[0][0]}")
    print(f"  Maker: {mk.result_set[0][0]}")

    for edge_type in ["MADE", "COMMENTED", "PARTICIPATED"]:
        r = graph.query(f"MATCH ()-[r:{edge_type}]->() RETURN count(r)")
        print(f"  {edge_type} edges: {r.result_set[0][0]}")

    # Status breakdown
    r = graph.query(
        "MATCH (d:SourceDoc) "
        "RETURN d.product_status, count(d) ORDER BY count(d) DESC"
    )
    print("\nProduct statuses:")
    for row in r.result_set:
        print(f"  {row[0] or '(empty)'}: {row[1]}")


def _default_data_root() -> Path:
    """Resolve data root: $SOLO_DATA_ROOT > ~/data."""
    import os
    return Path(os.environ.get("SOLO_DATA_ROOT", Path.home() / "data"))


def main():
    parser = argparse.ArgumentParser(description="Build ProductHunt FalkorDB graph")
    parser.add_argument(
        "--file", "-f",
        default=None,
        help="Path to merged JSONL file (default: ~/data/ph/final/ph_{year}_merged.jsonl)",
    )
    parser.add_argument(
        "--year", "-y",
        default="2026",
        help="Year for default file path (default: 2026)",
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Source name for FalkorDB (default: producthunt-{year})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no DB writes")
    parser.add_argument("--edges-only", action="store_true", help="Skip products, only rebuild edges")
    parser.add_argument("--clean", action="store_true", help="Wipe and rebuild from scratch")
    args = parser.parse_args()

    # Resolve defaults based on year
    data_root = _default_data_root()
    if args.file is None:
        args.file = str(data_root / "ph" / "final" / f"ph_{args.year}_merged.jsonl")
    if args.source is None:
        args.source = f"producthunt-{args.year}"

    print(f"Loading {args.file}...")
    items = load_items(args.file)
    print(f"Loaded {len(items)} items")

    build_graph(
        items=items,
        source=args.source,
        edges_only=args.edges_only,
        dry_run=args.dry_run,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
