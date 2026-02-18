#!/usr/bin/env bash
# ProductHunt Lead Pipeline — discover, enrich, profiles, csv
set -euo pipefail

CLI="uv run solograph-cli"
YEAR=2026

# --- Data root from ~/.solo/config.yaml ---
SOLO_CONFIG="${HOME}/.solo/config.yaml"
if [[ -n "${SOLO_DATA_ROOT:-}" ]]; then
    DATA_ROOT="${SOLO_DATA_ROOT}"
elif [[ -f "${SOLO_CONFIG}" ]]; then
    # Parse data_root from YAML (simple grep, no yq dependency)
    DATA_ROOT=$(grep '^data_root:' "${SOLO_CONFIG}" 2>/dev/null | sed 's/^data_root:[[:space:]]*//' | sed "s|~|${HOME}|g")
fi
DATA_ROOT="${DATA_ROOT:-${HOME}/data}"

PH="${DATA_ROOT}/ph"

# Defaults
CONCURRENCY=3
DELAY=1.5
MIN_POINTS=5
MAX_POINTS=1000
MIN_UPVOTES=5
MAX_UPVOTES=700

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Data root: ${PH}

Commands:
  discover    Crawl PH daily leaderboards for ${YEAR}
  enrich      Enrich API products with browser data
  profiles    Scrape maker/commenter profiles (LinkedIn, streak, points)
  csv         Generate filtered CSVs from enriched data
  merge       Merge enriched + API data into final file
  all         Run full pipeline: discover -> enrich -> profiles -> csv
  status      Show data file stats

Options:
  -j NUM      Parallel browser tabs (default: ${CONCURRENCY})
  -d SECS     Delay between requests (default: ${DELAY})
  --from DATE Start date for discover (default: ${YEAR}-01-01)
  --to DATE   End date for discover (default: today)
  --limit N   Max items to process

Env: SOLO_DATA_ROOT overrides config. Default: ~/data

Examples:
  $(basename "$0") all
  $(basename "$0") enrich -j 4
  $(basename "$0") csv
  $(basename "$0") discover --from 2026-02-01 --to 2026-02-17
  $(basename "$0") status
EOF
    exit 0
}

# Parse args
CMD="${1:-help}"
shift || true

FROM_DATE="${YEAR}-01-01"
TO_DATE=""
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j) CONCURRENCY="$2"; shift 2 ;;
        -d) DELAY="$2"; shift 2 ;;
        --from) FROM_DATE="$2"; shift 2 ;;
        --to) TO_DATE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "${PH}"/{raw,enriched,profiles,final/csv,tmp}

do_discover() {
    echo "=== Step 1a: Discover products from leaderboard ==="
    local args=(
        scrape-ph-browser --mode discover
        --from-date "${FROM_DATE}"
        -o "${PH}/raw/ph_${YEAR}_browser.jsonl"
        --resume -j "${CONCURRENCY}" --delay "${DELAY}"
    )
    [[ -n "${TO_DATE}" ]] && args+=(--to-date "${TO_DATE}")
    [[ -n "${LIMIT}" ]] && args+=(--limit "${LIMIT}")
    ${CLI} "${args[@]}"
}

do_enrich() {
    echo "=== Step 1b: Enrich API products with browser data ==="
    local source=""
    for f in "${PH}/raw/ph_${YEAR}_all_server_v2.jsonl" \
             "${PH}/raw/ph_${YEAR}_all_server.jsonl" \
             "${PH}/raw/ph_${YEAR}_all.jsonl"; do
        if [[ -f "${f}" ]]; then
            source="${f}"
            break
        fi
    done
    if [[ -z "${source}" ]]; then
        echo "ERROR: No source file found in ${PH}/raw/"
        exit 1
    fi
    echo "  Source: ${source}"
    local args=(
        scrape-ph-browser --mode enrich
        -s "${source}"
        -o "${PH}/enriched/ph_${YEAR}_enriched.jsonl"
        --resume -j "${CONCURRENCY}" --delay "${DELAY}"
    )
    [[ -n "${LIMIT}" ]] && args+=(--limit "${LIMIT}")
    ${CLI} "${args[@]}"
}

do_profiles() {
    echo "=== Step 2: Scrape maker/commenter profiles ==="
    # Use merged file (best coverage) or enriched as fallback
    local source=""
    for f in "${PH}/final/ph_${YEAR}_merged.jsonl" \
             "${PH}/enriched/ph_${YEAR}_enriched.jsonl" \
             "${PH}/enriched/ph_${YEAR}_enriched_v2.jsonl"; do
        if [[ -f "${f}" ]]; then
            source="${f}"
            break
        fi
    done
    if [[ -z "${source}" ]]; then
        echo "ERROR: No enriched/merged file. Run 'enrich' or 'merge' first."
        exit 1
    fi
    echo "  Source: ${source}"
    local profiles="${PH}/profiles/ph_${YEAR}_profiles.jsonl"

    # Run via uv to get solograph venv with codegraph_mcp
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${SCRIPT_DIR}"
    uv run python3 - "${source}" "${profiles}" "${CONCURRENCY}" "${DELAY}" "${LIMIT}" <<'PYEOF'
import sys, json
from pathlib import Path

source_path = sys.argv[1]
profiles_path = sys.argv[2]
concurrency = int(sys.argv[3])
delay = float(sys.argv[4])
limit = int(sys.argv[5]) if sys.argv[5] else 0

# Collect unique usernames from products
usernames = set()
for line in Path(source_path).read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        item = json.loads(line)
        for m in item.get("makers", []):
            u = m.get("username", "")
            if u and "[REDACTED]" not in u:
                usernames.add(u)
    except json.JSONDecodeError:
        pass

print(f"Found {len(usernames)} unique usernames")
usernames_list = sorted(usernames)

from codegraph_mcp.scrapers.producthunt_browser import run_ph_profile_scraper
run_ph_profile_scraper(
    usernames=usernames_list,
    resume_path=profiles_path,
    concurrency=concurrency,
    delay=delay,
    limit=limit or None,
)
PYEOF
}

do_merge() {
    echo "=== Merge: Combine enriched + API data ==="
    python3 - "${PH}" "${YEAR}" <<'PYEOF'
import json, sys
from pathlib import Path

ph = Path(sys.argv[1])
year = sys.argv[2]

def load_jsonl(filepath):
    items = {}
    if not filepath.exists():
        return items
    for line in filepath.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            slug = item.get("slug", "")
            if slug:
                items[slug] = item
        except json.JSONDecodeError:
            pass
    return items

# Load all enriched files (v2 has priority — has roles)
enriched_files = sorted(ph.glob(f"enriched/ph_{year}_enriched*.jsonl"), reverse=True)
api_files = sorted(ph.glob(f"raw/ph_{year}_all*.jsonl"), reverse=True)

all_enriched = {}
for f in enriched_files:
    data = load_jsonl(f)
    print(f"  Enriched: {f.name} → {len(data)} products")
    for slug, item in data.items():
        if slug not in all_enriched:
            all_enriched[slug] = item

all_api = {}
for f in api_files:
    data = load_jsonl(f)
    print(f"  API: {f.name} → {len(data)} products")
    for slug, item in data.items():
        if slug not in all_api:
            all_api[slug] = item

NEW_FIELDS = [
    "comments", "daily_rank", "weekly_rank", "rating",
    "reviews_count", "featured", "product_links",
    "comments_count", "created_at", "featured_at",
    "product_status", "launch_date",
]

merged = {}
from_browser = 0
from_api = 0

# Priority 1: enriched (browser data with real makers)
for slug, item in all_enriched.items():
    if slug in all_api:
        for field in NEW_FIELDS:
            api_val = all_api[slug].get(field)
            cur_val = item.get(field)
            if api_val and not cur_val:
                item[field] = api_val
        sv_up = all_api[slug].get("upvotes", 0) or 0
        if (sv_up) > (item.get("upvotes", 0) or 0):
            item["upvotes"] = sv_up
    item["_source"] = "browser"
    merged[slug] = item
    from_browser += 1

# Priority 2: API-only products
for slug, item in all_api.items():
    if slug not in merged:
        item["_source"] = "api_only"
        if not item.get("product_status"):
            item["product_status"] = "launched"
        merged[slug] = item
        from_api += 1

# Ensure all products have a status
for slug, item in merged.items():
    if not item.get("product_status"):
        item["product_status"] = "launched"

out = ph / "final" / f"ph_{year}_merged.jsonl"
tmp = out.with_suffix(".jsonl.tmp")
with open(tmp, "w") as f:
    for item in merged.values():
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
tmp.rename(out)

real_makers = sum(1 for i in merged.values()
    if i.get("makers") and not any("[REDACTED]" in str(m) for m in i["makers"]))

print(f"\nMerged: {len(merged)} products")
print(f"  browser enriched: {from_browser}")
print(f"  api only:         {from_api}")
print(f"  with real makers: {real_makers}")
print(f"Saved: {out}")
PYEOF
}

do_csv() {
    echo "=== Step 3: Generate filtered CSVs ==="
    local source="${PH}/final/ph_${YEAR}_merged.jsonl"
    local profiles="${PH}/profiles/ph_${YEAR}_profiles.jsonl"
    if [[ ! -f "${source}" ]]; then
        echo "ERROR: No merged file. Run 'merge' first."
        exit 1
    fi

    local csv_dir="${PH}/final/csv"

    echo "--- All (points ${MIN_POINTS}-${MAX_POINTS}, upvotes ${MIN_UPVOTES}-${MAX_UPVOTES}) ---"
    ${CLI} ph-makers-csv \
        -p "${source}" --profiles "${profiles}" \
        --min-points "${MIN_POINTS}" --max-points "${MAX_POINTS}" \
        --min-upvotes "${MIN_UPVOTES}" --max-upvotes "${MAX_UPVOTES}" \
        -o "${csv_dir}/ph_${YEAR}_makers.csv"

    echo "--- LinkedIn only ---"
    ${CLI} ph-makers-csv \
        -p "${source}" --profiles "${profiles}" \
        --min-points "${MIN_POINTS}" --max-points "${MAX_POINTS}" \
        --min-upvotes "${MIN_UPVOTES}" --max-upvotes "${MAX_UPVOTES}" \
        --linkedin-only \
        -o "${csv_dir}/ph_${YEAR}_makers_linkedin.csv"

    echo "--- LinkedIn + streak 5-400 ---"
    ${CLI} ph-makers-csv \
        -p "${source}" --profiles "${profiles}" \
        --min-points "${MIN_POINTS}" --max-points "${MAX_POINTS}" \
        --min-upvotes "${MIN_UPVOTES}" --max-upvotes "${MAX_UPVOTES}" \
        --linkedin-only --min-streak 5 --max-streak 400 \
        -o "${csv_dir}/ph_${YEAR}_makers_linkedin_streak.csv"

    echo ""
    echo "=== CSV files ==="
    ls -lh "${csv_dir}"/ph_${YEAR}_makers*.csv
}

do_status() {
    echo "=== ProductHunt Data: ${PH} ==="
    echo ""
    for dir in raw enriched profiles final final/csv tmp; do
        local full="${PH}/${dir}"
        [[ -d "${full}" ]] || continue
        local has_files=false
        for f in "${full}"/*; do
            [[ -f "$f" ]] || continue
            has_files=true
            local name
            name=$(basename "$f")
            local size
            size=$(ls -lh "$f" | awk '{print $5}')
            local lines=""
            if [[ "$f" == *.jsonl || "$f" == *.csv || "$f" == *.txt ]]; then
                lines=" ($(wc -l < "$f" | tr -d ' ') lines)"
            fi
            printf "  %-12s %-50s %6s%s\n" "${dir}/" "${name}" "${size}" "${lines}"
        done
        $has_files || true
    done
}

case "${CMD}" in
    discover) do_discover ;;
    enrich)   do_enrich ;;
    profiles) do_profiles ;;
    merge)    do_merge ;;
    csv)      do_csv ;;
    status)   do_status ;;
    all)
        do_discover
        do_enrich
        do_profiles
        do_merge
        do_csv
        echo ""
        echo "=== Pipeline complete ==="
        ;;
    help|-h|--help) usage ;;
    *) echo "Unknown command: ${CMD}"; usage ;;
esac
