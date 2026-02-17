#!/usr/bin/env bash
# ProductHunt Lead Pipeline — discover, enrich, profiles, csv
set -euo pipefail

SOURCES="${HOME}/.solo/sources"
CLI="uv run solograph-cli"
YEAR=2026

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

Commands:
  discover    Crawl PH daily leaderboards for ${YEAR}
  enrich      Enrich API products with browser data
  profiles    Scrape maker/commenter profiles (LinkedIn, streak, points)
  csv         Generate filtered CSVs from enriched data
  all         Run full pipeline: discover -> enrich -> profiles -> csv

Options:
  -j NUM      Parallel browser tabs (default: ${CONCURRENCY})
  -d SECS     Delay between requests (default: ${DELAY})
  --from DATE Start date for discover (default: ${YEAR}-01-01)
  --to DATE   End date for discover (default: today)
  --limit N   Max items to process

Examples:
  $(basename "$0") all
  $(basename "$0") enrich -j 4
  $(basename "$0") csv
  $(basename "$0") discover --from 2026-02-01 --to 2026-02-17
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

mkdir -p "${SOURCES}"

do_discover() {
    echo "=== Step 1a: Discover products from leaderboard ==="
    local args=(
        scrape-ph-browser --mode discover
        --from-date "${FROM_DATE}"
        -o "${SOURCES}/ph_${YEAR}_browser.jsonl"
        --resume -j "${CONCURRENCY}" --delay "${DELAY}"
    )
    [[ -n "${TO_DATE}" ]] && args+=(--to-date "${TO_DATE}")
    [[ -n "${LIMIT}" ]] && args+=(--limit "${LIMIT}")
    ${CLI} "${args[@]}"
}

do_enrich() {
    echo "=== Step 1b: Enrich API products with browser data ==="
    local source="${SOURCES}/ph_${YEAR}_all_server.jsonl"
    if [[ ! -f "${source}" ]]; then
        source="${SOURCES}/ph_${YEAR}_all.jsonl"
    fi
    if [[ ! -f "${source}" ]]; then
        echo "ERROR: No source file found. Expected:"
        echo "  ${SOURCES}/ph_${YEAR}_all_server.jsonl"
        echo "  ${SOURCES}/ph_${YEAR}_all.jsonl"
        exit 1
    fi
    local args=(
        scrape-ph-browser --mode enrich
        -s "${source}"
        -o "${SOURCES}/ph_${YEAR}_enriched.jsonl"
        --resume -j "${CONCURRENCY}" --delay "${DELAY}"
    )
    [[ -n "${LIMIT}" ]] && args+=(--limit "${LIMIT}")
    ${CLI} "${args[@]}"
}

do_profiles() {
    echo "=== Step 2: Scrape maker/commenter profiles ==="
    local enriched="${SOURCES}/ph_${YEAR}_enriched.jsonl"
    if [[ ! -f "${enriched}" ]]; then
        echo "ERROR: No enriched file. Run 'enrich' first."
        exit 1
    fi
    # Extract unique usernames and run profile scraper via Python
    python3 - "${enriched}" "${SOURCES}/ph_${YEAR}_profiles.jsonl" "${CONCURRENCY}" "${DELAY}" "${LIMIT}" <<'PYEOF'
import sys, json
from pathlib import Path

enriched_path = sys.argv[1]
profiles_path = sys.argv[2]
concurrency = int(sys.argv[3])
delay = float(sys.argv[4])
limit = int(sys.argv[5]) if sys.argv[5] else 0

# Collect unique usernames from enriched products
usernames = set()
for line in Path(enriched_path).read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        item = json.loads(line)
        for m in item.get("makers", []):
            u = m.get("username", "")
            if u:
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

do_csv() {
    echo "=== Step 3: Generate filtered CSVs ==="
    local enriched="${SOURCES}/ph_${YEAR}_enriched.jsonl"
    local profiles="${SOURCES}/ph_${YEAR}_profiles.jsonl"
    if [[ ! -f "${enriched}" ]]; then
        echo "ERROR: No enriched file. Run 'enrich' first."
        exit 1
    fi

    # All makers+commenters
    echo "--- All (points ${MIN_POINTS}-${MAX_POINTS}, upvotes ${MIN_UPVOTES}-${MAX_UPVOTES}) ---"
    ${CLI} ph-makers-csv \
        -p "${enriched}" --profiles "${profiles}" \
        --min-points "${MIN_POINTS}" --max-points "${MAX_POINTS}" \
        --min-upvotes "${MIN_UPVOTES}" --max-upvotes "${MAX_UPVOTES}" \
        -o "${SOURCES}/ph_${YEAR}_makers.csv"

    # LinkedIn only
    echo "--- LinkedIn only ---"
    ${CLI} ph-makers-csv \
        -p "${enriched}" --profiles "${profiles}" \
        --min-points "${MIN_POINTS}" --max-points "${MAX_POINTS}" \
        --min-upvotes "${MIN_UPVOTES}" --max-upvotes "${MAX_UPVOTES}" \
        --linkedin-only \
        -o "${SOURCES}/ph_${YEAR}_makers_linkedin.csv"

    # LinkedIn + streak
    echo "--- LinkedIn + streak 5-400 ---"
    ${CLI} ph-makers-csv \
        -p "${enriched}" --profiles "${profiles}" \
        --min-points "${MIN_POINTS}" --max-points "${MAX_POINTS}" \
        --min-upvotes "${MIN_UPVOTES}" --max-upvotes "${MAX_UPVOTES}" \
        --linkedin-only --min-streak 5 --max-streak 400 \
        -o "${SOURCES}/ph_${YEAR}_makers_linkedin_streak.csv"

    echo ""
    echo "=== CSV files ==="
    ls -lh "${SOURCES}"/ph_${YEAR}_makers*.csv
}

case "${CMD}" in
    discover) do_discover ;;
    enrich)   do_enrich ;;
    profiles) do_profiles ;;
    csv)      do_csv ;;
    all)
        do_discover
        do_enrich
        do_profiles
        do_csv
        echo ""
        echo "=== Pipeline complete ==="
        ;;
    help|-h|--help) usage ;;
    *) echo "Unknown command: ${CMD}"; usage ;;
esac
