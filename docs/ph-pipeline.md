# ProductHunt Lead Pipeline

End-to-end pipeline: scrape PH products, extract makers/commenters, enrich profiles with LinkedIn/socials, generate filtered CSVs for outreach.

## Architecture

```
Step 1: Discover/Enrich Products
  PH Leaderboard / API JSONL  -->  Playwright browser  -->  products JSONL
                                   (CF bypass, headed)       (makers with roles)

Step 2: Extract & Scrape Profiles
  products JSONL  -->  unique usernames  -->  Playwright  -->  profiles JSONL
                       (makers+commenters)    (/@username)     (LinkedIn, streak, points)

Step 3: Generate CSV
  products JSONL + profiles JSONL  -->  ph-makers-csv  -->  filtered CSV
                                       (cross-reference)    (profile_type, maker_of, commenter_on)
```

## Data Files

| File | Description |
|------|-------------|
| `~/.solo/sources/ph_2026_all_server.jsonl` | Raw API data from server (16965 products) |
| `~/.solo/sources/ph_2026_enriched.jsonl` | Browser-enriched products (makers, topics, upvotes) |
| `~/.solo/sources/ph_2026_browser.jsonl` | Discover-only products from leaderboard |
| `~/.solo/sources/ph_2026_profiles.jsonl` | Maker/commenter profile data |
| `~/.solo/sources/ph_2026_makers*.csv` | Filtered CSV exports |

## Step 1: Scrape Products

### Discover mode (leaderboard crawl)

Crawls daily leaderboard pages, extracts top products per day.

```bash
# Full 2026 discovery
scripts/ph-pipeline.sh discover

# Manual
solograph-cli scrape-ph-browser \
  --mode discover \
  --from-date 2026-01-01 \
  --to-date 2026-02-17 \
  -o ~/.solo/sources/ph_2026_browser.jsonl \
  --resume -j 3 --delay 1.5
```

### Enrich mode (API products)

Takes existing JSONL from API, visits each product page for non-redacted data.

```bash
# Enrich server file (16965 products)
scripts/ph-pipeline.sh enrich

# Manual
solograph-cli scrape-ph-browser \
  --mode enrich \
  -s ~/.solo/sources/ph_2026_all_server.jsonl \
  -o ~/.solo/sources/ph_2026_enriched.jsonl \
  --resume -j 4 --delay 1
```

**What enrichment adds:**
- Real maker usernames (not `[REDACTED]`)
- Maker/commenter/hunter roles (cross-referenced via `/products/{slug}/makers`)
- Topics, upvotes, featured_at, created_at
- product_status (featured/created), daily_rank, weekly_rank
- comments_count, reviews_count, reviews_rating

## Step 2: Scrape Profiles

Extract unique usernames from enriched products, scrape `/@username` pages.

```bash
# Run profile scraper
scripts/ph-pipeline.sh profiles

# Manual (from CLI)
solograph-cli scrape-ph-browser ...  # profiles are handled by scripts/ph-profiles.py
```

**Profile data extracted:**
- LinkedIn URL
- Twitter/X URL
- GitHub URL
- Website
- Streak (days)
- Points
- Followers count
- is_maker (PH global flag)
- Headline

## Step 3: Generate CSV

Cross-reference products (with roles) and profiles to produce filtered CSV.

```bash
# All makers+commenters with 5-1000 points, products 5-700 upvotes
scripts/ph-pipeline.sh csv

# LinkedIn-only with streak
solograph-cli ph-makers-csv \
  --linkedin-only \
  --min-streak 5 --max-streak 400

# Custom filters
solograph-cli ph-makers-csv \
  --min-points 100 --max-points 500 \
  --min-upvotes 10 --max-upvotes 200 \
  -o ~/custom_leads.csv
```

**CSV columns:**
| Column | Description |
|--------|-------------|
| `username` | PH username |
| `profile_type` | `maker` (if maker of ANY product) or `commenter` |
| `name` | Headline from profile |
| `points` | PH points |
| `streak` | Consecutive days active |
| `linkedin` | LinkedIn profile URL |
| `twitter` | Twitter/X URL |
| `github` | GitHub URL |
| `website` | Personal website |
| `is_maker_ph` | PH global maker flag |
| `followers` | Follower count |
| `max_upvotes` | Max upvotes across their products |
| `maker_of_count` | Number of products they made |
| `commenter_on_count` | Number of products they commented on |
| `maker_of` | Product names + URLs (maker role) |
| `commenter_on` | Product names + URLs (commenter role) |
| `profile_url` | PH profile URL |

## Full Pipeline (one command)

```bash
scripts/ph-pipeline.sh all
```

Runs: discover -> enrich -> profiles -> csv (with resume at each step).

## Role Detection Logic

1. Visit `/posts/{slug}` - collect ALL `/@username` links from page
2. JSON-LD `author[]` -> tagged as `maker` initially
3. "Hunted by" text -> tagged as `hunter`
4. Everyone else -> tagged as `commenter`
5. Visit `/products/{slug}/makers` - get definitive maker list
6. Cross-reference: on makers page = `maker`, rest keep original role
7. **Dedup**: within one product, each username appears once; maker wins over commenter

## Rate Limits & Tips

- Default delay: 2s between pages, -j 1 (sequential)
- Safe parallel: `-j 3 --delay 1.5` (~120 products/hour)
- Aggressive: `-j 5 --delay 0.8` (~225 products/hour)
- CF challenge solved once per session, cookies shared across tabs
- Always use `--resume` for long runs (atomic JSONL saves)
- Headed mode (macOS) required for CF bypass; headless fails
- Don't run other Chrome instances — may share profile state

## Updating Data

```bash
# 1. Download fresh API export from server
scp google:~/.solo/sources/ph_2026_all.jsonl ~/.solo/sources/ph_2026_all_server.jsonl

# 2. Re-enrich with new data
scripts/ph-pipeline.sh enrich

# 3. Update profiles (only new usernames)
scripts/ph-pipeline.sh profiles

# 4. Regenerate CSVs
scripts/ph-pipeline.sh csv
```
