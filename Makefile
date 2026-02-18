# Makefile for solograph
# Usage: make help

.PHONY: help lint format check typecheck test hooks clean \
	ph-discover ph-enrich ph-profiles ph-merge ph-csv ph-pipeline ph-status

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Code Quality ---

lint: ## Run ruff linter (with auto-fix)
	uvx ruff check --fix src/

format: ## Run ruff formatter
	uvx ruff format src/

check: ## Run all checks without modifying files (CI-friendly)
	uvx ruff check src/
	uvx ruff format --check src/

typecheck: ## Run ty type-checker (informational)
	uvx ty check src/

# --- Testing ---

test: ## Run tests (pytest)
	uv run pytest tests/ -v

# --- Hooks ---

hooks: ## Install pre-commit hooks
	uvx pre-commit install

hooks-run: ## Run pre-commit on all files
	uvx pre-commit run --all-files

# --- Build ---

build: ## Build package
	uv build

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info
	find src -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# --- ProductHunt Pipeline ---

J ?= 3
D ?= 1.5
YEAR ?= 2026
FROM ?= $(YEAR)-01-01
TO ?=
LIMIT ?=
MIN_PTS ?= 5
MAX_PTS ?= 1000
MIN_UP ?= 5
MAX_UP ?= 700

ph-discover: ## Crawl PH leaderboard (FROM=2026-01-01 TO= J=3)
	scripts/ph-pipeline.sh discover -j $(J) -d $(D) $(if $(FROM),--from $(FROM)) $(if $(TO),--to $(TO)) $(if $(LIMIT),--limit $(LIMIT))

ph-enrich: ## Enrich API products with browser data (J=3)
	scripts/ph-pipeline.sh enrich -j $(J) -d $(D) $(if $(LIMIT),--limit $(LIMIT))

ph-profiles: ## Scrape maker/commenter profiles (J=3)
	scripts/ph-pipeline.sh profiles -j $(J) -d $(D) $(if $(LIMIT),--limit $(LIMIT))

ph-merge: ## Merge enriched + API data into final file
	scripts/ph-pipeline.sh merge

ph-csv: ## Generate filtered CSVs (MIN_PTS=5 MAX_PTS=1000 MIN_UP=5 MAX_UP=700)
	scripts/ph-pipeline.sh csv

ph-pipeline: ## Full pipeline: discover -> enrich -> profiles -> csv
	scripts/ph-pipeline.sh all -j $(J) -d $(D) $(if $(FROM),--from $(FROM)) $(if $(TO),--to $(TO)) $(if $(LIMIT),--limit $(LIMIT))

ph-status: ## Show current PH data file stats
	@scripts/ph-pipeline.sh status
