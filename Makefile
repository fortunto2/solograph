# Makefile for solograph
# Usage: make help

.PHONY: help lint format check typecheck test hooks clean

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
