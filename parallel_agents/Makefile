.PHONY: help fix test default

help: ## Show available commands
	@echo "Available commands:"
	@echo "  make fix   - Ruff autofix + format"
	@echo "  make test  - Run tests"

fix: ## Auto-fix with ruff and format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run tests
	uv run pytest -v tests

default: help
