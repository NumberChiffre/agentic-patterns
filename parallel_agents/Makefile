.PHONY: help lint test default

help: ## Show available commands
	@echo "Available commands:"
	@echo "  make lint  - Auto-fix lint issues with ruff"

lint: ## Auto-fix lint issues with ruff
	uv run ruff check --fix .

test: ## Run tests
	uv run pytest -v tests

default: help
