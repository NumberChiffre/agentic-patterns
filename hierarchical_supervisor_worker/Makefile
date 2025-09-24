.PHONY: help fix test default

help: ## Show available commands
	@echo "Available commands:"
	@echo "  make fix   - Ruff autofix + format"
	@echo "  make test  - Run tests"

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.ba*" -delete
	@echo "✅ Cleanup complete"

fix: ## Auto-fix with ruff and format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run tests
	uv run pytest -v tests

default: help
