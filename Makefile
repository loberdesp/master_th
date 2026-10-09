.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-all cov check tb clean

FAST := -m "not slow and not gpu and not hardware"

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Sync the environment and install git hooks
	uv sync
	uv run pre-commit install

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Pyright
	uv run pyright

test: ## Fast tests (skips slow, gpu, hardware)
	uv run pytest $(FAST)

test-all: ## All tests
	uv run pytest

cov: ## Fast tests with coverage (terminal + htmlcov/)
	uv run pytest $(FAST) --cov --cov-report=term-missing --cov-report=html

check: lint typecheck test ## Lint + typecheck + fast tests

tb: ## TensorBoard on runs/
	uv run tensorboard --logdir runs

clean: ## Remove caches (never touches data/ or runs/)
	rm -rf .ruff_cache .pytest_cache .coverage .coverage.* coverage.xml htmlcov
	find . -path ./.venv -prune -o -path ./data -prune -o -path ./runs -prune -o \
		-type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -print -exec rm -rf {} +
