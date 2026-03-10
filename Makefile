.PHONY: install lint format typecheck test coverage clean help all audit docker lock docs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install in development mode
	pip install -e ".[dev]"

lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format code
	ruff format src/ tests/

typecheck:  ## Run mypy type checker
	mypy src/ts_autopilot/

format-check:  ## Check formatting without changes
	ruff format --check --diff src/ tests/

test:  ## Run tests with verbose output
	python -m pytest tests/ -v

coverage:  ## Run tests with coverage report
	python -m pytest tests/ --cov=ts_autopilot --cov-report=term-missing --cov-fail-under=80

all:  ## Run lint, typecheck, and tests
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

audit:  ## Run dependency vulnerability scan
	pip-audit --strict -r requirements-lock.txt

docker:  ## Build Docker image
	docker build -t tollama-eval:latest .

lock:  ## Regenerate dependency lock file
	uv pip compile pyproject.toml -o requirements-lock.txt

docs:  ## Generate API reference docs
	pdoc src/ts_autopilot/ -o docs/_build/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
