# Contributing to ts-autopilot

Thank you for considering contributing to ts-autopilot! This document provides
guidelines and instructions for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ychoi-atop/ts-autopilot.git
cd ts-autopilot

# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Verify installation
ts-autopilot --version
ts-autopilot doctor
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage report
make coverage

# Run a specific test file
pytest tests/test_pipeline.py -v

# Run a specific test
pytest tests/test_pipeline.py::test_run_benchmark_returns_result -v
```

### Code Quality

```bash
# Lint check
make lint

# Auto-format code
make format

# Type checking
make typecheck

# Run all checks
make lint && make typecheck && make test
```

### Code Style

- **Formatter**: ruff (line length 88, Python 3.10 target)
- **Linter rules**: E, F, I, UP, B, SIM, RUF
- **Type annotations**: Required for all public functions (mypy strict mode)
- **Docstrings**: Required for all public modules, classes, and functions

### Key Invariants

1. **Canonical data format**: `unique_id` (str), `ds` (datetime64[ns] naive), `y` (float64)
2. **results.json schema is frozen**: Do not rename fields. Additive changes only.
3. **CLI is a thin shell**: All business logic lives in `pipeline.py` and submodules.
4. **Shared test fixtures**: Live in `tests/conftest.py`.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, atomic commits
3. Ensure all checks pass: `make lint && make typecheck && make coverage`
4. Update documentation if needed (README, FEATURES.md, CHANGELOG.md)
5. Submit a pull request with a clear description

### PR Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Linting passes (`ruff check src/ tests/`)
- [ ] Type checking passes (`mypy src/ts_autopilot/`)
- [ ] Coverage gate met (80%+)
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] Documentation updated if applicable

## Architecture Overview

```
src/ts_autopilot/
├── cli.py           # Typer CLI (thin shell)
├── pipeline.py      # Orchestrator (core logic)
├── contracts.py     # Frozen data contracts
├── config.py        # YAML/JSON config loading
├── exceptions.py    # Exception hierarchy
├── events.py        # Structured event emission
├── logging_config.py# Structured logging
├── ingestion/       # CSV loading & profiling
├── evaluation/      # CV, metrics, diagnostics
├── runners/         # Model runners (base, statistical, optional, tollama)
├── reporting/       # HTML, PDF, exports
└── tollama/         # TSFM HTTP client
```

## Adding a New Model Runner

1. Create a new class inheriting from `BaseRunner` or `StatsForecastRunner`
2. Implement `name` property and `fit_predict` method
3. Add to `DEFAULT_RUNNERS` in `pipeline.py` or `get_optional_runners()`
4. Add tests in `tests/`
5. Update FEATURES.md model roster

## Reporting Issues

Please report bugs and feature requests via
[GitHub Issues](https://github.com/ychoi-atop/ts-autopilot/issues).
