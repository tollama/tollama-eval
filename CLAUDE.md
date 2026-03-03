# CLAUDE.md — Project Memory for Claude Code

## Current Milestone
MVP v0.1 — single command benchmark with SeasonalNaive + AutoETS.

## Build Order
Issues 2 → 3 → 4 → 6 → 7 → 5 → 8 → 10 → 9

## Key Invariants
- Canonical format: unique_id (str), ds (datetime64[ns] naive), y (float64)
- results.json schema is frozen — do not rename fields
- CLI calls pipeline functions, zero embedded logic in cli.py
- Shared test fixtures live in tests/conftest.py

## Commands
- `ruff check src/ tests/` — lint
- `pytest tests/ -v` — run tests
- `pip install -e .` — install in dev mode
