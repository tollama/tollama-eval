# CLAUDE.md — Project Memory for Claude Code

## Current Milestone
v0.2.0 Beta — 36+ models (statistical, ML, neural, foundation), AutoML, campaign mode, REST API server, fluent SDK, distributed execution, ensemble, anomaly detection, stability analysis.

## Key Invariants
- Canonical format: unique_id (str), ds (datetime64[ns] naive), y (float64)
- results.json schema is frozen — do not rename fields
- CLI calls pipeline functions, zero embedded logic in cli.py
- Shared test fixtures live in tests/conftest.py

## Commands
- `ruff check src/ tests/` — lint
- `pytest tests/ -v` — run tests
- `pip install -e .` — install in dev mode
