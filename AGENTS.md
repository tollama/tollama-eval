# AGENTS.md — Instructions for AI Coding Agents

## Project
ts-autopilot: automated time series benchmarking pipeline.
CSV in → model comparison → results.json + report.html out.

## Commands
- Lint: `ruff check src/ tests/`
- Test: `pytest tests/ -v`
- Install (dev): `pip install -e .` then `pip install pytest ruff`

## Canonical Data Format
All internal DataFrames use **long format** with exactly these columns:
- `unique_id`: `str`
- `ds`: `datetime64[ns]` (timezone-naive)
- `y`: `float64`

If input has timezones, strip to naive. If `unique_id` is missing, use `"series_1"`.

## Dependency Policy
- No new dependencies without updating `pyproject.toml` and adding at least one test that would fail without the new dependency.
- **Prohibited in MVP**: torch, neuralforecast, lightgbm, salesforce-merlion, stumpy, polars, plotly.

## Reserved Flags (not implemented)
- `--tollama-url`: URL for tollama service
- `--no-tollama`: Disable tollama integration

## results.json Schema (frozen — do not rename fields)
```json
{
  "profile": { ... },
  "config": { "horizon": int, "n_folds": int },
  "models": [{ "name": str, "runtime_sec": float, "folds": [...], "mean_mase": float, "std_mase": float }],
  "leaderboard": [{ "rank": int, "name": str, "mean_mase": float }]
}
```

## Architecture Rule
CLI calls pipeline functions. Zero logic in `cli.py`.
