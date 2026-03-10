# AGENTS.md — Instructions for AI Coding Agents

## Project
tollama-eval: automated time series benchmarking pipeline.
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
- Optional deps (torch, neuralforecast, lightgbm, plotly, etc.) live behind install extras; the core package must work without them.

## Tollama Integration (TSFM)
tollama is a **Time Series Foundation Model platform** (https://github.com/tollama/tollama),
NOT an LLM service. It provides access to models like Chronos-2, TimesFM, Moirai, etc.

- `--tollama-url`: Base URL of the tollama server (e.g. http://localhost:8000)
- `--tollama-models`: Comma-separated TSFM models to benchmark (e.g. chronos2,timesfm)
- `--no-tollama`: Disable tollama integration even if URL is provided
- TollamaRunner delegates to tollama's `POST /v1/forecast` endpoint
- Tollama models are zero-shot (no training step)
- Graceful degradation: if tollama is unreachable, NaN values are used

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
