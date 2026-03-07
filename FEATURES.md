# ts-autopilot Feature Inventory (v0.2.0)

Automated time series benchmarking — drop in a CSV, get ranked model results
with cross-validated metrics and a visual report.

## Current Features

### 1. CLI Interface (`ts-autopilot run`)
- Input CSV (`-i`), output directory (`-o`), forecast horizon (`-H`), CV folds (`-k`)
- Model selection (`-m`), parallel workers (`-j`)
- YAML/JSON config file support (`-c`)
- Verbose (`-v`) / quiet (`-q`) modes
- PDF report export (`--pdf`, requires weasyprint)
- Structured JSON logging (`--log-json`)
- Tollama integration flags (`--tollama-url`, `--tollama-models`, `--no-tollama`)
- Version flag (`-V`)

### 2. Data Ingestion & Validation
- **Long format** (canonical: `unique_id`, `ds`, `y`) and **wide format** auto-detection with conversion
- 500 MB CSV file size limit
- Automatic timezone stripping (naive UTC)
- NaN/Inf handling with warnings
- Duplicate `(unique_id, ds)` detection
- Negative value detection (informational)
- Missing data ratio tracking

### 3. Data Profiling
- Series count and total row count
- Frequency detection and season length guessing (hourly, daily, weekly, monthly, etc.)
- Series length statistics (min, max)
- Missing ratio calculation

### 4. Forecasting Models

#### Statistical Models (core, 5 models)
- **SeasonalNaive** — repeats last season (baseline)
- **AutoETS** — automated exponential smoothing
- **AutoARIMA** — automated ARIMA selection
- **AutoTheta** — automated Theta method
- **CES (AutoCES)** — complex exponential smoothing

#### Optional Models (extras, 4 models)
- **Prophet** — Facebook Prophet (`pip install "ts-autopilot[prophet]"`)
- **LightGBM** — gradient boosting via mlforecast (`pip install "ts-autopilot[lightgbm]"`)
- **NHITS** — neural model (`pip install "ts-autopilot[neural]"`)
- **NBEATS** — neural model (`pip install "ts-autopilot[neural]"`)

#### Foundation Models (via Tollama, 7 models)
- Chronos-2, TimesFM, Moirai, Granite-TTM, Lag-Llama, PatchTST, TIDE

### 5. Evaluation & Metrics
- **Expanding-window (walk-forward) cross-validation** with configurable folds
- Per-series minimum length validation and date gap detection
- **4 metrics**: MASE (primary ranking), SMAPE, RMSSE, MAE
- Per-series metric breakdowns

### 6. Residual Diagnostics
- Mean, std, skewness, kurtosis of residuals
- Ljung-Box autocorrelation test p-value
- Residual histogram with bin edges and counts
- Autocorrelation function (ACF) up to 20 lags
- Residuals vs fitted scatter plot data

### 7. Reporting & Visualization

#### Output Files
- `results.json` — frozen schema, machine-readable benchmark results
- `details.json` — forecast data and diagnostics for report reproducibility
- `report.html` — interactive HTML report with Plotly charts
- `report.pdf` — optional PDF export (requires weasyprint)

#### Report Sections
- Executive summary (auto-generated narrative, no LLM required)
- Data profile (series count, frequency, seasons, missing ratio)
- Warnings and data quality
- Leaderboard (ranked table + bar chart)
- Multi-metric radar chart (normalized MASE, SMAPE, RMSSE, MAE)
- Fold stability line chart (MASE per fold per model)
- Error distribution box plot (per-series MASE spread)
- Forecast vs actual plot (best model, top/bottom series)
- Residual diagnostics panel (histogram, ACF, residuals vs fitted)
- Per-series deep dive (collapsible)
- Methodology notes (CV strategy, metric definitions)
- Appendix: raw metrics table (all folds x models x series)

### 8. Tollama Integration (TSFM)
- Zero-shot forecasting via foundation models through HTTP API (`/v1/forecast`)
- Per-series independent forecasting
- Graceful degradation (NaN if tollama unavailable)

### 9. Configuration
- YAML/JSON config file support
- CLI flags override config file values
- Configurable retry attempts (default: 2) with exponential backoff
- Transient failure handling (RuntimeError, FloatingPointError, LinAlgError)

### 10. Logging & Observability
- Standard text logs with timestamps
- Optional structured JSON log format (one object per line)
- Configurable verbosity (DEBUG, INFO, WARNING)
- Model-level and fold-level progress callbacks
- Runtime tracking per model

### 11. Python API
- `run_benchmark(df, horizon, n_folds, runners, model_names, ...)` — full benchmark on DataFrame
- `run_from_csv(csv_path, horizon, n_folds, output_dir, ...)` — end-to-end CSV to results
- Individual metric functions: `mase()`, `smape()`, `rmsse()`, `mae()`
- Extensible via `BaseRunner` subclass for custom models

### 12. Data Contracts (Frozen Schema)
- DataProfile, BenchmarkConfig, ModelResult, FoldResult, LeaderboardEntry
- ResultMetadata (version, generated_at, total_runtime_sec)
- ForecastOutput, ForecastData, DiagnosticsResult
- JSON serialization with `to_json()` / `from_json()`

### 13. Error Handling
- `SchemaError` — CSV parsing failures
- `ConfigError` — invalid configuration
- `ModelFitError` — model fit failures after retries
- Distinct exit codes: 0 (success), 1 (schema), 2 (data), 3 (unexpected)

### 14. Development Infrastructure
- Makefile (install, lint, format, typecheck, test, coverage, clean)
- Pre-commit hooks (trailing whitespace, YAML/JSON check, large files, ruff)
- ruff linter (line-length 88, Python 3.10+)
- mypy strict mode (disallow_untyped_defs)
- pytest with 80% code coverage threshold
- GitHub Actions CI/CD

## Planned Features (Roadmap)

### v0.3 — Intelligence Phase
- LLM-powered result interpretation via Tollama
- Natural-language summaries in reports
- Model comparison narratives

### v0.4 — Scale Phase
- Multi-dataset campaigns (benchmark across many datasets)
- Ensemble recommendations
- Export to dashboard formats
- Cross-dataset analytics

## Architectural Principles
- **Zero-config default** — works out of the box with just a CSV
- **Canonical format** — all internal DataFrames use `(unique_id, ds, y)` long format
- **Thin CLI** — all logic in `pipeline.py`, not `cli.py`
- **Frozen schema** — `results.json` fields never renamed (additive only)
- **Graceful degradation** — missing optional deps skip models, don't error
- **Atomic writes** — temp file + rename to prevent corruption
- **Type safety** — full mypy compliance, typed contracts throughout
