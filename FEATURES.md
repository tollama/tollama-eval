# tollama-eval Feature Inventory (v0.2.0)

Automated time series benchmarking — drop in a CSV, get ranked model results
with cross-validated metrics and a visual report.

## Current Features

### 1. CLI Interface (`tollama-eval run`)
- Input CSV (`-i`), output directory (`-o`), forecast horizon (`-H`), CV folds (`-k`)
- Model selection (`-m`), parallel workers (`-j`)
- YAML/JSON config file support (`-c`)
- Verbose (`-v`) / quiet (`-q`) modes
- PDF report export (`--pdf`, requires weasyprint)
- Excel workbook export (`--excel`, requires openpyxl)
- Structured JSON logging (`--log-json`)
- Result caching (`--no-cache`, `--cache-dir`)
- Parallel model execution (`--parallel-models`)
- AutoML model selection (`--auto-select`)
- Optional model discovery (`--include-optional`, `--include-neural`)
- Anomaly detection (`--detect-anomalies`)
- Composite metric scoring (`--metric-weights`)
- Exogenous variable support (`--exog-cols`)
- Distributed execution (`--distributed`, requires Ray)
- Tollama integration flags (`--tollama-url`, `--tollama-models`, `--no-tollama`)
- Version flag (`-V`)

### 1b. Additional CLI Commands
- **`tollama-eval campaign`** — multi-dataset benchmarking across a directory of CSVs
- **`tollama-eval doctor`** — environment diagnostics (Python version, dependencies, hardware acceleration)
- **`tollama-eval serve`** — REST API server (FastAPI, requires `tollama-eval[server]`)

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

#### Extended Statistical Models (9 models)
- **MSTL** — multiple seasonal-trend decomposition using LOESS
- **DynamicOptimizedTheta (DOTheta)** — dynamic theta variant
- **Holt** — double exponential smoothing (linear trend)
- **HoltWinters** — triple exponential smoothing (trend + seasonality)
- **HistoricAverage** — simple mean baseline
- **Naive** — repeat last observation
- **RandomWalkWithDrift** — random walk with trend
- **WindowAverage** — simple moving average
- **SeasonalWindowAverage** — seasonal moving average

#### Intermittent Demand Models (6 models)
- **CrostonClassic** — classic Croston method
- **CrostonOptimized** — optimized Croston method
- **CrostonSBA** — Syntetos-Boylan Approximation (bias-corrected)
- **ADIDA** — Aggregate-Disaggregate Intermittent Demand Approach
- **IMAPA** — Intermittent Multiple Aggregation Prediction Algorithm
- **TSB** — Teunter-Syntetos-Babai method

#### Optional ML Models (extras)
- **Prophet** — Facebook Prophet (`pip install "tollama-eval[prophet]"`)
- **LightGBM** — gradient boosting via mlforecast (`pip install "tollama-eval[lightgbm]"`)
- **XGBoost** — gradient boosting via mlforecast (`pip install "tollama-eval[xgboost]"`)

#### Optional Neural Models (extras, `pip install "tollama-eval[neural]"`)
- **NHITS** — N-HiTS neural model
- **NBEATS** — N-BEATS neural model
- **TiDE** — Time-series Dense Encoder
- **DeepAR** — probabilistic autoregressive model
- **PatchTST** — Patch Time Series Transformer
- **TFT** — Temporal Fusion Transformer

#### Foundation Models (via Tollama, 7 models)
- Chronos-2, TimesFM, Moirai, Granite-TTM, Lag-Llama, PatchTST, TIDE

### 5. Evaluation & Metrics
- **Expanding-window (walk-forward) cross-validation** with configurable folds
- Per-series minimum length validation and date gap detection
- **6 metrics**: MASE (primary ranking), SMAPE, RMSSE, MAE, MSIS, Coverage
- Per-series metric breakdowns
- **Composite scoring** with configurable metric weights
- **Probabilistic metrics**: MSIS (Mean Scaled Interval Score), coverage probability

### 5b. AutoML & Intelligent Model Selection
- **Auto-model recommendation** based on data profile (series length, frequency, intermittency)
- Models categorized by priority: must-run, recommended, optional
- **Intermittency detection** using SBC (Syntetos-Boylan Classification): smooth, erratic, intermittent, lumpy
- Automatic intermittent demand model selection when zero ratio > 30%
- Configurable max model count

### 5c. Ensemble Construction
- **Simple average ensemble** — average predictions across all models
- **Inverse-MASE weighted ensemble** — weight by model accuracy
- **Best-per-series ensemble** — select best model per series
- Per-series best-model recommendation with win counts
- Virtual ensemble MASE computation

### 5d. Anomaly Detection
- **Z-score detector** — statistical threshold-based detection
- **IQR detector** — interquartile range outlier detection
- **Rolling detector** — rolling mean/std anomaly detection
- **Residual detector** — forecast residual-based anomaly detection
- Anomaly scoring, ranking, and reporting
- Multi-detector ensemble (`run_all_detectors`)

### 5e. Model Stability Analysis
- **Fold CV** — coefficient of variation across CV folds
- **Series CV** — coefficient of variation across series
- **Rank consistency** — how often a model maintains its rank
- **Composite stability score** (0-1) for leaderboard tie-breaking

### 5f. Speed Benchmarking & Pareto Analysis
- Per-model speed profiles (total runtime, per-series, throughput)
- **Pareto frontier** analysis (accuracy vs speed tradeoff)
- Identification of fastest and most efficient models

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
- `leaderboard.csv` — ranked model summary table
- `fold_details.csv` — fold-level metrics by model
- `per_series_scores.csv` — per-series error table
- `per_series_winners.csv` — per-series winner and margin summary
- `report.pdf` — optional PDF export (`--pdf`, requires weasyprint)
- `report.xlsx` — optional Excel workbook export (`--excel`, requires openpyxl)

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
- **Fluent SDK**: `TSAutopilot(df).with_models([...]).with_horizon(14).run()`
- Individual metric functions: `mase()`, `smape()`, `rmsse()`, `mae()`
- Extensible via `BaseRunner` subclass for custom models

### 11b. REST API Server (`tollama-eval serve`)
- FastAPI-based REST server with async job execution
- Endpoints: `POST /benchmark`, `GET /status/{run_id}`, `GET /results/{run_id}`, `GET /health`
- Authentication: API key, JWT, OIDC support
- Multi-tenancy, job persistence, retention policies
- Prometheus metrics, webhook notifications
- Rate limiting via slowapi

### 11c. Distributed Execution
- **Ray integration** (`--distributed`) for parallel fold execution across workers
- Safe hardware probe with Ray fallback

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
- LLM-powered result interpretation (via separate LLM API, not Tollama)
- Natural-language model comparison narratives
- Hierarchical reconciliation GA (currently experimental)
- Confidence band visualization in reports

### v0.4 — Scale Phase
- Export to PowerPoint and Jupyter notebook
- Streamlit dashboard GA (currently experimental)
- Custom model plugin system via entry points

## Architectural Principles
- **Zero-config default** — works out of the box with just a CSV
- **Canonical format** — all internal DataFrames use `(unique_id, ds, y)` long format
- **Thin CLI** — all logic in `pipeline.py`, not `cli.py`
- **Frozen schema** — `results.json` fields never renamed (additive only)
- **Graceful degradation** — missing optional deps skip models, don't error
- **Atomic writes** — temp file + rename to prevent corruption
- **Type safety** — full mypy compliance, typed contracts throughout
