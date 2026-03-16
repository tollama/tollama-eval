<p align="center">
  <h1 align="center">tollama-eval</h1>
  <p align="center">
    <strong>Stop guessing which forecasting model to use. Let the data decide.</strong>
  </p>
  <p align="center">
    Part of the <a href="https://www.tollama.com/">Tollama</a> time series platform.
  </p>
  <p align="center">
    <a href="https://pypi.org/project/tollama-eval/"><img src="https://img.shields.io/pypi/v/tollama-eval.svg" alt="PyPI version"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://github.com/tollama/tollama-eval/actions"><img src="https://img.shields.io/github/actions/workflow/status/tollama/tollama-eval/ci.yml?label=CI" alt="CI"></a>
  </p>
</p>

---

**tollama-eval** is a zero-config time series benchmarking tool. Drop in a CSV, get a ranked leaderboard of 36+ forecasting models — complete with cross-validated metrics and a visual report. No boilerplate. No notebook spaghetti. One command.

```bash
pip install tollama-eval
tollama-eval run -i sales.csv
```

That's it. Open `out/report.html` and see which model wins.

---

## Why tollama-eval?

Forecasting teams waste weeks on the same loop: load data, wrangle formats, fit models one at a time, compute metrics manually, copy results into slides. **tollama-eval collapses that entire workflow into a single command.**

| The old way | With tollama-eval |
|---|---|
| Write custom ingestion for every dataset | Auto-detects long & wide CSV formats |
| Manually split train/test, hope it's fair | Expanding-window cross-validation, configurable folds |
| Fit models one at a time in notebooks | 36+ models run automatically, from naive baselines to foundation models |
| Pick models manually, hope for the best | AutoML recommends models based on your data profile |
| Compute MAPE and call it a day | 6 metrics (MASE, SMAPE, RMSSE, MAE, MSIS, Coverage) with composite scoring |
| Paste results into a spreadsheet | Structured JSON + interactive HTML report, generated instantly |

### Who is this for?

- **Data scientists** who want a fast baseline before investing in custom models
- **ML engineers** evaluating which forecasting approach fits a new dataset
- **Analysts** who need defensible model comparisons without writing Python
- **Teams** standardizing how they benchmark across projects

---

## Quick Start

### 1. Install

```bash
pip install tollama-eval
```

### 2. Run

```bash
tollama-eval run -i your_data.csv
```

### 3. Read the results

Standard runs write a full artifact set to `out/`:

- **`results.json`** — Machine-readable benchmark results (schema is frozen across versions, safe to build on)
- **`details.json`** — Forecast, diagnostics, data-characteristics, and provenance context for rich reports
- **`report.html`** — Interactive visual report with leaderboard, forecasts, diagnostics, and per-series winner analysis
- **`leaderboard.csv`** — Ranked summary table
- **`fold_details.csv`** — Fold-level metrics by model
- **`per_series_scores.csv`** — Per-series model error table
- **`per_series_winners.csv`** — Per-series winner and margin summary

Add `--pdf` to also generate `report.pdf`. Add `--excel` to generate `report.xlsx`.

---

## Documentation

Use these guides for common workflows:

- [Troubleshooting](docs/troubleshooting.md)
- [Output Interpretation](docs/output-interpretation.md)
- [Recipe Cookbook](docs/recipes.md)
- [Tollama Setup Guide](docs/tollama-setup.md)
- [Feature Inventory](FEATURES.md)

---

## Installation Extras

The base install includes 20 built-in statistical models. Install optional extras for more:

```bash
pip install "tollama-eval[all]"          # Everything
pip install "tollama-eval[prophet]"      # Facebook Prophet
pip install "tollama-eval[lightgbm]"     # LightGBM via mlforecast
pip install "tollama-eval[xgboost]"      # XGBoost via mlforecast
pip install "tollama-eval[neural]"       # NHITS, NBEATS, TiDE, DeepAR, PatchTST, TFT
pip install "tollama-eval[pdf]"          # PDF report export
pip install "tollama-eval[excel]"        # Excel workbook export
pip install "tollama-eval[server]"       # REST API server (FastAPI)
pip install "tollama-eval[distributed]"  # Ray distributed execution
pip install "tollama-eval[dashboard]"    # Streamlit interactive dashboard
pip install "tollama-eval[hierarchical]" # Hierarchical forecast reconciliation
```

---

## Input Formats

tollama-eval auto-detects your CSV format. No configuration needed.

**Long format** (recommended):

| unique_id | ds         | y    |
|-----------|------------|------|
| store_1   | 2020-01-01 | 42.0 |
| store_1   | 2020-01-02 | 45.0 |
| store_2   | 2020-01-01 | 10.0 |

**Wide format** (dates in first column, series as columns):

| date       | store_1 | store_2 |
|------------|---------|---------|
| 2020-01-01 | 42.0    | 10.0    |
| 2020-01-02 | 45.0    | 12.0    |

Timezone-aware timestamps are handled automatically (stripped to naive UTC). Extra numeric columns can be used as exogenous variables via `--exog-cols`.

---

## Models

tollama-eval ships with 36+ models spanning from classical baselines to foundation models.

### Core Statistical (5 models) — Built-in

| Model | Description |
|---|---|
| SeasonalNaive | Repeats last season (baseline) |
| AutoETS | Automated exponential smoothing |
| AutoARIMA | Automated ARIMA model selection |
| AutoTheta | Automated Theta method |
| CES (AutoCES) | Complex exponential smoothing |

<details>
<summary><strong>Extended Statistical (9 models) — Built-in</strong></summary>

| Model | Description |
|---|---|
| MSTL | Multiple seasonal-trend decomposition (LOESS) |
| DynamicOptimizedTheta | Dynamic theta variant |
| Holt | Double exponential smoothing (linear trend) |
| HoltWinters | Triple exponential smoothing (trend + seasonality) |
| HistoricAverage | Simple mean baseline |
| Naive | Repeat last observation |
| RandomWalkWithDrift | Random walk with trend |
| WindowAverage | Simple moving average |
| SeasonalWindowAverage | Seasonal moving average |

</details>

<details>
<summary><strong>Intermittent Demand (6 models) — Built-in</strong></summary>

| Model | Description |
|---|---|
| CrostonClassic | Classic Croston method |
| CrostonOptimized | Optimized Croston method |
| CrostonSBA | Syntetos-Boylan Approximation (bias-corrected) |
| ADIDA | Aggregate-Disaggregate Intermittent Demand |
| IMAPA | Intermittent Multiple Aggregation Prediction |
| TSB | Teunter-Syntetos-Babai method |

</details>

### ML Models (3 models) — Optional extras

| Model | Install |
|---|---|
| Prophet | `pip install "tollama-eval[prophet]"` |
| LightGBM | `pip install "tollama-eval[lightgbm]"` |
| XGBoost | `pip install "tollama-eval[xgboost]"` |

### Neural Models (6 models) — `pip install "tollama-eval[neural]"`

| Model | Description |
|---|---|
| NHITS | N-HiTS architecture |
| NBEATS | N-BEATS architecture |
| TiDE | Time-series Dense Encoder |
| DeepAR | Probabilistic autoregressive model |
| PatchTST | Patch Time Series Transformer |
| TFT | Temporal Fusion Transformer |

### Foundation Models (7 models) — via Tollama

Zero-shot forecasting through time series foundation models via the Tollama TSFM server:

Chronos-2, TimesFM, Moirai, Granite-TTM, Lag-Llama, PatchTST, TIDE

```bash
tollama-eval run -i data.csv --tollama-url http://localhost:8000 --tollama-models chronos2,timesfm
```

---

Run specific models with `-m`:

```bash
tollama-eval run -i data.csv -m SeasonalNaive,AutoETS,Prophet
```

Want everything? `pip install "tollama-eval[all]"`

---

## Evaluation

Models are ranked by **MASE** (Mean Absolute Scaled Error) — the gold-standard metric for time series comparison. A MASE of 1.0 means the model performs exactly like a seasonal naive baseline. Below 1.0 means it beats the baseline.

| Metric | What it measures | Interpretation |
|--------|-----------------|----------------|
| **MASE** | Accuracy relative to naive baseline | < 1.0 = better than naive |
| **SMAPE** | Symmetric percentage error | 0-200%, lower is better |
| **RMSSE** | Scaled RMSE | < 1.0 = better than naive |
| **MAE** | Raw absolute error | 0 = perfect |
| **MSIS** | Prediction interval quality | Lower = tighter, more accurate intervals |
| **Coverage** | Interval capture rate | Should match nominal level (e.g., 0.9) |

All metrics are computed per series, per fold, then aggregated — giving you honest, cross-validated scores instead of a single optimistic number.

Combine metrics with custom weights for composite scoring:

```bash
tollama-eval run -i data.csv --metric-weights 'mase=0.5,smape=0.3,speed=0.2'
```

---

## Advanced Features

### AutoML

Use `--auto-select` to let tollama-eval analyze your data profile (series length, frequency, intermittency) and automatically recommend the best subset of models. Intermittent demand patterns are detected using Syntetos-Boylan Classification.

### Anomaly Detection

Use `--detect-anomalies` to run 4 detection methods on your input data: Z-score, IQR, rolling statistics, and forecast residuals. Results are combined via a multi-detector ensemble with anomaly scoring.

### Ensemble Construction

After benchmarking, tollama-eval can construct ensembles using 3 strategies: simple average, inverse-MASE weighted average, and best-per-series selection. Per-series model recommendations show which model wins for each series.

### Stability Analysis

Models are assessed for consistency across CV folds and series. Fold CV, series CV, and rank consistency scores produce a composite stability score (0-1) used as a tiebreaker in the leaderboard.

### Speed Benchmarking

Per-model speed profiles track runtime, throughput (series/sec), and per-fold time. A Pareto frontier analysis identifies which models offer the best accuracy-speed tradeoff.

See [FEATURES.md](FEATURES.md) for full details on all features.

---

## CLI Reference

### `tollama-eval run`

```
tollama-eval run [OPTIONS]

Core:
  -i, --input PATH         Input CSV file (required)
  -o, --output PATH        Output directory (default: out/)
  -H, --horizon INT        Forecast horizon (default: 14)
  -k, --n-folds INT        Number of CV folds (default: 3)
  -m, --models TEXT         Comma-separated model list
  -j, --n-jobs INT         Parallel workers (default: 1)
  -c, --config PATH        YAML or JSON config file
  -v, --verbose            Show data profile and fold-level progress
  -q, --quiet              Suppress all output except errors
  -V, --version            Show version

Advanced:
  --pdf                    Generate PDF report (requires weasyprint)
  --excel                  Generate Excel workbook (requires openpyxl)
  --log-json               Emit structured JSON logs to stderr
  --no-cache               Disable result caching
  --cache-dir PATH         Custom cache directory
  --parallel-models        Run models in parallel processes
  --detect-anomalies       Run anomaly detection on input data
  --metric-weights TEXT    Custom metric weights (e.g. 'mase=0.5,smape=0.3,speed=0.2')
  --auto-select            Auto-select models based on data characteristics
  --include-optional       Safely enable optional non-core models when available
  --include-neural         Safely enable optional neural models when available
  --exog-cols TEXT          Comma-separated exogenous column names
  --distributed            Use Ray for distributed fold execution

Integration:
  --tollama-url URL        Tollama TSFM server URL
  --tollama-models TEXT    Comma-separated tollama model names
  --no-tollama             Disable tollama integration
```

### `tollama-eval campaign`

Benchmark across multiple CSV files in a directory. Outputs a `campaign_summary.csv` with per-dataset results.

```bash
tollama-eval campaign -d data_dir/ -o results/ -H 14 -k 3
```

### `tollama-eval doctor`

Run diagnostic checks on your environment. It verifies Python version, core/optional dependencies, output directory writability, and hardware acceleration via a safe probe.

```bash
tollama-eval doctor
```

### Dashboard

Launch the Streamlit dashboard to run new benchmarks or inspect saved artifacts:

```bash
streamlit run -m ts_autopilot.reporting.dashboard
streamlit run -m ts_autopilot.reporting.dashboard -- --artifact-dir out/
streamlit run -m ts_autopilot.reporting.dashboard -- --results out/results.json --details out/details.json
```

Requires `pip install "tollama-eval[dashboard]"`.

### `tollama-eval serve`

Start a REST API server for remote benchmarking.

```bash
tollama-eval serve --port 8000 --host 0.0.0.0 --output-dir out/server
```

Requires `pip install "tollama-eval[server]"`. Endpoints: `POST /benchmark`, `GET /status/{run_id}`, `GET /results/{run_id}`, `GET /health`.

Or run as a Python module:

```bash
python -m ts_autopilot run -i data.csv
```

---

## Config Files

For reproducible benchmarks, use a YAML or JSON config instead of CLI flags:

```yaml
# benchmark.yaml
input: data.csv
output: results/
horizon: 7
n_folds: 5
models:
  - SeasonalNaive
  - AutoETS
  - AutoARIMA
  - AutoCES
n_jobs: 4
parallel_models: true
model_timeout_sec: 300
report_title: "Q1 Forecast Benchmark"
```

```bash
tollama-eval run -c benchmark.yaml
```

CLI flags override config file values when both are provided.

---

## Python API

### Fluent SDK

The recommended way to use tollama-eval as a library:

```python
from ts_autopilot.sdk import TSAutopilot

result = (
    TSAutopilot(df)
    .with_models(["SeasonalNaive", "AutoETS", "AutoARIMA"])
    .with_horizon(14)
    .with_folds(3)
    .with_n_jobs(4)
    .run()
)

# Inspect the leaderboard
for entry in result.leaderboard:
    print(f"#{entry.rank} {entry.name}: MASE={entry.mean_mase:.4f}")
```

Auto-select models and save results to disk:

```python
output_dir = (
    TSAutopilot(df)
    .with_auto_select()
    .with_horizon(14)
    .save("out/")
)
```

### Pipeline Functions

For lower-level control:

```python
from ts_autopilot.pipeline import run_benchmark, run_from_csv

# One-liner from CSV
result = run_from_csv("data.csv", horizon=14, n_folds=3, output_dir="out/")

# From an existing DataFrame
result = run_benchmark(df, horizon=14, n_folds=3)

# Serialize / deserialize
json_str = result.to_json(indent=2)
loaded = BenchmarkResult.from_json(json_str)
```

### Custom Runners

Extend tollama-eval with your own models:

```python
from ts_autopilot.runners.base import BaseRunner
from ts_autopilot.contracts import ForecastOutput

class MyModelRunner(BaseRunner):
    @property
    def name(self) -> str:
        return "MyModel"

    def fit_predict(self, train, horizon, freq, season_length, n_jobs=1):
        # Your forecasting logic here
        return ForecastOutput(
            unique_id=[...], ds=[...], y_hat=[...],
            model_name=self.name, runtime_sec=0.0,
        )

# Plug it in
result = run_benchmark(df, horizon=14, n_folds=3, runners=[MyModelRunner()])
```

### Metrics API

```python
from ts_autopilot.evaluation.metrics import mase, smape, rmsse, mae
import numpy as np

y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_true  = np.array([11, 12, 13])
y_pred  = np.array([10.5, 12.2, 13.1])

print(f"MASE:  {mase(y_true, y_pred, y_train):.4f}")
print(f"SMAPE: {smape(y_true, y_pred):.2f}%")
print(f"RMSSE: {rmsse(y_true, y_pred, y_train):.4f}")
print(f"MAE:   {mae(y_true, y_pred):.4f}")
```

### REST API

Start a server and submit benchmarks programmatically:

```bash
tollama-eval serve --port 8000
```

```python
import httpx

# Submit a benchmark
with open("data.csv", "rb") as f:
    resp = httpx.post("http://localhost:8000/benchmark", files={"file": f})
run_id = resp.json()["run_id"]

# Check status
status = httpx.get(f"http://localhost:8000/status/{run_id}").json()

# Get results
results = httpx.get(f"http://localhost:8000/results/{run_id}").json()
```

---

## Output Schema

The `results.json` schema is **frozen** — field names will never change across versions, so it's safe to build pipelines and dashboards on top of it.

Output files:
- **`results.json`** — Machine-readable benchmark results
- **`details.json`** — Forecast data, diagnostics, data characteristics, and optional-model environment context
- **`report.html`** — Interactive visual report (Plotly charts)
- **`leaderboard.csv`** — Ranked model summary
- **`fold_details.csv`** — Fold-level metrics by model
- **`per_series_scores.csv`** — Per-series error table
- **`per_series_winners.csv`** — Per-series winner summary
- **`report.pdf`** — PDF export (optional, with `--pdf`)
- **`report.xlsx`** — Excel export (optional, with `--excel`)

```json
{
  "metadata": {
    "version": "0.2.0",
    "generated_at": "2024-01-15T10:30:00+00:00",
    "total_runtime_sec": 12.34
  },
  "profile": {
    "n_series": 10,
    "frequency": "D",
    "missing_ratio": 0.0,
    "season_length_guess": 7,
    "min_length": 365,
    "max_length": 730,
    "total_rows": 5000
  },
  "config": { "horizon": 14, "n_folds": 3 },
  "models": [
    {
      "name": "AutoETS",
      "runtime_sec": 3.21,
      "folds": [
        { "fold": 1, "cutoff": "2023-12-17", "mase": 0.85, "series_scores": {} }
      ],
      "mean_mase": 0.85,
      "std_mase": 0.02,
      "mean_smape": 12.3,
      "mean_rmsse": 0.78,
      "mean_mae": 2.15
    }
  ],
  "leaderboard": [
    { "rank": 1, "name": "AutoETS", "mean_mase": 0.85 }
  ]
}
```

---

## Roadmap

| Milestone | Status | Highlights |
|-----------|--------|------------|
| **v0.1 — MVP** | Done | SeasonalNaive + AutoETS, CLI, cross-validation, results.json |
| **v0.2 — Beta** | Done | 36+ models (statistical + intermittent + ML + neural + foundation), AutoML, anomaly detection, ensemble construction, Tollama TSFM integration, campaign mode, fluent SDK, REST API server, Ray distributed execution, PDF reports, residual diagnostics, stability analysis, speed benchmarking, YAML config, interactive HTML reports with Plotly |
| **v0.3 — Intelligence** | Planned | LLM-powered result narratives, hierarchical reconciliation GA, confidence band visualization |
| **v0.4 — Scale** | Planned | Export to PowerPoint/Excel/Jupyter, Streamlit dashboard GA, custom model plugin ecosystem |

---

## Architecture

```
src/ts_autopilot/
├── cli.py              # Typer CLI (run, campaign, doctor, serve)
├── pipeline.py         # Benchmark orchestrator
├── sdk.py              # Fluent builder API (TSAutopilot class)
├── contracts.py        # Frozen data contracts
├── config.py           # YAML/JSON config loading
├── cache.py            # Result caching
├── events.py           # Structured event emission
├── logging_config.py   # Logging setup (text + JSON)
├── tracing.py          # OpenTelemetry integration
├── ingestion/          # CSV loading + data profiling
├── evaluation/         # Metrics, CV, ensemble, stability, speed
├── runners/            # Model runners (statistical, optional, tollama)
├── anomaly/            # Anomaly detection (Z-score, IQR, rolling, residual)
├── automl/             # Intelligent model selection + tuning
├── reporting/          # HTML, PDF, executive summary, dashboard, export
├── tollama/            # Foundation model integration (TSFM)
├── server/             # FastAPI REST server
├── distributed/        # Ray distributed execution
└── hierarchy/          # Hierarchical forecast reconciliation
```

**Design principles:**
- CLI is a thin shell — all logic lives in the pipeline
- Canonical data format (`unique_id`, `ds`, `y`) enforced everywhere
- Output schema is frozen for version stability
- Optional dependencies are gracefully degraded (missing models are skipped, not errors)
- Enterprise-grade reliability — input validation, per-model timeouts, signal handling, structured logging

---

## Development

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Check environment
tollama-eval doctor
```

Powered by the [Nixtla](https://github.com/nixtla) ecosystem — statsforecast, mlforecast, and neuralforecast.

---

## License

MIT
