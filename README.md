<p align="center">
  <h1 align="center">ts-autopilot</h1>
  <p align="center">
    <strong>Stop guessing which forecasting model to use. Let the data decide.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/ts-autopilot/"><img src="https://img.shields.io/pypi/v/ts-autopilot.svg" alt="PyPI version"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://github.com/ychoi-atop/ts-autopilot/actions"><img src="https://img.shields.io/github/actions/workflow/status/ychoi-atop/ts-autopilot/ci.yml?label=CI" alt="CI"></a>
  </p>
</p>

---

**ts-autopilot** is a zero-config time series benchmarking tool. Drop in a CSV, get a ranked leaderboard of forecasting models — complete with cross-validated metrics and a visual report. No boilerplate. No notebook spaghetti. One command.

```bash
pip install ts-autopilot
ts-autopilot run -i sales.csv
```

That's it. Open `out/report.html` and see which model wins.

---

## Why ts-autopilot?

Forecasting teams waste weeks on the same loop: load data, wrangle formats, fit models one at a time, compute metrics manually, copy results into slides. **ts-autopilot collapses that entire workflow into a single command.**

| The old way | With ts-autopilot |
|---|---|
| Write custom ingestion for every dataset | Auto-detects long & wide CSV formats |
| Manually split train/test, hope it's fair | Expanding-window cross-validation, configurable folds |
| Fit models one at a time in notebooks | 9 models run automatically, from naive baselines to deep learning |
| Compute MAPE and call it a day | 4 metrics (MASE, SMAPE, RMSSE, MAE) with per-series breakdowns |
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
pip install ts-autopilot
```

### 2. Run

```bash
ts-autopilot run -i your_data.csv
```

### 3. Read the results

Two files appear in `out/`:

- **`results.json`** — Machine-readable benchmark results (schema is frozen across versions, safe to build on)
- **`report.html`** — Interactive visual report with leaderboard, per-fold breakdown, and per-series scores

---

## Input Formats

ts-autopilot auto-detects your CSV format. No configuration needed.

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

Timezone-aware timestamps are handled automatically (stripped to naive UTC).

---

## Models

ts-autopilot ships with 5 built-in models and supports 4 optional ones — spanning from classical baselines to neural networks.

| Model         | Type        | Description                         | Install |
|---------------|-------------|-------------------------------------|---------|
| SeasonalNaive | Baseline    | Repeats last season                 | Built-in |
| AutoETS       | Statistical | Automated exponential smoothing     | Built-in |
| AutoARIMA     | Statistical | Automated ARIMA model selection     | Built-in |
| AutoTheta     | Statistical | Automated Theta method              | Built-in |
| AutoCES       | Statistical | Complex exponential smoothing       | Built-in |
| Prophet       | Statistical | Facebook Prophet                    | `pip install "ts-autopilot[prophet]"` |
| LightGBM      | ML          | Gradient boosting via mlforecast    | `pip install "ts-autopilot[lightgbm]"` |
| NHITS         | Neural      | N-HiTS deep learning architecture   | `pip install "ts-autopilot[neural]"` |
| NBEATS        | Neural      | N-BEATS deep learning architecture  | `pip install "ts-autopilot[neural]"` |

Want everything? `pip install "ts-autopilot[all]"`

Run specific models with `-m`:

```bash
ts-autopilot run -i data.csv -m SeasonalNaive,AutoETS,Prophet
```

---

## Evaluation

Models are ranked by **MASE** (Mean Absolute Scaled Error) — the gold-standard metric for time series comparison. A MASE of 1.0 means the model performs exactly like a seasonal naive baseline. Below 1.0 means it beats the baseline.

| Metric | What it measures | Interpretation |
|--------|-----------------|----------------|
| **MASE** | Accuracy relative to naive baseline | < 1.0 = better than naive |
| **SMAPE** | Symmetric percentage error | 0-200%, lower is better |
| **RMSSE** | Scaled RMSE | < 1.0 = better than naive |
| **MAE** | Raw absolute error | 0 = perfect |

All metrics are computed per series, per fold, then aggregated — giving you honest, cross-validated scores instead of a single optimistic number.

---

## CLI Reference

```
ts-autopilot run [OPTIONS]

Options:
  -i, --input PATH       Input CSV file (required)
  -o, --output PATH      Output directory (default: out/)
  -H, --horizon INT      Forecast horizon (default: 14)
  -k, --n-folds INT      Number of CV folds (default: 3)
  -m, --models TEXT       Comma-separated model list
  -j, --n-jobs INT       Parallel workers (default: 1)
  -c, --config PATH      YAML or JSON config file
  -v, --verbose          Show data profile and fold-level progress
  -q, --quiet            Suppress all output except errors
  -V, --version          Show version
```

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
```

```bash
ts-autopilot run -c benchmark.yaml
```

CLI flags override config file values when both are provided.

---

## Python API

Use ts-autopilot as a library for tighter integration with your workflows.

```python
from ts_autopilot.pipeline import run_benchmark, run_from_csv

# One-liner from CSV
result = run_from_csv("data.csv", horizon=14, n_folds=3, output_dir="out/")

# From an existing DataFrame
result = run_benchmark(df, horizon=14, n_folds=3)

# Inspect the leaderboard
for entry in result.leaderboard:
    print(f"#{entry.rank} {entry.name}: MASE={entry.mean_mase:.4f}")

# Serialize / deserialize
json_str = result.to_json(indent=2)
loaded = BenchmarkResult.from_json(json_str)
```

### Custom Runners

Extend ts-autopilot with your own models:

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

---

## Output Schema

The `results.json` schema is **frozen** — field names will never change across versions, so it's safe to build pipelines and dashboards on top of it.

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
      "std_mase": 0.02
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
| **v0.2 — Beta** | Done | 9 models (statistical + ML + neural), YAML config, HTML reports, per-series breakdown, parallel execution |
| **v0.3 — Intelligence** | Planned | LLM-powered result interpretation via Tollama, natural-language summaries in reports |
| **v0.4 — Scale** | Planned | Multi-dataset campaigns, ensemble recommendations, export to dashboard formats |

---

## Architecture

```
src/ts_autopilot/
├── cli.py              # Typer CLI — calls pipeline, zero embedded logic
├── pipeline.py         # Benchmark orchestrator
├── contracts.py        # Frozen data contracts
├── config.py           # YAML/JSON config loading
├── ingestion/          # CSV loading + data profiling
├── evaluation/         # Metrics + cross-validation
├── runners/            # Model runners (base + statistical + optional)
├── reporting/          # HTML report generation (Jinja2)
└── tollama/            # LLM interpretation client
```

**Design principles:**
- CLI is a thin shell — all logic lives in the pipeline
- Canonical data format (`unique_id`, `ds`, `y`) enforced everywhere
- Output schema is frozen for version stability
- Optional dependencies are gracefully degraded (missing models are skipped, not errors)

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
```

Powered by the [Nixtla](https://github.com/nixtla) ecosystem — statsforecast, mlforecast, and neuralforecast.

---

## License

MIT
