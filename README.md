# ts-autopilot

[![PyPI version](https://img.shields.io/pypi/v/ts-autopilot.svg)](https://pypi.org/project/ts-autopilot/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Automated time series benchmarking powered by the [Nixtla](https://github.com/nixtla) ecosystem (statsforecast, mlforecast, neuralforecast). Drop in a CSV, get ranked model results with cross-validated metrics.

## Quick Start

```bash
pip install ts-autopilot
ts-autopilot run -i data.csv -o results/
```

That's it. You get:
- **results.json** — structured benchmark results (frozen schema)
- **report.html** — visual report with leaderboard, fold details, and per-series breakdown

## Installation

```bash
# Basic install
pip install ts-autopilot

# With optional models
pip install "ts-autopilot[prophet]"    # Facebook Prophet
pip install "ts-autopilot[lightgbm]"   # LightGBM via mlforecast
pip install "ts-autopilot[neural]"     # NHITS + NBEATS via neuralforecast
pip install "ts-autopilot[all]"        # Everything

# Development
pip install -e ".[dev]"
```

## Input Formats

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

Both formats are auto-detected. Timezone-aware timestamps are stripped to naive UTC.

## CLI Reference

```
ts-autopilot run [OPTIONS]

Options:
  -i, --input PATH       Input CSV file (required)
  -o, --output PATH      Output directory (default: out/)
  -H, --horizon INT      Forecast horizon (default: 14)
  -k, --n-folds INT      CV folds (default: 3)
  -m, --models TEXT       Comma-separated model list
  -j, --n-jobs INT       Parallel workers (default: 1)
  -c, --config PATH      YAML or JSON config file
  -v, --verbose          Show data profile and fold-level progress
  -q, --quiet            Suppress all output except errors
  -V, --version          Show version
  --tollama-url URL      Tollama LLM interpretation service URL
  --no-tollama           Disable tollama even if URL is set
```

You can also run as a Python module:

```bash
python -m ts_autopilot run -i data.csv
```

## Models

| Model         | Type        | Description                         | Dependency       |
|---------------|-------------|-------------------------------------|------------------|
| SeasonalNaive | Baseline    | Repeats last season                 | Built-in         |
| AutoETS       | Statistical | Automated exponential smoothing     | Built-in         |
| AutoARIMA     | Statistical | Automated ARIMA model selection     | Built-in         |
| AutoTheta     | Statistical | Automated Theta method              | Built-in         |
| AutoCES       | Statistical | Complex exponential smoothing       | Built-in         |
| Prophet       | Statistical | Facebook Prophet                    | `ts-autopilot[prophet]`  |
| LightGBM      | ML          | Gradient boosting via mlforecast    | `ts-autopilot[lightgbm]` |
| NHITS         | Neural      | N-HiTS deep learning model          | `ts-autopilot[neural]`   |
| NBEATS        | Neural      | N-BEATS deep learning model         | `ts-autopilot[neural]`   |

Run specific models with `-m`:

```bash
ts-autopilot run -i data.csv -m SeasonalNaive,AutoETS,AutoCES
```

## Evaluation Metrics

| Metric | Description                          | Range          |
|--------|--------------------------------------|----------------|
| MASE   | Mean Absolute Scaled Error           | 0 = perfect, 1.0 = naive baseline |
| SMAPE  | Symmetric Mean Absolute % Error      | 0-200%         |
| RMSSE  | Root Mean Squared Scaled Error       | 0 = perfect, 1.0 = naive baseline |
| MAE    | Mean Absolute Error                  | 0 = perfect    |

The leaderboard ranks models by MASE (lower is better). A MASE < 1.0 means the model beats the seasonal naive baseline.

## Config Files

Instead of CLI flags, use a YAML or JSON config:

```yaml
# config.yaml
input: data.csv
output: results/
horizon: 7
n_folds: 5
models:
  - SeasonalNaive
  - AutoETS
  - AutoCES
n_jobs: 4
```

```bash
ts-autopilot run -c config.yaml
```

CLI flags override config file values.

## Python API

```python
from ts_autopilot.pipeline import run_benchmark, run_from_csv

# From CSV file
result = run_from_csv("data.csv", horizon=14, n_folds=3, output_dir="out/")

# From DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
result = run_benchmark(df, horizon=14, n_folds=3)

# Access results
for entry in result.leaderboard:
    print(f"#{entry.rank} {entry.name}: MASE={entry.mean_mase:.4f}")

# Use specific models
result = run_benchmark(df, horizon=14, n_folds=3, model_names=["SeasonalNaive", "AutoETS"])

# Serialize
json_str = result.to_json(indent=2)
loaded = BenchmarkResult.from_json(json_str)
```

### Custom Runners

```python
from ts_autopilot.runners.base import BaseRunner, StatsForecastRunner
from ts_autopilot.contracts import ForecastOutput

# Option 1: Wrap a StatsForecast model
class MyModelRunner(StatsForecastRunner):
    @property
    def name(self) -> str:
        return "MyModel"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import Naive
        return Naive()

# Option 2: Fully custom runner
class CustomRunner(BaseRunner):
    @property
    def name(self) -> str:
        return "Custom"

    def fit_predict(self, train, horizon, freq, season_length, n_jobs=1):
        # Your forecasting logic here
        return ForecastOutput(
            unique_id=[...], ds=[...], y_hat=[...],
            model_name=self.name, runtime_sec=0.0,
        )

# Use custom runners
result = run_benchmark(df, horizon=14, n_folds=3, runners=[MyModelRunner(), CustomRunner()])
```

### Metrics API

```python
from ts_autopilot.evaluation.metrics import mase, smape, rmsse, mae
import numpy as np

y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_true = np.array([11, 12, 13])
y_pred = np.array([10.5, 12.2, 13.1])

print(f"MASE:  {mase(y_true, y_pred, y_train):.4f}")
print(f"SMAPE: {smape(y_true, y_pred):.2f}%")
print(f"RMSSE: {rmsse(y_true, y_pred, y_train):.4f}")
print(f"MAE:   {mae(y_true, y_pred):.4f}")
```

## Output Schema

The `results.json` schema is frozen -- field names will not change across versions.

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

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Architecture

```
src/ts_autopilot/
├── cli.py              # Typer CLI (calls pipeline, no logic)
├── pipeline.py         # Benchmark orchestrator
├── contracts.py        # Frozen data contracts
├── config.py           # YAML/JSON config loading
├── ingestion/          # CSV loading + profiling
├── evaluation/         # Metrics + cross-validation
├── runners/            # Model runners (base + statistical + optional)
├── reporting/          # HTML report generation
└── tollama/            # LLM interpretation client
```

## License

MIT
