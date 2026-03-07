# ts-autopilot

Automated time series benchmarking. Drop in a CSV, get ranked model results with cross-validated MASE scores.

## Quick start

```bash
pip install -e .
ts-autopilot run -i data.csv -o results/ -H 14 -k 3
```

## Input formats

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

## CLI options

```
ts-autopilot run [OPTIONS]

  -i, --input PATH       Input CSV file (required)
  -o, --output PATH      Output directory (default: out/)
  -H, --horizon INT      Forecast horizon (default: 14)
  -k, --n-folds INT      CV folds (default: 3)
  -m, --models TEXT       Comma-separated model list (e.g. SeasonalNaive,AutoETS)
  -v, --verbose           Show data profile and fold-level progress
  -q, --quiet             Suppress all output except errors
  -V, --version           Show version
```

You can also run as a Python module:

```bash
python -m ts_autopilot
```

## Output

Results are written to the output directory:

- **results.json** -- structured benchmark results (frozen schema)
- **report.html** -- visual report with leaderboard and fold details

## Models

| Model         | Type        | Description                     |
|---------------|-------------|---------------------------------|
| SeasonalNaive | Baseline    | Repeats last season             |
| AutoETS       | Statistical | Automated exponential smoothing |

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest tests/ -v
```

## License

MIT
