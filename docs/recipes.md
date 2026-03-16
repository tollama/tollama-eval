# Recipes

Practical command patterns for common `tollama-eval` workflows.

## 1) Fast Baseline Run

Use this when you want a quick benchmark with defaults.

```bash
tollama-eval run -i data.csv
```

Output defaults to `out/` with `results.json`, `details.json`, `report.html`, `leaderboard.csv`, `fold_details.csv`, `per_series_scores.csv`, and `per_series_winners.csv`.

## 2) Run Specific Models Only

Use this when you want a controlled comparison.

```bash
tollama-eval run -i data.csv -m SeasonalNaive,AutoETS,AutoARIMA
```

## 3) Increase Forecast Horizon and Folds

Use this when you need stronger validation.

```bash
tollama-eval run -i data.csv -H 28 -k 5
```

## 4) Use Exogenous Variables

Use this when your CSV includes extra numeric predictors.

```bash
tollama-eval run -i data_with_exog.csv --exog-cols promo,price,holiday_flag
```

## 5) Auto-Select Models from Data Profile

Use this when you want a smaller, data-aware model subset.

```bash
tollama-eval run -i data.csv --auto-select
```

## 6) Add Composite Scoring

Use this when you want ranking to include speed and multiple metrics.

```bash
tollama-eval run -i data.csv --metric-weights "mase=0.5,smape=0.3,speed=0.2"
```

## 7) Generate PDF and Excel Reports

Install extras first:

```bash
pip install "tollama-eval[pdf,excel]"
```

Then run:

```bash
tollama-eval run -i data.csv --pdf --excel
```

## 8) Include Optional Models Safely

Use this when you want to opt in to optional extras without breaking core runs in partially installed environments.

```bash
tollama-eval run -i data.csv --include-optional
tollama-eval run -i data.csv --include-neural
```

The CLI prints which optional groups were enabled or skipped and why.

## 9) Run Tollama TSFM Models (Remote URL)

Use this when your Tollama server is reachable via a non-private URL.

```bash
tollama-eval run -i data.csv \
  --tollama-url https://your-tollama.example.com \
  --tollama-models chronos2,timesfm,moirai
```

## 10) Run Tollama TSFM Models (Local/Private URL)

Private/local URLs are blocked by default. Use config with `allow_private_urls: true`.

```yaml
# local_tollama.yaml
input: data.csv
output: out/
horizon: 14
n_folds: 3
tollama_url: http://127.0.0.1:8000
tollama_models:
  - chronos2
  - timesfm
allow_private_urls: true
```

```bash
tollama-eval run -c local_tollama.yaml
```

## 11) Benchmark a Directory of CSVs

Use campaign mode for many datasets.

```bash
tollama-eval campaign -d datasets/ -o campaign_out/ -H 14 -k 3
```

## 12) Run in Distributed Mode

Install distributed extra:

```bash
pip install "tollama-eval[distributed]"
```

Run:

```bash
tollama-eval run -i data.csv --distributed
```

If Ray is unavailable or cannot initialize in the current environment, the command falls back to the standard local execution path.

## 13) Open the Dashboard on Saved Results

Install the dashboard extra:

```bash
pip install "tollama-eval[dashboard]"
```

Then open a previously generated artifact directory:

```bash
streamlit run -m ts_autopilot.reporting.dashboard -- --artifact-dir out/
```

Or open specific files:

```bash
streamlit run -m ts_autopilot.reporting.dashboard -- --results out/results.json --details out/details.json
```

## 14) Reproducible Config-Based Runs

Use a config file instead of long CLI commands:

```yaml
# benchmark.yaml
input: data.csv
output: out_q1/
horizon: 14
n_folds: 3
models:
  - SeasonalNaive
  - AutoETS
  - AutoARIMA
n_jobs: 4
parallel_models: true
model_timeout_sec: 300
```

```bash
tollama-eval run -c benchmark.yaml
```

## 15) Python API One-Liner

```python
from ts_autopilot.pipeline import run_from_csv

result = run_from_csv(
    csv_path="data.csv",
    horizon=14,
    n_folds=3,
    output_dir="out/",
)
print(result.leaderboard[0].name, result.leaderboard[0].mean_mase)
```
