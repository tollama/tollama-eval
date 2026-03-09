# Output Interpretation

After a run, `ts-autopilot` writes files to your output directory (default: `out/`).

## Output Files

- `results.json`: core benchmark result object (stable schema)
- `details.json`: extra forecast and diagnostics data used for report reproducibility
- `report.html`: interactive report for human review
- `report.pdf`: optional PDF export when `--pdf` is enabled
- `report.xlsx`: optional Excel workbook when `--excel` is enabled

## How to Read `results.json`

The core top-level sections are:

1. `profile`
- Data profile summary: number of series, inferred frequency, missing ratio, length stats

2. `config`
- Run configuration used for the benchmark:
- `horizon`
- `n_folds`

3. `models`
- Per-model details:
- `name`
- `runtime_sec`
- `folds` (per-fold metrics and cutoff timestamps)
- `mean_mase`
- `std_mase`
- Optional companion metrics like `mean_smape`, `mean_rmsse`, `mean_mae`

4. `leaderboard`
- Ranked model summary:
- `rank`
- `name`
- `mean_mase` (lower is better)

## Metric Interpretation

- `MASE`: primary ranking metric
- `< 1.0` means better than seasonal naive baseline
- `~1.0` means similar to baseline
- `> 1.0` means worse than baseline

- `SMAPE`: percentage-like error, lower is better
- `RMSSE`: scaled RMSE, lower is better
- `MAE`: absolute error in target units

## Fold-Level Interpretation

Each model includes per-fold metrics. Use folds to assess robustness:

- Tight fold scores: stable model behavior
- Wide fold spread: model is sensitive to cutoff period
- Very low `std_mase`: consistent performance

## Runtime Interpretation

Use `runtime_sec` to evaluate speed/accuracy tradeoffs:

- If two models have similar MASE, prefer lower runtime in production workflows
- If one model is more accurate but much slower, decide based on SLA and retrain cadence

## Report Interpretation (`report.html`)

The HTML report is best for quick model selection:

1. Start with leaderboard and runtime table
2. Check fold heatmap for consistency
3. Review per-series errors (box plots, distributions)
4. If available, review significance and critical difference sections
5. Validate diagnostics for the winner model before deployment

## NaN Values in Results

If you see NaNs in forecasts or aggregate metrics:

- A model may have failed on one or more folds/series
- Optional model dependencies may be missing
- Tollama requests may have failed and used NaN fallback

Investigate with:

```bash
ts-autopilot run -i your_data.csv -v --no-cache
```

## Choosing a Winner

Use this order:

1. Lowest `mean_mase` on the leaderboard
2. Acceptable fold stability (`std_mase`, fold spread)
3. Runtime constraints (`runtime_sec`)
4. Business constraints (interpretability, deployability, dependency policy)
