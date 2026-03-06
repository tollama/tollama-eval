"""Benchmark pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.evaluation.cross_validation import make_expanding_splits
from ts_autopilot.evaluation.metrics import mean_mase_per_fold
from ts_autopilot.ingestion.loader import load_csv
from ts_autopilot.ingestion.profiler import profile_dataframe
from ts_autopilot.runners.base import BaseRunner
from ts_autopilot.runners.statistical import AutoETSRunner, SeasonalNaiveRunner

DEFAULT_RUNNERS: list[BaseRunner] = [
    SeasonalNaiveRunner(),
    AutoETSRunner(),
]


def generate_warnings(
    profile: DataProfile, horizon: int, n_folds: int
) -> list[str]:
    """Produce user-facing warnings based on data profile and config."""
    warnings: list[str] = []

    min_needed = (n_folds + 2) * horizon
    if profile.min_length < min_needed:
        warnings.append(
            f"Shortest series has {profile.min_length} rows but "
            f"{min_needed} are recommended for horizon={horizon}, "
            f"n_folds={n_folds}. Results may be unreliable."
        )

    if profile.missing_ratio > 0.1:
        warnings.append(
            f"Missing ratio is {profile.missing_ratio:.1%}. "
            "Consider imputing or removing series with gaps."
        )

    if profile.n_series == 1:
        warnings.append(
            "Only 1 series found. Cross-series patterns cannot be leveraged."
        )

    return warnings


def run_benchmark(
    df: pd.DataFrame,
    horizon: int,
    n_folds: int,
    runners: list[BaseRunner] | None = None,
    model_names: list[str] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> BenchmarkResult:
    """Run the full benchmark pipeline (pure logic, no I/O).

    Steps:
      1. Profile the DataFrame
      2. Make CV splits
      3. For each runner x fold: fit_predict → compute MASE
      4. Aggregate into ModelResult + leaderboard
      5. Return BenchmarkResult
    """
    if runners is None:
        runners = list(DEFAULT_RUNNERS)

    if model_names is not None:
        available = {r.name for r in runners}
        unknown = set(model_names) - available
        if unknown:
            raise ValueError(
                f"Unknown model(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(available))}"
            )
        runners = [r for r in runners if r.name in model_names]

    profile = profile_dataframe(df)
    config = BenchmarkConfig(horizon=horizon, n_folds=n_folds)
    warnings = generate_warnings(profile, horizon, n_folds)
    splits = make_expanding_splits(df, horizon=horizon, n_folds=n_folds)

    model_results: list[ModelResult] = []

    for runner_idx, runner in enumerate(runners):
        if progress_callback is not None:
            progress_callback("model", runner_idx + 1, len(runners))

        fold_results: list[FoldResult] = []
        total_runtime = 0.0

        for fold_idx, split in enumerate(splits):
            if progress_callback is not None:
                progress_callback("fold", fold_idx + 1, len(splits))

            output = runner.fit_predict(
                train=split.train,
                horizon=horizon,
                freq=profile.frequency,
                season_length=profile.season_length_guess,
            )
            total_runtime += output.runtime_sec

            # Convert ForecastOutput to DataFrame for metric computation
            pred_df = pd.DataFrame(
                {
                    "unique_id": output.unique_id,
                    "ds": pd.to_datetime(output.ds),
                    output.model_name: output.y_hat,
                }
            )

            fold_mase = mean_mase_per_fold(
                forecast_df=pred_df,
                actuals_df=split.test,
                train_df=split.train,
                season_length=profile.season_length_guess,
                model_col=output.model_name,
            )
            fold_results.append(
                FoldResult(
                    fold=split.fold,
                    cutoff=split.cutoff.isoformat(),
                    mase=round(fold_mase, 6),
                )
            )

        mase_values = [f.mase for f in fold_results]
        model_results.append(
            ModelResult(
                name=runner.name,
                runtime_sec=round(total_runtime, 4),
                folds=fold_results,
                mean_mase=round(float(np.mean(mase_values)), 6),
                std_mase=round(float(np.std(mase_values)), 6),
            )
        )

    # Build leaderboard: rank by mean_mase ascending
    sorted_models = sorted(model_results, key=lambda m: m.mean_mase)
    leaderboard = [
        LeaderboardEntry(rank=i + 1, name=m.name, mean_mase=m.mean_mase)
        for i, m in enumerate(sorted_models)
    ]

    return BenchmarkResult(
        profile=profile,
        config=config,
        models=model_results,
        leaderboard=leaderboard,
        warnings=warnings,
    )


def run_from_csv(
    csv_path: str | Path,
    horizon: int,
    n_folds: int,
    output_dir: str | Path,
    model_names: list[str] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> BenchmarkResult:
    """Full end-to-end pipeline: CSV → results.json + report.html."""
    from ts_autopilot.reporting.html_report import render_report

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    result = run_benchmark(
        df,
        horizon=horizon,
        n_folds=n_folds,
        model_names=model_names,
        progress_callback=progress_callback,
    )

    # Write results.json
    results_path = output_dir / "results.json"
    results_path.write_text(result.to_json(indent=2))

    # Write report.html
    report_path = output_dir / "report.html"
    report_path.write_text(render_report(result))

    return result
