"""Distributed benchmark execution via Ray.

Distributes CV folds across Ray workers for parallel execution.

Requires: ``pip install "tollama-eval[distributed]"``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ts_autopilot.logging_config import get_logger

if TYPE_CHECKING:
    import pandas as pd

    from ts_autopilot.contracts import BenchmarkResult
    from ts_autopilot.runners.base import BaseRunner

logger = get_logger("distributed")

try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def is_available() -> bool:
    """Check if Ray is installed and importable."""
    return HAS_RAY


def run_benchmark_distributed(
    df: Any,
    horizon: int,
    n_folds: int,
    runners: list[BaseRunner] | None = None,
    model_names: list[str] | None = None,
    n_jobs: int = 1,
    ray_address: str | None = None,
    num_cpus: int | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Run benchmark with distributed fold execution via Ray.

    Falls back to local execution if Ray is not available.

    Args:
        df: Canonical long-format DataFrame.
        horizon: Forecast horizon.
        n_folds: Number of CV folds.
        runners: List of model runners.
        model_names: Optional model name filter.
        n_jobs: Per-model parallelism.
        ray_address: Ray cluster address (None for local).
        num_cpus: Number of CPUs for Ray local cluster.
        **kwargs: Additional arguments passed to run_benchmark.

    Returns:
        BenchmarkResult.
    """
    if not HAS_RAY:
        logger.warning(
            "Ray not installed, falling back to local execution. "
            'Install with: pip install "tollama-eval[distributed]"'
        )
        from ts_autopilot.pipeline import run_benchmark

        return run_benchmark(
            df=df,
            horizon=horizon,
            n_folds=n_folds,
            runners=runners,
            model_names=model_names,
            n_jobs=n_jobs,
            **kwargs,
        )

    # Initialize Ray
    if not ray.is_initialized():
        init_kwargs: dict[str, Any] = {}
        if ray_address:
            init_kwargs["address"] = ray_address
        if num_cpus:
            init_kwargs["num_cpus"] = num_cpus
        ray.init(**init_kwargs)
        logger.info("Ray initialized: %s", ray.cluster_resources())

    return _run_distributed(
        df=df,
        horizon=horizon,
        n_folds=n_folds,
        runners=runners,
        model_names=model_names,
        n_jobs=n_jobs,
        **kwargs,
    )


def _run_distributed(
    df: Any,
    horizon: int,
    n_folds: int,
    runners: list[BaseRunner] | None = None,
    model_names: list[str] | None = None,
    n_jobs: int = 1,
    **kwargs: Any,
) -> BenchmarkResult:
    """Internal: distribute folds across Ray workers.

    Strategy: distribute (model, fold) pairs as Ray remote tasks,
    then collect and aggregate locally.
    """
    import numpy as np
    import pandas as pd

    from ts_autopilot.contracts import (
        BenchmarkConfig,
        BenchmarkResult,
        FoldResult,
        LeaderboardEntry,
        ModelResult,
    )
    from ts_autopilot.evaluation.cross_validation import make_expanding_splits
    from ts_autopilot.evaluation.metrics import (
        per_series_mae,
        per_series_mase,
        per_series_rmsse,
        per_series_smape,
    )
    from ts_autopilot.ingestion.profiler import compute_data_characteristics
    from ts_autopilot.pipeline import (
        DEFAULT_RUNNERS,
        generate_warnings,
        profile_dataframe,
    )

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
    data_chars = compute_data_characteristics(df, profile.season_length_guess)
    splits = make_expanding_splits(df, horizon=horizon, n_folds=n_folds)
    warnings = generate_warnings(profile, horizon, n_folds)

    @ray.remote
    def _run_fold(
        runner: BaseRunner,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold_idx: int,
        cutoff: str,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int,
    ) -> dict:
        """Run a single (runner, fold) pair."""
        import time

        t0 = time.perf_counter()
        output = runner.fit_predict(
            train=train_data,
            horizon=horizon,
            freq=freq,
            season_length=season_length,
            n_jobs=n_jobs,
        )
        elapsed = time.perf_counter() - t0

        pred_df = pd.DataFrame(
            {
                "unique_id": output.unique_id,
                "ds": pd.to_datetime(output.ds),
                output.model_name: output.y_hat,
            }
        )

        series_scores = per_series_mase(
            forecast_df=pred_df,
            actuals_df=test_data,
            train_df=train_data,
            season_length=season_length,
            model_col=output.model_name,
        )
        fold_mase = float(np.mean(list(series_scores.values())))

        smape_scores = per_series_smape(
            forecast_df=pred_df,
            actuals_df=test_data,
            model_col=output.model_name,
        )
        fold_smape = float(np.mean(list(smape_scores.values())))

        rmsse_scores = per_series_rmsse(
            forecast_df=pred_df,
            actuals_df=test_data,
            train_df=train_data,
            season_length=season_length,
            model_col=output.model_name,
        )
        fold_rmsse = float(np.mean(list(rmsse_scores.values())))

        mae_scores = per_series_mae(
            forecast_df=pred_df,
            actuals_df=test_data,
            model_col=output.model_name,
        )
        fold_mae = float(np.mean(list(mae_scores.values())))

        return {
            "runner_name": runner.name,
            "fold": fold_idx + 1,
            "cutoff": cutoff,
            "mase": round(fold_mase, 6),
            "smape": round(fold_smape, 4),
            "rmsse": round(fold_rmsse, 6),
            "mae": round(fold_mae, 6),
            "series_scores": {k: round(v, 6) for k, v in series_scores.items()},
            "runtime": round(elapsed, 4),
        }

    # Submit all (runner, fold) pairs to Ray
    futures = []
    for runner in runners:
        for fold_idx, split in enumerate(splits):
            futures.append(
                _run_fold.remote(
                    runner,
                    split.train,
                    split.test,
                    fold_idx,
                    split.cutoff.isoformat(),
                    horizon,
                    profile.frequency,
                    profile.season_length_guess,
                    n_jobs,
                )
            )

    # Collect results
    results = ray.get(futures)

    # Aggregate by model
    model_folds: dict[str, list[dict]] = {}
    for r in results:
        model_folds.setdefault(r["runner_name"], []).append(r)

    model_results: list[ModelResult] = []
    for runner in runners:
        folds_data = model_folds.get(runner.name, [])
        if not folds_data:
            continue
        fold_results = [
            FoldResult(
                fold=fd["fold"],
                cutoff=fd["cutoff"],
                mase=fd["mase"],
                smape=fd["smape"],
                rmsse=fd["rmsse"],
                mae=fd["mae"],
                series_scores=fd["series_scores"],
            )
            for fd in sorted(folds_data, key=lambda x: x["fold"])
        ]
        mase_values = [f.mase for f in fold_results]
        total_runtime = sum(fd["runtime"] for fd in folds_data)
        model_results.append(
            ModelResult(
                name=runner.name,
                runtime_sec=round(total_runtime, 4),
                folds=fold_results,
                mean_mase=round(float(np.mean(mase_values)), 6),
                std_mase=round(float(np.std(mase_values)), 6),
                mean_smape=round(float(np.mean([f.smape for f in fold_results])), 4),
                mean_rmsse=round(float(np.mean([f.rmsse for f in fold_results])), 6),
                mean_mae=round(float(np.mean([f.mae for f in fold_results])), 6),
            )
        )

    # Build leaderboard
    successful = [m for m in model_results if not np.isnan(m.mean_mase)]
    sorted_models = sorted(successful, key=lambda m: m.mean_mase)
    leaderboard = [
        LeaderboardEntry(
            rank=i + 1,
            name=m.name,
            mean_mase=m.mean_mase,
            mean_smape=m.mean_smape,
            mean_rmsse=m.mean_rmsse,
            mean_mae=m.mean_mae,
        )
        for i, m in enumerate(sorted_models)
    ]

    return BenchmarkResult(
        profile=profile,
        config=BenchmarkConfig(horizon=horizon, n_folds=n_folds),
        models=model_results,
        leaderboard=leaderboard,
        warnings=warnings,
        data_characteristics=data_chars,
    )
