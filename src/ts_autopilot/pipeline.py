"""Benchmark pipeline orchestrator."""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    ForecastOutput,
    LeaderboardEntry,
    ModelResult,
    ResultMetadata,
)
from ts_autopilot.evaluation.cross_validation import make_expanding_splits
from ts_autopilot.evaluation.metrics import mean_mase_per_fold
from ts_autopilot.ingestion.loader import load_csv
from ts_autopilot.ingestion.profiler import profile_dataframe
from ts_autopilot.logging_config import get_logger
from ts_autopilot.runners.base import BaseRunner
from ts_autopilot.runners.statistical import AutoETSRunner, SeasonalNaiveRunner

logger = get_logger("pipeline")

DEFAULT_RUNNERS: tuple[BaseRunner, ...] = (
    SeasonalNaiveRunner(),
    AutoETSRunner(),
)


def _validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Validate input DataFrame and return warnings for data quality issues.

    Raises ValueError for unrecoverable problems.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty — nothing to benchmark.")

    warnings: list[str] = []

    # Check for NaN/inf in y column
    nan_count = int(df["y"].isna().sum())
    if nan_count > 0:
        warnings.append(
            f"Found {nan_count} NaN values in target column 'y'. "
            "These rows will degrade metric accuracy."
        )

    inf_mask = np.isinf(df["y"].values)
    inf_count = int(inf_mask.sum())
    if inf_count > 0:
        raise ValueError(
            f"Found {inf_count} infinite values in column 'y'. "
            "Remove or replace them before benchmarking."
        )

    # Check for duplicate (unique_id, ds) pairs
    dup_count = int(df.duplicated(subset=["unique_id", "ds"]).sum())
    if dup_count > 0:
        raise ValueError(
            f"Found {dup_count} duplicate (unique_id, ds) rows. "
            "Each series must have unique timestamps."
        )

    # Check for negative values (informational)
    neg_count = int((df["y"] < 0).sum())
    if neg_count > 0:
        warnings.append(
            f"Found {neg_count} negative values in 'y'. "
            "Verify this is expected for your domain."
        )

    return warnings


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

    # Detect gaps in date sequences
    return warnings


def _detect_date_gaps(df: pd.DataFrame, freq: str) -> list[str]:
    """Check for unexpected gaps in time series dates."""
    warnings: list[str] = []
    gap_series: list[str] = []

    for uid, group in df.groupby("unique_id"):
        dates = group["ds"].sort_values()
        if len(dates) < 3:
            continue
        try:
            expected = pd.date_range(
                start=dates.iloc[0], end=dates.iloc[-1], freq=freq
            )
            actual_set = set(dates)
            missing = set(expected) - actual_set
            if missing:
                gap_series.append(str(uid))
        except ValueError:
            continue

    if gap_series:
        n = len(gap_series)
        sample = ", ".join(gap_series[:3])
        suffix = f" and {n - 3} more" if n > 3 else ""
        warnings.append(
            f"Date gaps detected in {n} series ({sample}{suffix}). "
            "Missing timesteps may affect forecast quality."
        )

    return warnings


_MAX_RETRIES = 2
_RETRY_BACKOFF_SEC = 1.0


def _fit_predict_with_retry(
    runner: BaseRunner,
    train: pd.DataFrame,
    horizon: int,
    freq: str,
    season_length: int,
) -> ForecastOutput:
    """Call runner.fit_predict with retry on transient failures.

    Retries up to _MAX_RETRIES times with exponential backoff.
    Only retries on RuntimeError (e.g. numerical convergence issues);
    ValueError and similar are raised immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return runner.fit_predict(
                train=train,
                horizon=horizon,
                freq=freq,
                season_length=season_length,
            )
        except (RuntimeError, FloatingPointError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF_SEC * (2**attempt)
                logger.warning(
                    "Model %s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    runner.name,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Model %s failed after %d attempts: %s",
                    runner.name,
                    _MAX_RETRIES + 1,
                    exc,
                )
        except Exception:
            raise
    raise RuntimeError(
        f"Model {runner.name} failed after {_MAX_RETRIES + 1} attempts"
    ) from last_exc


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
      1. Validate input data
      2. Profile the DataFrame
      3. Make CV splits
      4. For each runner x fold: fit_predict → compute MASE
      5. Aggregate into ModelResult + leaderboard
      6. Return BenchmarkResult
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

    # Validate input data
    validation_warnings = _validate_dataframe(df)

    profile = profile_dataframe(df)
    config = BenchmarkConfig(horizon=horizon, n_folds=n_folds)
    warnings = generate_warnings(profile, horizon, n_folds)
    warnings.extend(validation_warnings)

    # Detect date gaps
    gap_warnings = _detect_date_gaps(df, profile.frequency)
    warnings.extend(gap_warnings)

    logger.info(
        "Starting benchmark: %d series, horizon=%d, n_folds=%d, models=%s",
        profile.n_series,
        horizon,
        n_folds,
        [r.name for r in runners],
    )

    splits = make_expanding_splits(df, horizon=horizon, n_folds=n_folds)

    model_results: list[ModelResult] = []
    pipeline_t0 = time.perf_counter()

    for runner_idx, runner in enumerate(runners):
        if progress_callback is not None:
            progress_callback("model", runner_idx + 1, len(runners))

        logger.info(
            "Running model %d/%d: %s",
            runner_idx + 1,
            len(runners),
            runner.name,
        )

        fold_results: list[FoldResult] = []
        total_runtime = 0.0

        for fold_idx, split in enumerate(splits):
            if progress_callback is not None:
                progress_callback("fold", fold_idx + 1, len(splits))

            logger.debug(
                "  %s fold %d/%d (train=%d rows, test=%d rows)",
                runner.name,
                fold_idx + 1,
                len(splits),
                len(split.train),
                len(split.test),
            )

            output = _fit_predict_with_retry(
                runner=runner,
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

            logger.debug(
                "  %s fold %d MASE=%.6f (%.4fs)",
                runner.name,
                split.fold,
                fold_mase,
                output.runtime_sec,
            )

        mase_values = [f.mase for f in fold_results]
        mean_val = round(float(np.mean(mase_values)), 6)
        model_results.append(
            ModelResult(
                name=runner.name,
                runtime_sec=round(total_runtime, 4),
                folds=fold_results,
                mean_mase=mean_val,
                std_mase=round(float(np.std(mase_values)), 6),
            )
        )

        logger.info(
            "Completed %s: mean_mase=%.6f, runtime=%.4fs",
            runner.name,
            mean_val,
            total_runtime,
        )

    pipeline_elapsed = time.perf_counter() - pipeline_t0

    # Build leaderboard: rank by mean_mase ascending
    sorted_models = sorted(model_results, key=lambda m: m.mean_mase)
    leaderboard = [
        LeaderboardEntry(rank=i + 1, name=m.name, mean_mase=m.mean_mase)
        for i, m in enumerate(sorted_models)
    ]

    logger.info(
        "Benchmark complete in %.2fs. Winner: %s (MASE=%.6f)",
        pipeline_elapsed,
        leaderboard[0].name if leaderboard else "N/A",
        leaderboard[0].mean_mase if leaderboard else 0.0,
    )

    return BenchmarkResult(
        profile=profile,
        config=config,
        models=model_results,
        leaderboard=leaderboard,
        warnings=warnings,
    )


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a file atomically via temp file + rename.

    Prevents corrupted output if the process is interrupted mid-write.
    """
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=f".{path.name}."
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


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

    logger.info("Loading CSV: %s", csv_path)
    t0 = time.perf_counter()
    df = load_csv(csv_path)
    logger.info("Loaded %d rows, %d series", len(df), df["unique_id"].nunique())

    result = run_benchmark(
        df,
        horizon=horizon,
        n_folds=n_folds,
        model_names=model_names,
        progress_callback=progress_callback,
    )

    # Attach metadata
    total_runtime = time.perf_counter() - t0
    metadata = ResultMetadata.create_now()
    metadata.total_runtime_sec = round(total_runtime, 4)
    result.metadata = metadata

    # Atomic write results.json
    results_path = output_dir / "results.json"
    _atomic_write(results_path, result.to_json(indent=2))
    logger.info("Wrote %s", results_path)

    # Atomic write report.html
    report_path = output_dir / "report.html"
    _atomic_write(report_path, render_report(result))
    logger.info("Wrote %s", report_path)

    return result
