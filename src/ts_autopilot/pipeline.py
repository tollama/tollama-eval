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
    DiagnosticsResult,
    FoldResult,
    ForecastData,
    ForecastOutput,
    LeaderboardEntry,
    ModelResult,
    ResultMetadata,
)
from ts_autopilot.evaluation.cross_validation import make_expanding_splits
from ts_autopilot.evaluation.metrics import (
    per_series_mae,
    per_series_mase,
    per_series_rmsse,
    per_series_smape,
)
from ts_autopilot.ingestion.loader import load_csv
from ts_autopilot.ingestion.profiler import profile_dataframe
from ts_autopilot.logging_config import get_logger
from ts_autopilot.runners.base import BaseRunner
from ts_autopilot.runners.optional import get_optional_runners
from ts_autopilot.runners.statistical import (
    AutoARIMARunner,
    AutoCESRunner,
    AutoETSRunner,
    AutoThetaRunner,
    SeasonalNaiveRunner,
)

logger = get_logger("pipeline")

DEFAULT_RUNNERS: tuple[BaseRunner, ...] = (
    SeasonalNaiveRunner(),
    AutoETSRunner(),
    AutoARIMARunner(),
    AutoThetaRunner(),
    AutoCESRunner(),
    *get_optional_runners(),
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


def generate_warnings(profile: DataProfile, horizon: int, n_folds: int) -> list[str]:
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
            expected = pd.date_range(start=dates.iloc[0], end=dates.iloc[-1], freq=freq)
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
    n_jobs: int = 1,
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
                n_jobs=n_jobs,
            )
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
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
    n_jobs: int = 1,
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

    # Deduplicate warnings while preserving order
    warnings = list(dict.fromkeys(warnings))

    logger.info(
        "Starting benchmark: %d series, horizon=%d, n_folds=%d, models=%s",
        profile.n_series,
        horizon,
        n_folds,
        [r.name for r in runners],
    )

    splits = make_expanding_splits(df, horizon=horizon, n_folds=n_folds)

    model_results: list[ModelResult] = []
    all_forecast_data: list[ForecastData] = []
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
                n_jobs=n_jobs,
            )
            total_runtime += output.runtime_sec

            # Capture forecast vs actual data for report visualizations
            test_sorted = split.test.sort_values(["unique_id", "ds"])
            tail_len = min(2 * horizon, len(split.train))
            train_tail = (
                split.train.sort_values(["unique_id", "ds"])
                .groupby("unique_id")
                .tail(tail_len)
                .sort_values(["unique_id", "ds"])
            )
            all_forecast_data.append(
                ForecastData(
                    model_name=runner.name,
                    fold=split.fold,
                    unique_id=output.unique_id,
                    ds=output.ds,
                    y_hat=output.y_hat,
                    y_actual=test_sorted["y"].tolist(),
                    ds_train_tail=[
                        d.isoformat() for d in train_tail["ds"]
                    ],
                    y_train_tail=train_tail["y"].tolist(),
                )
            )

            # Convert ForecastOutput to DataFrame for metric computation
            pred_df = pd.DataFrame(
                {
                    "unique_id": output.unique_id,
                    "ds": pd.to_datetime(output.ds),
                    output.model_name: output.y_hat,
                }
            )

            series_scores = per_series_mase(
                forecast_df=pred_df,
                actuals_df=split.test,
                train_df=split.train,
                season_length=profile.season_length_guess,
                model_col=output.model_name,
            )
            fold_mase = float(np.mean(list(series_scores.values())))

            smape_scores = per_series_smape(
                forecast_df=pred_df,
                actuals_df=split.test,
                model_col=output.model_name,
            )
            fold_smape = float(np.mean(list(smape_scores.values())))

            rmsse_scores = per_series_rmsse(
                forecast_df=pred_df,
                actuals_df=split.test,
                train_df=split.train,
                season_length=profile.season_length_guess,
                model_col=output.model_name,
            )
            fold_rmsse = float(np.mean(list(rmsse_scores.values())))

            mae_scores = per_series_mae(
                forecast_df=pred_df,
                actuals_df=split.test,
                model_col=output.model_name,
            )
            fold_mae = float(np.mean(list(mae_scores.values())))

            fold_results.append(
                FoldResult(
                    fold=split.fold,
                    cutoff=split.cutoff.isoformat(),
                    mase=round(fold_mase, 6),
                    smape=round(fold_smape, 4),
                    rmsse=round(fold_rmsse, 6),
                    mae=round(fold_mae, 6),
                    series_scores={k: round(v, 6) for k, v in series_scores.items()},
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
                mean_smape=round(float(np.mean([f.smape for f in fold_results])), 4),
                mean_rmsse=round(float(np.mean([f.rmsse for f in fold_results])), 6),
                mean_mae=round(float(np.mean([f.mae for f in fold_results])), 6),
            )
        )

        logger.info(
            "Completed %s: mean_mase=%.6f, runtime=%.4fs",
            runner.name,
            mean_val,
            total_runtime,
        )

    pipeline_elapsed = time.perf_counter() - pipeline_t0

    # Compute residual diagnostics for each model
    all_diagnostics: list[DiagnosticsResult] = []
    for model in model_results:
        model_forecasts = [
            fd for fd in all_forecast_data if fd.model_name == model.name
        ]
        all_residuals = []
        all_fitted = []
        for fd in model_forecasts:
            residuals = [
                a - p for a, p in zip(fd.y_actual, fd.y_hat, strict=False)
            ]
            all_residuals.extend(residuals)
            all_fitted.extend(fd.y_hat)
        if all_residuals:
            from ts_autopilot.evaluation.diagnostics import compute_diagnostics

            diag = compute_diagnostics(
                model.name, np.array(all_residuals), np.array(all_fitted)
            )
            all_diagnostics.append(diag)

    # Build leaderboard: rank by mean_mase ascending
    sorted_models = sorted(model_results, key=lambda m: m.mean_mase)
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
        forecast_data=all_forecast_data,
        diagnostics=all_diagnostics,
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
    tollama_interpretation: str | None = None,
    tollama_url: str | None = None,
    n_jobs: int = 1,
    generate_pdf: bool = False,
) -> BenchmarkResult:
    """Full end-to-end pipeline: CSV → results.json + report.html.

    If tollama_url is provided, requests an LLM interpretation after
    benchmarking and includes it in the HTML report.
    """
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
        n_jobs=n_jobs,
    )

    # Attach metadata
    total_runtime = time.perf_counter() - t0
    metadata = ResultMetadata.create_now()
    metadata.total_runtime_sec = round(total_runtime, 4)
    result.metadata = metadata

    # Tollama LLM interpretation
    if tollama_url and tollama_interpretation is None:
        from ts_autopilot.tollama.client import interpret

        logger.info("Requesting LLM interpretation from %s", tollama_url)
        response = interpret(result, tollama_url)
        if response is not None:
            tollama_interpretation = response.interpretation
        else:
            logger.warning("Tollama unavailable — skipping interpretation.")

    # Atomic write results.json
    results_path = output_dir / "results.json"
    _atomic_write(results_path, result.to_json(indent=2))
    logger.info("Wrote %s", results_path)

    # Atomic write report.html
    report_path = output_dir / "report.html"
    _atomic_write(
        report_path,
        render_report(result, tollama_interpretation=tollama_interpretation),
    )
    logger.info("Wrote %s", report_path)

    # Optional PDF generation
    if generate_pdf:
        from ts_autopilot.reporting.pdf_export import generate_pdf as make_pdf
        from ts_autopilot.reporting.pdf_export import is_available

        if is_available():
            pdf_path = output_dir / "report.pdf"
            if make_pdf(report_path, pdf_path):
                logger.info("Wrote %s", pdf_path)
        else:
            logger.info(
                "PDF export requested but weasyprint not installed. "
                'Install with: pip install "ts-autopilot[pdf]"'
            )

    return result
