"""Benchmark pipeline orchestrator."""

from __future__ import annotations

import contextlib
import os
import signal
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ts_autopilot.cache import ResultCache

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
from ts_autopilot.exceptions import ModelFitError, ModelTimeoutError
from ts_autopilot.logging_config import get_logger
from ts_autopilot.runners.base import BaseRunner
from ts_autopilot.runners.optional import get_optional_runners
from ts_autopilot.runners.statistical import (
    ALL_STATISTICAL_RUNNERS,
    CORE_RUNNERS,
)

logger = get_logger("pipeline")

DEFAULT_RUNNERS: tuple[BaseRunner, ...] = (
    *CORE_RUNNERS,
    *get_optional_runners(),
)

# Extended runner set (includes all statistical + optional)
EXTENDED_DEFAULT_RUNNERS: tuple[BaseRunner, ...] = (
    *ALL_STATISTICAL_RUNNERS,
    *get_optional_runners(),
)

DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SEC = 1.0
DEFAULT_MODEL_TIMEOUT_SEC = 300.0

# Global shutdown event for graceful signal handling
_shutdown_event = threading.Event()


def _install_signal_handlers() -> None:
    """Install SIGINT/SIGTERM handlers that set the shutdown event."""

    def _handler(signum: int, frame: FrameType | None) -> None:
        logger.warning("Received signal %d, shutting down gracefully...", signum)
        _shutdown_event.set()

    # Only install in main thread
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handler)
        # SIGINT is handled by Python's default KeyboardInterrupt,
        # but we also set the event so the pipeline loop can break cleanly
        prev_handler = signal.getsignal(signal.SIGINT)

        def _int_handler(signum: int, frame: FrameType | None) -> None:
            _shutdown_event.set()
            is_custom = (
                callable(prev_handler)
                and prev_handler is not signal.default_int_handler
            )
            if is_custom:
                handler = cast(
                    Callable[[int, FrameType | None], object],
                    prev_handler,
                )
                handler(signum, frame)
            else:
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _int_handler)


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


def _fit_predict_with_retry(
    runner: BaseRunner,
    train: pd.DataFrame,
    horizon: int,
    freq: str,
    season_length: int,
    n_jobs: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF_SEC,
    model_timeout_sec: float = DEFAULT_MODEL_TIMEOUT_SEC,
) -> ForecastOutput:
    """Call runner.fit_predict with retry on transient failures.

    Retries up to *max_retries* times with exponential backoff.
    Only retries on RuntimeError (e.g. numerical convergence issues);
    ValueError and similar are raised immediately.

    If model_timeout_sec > 0, each attempt is wrapped in a timeout.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if model_timeout_sec > 0:
                output = _run_with_timeout(
                    runner,
                    train,
                    horizon,
                    freq,
                    season_length,
                    n_jobs,
                    model_timeout_sec,
                )
            else:
                output = runner.fit_predict(
                    train=train,
                    horizon=horizon,
                    freq=freq,
                    season_length=season_length,
                    n_jobs=n_jobs,
                )
            return output
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = retry_backoff * (2**attempt)
                logger.warning(
                    "Model %s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    runner.name,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Model %s failed after %d attempts: %s",
                    runner.name,
                    max_retries + 1,
                    exc,
                )
        except ModelTimeoutError:
            raise
        except Exception:
            raise
    raise ModelFitError(runner.name, max_retries + 1, last_exc)


def _run_with_timeout(
    runner: BaseRunner,
    train: pd.DataFrame,
    horizon: int,
    freq: str,
    season_length: int,
    n_jobs: int,
    timeout_sec: float,
) -> ForecastOutput:
    """Run fit_predict in a thread with timeout."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            runner.fit_predict,
            train=train,
            horizon=horizon,
            freq=freq,
            season_length=season_length,
            n_jobs=n_jobs,
        )
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError as exc:
            raise ModelTimeoutError(runner.name, timeout_sec) from exc


def run_benchmark(
    df: pd.DataFrame,
    horizon: int,
    n_folds: int,
    runners: list[BaseRunner] | None = None,
    model_names: list[str] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    n_jobs: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF_SEC,
    model_timeout_sec: float = DEFAULT_MODEL_TIMEOUT_SEC,
    run_id: str | None = None,
    cache: ResultCache | None = None,
    data_hash: str | None = None,
    parallel_models: bool = False,
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
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    logger.info("[run_id=%s] Benchmark starting", run_id)

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
    data_chars = compute_data_characteristics(df, profile.season_length_guess)
    config = BenchmarkConfig(horizon=horizon, n_folds=n_folds)
    warnings = generate_warnings(profile, horizon, n_folds)
    warnings.extend(validation_warnings)

    # Detect date gaps
    gap_warnings = _detect_date_gaps(df, profile.frequency)
    warnings.extend(gap_warnings)

    # Deduplicate warnings while preserving order
    warnings = list(dict.fromkeys(warnings))

    logger.info(
        "[run_id=%s] Starting benchmark: %d series, horizon=%d, n_folds=%d, models=%s",
        run_id,
        profile.n_series,
        horizon,
        n_folds,
        [r.name for r in runners],
    )

    splits = make_expanding_splits(df, horizon=horizon, n_folds=n_folds)

    model_results: list[ModelResult] = []
    all_forecast_data: list[ForecastData] = []
    pipeline_t0 = time.perf_counter()

    def _run_single_model(
        runner: BaseRunner,
        runner_idx: int,
    ) -> tuple[ModelResult, list[ForecastData]]:
        """Run all folds for a single model. Returns (ModelResult, forecasts)."""
        logger.info(
            "[run_id=%s] Running model %d/%d: %s",
            run_id,
            runner_idx + 1,
            len(runners),
            runner.name,
        )

        fold_results_inner: list[FoldResult] = []
        forecast_data: list[ForecastData] = []
        total_runtime = 0.0
        model_failed = False

        for fold_idx, split in enumerate(splits):
            if _shutdown_event.is_set():
                break

            if not parallel_models and progress_callback is not None:
                progress_callback("fold", fold_idx + 1, len(splits))

            logger.debug(
                "  %s fold %d/%d (train=%d rows, test=%d rows)",
                runner.name,
                fold_idx + 1,
                len(splits),
                len(split.train),
                len(split.test),
            )

            # Check cache before running the model
            if cache is not None and data_hash is not None:
                cached = cache.get(data_hash, runner.name, split.fold)
                if cached is not None:
                    fold_results_inner.append(cached)
                    continue

            try:
                output = _fit_predict_with_retry(
                    runner=runner,
                    train=split.train,
                    horizon=horizon,
                    freq=profile.frequency,
                    season_length=profile.season_length_guess,
                    n_jobs=n_jobs,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                    model_timeout_sec=model_timeout_sec,
                )
            except (ModelTimeoutError, ModelFitError) as exc:
                logger.warning(
                    "[run_id=%s] Skipping model %s: %s",
                    run_id,
                    runner.name,
                    exc,
                )
                model_failed = True
                break

            total_runtime += output.runtime_sec

            # Capture forecast vs actual data for report visualizations
            # Invariant: splits are pre-sorted by (unique_id, ds)
            # from make_expanding_splits, so no re-sorting needed.
            test_sorted = split.test
            tail_len = min(2 * horizon, len(split.train))
            train_tail = split.train.groupby("unique_id", sort=False).tail(tail_len)
            forecast_data.append(
                ForecastData(
                    model_name=runner.name,
                    fold=split.fold,
                    unique_id=output.unique_id,
                    ds=output.ds,
                    y_hat=output.y_hat,
                    y_actual=test_sorted["y"].tolist(),
                    ds_train_tail=[d.isoformat() for d in train_tail["ds"]],
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

            fold_result = FoldResult(
                fold=split.fold,
                cutoff=split.cutoff.isoformat(),
                mase=round(fold_mase, 6),
                smape=round(fold_smape, 4),
                rmsse=round(fold_rmsse, 6),
                mae=round(fold_mae, 6),
                series_scores={k: round(v, 6) for k, v in series_scores.items()},
            )
            fold_results_inner.append(fold_result)

            # Store in cache
            if cache is not None and data_hash is not None:
                cache.put(data_hash, runner.name, split.fold, fold_result)

            logger.debug(
                "  %s fold %d MASE=%.6f (%.4fs)",
                runner.name,
                split.fold,
                fold_mase,
                output.runtime_sec,
            )

        if model_failed or not fold_results_inner:
            return (
                ModelResult(
                    name=runner.name,
                    runtime_sec=round(total_runtime, 4),
                    folds=[],
                    mean_mase=float("nan"),
                    std_mase=float("nan"),
                    mean_smape=float("nan"),
                    mean_rmsse=float("nan"),
                    mean_mae=float("nan"),
                ),
                forecast_data,
            )

        mase_values = [f.mase for f in fold_results_inner]
        mean_val = round(float(np.mean(mase_values)), 6)
        mr = ModelResult(
            name=runner.name,
            runtime_sec=round(total_runtime, 4),
            folds=fold_results_inner,
            mean_mase=mean_val,
            std_mase=round(float(np.std(mase_values)), 6),
            mean_smape=round(float(np.mean([f.smape for f in fold_results_inner])), 4),
            mean_rmsse=round(float(np.mean([f.rmsse for f in fold_results_inner])), 6),
            mean_mae=round(float(np.mean([f.mae for f in fold_results_inner])), 6),
        )

        logger.info(
            "[run_id=%s] Completed %s: mean_mase=%.6f, runtime=%.4fs",
            run_id,
            runner.name,
            mean_val,
            total_runtime,
        )
        return mr, forecast_data

    # Run models: parallel or sequential
    if parallel_models and len(runners) > 1:
        logger.info(
            "[run_id=%s] Running %d models in parallel",
            run_id,
            len(runners),
        )
        with ThreadPoolExecutor(max_workers=len(runners)) as executor:
            futures = {
                executor.submit(_run_single_model, runner, idx): runner
                for idx, runner in enumerate(runners)
            }
            for future in as_completed(futures):
                if _shutdown_event.is_set():
                    break
                mr, fds = future.result()
                model_results.append(mr)
                all_forecast_data.extend(fds)
    else:
        for runner_idx, runner in enumerate(runners):
            if _shutdown_event.is_set():
                logger.warning(
                    "[run_id=%s] Shutdown requested after %d/%d models",
                    run_id,
                    runner_idx,
                    len(runners),
                )
                break

            if progress_callback is not None:
                progress_callback("model", runner_idx + 1, len(runners))

            mr, fds = _run_single_model(runner, runner_idx)
            model_results.append(mr)
            all_forecast_data.extend(fds)

    pipeline_elapsed = time.perf_counter() - pipeline_t0

    # Compute residual diagnostics for each model
    all_diagnostics: list[DiagnosticsResult] = []
    for model in model_results:
        if not model.folds:
            continue
        model_forecasts = [
            fd for fd in all_forecast_data if fd.model_name == model.name
        ]
        all_residuals = []
        all_fitted = []
        for fd in model_forecasts:
            residuals = [a - p for a, p in zip(fd.y_actual, fd.y_hat, strict=False)]
            all_residuals.extend(residuals)
            all_fitted.extend(fd.y_hat)
        if all_residuals:
            from ts_autopilot.evaluation.diagnostics import compute_diagnostics

            diag = compute_diagnostics(
                model.name, np.array(all_residuals), np.array(all_fitted)
            )
            all_diagnostics.append(diag)

    # Build leaderboard: rank by mean_mase ascending, excluding NaN models
    successful_models = [m for m in model_results if not np.isnan(m.mean_mase)]
    sorted_models = sorted(successful_models, key=lambda m: m.mean_mase)
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
        "[run_id=%s] Benchmark complete in %.2fs. Winner: %s (MASE=%.6f)",
        run_id,
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
        data_characteristics=data_chars,
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
    tollama_url: str | None = None,
    tollama_models: list[str] | None = None,
    n_jobs: int = 1,
    generate_pdf: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF_SEC,
    report_title: str | None = None,
    report_lang: str | None = None,
    model_timeout_sec: float = DEFAULT_MODEL_TIMEOUT_SEC,
    memory_limit_mb: int = 2048,
    allow_private_urls: bool = False,
    run_id: str | None = None,
    no_cache: bool = False,
    cache_dir: Path | None = None,
    parallel_models: bool = False,
) -> BenchmarkResult:
    """Full end-to-end pipeline: CSV → results.json + report.html.

    If tollama_url and tollama_models are provided, creates TollamaRunners
    to benchmark TSFM models alongside statistical models.
    """
    from ts_autopilot.reporting.html_report import render_report

    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    # Install signal handlers for graceful shutdown
    _install_signal_handlers()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[run_id=%s] Loading CSV: %s", run_id, csv_path)
    t0 = time.perf_counter()
    df = load_csv(csv_path, max_memory_mb=memory_limit_mb)
    logger.info(
        "[run_id=%s] Loaded %d rows, %d series",
        run_id,
        len(df),
        df["unique_id"].nunique(),
    )

    # Build extra tollama runners if configured
    extra_runners: list[BaseRunner] | None = None
    if tollama_url and tollama_models:
        from ts_autopilot.tollama.client import validate_tollama_url

        validate_tollama_url(tollama_url, allow_private=allow_private_urls)

        from ts_autopilot.runners.tollama import get_tollama_runners

        tollama_runners = get_tollama_runners(tollama_url, tollama_models)
        if tollama_runners:
            base: list[BaseRunner] = list(DEFAULT_RUNNERS)
            base.extend(tollama_runners)
            extra_runners = base
            logger.info(
                "[run_id=%s] Added %d tollama model(s): %s",
                run_id,
                len(tollama_runners),
                [r.name for r in tollama_runners],
            )

    # Set up result cache
    cache_instance: ResultCache | None = None
    computed_hash: str | None = None
    if not no_cache:
        from ts_autopilot.cache import ResultCache as _ResultCache
        from ts_autopilot.cache import _compute_data_hash

        effective_cache_dir = cache_dir or (output_dir / ".cache")
        cache_instance = _ResultCache(effective_cache_dir)
        computed_hash = _compute_data_hash(df, horizon, n_folds)
        logger.info(
            "[run_id=%s] Cache enabled: dir=%s, hash=%s",
            run_id,
            effective_cache_dir,
            computed_hash,
        )

    result = run_benchmark(
        df,
        horizon=horizon,
        n_folds=n_folds,
        runners=extra_runners,
        model_names=model_names,
        progress_callback=progress_callback,
        n_jobs=n_jobs,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        model_timeout_sec=model_timeout_sec,
        run_id=run_id,
        cache=cache_instance,
        data_hash=computed_hash,
        parallel_models=parallel_models,
    )

    # Attach metadata
    total_runtime = time.perf_counter() - t0
    metadata = ResultMetadata.create_now()
    metadata.total_runtime_sec = round(total_runtime, 4)
    metadata.run_id = run_id
    result.metadata = metadata

    # Check if we were interrupted
    if _shutdown_event.is_set():
        partial_path = output_dir / "results.partial.json"
        _atomic_write(partial_path, result.to_json(indent=2))
        logger.warning(
            "[run_id=%s] Wrote partial results to %s (interrupted)",
            run_id,
            partial_path,
        )
        return result

    # Atomic write results.json
    results_path = output_dir / "results.json"
    _atomic_write(results_path, result.to_json(indent=2))
    logger.info("[run_id=%s] Wrote %s", run_id, results_path)

    # Atomic write details.json (forecast data + diagnostics for report reproducibility)
    details = result.to_details_dict()
    if details:
        details_path = output_dir / "details.json"
        _atomic_write(details_path, result.to_details_json(indent=2))
        logger.info("[run_id=%s] Wrote %s", run_id, details_path)

    # Atomic write report.html
    report_path = output_dir / "report.html"
    _atomic_write(
        report_path,
        render_report(result, report_title=report_title, report_lang=report_lang),
    )
    logger.info("[run_id=%s] Wrote %s", run_id, report_path)

    # Optional PDF generation
    if generate_pdf:
        from ts_autopilot.reporting.pdf_export import generate_pdf as make_pdf
        from ts_autopilot.reporting.pdf_export import is_available

        if is_available():
            pdf_path = output_dir / "report.pdf"
            if make_pdf(report_path, pdf_path):
                logger.info("[run_id=%s] Wrote %s", run_id, pdf_path)
        else:
            logger.info(
                "PDF export requested but weasyprint not installed. "
                'Install with: pip install "ts-autopilot[pdf]"'
            )

    return result


# Re-import for backward compatibility (used by cli.py)
from ts_autopilot.ingestion.loader import load_csv  # noqa: E402
from ts_autopilot.ingestion.profiler import (  # noqa: E402
    compute_data_characteristics,
    profile_dataframe,
)

__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL_TIMEOUT_SEC",
    "DEFAULT_RETRY_BACKOFF_SEC",
    "generate_warnings",
    "run_benchmark",
    "run_from_csv",
]
