"""CLI entry point for ts-autopilot."""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import importlib
import sys
import time
import traceback
from enum import IntEnum
from pathlib import Path
from typing import Any

import typer

from ts_autopilot import __version__

app = typer.Typer(
    name="ts-autopilot",
    help="Automated time series benchmarking.",
    no_args_is_help=True,
)


class ExitCode(IntEnum):
    """Distinct exit codes for different failure modes."""

    SUCCESS = 0
    SCHEMA_ERROR = 1
    DATA_ERROR = 2
    UNEXPECTED_ERROR = 3


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ts-autopilot {__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Automated time series benchmarking."""


_INPUT_DIR_OPTION = typer.Option(
    ...,
    "--input-dir",
    "-d",
    help="Directory containing CSV files to benchmark.",
)
_INPUT_OPTION = typer.Option(
    None,
    "--input",
    "-i",
    help="Path to input CSV file (long or wide format).",
)
_OUTPUT_OPTION = typer.Option(
    Path("out/"),
    "--output",
    "-o",
    help="Output directory for results.json and report.html.",
)
_CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to YAML or JSON config file.",
)


def _try_rich() -> bool:
    """Check if rich is available."""
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def _make_rich_progress_cb(
    quiet: bool, verbose: bool
) -> tuple[Any | None, Any | None, Any | None, Any]:
    """Create rich-based progress tracking.

    Returns (progress, model_task_id, fold_task_id, callback_fn).
    """
    if quiet:
        return None, None, None, lambda step, current, total: None

    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        )
        model_task = progress.add_task("Models", total=None)
        fold_task = progress.add_task("  Folds", total=None, visible=verbose)

        def cb(step: str, current: int, total: int) -> None:
            if step == "model":
                progress.update(model_task, completed=current - 1, total=total)
            elif step == "fold" and verbose:
                progress.update(fold_task, completed=current - 1, total=total)

        return progress, model_task, fold_task, cb
    except ImportError:
        return None, None, None, _make_plain_progress_cb(quiet, verbose)


def _make_plain_progress_cb(quiet: bool, verbose: bool) -> object:
    """Create plain text progress callback (fallback)."""

    def cb(step: str, current: int, total: int) -> None:
        if quiet:
            return
        if step == "model":
            typer.echo(f"  Running model {current}/{total}...")
        elif step == "fold" and verbose:
            typer.echo(f"    Fold {current}/{total}")

    return cb


def _merge_config(
    file_cfg: Any,
    *,
    cli_input: Path | None,
    cli_output: Path,
    cli_horizon: int,
    cli_n_folds: int,
    cli_models: str | None,
    cli_tollama_url: str | None,
    cli_tollama_models: str | None,
    cli_n_jobs: int,
    cli_no_cache: bool,
    cli_cache_dir: Path | None,
    cli_parallel_models: bool,
    default_timeout: float,
) -> dict[str, Any]:
    """Merge file config with CLI args. CLI non-default values win."""
    from ts_autopilot.pipeline import (
        DEFAULT_MAX_RETRIES,
        DEFAULT_RETRY_BACKOFF_SEC,
    )

    merged: dict[str, Any] = {}

    # Input/output paths: CLI wins if provided
    merged["input"] = (
        cli_input
        if cli_input is not None
        else Path(file_cfg.input)
        if file_cfg.input
        else None
    )
    merged["output"] = (
        Path(file_cfg.output)
        if file_cfg.output and cli_output == Path("out/")
        else cli_output
    )

    # Numeric params: CLI wins if not at default
    merged["horizon"] = (
        file_cfg.horizon
        if file_cfg.horizon is not None and cli_horizon == 14
        else cli_horizon
    )
    merged["n_folds"] = (
        file_cfg.n_folds
        if file_cfg.n_folds is not None and cli_n_folds == 3
        else cli_n_folds
    )
    merged["n_jobs"] = (
        file_cfg.n_jobs
        if file_cfg.n_jobs is not None and cli_n_jobs == 1
        else cli_n_jobs
    )

    # String/list params: CLI wins if provided
    merged["models"] = (
        cli_models
        if cli_models is not None
        else ",".join(file_cfg.models)
        if file_cfg.models
        else None
    )
    merged["tollama_url"] = (
        cli_tollama_url if cli_tollama_url is not None else file_cfg.tollama_url
    )
    merged["tollama_models"] = (
        cli_tollama_models
        if cli_tollama_models is not None
        else ",".join(file_cfg.tollama_models)
        if file_cfg.tollama_models
        else None
    )

    # Report settings (config file only)
    merged["report_title"] = file_cfg.report_title
    merged["report_lang"] = file_cfg.report_lang

    # Timeout/memory (config file overrides defaults)
    merged["model_timeout_sec"] = (
        file_cfg.model_timeout_sec
        if file_cfg.model_timeout_sec is not None
        else default_timeout
    )
    merged["memory_limit_mb"] = (
        file_cfg.memory_limit_mb if file_cfg.memory_limit_mb is not None else 2048
    )
    merged["allow_private_urls"] = file_cfg.allow_private_urls

    # Retry settings (config file only)
    merged["max_retries"] = (
        file_cfg.max_retries
        if file_cfg.max_retries is not None
        else DEFAULT_MAX_RETRIES
    )
    merged["retry_backoff"] = (
        file_cfg.retry_backoff
        if file_cfg.retry_backoff is not None
        else DEFAULT_RETRY_BACKOFF_SEC
    )

    # Cache settings: CLI wins if non-default
    merged["no_cache"] = cli_no_cache or file_cfg.no_cache
    merged["cache_dir"] = (
        cli_cache_dir
        if cli_cache_dir is not None
        else Path(file_cfg.cache_dir)
        if file_cfg.cache_dir
        else None
    )

    # Parallel models: CLI wins if set
    merged["parallel_models"] = cli_parallel_models or file_cfg.parallel_models

    return merged


@app.command()
def run(
    input: Path | None = _INPUT_OPTION,
    horizon: int = typer.Option(
        14,
        "--horizon",
        "-H",
        help="Forecast horizon (number of steps ahead).",
        min=1,
    ),
    n_folds: int = typer.Option(
        3,
        "--n-folds",
        "-k",
        help="Number of cross-validation folds.",
        min=1,
    ),
    output: Path = _OUTPUT_OPTION,
    models: str | None = typer.Option(
        None,
        "--models",
        "-m",
        help="Comma-separated list of models to run (e.g. SeasonalNaive,AutoETS).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress and data profile.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress all output except errors.",
    ),
    tollama_url: str | None = typer.Option(
        None,
        "--tollama-url",
        help="URL for tollama TSFM server (e.g. http://localhost:8000).",
    ),
    tollama_models: str | None = typer.Option(
        None,
        "--tollama-models",
        help="Comma-separated tollama models to benchmark (e.g. chronos2,timesfm).",
    ),
    no_tollama: bool = typer.Option(
        False,
        "--no-tollama",
        help="Disable tollama integration even if URL is provided.",
    ),
    n_jobs: int = typer.Option(
        1,
        "--n-jobs",
        "-j",
        help="Number of parallel workers for model fitting.",
        min=1,
    ),
    pdf: bool = typer.Option(
        False,
        "--pdf",
        help="Generate PDF report (requires weasyprint).",
    ),
    excel: bool = typer.Option(
        False,
        "--excel",
        help="Generate Excel workbook (requires openpyxl).",
    ),
    log_json: bool = typer.Option(
        False,
        "--log-json",
        help="Emit structured JSON logs to stderr.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable result caching.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help="Directory for cached results.",
    ),
    parallel_models: bool = typer.Option(
        False,
        "--parallel-models",
        help="Run models in parallel processes.",
    ),
    detect_anomalies: bool = typer.Option(
        False,
        "--detect-anomalies",
        help="Run anomaly detection on input data before benchmarking.",
    ),
    metric_weights: str | None = typer.Option(
        None,
        "--metric-weights",
        help=(
            "Custom metric weights for composite scoring. "
            "Format: 'mase=0.5,smape=0.3,speed=0.2'."
        ),
    ),
    auto_select: bool = typer.Option(
        False,
        "--auto-select",
        help="Automatically select models based on data characteristics.",
    ),
    exog_cols: str | None = typer.Option(
        None,
        "--exog-cols",
        help=(
            "Comma-separated exogenous column names. "
            "If omitted, extra columns are auto-detected."
        ),
    ),
    distributed: bool = typer.Option(
        False,
        "--distributed",
        help="Use Ray for distributed fold execution (requires ray).",
    ),
    config: Path | None = _CONFIG_OPTION,
) -> None:
    """Run automated time series benchmarking on a CSV file."""
    from ts_autopilot.exceptions import ConfigError, ModelFitError, SchemaError
    from ts_autopilot.logging_config import setup_logging
    from ts_autopilot.pipeline import (
        DEFAULT_MAX_RETRIES,
        DEFAULT_MODEL_TIMEOUT_SEC,
        DEFAULT_RETRY_BACKOFF_SEC,
        run_from_csv,
    )

    # Merge config file with CLI flags
    if config is not None:
        from ts_autopilot.config import load_config

        try:
            file_cfg = load_config(config)
        except (FileNotFoundError, ConfigError, ValueError) as exc:
            typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=ExitCode.DATA_ERROR) from exc

        merged_cfg = _merge_config(
            file_cfg,
            cli_input=input,
            cli_output=output,
            cli_horizon=horizon,
            cli_n_folds=n_folds,
            cli_models=models,
            cli_tollama_url=tollama_url,
            cli_tollama_models=tollama_models,
            cli_n_jobs=n_jobs,
            cli_no_cache=no_cache,
            cli_cache_dir=cache_dir,
            cli_parallel_models=parallel_models,
            default_timeout=DEFAULT_MODEL_TIMEOUT_SEC,
        )
        input = merged_cfg["input"]
        output = merged_cfg["output"]
        horizon = merged_cfg["horizon"]
        n_folds = merged_cfg["n_folds"]
        n_jobs = merged_cfg["n_jobs"]
        models = merged_cfg["models"]
        tollama_url = merged_cfg["tollama_url"]
        tollama_models = merged_cfg["tollama_models"]
        report_title = merged_cfg["report_title"]
        report_lang = merged_cfg["report_lang"]
        model_timeout_sec = merged_cfg["model_timeout_sec"]
        memory_limit_mb = merged_cfg["memory_limit_mb"]
        allow_private_urls = merged_cfg["allow_private_urls"]
        max_retries = merged_cfg["max_retries"]
        retry_backoff = merged_cfg["retry_backoff"]
        no_cache = merged_cfg["no_cache"]
        cache_dir = merged_cfg["cache_dir"]
        parallel_models = merged_cfg["parallel_models"]
    else:
        report_title = None
        report_lang = None
        model_timeout_sec = DEFAULT_MODEL_TIMEOUT_SEC
        memory_limit_mb = 2048
        allow_private_urls = False
        max_retries = DEFAULT_MAX_RETRIES
        retry_backoff = DEFAULT_RETRY_BACKOFF_SEC

    if input is None:
        typer.secho(
            "Error: --input is required (via CLI flag or config file).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=ExitCode.DATA_ERROR)

    if not input.exists():
        typer.secho(
            f"Error: Input file not found: {input}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=ExitCode.DATA_ERROR)

    # Initialize structured logging
    setup_logging(verbose=verbose, quiet=quiet, log_json=log_json)

    model_names = None
    if models is not None:
        model_names = [m.strip() for m in models.split(",") if m.strip()]

    # Parse metric weights if provided
    parsed_weights: dict[str, float] | None = None
    if metric_weights is not None:
        from ts_autopilot.evaluation.metrics import parse_metric_weights

        try:
            parsed_weights = parse_metric_weights(metric_weights)
        except ValueError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=ExitCode.DATA_ERROR) from exc

    # Set up progress display (rich if available, plain text otherwise)
    use_rich = _try_rich() and not log_json
    if use_rich:
        progress, _mtask, _ftask, progress_cb = _make_rich_progress_cb(quiet, verbose)
    else:
        progress = None
        progress_cb = _make_plain_progress_cb(quiet, verbose)

    if not quiet:
        if use_rich:
            try:
                from rich.console import Console
                from rich.panel import Panel

                console = Console()
                console.print(
                    Panel(
                        f"[bold]input[/bold]={input}  "
                        f"[bold]horizon[/bold]={horizon}  "
                        f"[bold]n_folds[/bold]={n_folds}",
                        title="[bold blue]ts-autopilot benchmark[/bold blue]",
                        border_style="blue",
                    )
                )
            except ImportError:
                msg = f"Running benchmark: input={input}, "
                msg += f"horizon={horizon}, n_folds={n_folds}"
                typer.secho(msg, bold=True)
        else:
            msg = f"Running benchmark: input={input}, "
            msg += f"horizon={horizon}, n_folds={n_folds}"
            typer.secho(msg, bold=True)

    t0 = time.perf_counter()

    # Auto-select models if requested (runs profiling early)
    if auto_select and model_names is None:
        from ts_autopilot.automl.selector import AutoSelector
        from ts_autopilot.ingestion.loader import load_csv as _load_csv
        from ts_autopilot.ingestion.profiler import profile_dataframe
        from ts_autopilot.pipeline import DEFAULT_RUNNERS

        _df_tmp = _load_csv(input, max_memory_mb=memory_limit_mb)
        _profile_tmp = profile_dataframe(_df_tmp)
        selector = AutoSelector(
            profile=_profile_tmp,
            include_extended=True,
            include_neural=False,
        )
        recommended = selector.recommended_model_names()
        # Filter to models available in the default runner set
        available = {r.name for r in DEFAULT_RUNNERS}
        model_names = [m for m in recommended if m in available]
        if not quiet:
            typer.echo(selector.summary())

    effective_tollama_url = tollama_url if not no_tollama else None
    effective_tollama_models: list[str] | None = None
    if tollama_models and effective_tollama_url:
        effective_tollama_models = [
            m.strip() for m in tollama_models.split(",") if m.strip()
        ]

    try:
        if progress is not None:
            progress.start()

        try:
            if distributed:
                from ts_autopilot.distributed.ray_runner import (
                    is_available as ray_available,
                )
                from ts_autopilot.distributed.ray_runner import (
                    run_benchmark_distributed,
                )
                from ts_autopilot.ingestion.loader import load_csv as _load_csv3

                if not ray_available():
                    typer.secho(
                        "Warning: Ray not installed, falling back to local.",
                        fg=typer.colors.YELLOW,
                    )

                _dist_df = _load_csv3(input, max_memory_mb=memory_limit_mb)
                result = run_benchmark_distributed(
                    df=_dist_df,
                    horizon=horizon,
                    n_folds=n_folds,
                    model_names=model_names,
                    n_jobs=n_jobs,
                )
                # Write output files
                output.mkdir(parents=True, exist_ok=True)
                (output / "results.json").write_text(result.to_json(indent=2))
                from ts_autopilot.reporting.html_report import render_report

                (output / "report.html").write_text(render_report(result))
            else:
                result = run_from_csv(
                    csv_path=input,
                    horizon=horizon,
                    n_folds=n_folds,
                    output_dir=output,
                    model_names=model_names,
                    progress_callback=progress_cb,
                    tollama_url=effective_tollama_url,
                    tollama_models=effective_tollama_models,
                    n_jobs=n_jobs,
                    generate_pdf=pdf,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                    report_title=report_title,
                    report_lang=report_lang,
                    model_timeout_sec=model_timeout_sec,
                    memory_limit_mb=memory_limit_mb,
                    allow_private_urls=allow_private_urls,
                    no_cache=no_cache,
                    cache_dir=cache_dir,
                    parallel_models=parallel_models,
                )
        finally:
            if progress is not None:
                progress.stop()

    except ModelFitError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        typer.echo(
            "Hint: The model failed repeatedly. Try increasing max_retries "
            "in the config file or removing the problematic model.",
            err=True,
        )
        raise typer.Exit(code=ExitCode.UNEXPECTED_ERROR) from exc
    except SchemaError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        typer.echo(
            "Hint: Ensure your CSV has columns (unique_id, ds, y) "
            "or is in wide format with dates in the first column.",
            err=True,
        )
        raise typer.Exit(code=ExitCode.SCHEMA_ERROR) from exc
    except ValueError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        typer.echo(
            "Hint: Check that your series are long enough for the "
            "requested horizon and number of folds.",
            err=True,
        )
        raise typer.Exit(code=ExitCode.DATA_ERROR) from exc
    except ZeroDivisionError as exc:
        typer.secho(
            "Error: One or more training series has zero variation (constant values).",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo(
            "Hint: Remove constant series from your dataset.",
            err=True,
        )
        raise typer.Exit(code=ExitCode.DATA_ERROR) from exc
    except Exception as exc:
        typer.secho(f"Unexpected error: {exc}", fg=typer.colors.RED, err=True)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        else:
            typer.echo("Hint: Use --verbose for full traceback.", err=True)
        raise typer.Exit(code=ExitCode.UNEXPECTED_ERROR) from exc

    # Anomaly detection (post-benchmark on input data)
    if detect_anomalies:
        from ts_autopilot.anomaly.detector import run_all_detectors
        from ts_autopilot.ingestion.loader import load_csv as _load_csv2

        _anom_df = _load_csv2(input, max_memory_mb=memory_limit_mb)
        reports = run_all_detectors(_anom_df)
        if not quiet:
            for report in reports:
                typer.echo("")
                typer.echo(report.summary())

    # Composite score display
    if parsed_weights is not None and result.leaderboard:
        from ts_autopilot.evaluation.metrics import composite_score

        if not quiet:
            typer.echo("")
            typer.secho("Composite Scores:", bold=True)
        model_by_name = {
            model_result.name: model_result for model_result in result.models
        }
        for entry in result.leaderboard:
            model_result = model_by_name.get(entry.name)
            if model_result is None:
                continue
            cs = composite_score(
                mase_val=model_result.mean_mase,
                smape_val=model_result.mean_smape,
                rmsse_val=model_result.mean_rmsse,
                mae_val=model_result.mean_mae,
                runtime_sec=model_result.runtime_sec,
                weights=parsed_weights,
            )
            if not quiet:
                typer.echo(f"  {entry.name}: composite={cs:.4f}")

    elapsed = time.perf_counter() - t0

    if quiet:
        return

    # Print warnings
    for warning in result.warnings:
        typer.secho(f"[WARNING] {warning}", fg=typer.colors.YELLOW)

    # Verbose: data profile summary
    if verbose:
        p = result.profile
        typer.echo("")
        typer.secho("Dataset Profile:", bold=True)
        typer.echo(f"  Series: {p.n_series}")
        typer.echo(f"  Total rows: {p.total_rows}")
        typer.echo(f"  Frequency: {p.frequency}")
        typer.echo(f"  Season length: {p.season_length_guess}")
        typer.echo(f"  Series lengths: {p.min_length}\u2013{p.max_length}")
        typer.echo(f"  Missing ratio: {p.missing_ratio:.2%}")

    # Excel export
    if excel:
        try:
            from ts_autopilot.reporting.export import export_excel

            xlsx_path = output / "report.xlsx"
            export_excel(result, xlsx_path)
            if not quiet:
                typer.echo(f"Excel workbook written to {xlsx_path.resolve()}")
        except ImportError:
            typer.secho(
                "openpyxl not installed. Install with: "
                'pip install "ts-autopilot[excel]"',
                fg=typer.colors.YELLOW,
                err=True,
            )

    abs_output = output.resolve()
    typer.echo("")
    typer.echo(f"Results written to {abs_output}/results.json")
    typer.echo(f"Report written to {abs_output}/report.html")

    # Leaderboard
    std_by_name = {m.name: m.std_mase for m in result.models}
    typer.echo("")
    typer.secho("Leaderboard:", bold=True)
    for entry in result.leaderboard:
        std = std_by_name.get(entry.name, 0.0)
        line = (
            f"  #{entry.rank} {entry.name}: MASE={entry.mean_mase:.4f} \u00b1 {std:.4f}"
        )
        if entry.rank == 1:
            typer.secho(line, fg=typer.colors.GREEN, bold=True)
        else:
            typer.echo(line)

    # Summary
    typer.echo("")
    if result.leaderboard:
        winner = result.leaderboard[0]
        typer.secho(
            f"Best model: {winner.name} (MASE={winner.mean_mase:.4f})",
            fg=typer.colors.GREEN,
        )
    typer.echo(f"Completed in {elapsed:.2f}s")


@app.command()
def campaign(
    input_dir: Path = _INPUT_DIR_OPTION,
    output: Path = _OUTPUT_OPTION,
    horizon: int = typer.Option(
        14,
        "--horizon",
        "-H",
        help="Forecast horizon.",
        min=1,
    ),
    n_folds: int = typer.Option(
        3,
        "--n-folds",
        "-k",
        help="Number of CV folds.",
        min=1,
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        "-m",
        help="Comma-separated list of models.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress output.",
    ),
) -> None:
    """Run benchmarks across multiple CSV files in a directory."""
    from ts_autopilot.pipeline import run_from_csv

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        typer.secho(
            f"No CSV files found in {input_dir}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=ExitCode.DATA_ERROR)

    model_names = None
    if models is not None:
        model_names = [m.strip() for m in models.split(",") if m.strip()]

    if not quiet:
        typer.secho(
            f"Campaign: {len(csv_files)} datasets in {input_dir}",
            bold=True,
        )

    results_summary: list[dict] = []
    t0 = time.perf_counter()

    for i, csv_path in enumerate(csv_files, 1):
        dataset_name = csv_path.stem
        dataset_out = output / dataset_name

        if not quiet:
            typer.echo(f"\n[{i}/{len(csv_files)}] {csv_path.name}")

        try:
            result = run_from_csv(
                csv_path=csv_path,
                horizon=horizon,
                n_folds=n_folds,
                output_dir=dataset_out,
                model_names=model_names,
            )
            winner = result.leaderboard[0] if result.leaderboard else None
            entry = {
                "dataset": dataset_name,
                "status": "ok",
                "n_series": result.profile.n_series,
                "best_model": winner.name if winner else "N/A",
                "best_mase": winner.mean_mase if winner else None,
            }
            results_summary.append(entry)
            if not quiet and winner:
                typer.secho(
                    f"  Best: {winner.name} (MASE={winner.mean_mase:.4f})",
                    fg=typer.colors.GREEN,
                )
        except Exception as exc:
            results_summary.append(
                {
                    "dataset": dataset_name,
                    "status": f"error: {exc}",
                    "n_series": 0,
                    "best_model": "N/A",
                    "best_mase": None,
                }
            )
            if not quiet:
                typer.secho(
                    f"  Error: {exc}",
                    fg=typer.colors.RED,
                )

    elapsed = time.perf_counter() - t0

    # Write campaign summary CSV
    import pandas as pd

    summary_df = pd.DataFrame(results_summary)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "campaign_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not quiet:
        typer.echo("")
        typer.secho("Campaign Summary:", bold=True)
        ok = sum(1 for r in results_summary if r["status"] == "ok")
        typer.echo(f"  {ok}/{len(csv_files)} datasets succeeded")
        typer.echo(f"  Results: {summary_path.resolve()}")
        typer.echo(f"  Completed in {elapsed:.2f}s")


@app.command()
def doctor() -> None:
    """Run diagnostic checks on the environment."""
    checks: list[tuple[str, bool, str]] = []

    # Python version
    vi = sys.version_info
    py_ver = f"{vi.major}.{vi.minor}.{vi.micro}"
    py_ok = sys.version_info >= (3, 10)
    checks.append(
        (
            "Python version",
            py_ok,
            f"{py_ver} {'(OK)' if py_ok else '(need 3.10+)'}",
        )
    )

    # Core dependencies
    core_deps = [
        "pandas",
        "numpy",
        "statsforecast",
        "typer",
        "jinja2",
        "httpx",
        "yaml",
    ]
    for dep in core_deps:
        try:
            mod = importlib.import_module(dep)
            ver = getattr(mod, "__version__", "installed")
            checks.append((f"Core: {dep}", True, str(ver)))
        except ImportError:
            checks.append((f"Core: {dep}", False, "NOT INSTALLED"))

    # Optional dependencies
    optional_deps = {
        "prophet": "Prophet",
        "lightgbm": "LightGBM",
        "neuralforecast": "NeuralForecast",
        "weasyprint": "PDF export",
        "streamlit": "Dashboard",
    }
    for dep, label in optional_deps.items():
        try:
            importlib.import_module(dep)
            checks.append((f"Optional: {label}", True, "available"))
        except ImportError:
            checks.append((f"Optional: {label}", False, "not installed"))

    # Output directory
    out_dir = Path("out/")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".doctor_test"
        test_file.write_text("ok")
        test_file.unlink()
        checks.append(("Output dir writable", True, str(out_dir.resolve())))
    except OSError as exc:
        checks.append(("Output dir writable", False, str(exc)))

    # Accelerator check (GPU/MPS)
    try:
        import torch

        acc = "cpu"
        if torch.backends.mps.is_available():
            acc = "MPS (Metal)"
        elif torch.cuda.is_available():
            acc = f"CUDA ({torch.cuda.get_device_name(0)})"
        checks.append(("Hardware acceleration", True, acc))
    except ImportError:
        checks.append(("Hardware acceleration", True, "cpu (torch not installed)"))


    # Print results
    typer.secho("\nts-autopilot doctor", bold=True)
    typer.secho("=" * 50)
    passed = 0
    failed = 0
    for name, ok, detail in checks:
        if ok:
            status = typer.style("PASS", fg=typer.colors.GREEN)
            passed += 1
        else:
            status = typer.style("FAIL", fg=typer.colors.RED)
            failed += 1
        typer.echo(f"  [{status}] {name}: {detail}")

    typer.echo("")
    summary_color = typer.colors.GREEN if failed == 0 else typer.colors.YELLOW
    typer.secho(f"  {passed} passed, {failed} failed", fg=summary_color, bold=True)


@app.command()
def serve(
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to listen on.",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind to.",
    ),
    output_dir: Path = typer.Option(
        Path("out/server"),
        "--output-dir",
        "-o",
        help="Directory for storing benchmark results.",
    ),
) -> None:
    """Start the REST API server for remote benchmarking."""
    try:
        from ts_autopilot.server.app import create_app
    except ImportError as exc:
        typer.secho(
            "Error: FastAPI and uvicorn are required for the REST server.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo(
            'Install with: pip install "ts-autopilot[server]"',
            err=True,
        )
        raise typer.Exit(code=ExitCode.UNEXPECTED_ERROR) from exc

    try:
        import uvicorn
    except ImportError as exc:
        typer.secho(
            "Error: uvicorn is required for the REST server.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo(
            'Install with: pip install "ts-autopilot[server]"',
            err=True,
        )
        raise typer.Exit(code=ExitCode.UNEXPECTED_ERROR) from exc

    app_instance = create_app(results_dir=output_dir)
    typer.secho(
        f"Starting ts-autopilot server on {host}:{port}",
        bold=True,
    )
    uvicorn.run(app_instance, host=host, port=port)


if __name__ == "__main__":
    app()
