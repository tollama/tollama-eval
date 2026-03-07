"""CLI entry point for ts-autopilot."""

from __future__ import annotations

import sys
import time
import traceback
from enum import IntEnum
from pathlib import Path

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
) -> tuple[object | None, object | None, object | None, object]:
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
    log_json: bool = typer.Option(
        False,
        "--log-json",
        help="Emit structured JSON logs to stderr.",
    ),
    config: Path | None = _CONFIG_OPTION,
) -> None:
    """Run automated time series benchmarking on a CSV file."""
    from ts_autopilot.exceptions import ConfigError, ModelFitError, SchemaError
    from ts_autopilot.logging_config import setup_logging
    from ts_autopilot.pipeline import (
        DEFAULT_MAX_RETRIES,
        DEFAULT_RETRY_BACKOFF_SEC,
        run_from_csv,
    )

    # Load config file and merge with CLI flags (CLI wins)
    report_title: str | None = None
    report_lang: str | None = None
    if config is not None:
        from ts_autopilot.config import load_config

        try:
            file_cfg = load_config(config)
        except (FileNotFoundError, ConfigError, ValueError) as exc:
            typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=ExitCode.DATA_ERROR) from exc

        if input is None and file_cfg.input:
            input = Path(file_cfg.input)
        if file_cfg.output and output == Path("out/"):
            output = Path(file_cfg.output)
        if file_cfg.horizon is not None and horizon == 14:
            horizon = file_cfg.horizon
        if file_cfg.n_folds is not None and n_folds == 3:
            n_folds = file_cfg.n_folds
        if file_cfg.models and models is None:
            models = ",".join(file_cfg.models)
        if file_cfg.tollama_url and tollama_url is None:
            tollama_url = file_cfg.tollama_url
        if file_cfg.tollama_models and tollama_models is None:
            tollama_models = ",".join(file_cfg.tollama_models)
        if file_cfg.n_jobs is not None and n_jobs == 1:
            n_jobs = file_cfg.n_jobs
        report_title = file_cfg.report_title
        report_lang = file_cfg.report_lang

    # Retry settings (config file only, no CLI flags needed)
    max_retries = DEFAULT_MAX_RETRIES
    retry_backoff = DEFAULT_RETRY_BACKOFF_SEC
    if config is not None:
        if file_cfg.max_retries is not None:
            max_retries = file_cfg.max_retries
        if file_cfg.retry_backoff is not None:
            retry_backoff = file_cfg.retry_backoff

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

    # Set up progress display (rich if available, plain text otherwise)
    use_rich = _try_rich() and not log_json
    if use_rich:
        progress, _mtask, _ftask, progress_cb = _make_rich_progress_cb(
            quiet, verbose
        )
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
        14, "--horizon", "-H", help="Forecast horizon.", min=1,
    ),
    n_folds: int = typer.Option(
        3, "--n-folds", "-k", help="Number of CV folds.", min=1,
    ),
    models: str | None = typer.Option(
        None, "--models", "-m",
        help="Comma-separated list of models.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress output.",
    ),
) -> None:
    """Run benchmarks across multiple CSV files in a directory."""
    from ts_autopilot.pipeline import run_from_csv

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        typer.secho(
            f"No CSV files found in {input_dir}",
            fg=typer.colors.RED, err=True,
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
                    f"  Best: {winner.name} "
                    f"(MASE={winner.mean_mase:.4f})",
                    fg=typer.colors.GREEN,
                )
        except Exception as exc:
            results_summary.append({
                "dataset": dataset_name,
                "status": f"error: {exc}",
                "n_series": 0,
                "best_model": "N/A",
                "best_mase": None,
            })
            if not quiet:
                typer.secho(
                    f"  Error: {exc}", fg=typer.colors.RED,
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


if __name__ == "__main__":
    app()
