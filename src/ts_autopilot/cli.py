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
        help="URL for tollama LLM interpretation service.",
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
    config: Path | None = _CONFIG_OPTION,
) -> None:
    """Run automated time series benchmarking on a CSV file."""
    from ts_autopilot.ingestion.loader import SchemaError
    from ts_autopilot.logging_config import setup_logging
    from ts_autopilot.pipeline import run_from_csv

    # Load config file and merge with CLI flags (CLI wins)
    if config is not None:
        from ts_autopilot.config import load_config

        try:
            file_cfg = load_config(config)
        except (FileNotFoundError, ValueError) as exc:
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
        if file_cfg.n_jobs is not None and n_jobs == 1:
            n_jobs = file_cfg.n_jobs

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
    setup_logging(verbose=verbose, quiet=quiet)

    model_names = None
    if models is not None:
        model_names = [m.strip() for m in models.split(",") if m.strip()]

    def _progress_cb(step: str, current: int, total: int) -> None:
        if quiet:
            return
        if step == "model":
            typer.echo(f"  Running model {current}/{total}...")
        elif step == "fold" and verbose:
            typer.echo(f"    Fold {current}/{total}")

    if not quiet:
        typer.secho(
            f"Running benchmark: input={input}, horizon={horizon}, "
            f"n_folds={n_folds}",
            bold=True,
        )

    t0 = time.perf_counter()

    try:
        result = run_from_csv(
            csv_path=input,
            horizon=horizon,
            n_folds=n_folds,
            output_dir=output,
            model_names=model_names,
            progress_callback=_progress_cb,
            n_jobs=n_jobs,
        )
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
            "Error: One or more training series has zero variation "
            "(constant values).",
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
            typer.echo(
                "Hint: Use --verbose for full traceback.", err=True
            )
        raise typer.Exit(code=ExitCode.UNEXPECTED_ERROR) from exc

    elapsed = time.perf_counter() - t0

    # Tollama LLM interpretation
    tollama_response = None
    if tollama_url and not no_tollama:
        from ts_autopilot.tollama.client import interpret

        if not quiet:
            typer.echo("  Requesting LLM interpretation...")
        tollama_response = interpret(result, tollama_url)
        if tollama_response is not None:
            # Re-render report with interpretation
            from ts_autopilot.pipeline import _atomic_write
            from ts_autopilot.reporting.html_report import render_report

            report_path = output.resolve() / "report.html"
            _atomic_write(
                report_path,
                render_report(
                    result,
                    tollama_interpretation=tollama_response.interpretation,
                ),
            )
        elif not quiet:
            typer.secho(
                "[WARNING] Tollama unavailable — skipping interpretation.",
                fg=typer.colors.YELLOW,
            )

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
            f"  #{entry.rank} {entry.name}: "
            f"MASE={entry.mean_mase:.4f} \u00b1 {std:.4f}"
        )
        if entry.rank == 1:
            typer.secho(line, fg=typer.colors.GREEN, bold=True)
        else:
            typer.echo(line)

    # Summary
    typer.echo("")
    winner = result.leaderboard[0]
    typer.secho(
        f"Best model: {winner.name} (MASE={winner.mean_mase:.4f})",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Completed in {elapsed:.2f}s")

    # Tollama interpretation
    if tollama_response is not None:
        typer.echo("")
        typer.secho("LLM Interpretation:", bold=True)
        typer.echo(f"  {tollama_response.interpretation}")


if __name__ == "__main__":
    app()
