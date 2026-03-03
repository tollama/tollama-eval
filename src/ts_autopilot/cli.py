"""CLI entry point for ts-autopilot."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="ts-autopilot",
    help="Automated time series benchmarking.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Automated time series benchmarking."""


@app.command()
def run(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to input CSV file (long or wide format).",
        exists=True,
        file_okay=True,
        readable=True,
    ),
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
    output: Path = typer.Option(
        Path("out/"),
        "--output",
        "-o",
        help="Output directory for results.json and report.html.",
    ),
    tollama_url: str | None = typer.Option(
        None,
        "--tollama-url",
        help="[Reserved] URL for tollama service. Not yet implemented.",
    ),
    no_tollama: bool = typer.Option(
        False,
        "--no-tollama",
        help="[Reserved] Disable tollama integration. Not yet implemented.",
    ),
) -> None:
    """Run automated time series benchmarking on a CSV file."""
    from ts_autopilot.pipeline import run_from_csv

    if tollama_url is not None:
        typer.echo(
            "[WARNING] --tollama-url is reserved and not yet implemented.", err=True
        )
    if no_tollama:
        typer.echo(
            "[WARNING] --no-tollama is reserved and not yet implemented.", err=True
        )

    typer.echo(
        f"Running benchmark: input={input}, horizon={horizon}, n_folds={n_folds}"
    )

    result = run_from_csv(
        csv_path=input,
        horizon=horizon,
        n_folds=n_folds,
        output_dir=output,
    )

    typer.echo(f"Results written to {output}/results.json")
    typer.echo(f"Report written to {output}/report.html")
    typer.echo("Leaderboard:")
    for entry in result.leaderboard:
        typer.echo(f"  #{entry.rank} {entry.name}: mean_mase={entry.mean_mase:.4f}")


if __name__ == "__main__":
    app()
