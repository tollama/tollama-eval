"""Tests for CLI run command."""

import pandas as pd
from typer.testing import CliRunner

from ts_autopilot.cli import app

runner = CliRunner()


def _make_csv(tmp_path):
    """Create a small CSV for CLI tests."""
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": str(d.date()), "y": float(i)})
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_cli_run_produces_output_files(tmp_path):
    csv_path = _make_csv(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(csv_path),
            "--horizon",
            "7",
            "--n-folds",
            "2",
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "results.json").exists()
    assert (out_dir / "report.html").exists()


def test_cli_leaderboard_printed(tmp_path):
    csv_path = _make_csv(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(csv_path),
            "--horizon",
            "7",
            "--n-folds",
            "2",
            "--output",
            str(out_dir),
        ],
    )
    assert "Leaderboard" in result.output
    assert "#1" in result.output


def test_cli_missing_input_fails():
    result = runner.invoke(app, ["run", "--input", "nonexistent.csv"])
    assert result.exit_code != 0


def test_cli_help_shows_reserved_flags():
    result = runner.invoke(app, ["run", "--help"])
    assert "--tollama-url" in result.output
    assert "--no-tollama" in result.output
