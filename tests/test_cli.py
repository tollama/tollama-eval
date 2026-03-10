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


def test_cli_help_shows_reserved_flags(monkeypatch):
    import re
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: (200, 50))
    result = runner.invoke(app, ["run", "--help"])
    clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--tollama-url" in clean
    assert "--tollama-models" in clean
    assert "--no-tollama" in clean


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "tollama-eval" in result.output
    assert "0.2.0" in result.output


def test_cli_version_short_flag():
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "tollama-eval" in result.output


def test_cli_quiet_suppresses_output(tmp_path):
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
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    assert "Leaderboard" not in result.output


def test_cli_verbose_shows_profile(tmp_path):
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
            "--verbose",
        ],
    )
    assert result.exit_code == 0
    assert "Dataset Profile" in result.output
    assert "Series:" in result.output
    assert "Frequency:" in result.output


def test_cli_models_flag_filters(tmp_path):
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
            "--models",
            "SeasonalNaive",
        ],
    )
    assert result.exit_code == 0
    assert "SeasonalNaive" in result.output
    assert "AutoETS" not in result.output


def test_cli_models_unknown_exits_with_data_error(tmp_path):
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
            "--models",
            "NonExistentModel",
        ],
    )
    assert result.exit_code != 0


def test_cli_bad_csv_shows_friendly_error(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\nabc,def\n")
    result = runner.invoke(
        app,
        ["run", "--input", str(bad), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 1
    assert "Hint:" in result.output


def test_cli_summary_shows_best_model(tmp_path):
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
    assert result.exit_code == 0
    assert "Best model:" in result.output
    assert "Completed in" in result.output


def test_cli_help_shows_models_flag(monkeypatch):
    import re
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: (200, 50))
    result = runner.invoke(app, ["run", "--help"])
    clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--models" in clean
    assert "--verbose" in clean
    assert "--quiet" in clean


def test_cli_leaderboard_shows_std_mase(tmp_path):
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
    assert result.exit_code == 0
    assert str(out_dir.resolve()) in result.output


def test_python_m_module_exists():
    """__main__.py exists so ``python -m ts_autopilot`` works."""
    from pathlib import Path

    import ts_autopilot

    pkg_dir = Path(ts_autopilot.__file__).parent
    assert (pkg_dir / "__main__.py").exists()
