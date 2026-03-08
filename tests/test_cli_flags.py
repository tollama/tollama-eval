"""Tests for new CLI flags: --detect-anomalies, --metric-weights, --auto-select."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ts_autopilot.cli import app

runner = CliRunner()


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal CSV for testing CLI flags."""
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for sid in ["s1", "s2"]:
        for d in dates:
            rows.append({"unique_id": sid, "ds": d, "y": np.random.rand() * 100})
    df = pd.DataFrame(rows)
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


def test_auto_select_flag(sample_csv: Path, tmp_path: Path) -> None:
    """--auto-select should select models and run successfully."""
    result = runner.invoke(
        app,
        [
            "run",
            "-i",
            str(sample_csv),
            "-o",
            str(tmp_path / "out"),
            "-H",
            "7",
            "-k",
            "2",
            "--auto-select",
        ],
    )
    assert result.exit_code == 0
    assert "Auto-selected" in result.stdout


def test_metric_weights_flag(sample_csv: Path, tmp_path: Path) -> None:
    """--metric-weights should parse and display composite scores."""
    result = runner.invoke(
        app,
        [
            "run",
            "-i",
            str(sample_csv),
            "-o",
            str(tmp_path / "out"),
            "-H",
            "7",
            "-k",
            "2",
            "-m",
            "SeasonalNaive",
            "--metric-weights",
            "mase=0.5,smape=0.3,rmsse=0.2",
        ],
    )
    assert result.exit_code == 0
    assert "Composite Scores:" in result.stdout


def test_metric_weights_invalid(sample_csv: Path, tmp_path: Path) -> None:
    """--metric-weights with bad format should show error."""
    result = runner.invoke(
        app,
        [
            "run",
            "-i",
            str(sample_csv),
            "-o",
            str(tmp_path / "out"),
            "--metric-weights",
            "bad_format",
        ],
    )
    assert result.exit_code != 0


def test_detect_anomalies_flag(sample_csv: Path, tmp_path: Path) -> None:
    """--detect-anomalies should run detectors and print summary."""
    result = runner.invoke(
        app,
        [
            "run",
            "-i",
            str(sample_csv),
            "-o",
            str(tmp_path / "out"),
            "-H",
            "7",
            "-k",
            "2",
            "-m",
            "SeasonalNaive",
            "--detect-anomalies",
        ],
    )
    assert result.exit_code == 0
    assert "Anomaly Detection Report" in result.stdout


def test_distributed_flag_fallback(sample_csv: Path, tmp_path: Path) -> None:
    """--distributed should fall back to local if Ray not installed."""
    result = runner.invoke(
        app,
        [
            "run",
            "-i",
            str(sample_csv),
            "-o",
            str(tmp_path / "out"),
            "-H",
            "7",
            "-k",
            "2",
            "-m",
            "SeasonalNaive",
            "--distributed",
        ],
    )
    # Should succeed (falls back to local execution)
    assert result.exit_code == 0
