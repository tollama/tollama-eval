"""Tests for new CLI flags: --detect-anomalies, --metric-weights, --auto-select."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ts_autopilot.cli import app
from ts_autopilot.runners.optional import OptionalRunnerStatus

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


def test_include_optional_prints_discovery_summary(
    sample_csv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ts_autopilot.runners.optional.inspect_optional_runner_status",
        lambda include_neural=True, safe_mode=True: [
            OptionalRunnerStatus(
                label="Prophet",
                available=True,
                reason="available",
                runner_names=["Prophet"],
            ),
            OptionalRunnerStatus(
                label="LightGBM",
                available=False,
                reason="missing dependency: lightgbm",
                runner_names=["LightGBM"],
            ),
            OptionalRunnerStatus(
                label="XGBoost",
                available=False,
                reason="missing dependency: xgboost",
                runner_names=["XGBoost"],
            ),
            OptionalRunnerStatus(
                label="NeuralForecast",
                available=False,
                reason="not requested",
                runner_names=["NHITS", "NBEATS", "TiDE", "DeepAR", "PatchTST", "TFT"],
            ),
        ],
    )
    monkeypatch.setattr(
        "ts_autopilot.pipeline.get_optional_runners",
        lambda include_neural=True, safe_mode=True: [],
    )

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
            "--include-optional",
        ],
    )

    assert result.exit_code == 0
    assert "Optional Models:" in result.stdout
    assert "enabled Prophet: Prophet" in result.stdout
    assert "skipped LightGBM: missing dependency: lightgbm" in result.stdout
    assert "skipped NeuralForecast: not requested" in result.stdout
