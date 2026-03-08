"""Tests for exogenous variable support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_detect_exog_columns() -> None:
    """detect_exog_columns should find numeric non-canonical columns."""
    from ts_autopilot.ingestion.loader import detect_exog_columns

    df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 5,
            "ds": pd.date_range("2020-01-01", periods=5),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "temperature": [20.0, 21.0, 22.0, 23.0, 24.0],
            "humidity": [50.0, 55.0, 60.0, 65.0, 70.0],
            "label": ["a", "b", "c", "d", "e"],  # non-numeric, skipped
        }
    )
    exog = detect_exog_columns(df)
    assert "temperature" in exog
    assert "humidity" in exog
    assert "label" not in exog
    assert "unique_id" not in exog
    assert "ds" not in exog
    assert "y" not in exog


def test_detect_exog_columns_empty() -> None:
    """detect_exog_columns returns empty list for canonical-only data."""
    from ts_autopilot.ingestion.loader import detect_exog_columns

    df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 3,
            "ds": pd.date_range("2020-01-01", periods=3),
            "y": [1.0, 2.0, 3.0],
        }
    )
    assert detect_exog_columns(df) == []


def test_load_csv_preserves_exog(tmp_path: Path) -> None:
    """load_csv should keep exogenous columns in the output DataFrame."""
    from ts_autopilot.ingestion.loader import load_csv

    df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10),
            "y": np.random.rand(10),
            "temp": np.random.rand(10),
        }
    )
    path = tmp_path / "exog.csv"
    df.to_csv(path, index=False)

    result = load_csv(path)
    assert "temp" in result.columns
    assert result["temp"].dtype == np.float64


def test_load_csv_explicit_exog_cols(tmp_path: Path) -> None:
    """load_csv with explicit exog_cols keeps only specified columns."""
    from ts_autopilot.ingestion.loader import load_csv

    df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10),
            "y": np.random.rand(10),
            "temp": np.random.rand(10),
            "wind": np.random.rand(10),
        }
    )
    path = tmp_path / "exog2.csv"
    df.to_csv(path, index=False)

    result = load_csv(path, exog_cols=["temp"])
    assert "temp" in result.columns
    assert "wind" not in result.columns


def test_base_runner_supports_exog_default() -> None:
    """BaseRunner.supports_exog defaults to False."""
    from ts_autopilot.runners.statistical import SeasonalNaiveRunner

    runner = SeasonalNaiveRunner()
    assert runner.supports_exog is False


def test_data_profile_exog_columns() -> None:
    """DataProfile should support exog_columns field."""
    from ts_autopilot.contracts import DataProfile

    profile = DataProfile(
        n_series=1,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=100,
        max_length=100,
        total_rows=100,
        exog_columns=["temp", "humidity"],
    )
    d = profile.to_dict()
    assert d["exog_columns"] == ["temp", "humidity"]

    restored = DataProfile.from_dict(d)
    assert restored.exog_columns == ["temp", "humidity"]


def test_data_profile_exog_backward_compat() -> None:
    """DataProfile without exog_columns should still work."""
    from ts_autopilot.contracts import DataProfile

    d = {
        "n_series": 1,
        "frequency": "D",
        "missing_ratio": 0.0,
        "season_length_guess": 7,
        "min_length": 100,
        "max_length": 100,
        "total_rows": 100,
    }
    profile = DataProfile.from_dict(d)
    assert profile.exog_columns == []

    # to_dict should omit empty exog_columns
    assert "exog_columns" not in profile.to_dict()
