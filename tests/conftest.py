"""Shared test fixtures for ts-autopilot."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def long_csv(tmp_path):
    """Long-format CSV with 2 series, 4 rows."""
    df = pd.DataFrame(
        {
            "unique_id": ["s1", "s1", "s2", "s2"],
            "ds": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    p = tmp_path / "long.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def wide_csv(tmp_path):
    """Wide-format CSV: dates as index, series as columns."""
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "series_a": [1.0, 1.5],
            "series_b": [2.0, 2.5],
        }
    ).set_index("date")
    p = tmp_path / "wide.csv"
    df.to_csv(p)
    return p


@pytest.fixture
def bad_csv(tmp_path):
    """CSV that cannot be parsed as time series."""
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar\nabc,def\nghi,jkl\n")
    return p


@pytest.fixture
def tiny_long_df():
    """Canonical long-format DataFrame: 2 series x 60 daily rows each."""
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)
    rows = []
    for uid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": d, "y": 10.0 + i + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)
