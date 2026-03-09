"""Tests for time series profiler."""

import numpy as np
import pandas as pd

from ts_autopilot.ingestion.profiler import (
    _guess_season_length,
    compute_data_characteristics,
    profile_dataframe,
)


def test_profile_n_series(tiny_long_df):
    profile = profile_dataframe(tiny_long_df)
    assert profile.n_series == 2


def test_profile_frequency_daily(tiny_long_df):
    profile = profile_dataframe(tiny_long_df)
    assert profile.frequency == "D"


def test_profile_missing_ratio_zero(tiny_long_df):
    profile = profile_dataframe(tiny_long_df)
    assert profile.missing_ratio == 0.0


def test_profile_missing_ratio_nonzero(tiny_long_df):
    df = tiny_long_df.copy()
    df.loc[0, "y"] = np.nan
    profile = profile_dataframe(df)
    assert profile.missing_ratio > 0.0


def test_profile_season_length_daily(tiny_long_df):
    profile = profile_dataframe(tiny_long_df)
    assert profile.season_length_guess == 7  # D → 7


def test_profile_series_lengths(tiny_long_df):
    profile = profile_dataframe(tiny_long_df)
    assert profile.min_length == 60
    assert profile.max_length == 60


def test_profile_total_rows(tiny_long_df):
    profile = profile_dataframe(tiny_long_df)
    assert profile.total_rows == 120  # 2 series x 60


def test_guess_season_length_mapping():
    assert _guess_season_length("D") == 7
    assert _guess_season_length("W") == 52
    assert _guess_season_length("W-SUN") == 52
    assert _guess_season_length("ME") == 12
    assert _guess_season_length("h") == 24
    assert _guess_season_length("unknown") == 1


def test_profile_weekly_data():
    dates = pd.date_range("2020-01-06", periods=60, freq="W-MON")
    df = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": range(60)})
    profile = profile_dataframe(df)
    assert profile.season_length_guess == 52


def test_guess_season_length_unknown_freq_falls_back():
    assert _guess_season_length("XYZ") == 1
    assert _guess_season_length("") == 1


def test_profile_infers_fallback_freq_when_irregular():
    """Irregular dates should fall back to 'D'."""
    dates = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-20"])
    df = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": [1.0, 2.0, 3.0]})
    profile = profile_dataframe(df)
    assert profile.frequency == "D"  # fallback


def test_profile_single_row_series():
    df = pd.DataFrame(
        {
            "unique_id": ["s1"],
            "ds": pd.to_datetime(["2020-01-01"]),
            "y": [1.0],
        }
    )
    profile = profile_dataframe(df)
    assert profile.n_series == 1
    assert profile.min_length == 1
    assert profile.total_rows == 1


def test_frequency_fallback_logs_warning(caplog):
    """When frequency can't be inferred, a warning should be logged."""
    import logging

    root_logger = logging.getLogger("ts_autopilot")
    old_propagate = root_logger.propagate
    root_logger.propagate = True
    try:
        dates = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-20"])
        df = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": [1.0, 2.0, 3.0]})
        with caplog.at_level(logging.WARNING, logger="ts_autopilot"):
            profile = profile_dataframe(df)
        assert profile.frequency == "D"
        assert any("defaulting" in r.message.lower() for r in caplog.records)
    finally:
        root_logger.propagate = old_propagate


# --- Data Characteristics tests ---


def test_data_characteristics_basic(tiny_long_df):
    """Data characteristics computes all fields from a 2-series daily DataFrame."""
    chars = compute_data_characteristics(tiny_long_df, season_length=7)
    assert chars.y_mean != 0.0
    assert chars.y_std > 0.0
    assert chars.y_min < chars.y_max
    assert chars.y_median != 0.0
    assert chars.mean_cv > 0.0
    assert chars.series_heterogeneity >= 0.0


def test_data_characteristics_single_row():
    """Handles edge case of single-row series."""
    df = pd.DataFrame(
        {
            "unique_id": ["s1"],
            "ds": pd.to_datetime(["2020-01-01"]),
            "y": [5.0],
        }
    )
    chars = compute_data_characteristics(df, season_length=7)
    assert chars.y_mean == 5.0
    assert chars.y_std == 0.0
    assert chars.trend_strength == 0.0


def test_data_characteristics_empty():
    """Handles empty DataFrame."""
    df = pd.DataFrame({
        "unique_id": pd.Series([], dtype=str),
        "ds": pd.Series([], dtype="datetime64[ns]"),
        "y": pd.Series([], dtype=float),
    })
    chars = compute_data_characteristics(df, season_length=7)
    assert chars.y_mean == 0.0
    assert chars.trend_strength == 0.0


def test_data_characteristics_constant_series():
    """Constant series should have zero CV, zero trend, zero seasonality."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": 42.0})
    chars = compute_data_characteristics(df, season_length=7)
    assert chars.y_std == 0.0
    assert chars.mean_cv == 0.0
    assert chars.trend_strength == 0.0
    assert chars.seasonality_strength == 0.0
