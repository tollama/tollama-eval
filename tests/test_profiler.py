"""Tests for time series profiler."""

import numpy as np
import pandas as pd

from ts_autopilot.ingestion.profiler import _guess_season_length, profile_dataframe


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
