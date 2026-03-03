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
    df = pd.DataFrame(
        {"unique_id": "s1", "ds": dates, "y": range(60)}
    )
    profile = profile_dataframe(df)
    assert profile.season_length_guess == 52
