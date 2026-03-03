"""Minimal time series profiler."""

from __future__ import annotations

import pandas as pd

from ts_autopilot.contracts import DataProfile

# Mapping from pandas inferred freq strings to integer season lengths
FREQ_TO_SEASON: dict[str, int] = {
    "h": 24,
    "H": 24,
    "D": 7,
    "W": 52,
    "W-SUN": 52,
    "W-MON": 52,
    "ME": 12,
    "MS": 12,
    "QE": 4,
    "QS": 4,
    "YE": 1,
    "YS": 1,
    "T": 60,
    "min": 60,
}


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Compute a minimal profile of a canonical long-format DataFrame.

    Args:
        df: DataFrame with columns (unique_id, ds, y).

    Returns:
        DataProfile with n_series, frequency, missing_ratio,
        season_length_guess, min_length, max_length, total_rows.
    """
    series_lengths = df.groupby("unique_id").size()
    freq = _infer_frequency(df)

    return DataProfile(
        n_series=int(df["unique_id"].nunique()),
        frequency=freq,
        missing_ratio=float(df["y"].isna().mean()),
        season_length_guess=_guess_season_length(freq),
        min_length=int(series_lengths.min()),
        max_length=int(series_lengths.max()),
        total_rows=len(df),
    )


def _infer_frequency(df: pd.DataFrame) -> str:
    """Infer frequency from the first series. Falls back to 'D'."""
    first_uid = df["unique_id"].iloc[0]
    dates = df.loc[df["unique_id"] == first_uid, "ds"].sort_values()
    try:
        freq = pd.infer_freq(dates)
        return freq if freq is not None else "D"
    except Exception:
        return "D"


def _guess_season_length(freq: str) -> int:
    """Map a pandas freq string to a season length integer."""
    if freq in FREQ_TO_SEASON:
        return FREQ_TO_SEASON[freq]
    # Strip suffix variants: 'W-SUN' → 'W'
    base = freq.split("-")[0] if "-" in freq else freq
    return FREQ_TO_SEASON.get(base, 1)
