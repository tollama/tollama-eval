"""Minimal time series profiler."""

from __future__ import annotations

import pandas as pd

from ts_autopilot.contracts import DataCharacteristics, DataProfile

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

    # Series length distribution (histogram bin counts)
    length_dist: list[int] = []
    if len(series_lengths) > 1:
        import numpy as np

        counts, _ = np.histogram(
            series_lengths.values, bins=min(10, len(series_lengths))
        )
        length_dist = [int(c) for c in counts]

    # Zero ratio for intermittent demand detection
    zero_ratio = detect_zero_ratio(df)

    return DataProfile(
        n_series=int(df["unique_id"].nunique()),
        frequency=freq,
        missing_ratio=float(df["y"].isna().mean()),
        season_length_guess=_guess_season_length(freq),
        min_length=int(series_lengths.min()),
        max_length=int(series_lengths.max()),
        total_rows=len(df),
        series_length_distribution=length_dist,
        zero_ratio=zero_ratio,
    )


def _infer_frequency(df: pd.DataFrame) -> str:
    """Infer frequency from the first series. Falls back to 'D'."""
    from ts_autopilot.logging_config import get_logger

    _logger = get_logger("profiler")
    first_uid = df["unique_id"].iloc[0]
    dates = df.loc[df["unique_id"] == first_uid, "ds"].sort_values()
    try:
        freq = pd.infer_freq(dates)
        if freq is not None:
            return str(freq)
        _logger.warning(
            "Could not infer frequency for series '%s'; defaulting to 'D'.",
            first_uid,
        )
        return "D"
    except Exception:
        _logger.warning(
            "Frequency inference failed for series '%s'; defaulting to 'D'.",
            first_uid,
        )
        return "D"


def _guess_season_length(freq: str) -> int:
    """Map a pandas freq string to a season length integer."""
    if freq in FREQ_TO_SEASON:
        return FREQ_TO_SEASON[freq]
    # Strip suffix variants: 'W-SUN' → 'W'
    base = freq.split("-")[0] if "-" in freq else freq
    return FREQ_TO_SEASON.get(base, 1)


def compute_data_characteristics(
    df: pd.DataFrame,
    season_length: int,
) -> DataCharacteristics:
    """Compute descriptive data characteristics from a canonical DataFrame.

    Args:
        df: DataFrame with columns (unique_id, ds, y).
        season_length: Guessed season length from profile.

    Returns:
        DataCharacteristics with distribution stats and complexity measures.
    """
    import numpy as np

    y = df["y"].dropna().values.astype(np.float64)
    n = len(y)

    if n == 0:
        return DataCharacteristics()

    # 1. Value distribution (global)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y, ddof=1)) if n > 1 else 0.0
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_median = float(np.median(y))

    # Skewness & kurtosis (numpy-only)
    if y_std > 0 and n > 2:
        centered = y - y_mean
        y_skewness = float(np.mean(centered**3) / y_std**3)
        y_kurtosis = float(np.mean(centered**4) / y_std**4 - 3.0)
    else:
        y_skewness = 0.0
        y_kurtosis = 0.0

    # Per-series statistics
    grouped = df.groupby("unique_id")["y"]
    series_means: list[float] = []
    series_cvs: list[float] = []
    trend_r2s: list[float] = []
    season_acfs: list[float] = []

    for _uid, group_y in grouped:
        vals = group_y.dropna().values.astype(np.float64)
        m = len(vals)
        if m < 2:
            continue

        s_mean = float(np.mean(vals))
        s_std = float(np.std(vals, ddof=1))
        series_means.append(s_mean)

        # Per-series CV
        if abs(s_mean) > 1e-12:
            series_cvs.append(s_std / abs(s_mean))

        # Trend: R-squared of linear regression y ~ t
        t = np.arange(m, dtype=np.float64)
        t_mean = np.mean(t)
        y_mean_s = np.mean(vals)
        ss_tt = float(np.sum((t - t_mean) ** 2))
        ss_yy = float(np.sum((vals - y_mean_s) ** 2))
        if ss_tt > 0 and ss_yy > 0:
            ss_ty = float(np.sum((t - t_mean) * (vals - y_mean_s)))
            r2 = (ss_ty**2) / (ss_tt * ss_yy)
            trend_r2s.append(float(np.clip(r2, 0.0, 1.0)))

        # Seasonality: autocorrelation at seasonal lag
        if m > season_length and season_length > 0:
            centered_s = vals - np.mean(vals)
            var_s = float(np.sum(centered_s**2) / m)
            if var_s > 0:
                acf_lag = float(
                    np.sum(centered_s[season_length:] * centered_s[:-season_length])
                    / (m * var_s)
                )
                season_acfs.append(float(np.clip(acf_lag, 0.0, 1.0)))

    mean_cv = float(np.mean(series_cvs)) if series_cvs else 0.0
    trend_strength = float(np.mean(trend_r2s)) if trend_r2s else 0.0
    seasonality_strength = float(np.mean(season_acfs)) if season_acfs else 0.0

    # Series heterogeneity: CV of series means
    series_heterogeneity = 0.0
    if len(series_means) > 1:
        sm_arr = np.array(series_means)
        sm_mean = float(np.mean(sm_arr))
        if abs(sm_mean) > 1e-12:
            series_heterogeneity = float(np.std(sm_arr, ddof=1) / abs(sm_mean))

    return DataCharacteristics(
        y_mean=round(y_mean, 4),
        y_std=round(y_std, 4),
        y_min=round(y_min, 4),
        y_max=round(y_max, 4),
        y_median=round(y_median, 4),
        y_skewness=round(y_skewness, 4),
        y_kurtosis=round(y_kurtosis, 4),
        mean_cv=round(mean_cv, 4),
        trend_strength=round(trend_strength, 4),
        seasonality_strength=round(seasonality_strength, 4),
        series_heterogeneity=round(series_heterogeneity, 4),
    )


def detect_zero_ratio(df: pd.DataFrame) -> float:
    """Compute the ratio of zero values in the target column.

    Useful for detecting intermittent demand patterns.

    Args:
        df: Canonical long-format DataFrame.

    Returns:
        Fraction of zero values (0.0 to 1.0).
    """
    if df.empty or "y" not in df.columns:
        return 0.0
    return float((df["y"] == 0).mean())
