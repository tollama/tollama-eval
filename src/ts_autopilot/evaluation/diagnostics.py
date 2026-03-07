"""Residual diagnostics for forecast models."""

from __future__ import annotations

import numpy as np

from ts_autopilot.contracts import DiagnosticsResult


def _ljung_box_p(residuals: np.ndarray, max_lag: int = 10) -> float:
    """Approximate Ljung-Box test p-value for residual autocorrelation.

    Uses chi-squared approximation. p > 0.05 suggests residuals are white noise.
    """
    n = len(residuals)
    if n < max_lag + 1:
        return 1.0  # Not enough data to test

    mean = np.mean(residuals)
    centered = residuals - mean
    var = np.sum(centered**2) / n

    if var == 0:
        return 1.0

    q_stat = 0.0
    for k in range(1, max_lag + 1):
        rk = np.sum(centered[k:] * centered[:-k]) / (n * var)
        q_stat += (rk**2) / (n - k)
    q_stat *= n * (n + 2)

    # Chi-squared CDF approximation using regularized incomplete gamma
    # For df=max_lag, use simple approximation
    df = max_lag
    # Wilson-Hilferty approximation for chi-squared CDF
    z = ((q_stat / df) ** (1.0 / 3) - (1 - 2.0 / (9 * df))) / np.sqrt(
        2.0 / (9 * df)
    )
    # Standard normal CDF approximation
    p_value = 1.0 - 0.5 * (1 + np.sign(z) * (1 - np.exp(-2 * z * z / np.pi)) ** 0.5)
    return float(np.clip(p_value, 0.0, 1.0))


def _acf(residuals: np.ndarray, max_lag: int = 20) -> tuple[list[int], list[float]]:
    """Compute autocorrelation function up to max_lag."""
    n = len(residuals)
    max_lag = min(max_lag, n - 1)
    if max_lag < 1:
        return [], []

    mean = np.mean(residuals)
    centered = residuals - mean
    var = np.sum(centered**2) / n

    if var == 0:
        return list(range(1, max_lag + 1)), [0.0] * max_lag

    lags = []
    values = []
    for k in range(1, max_lag + 1):
        rk = np.sum(centered[k:] * centered[:-k]) / (n * var)
        lags.append(k)
        values.append(float(rk))

    return lags, values


def compute_diagnostics(
    model_name: str,
    residuals: np.ndarray,
    fitted: np.ndarray,
) -> DiagnosticsResult:
    """Compute residual diagnostics for a model's forecast errors.

    Args:
        model_name: Name of the model.
        residuals: Array of (actual - predicted) values.
        fitted: Array of predicted values.

    Returns:
        DiagnosticsResult with stats, histogram, ACF, and scatter data.
    """
    residuals = np.asarray(residuals, dtype=float)
    fitted = np.asarray(fitted, dtype=float)

    # Remove NaN/inf
    mask = np.isfinite(residuals)
    residuals = residuals[mask]
    fitted = fitted[mask]

    n = len(residuals)
    if n == 0:
        return DiagnosticsResult(
            model_name=model_name,
            residual_mean=0.0,
            residual_std=0.0,
            residual_skew=0.0,
            residual_kurtosis=0.0,
            ljung_box_p=1.0,
        )

    mean = float(np.mean(residuals))
    std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0

    # Skewness and kurtosis
    if std > 0 and n > 2:
        centered = residuals - mean
        skew = float(np.mean(centered**3) / std**3)
        kurtosis = float(np.mean(centered**4) / std**4 - 3.0)  # excess kurtosis
    else:
        skew = 0.0
        kurtosis = 0.0

    # Ljung-Box test
    lb_p = _ljung_box_p(residuals)

    # Histogram
    n_bins = min(50, max(10, n // 5))
    counts, bin_edges = np.histogram(residuals, bins=n_bins)
    histogram_bins = [float(b) for b in bin_edges]
    histogram_counts = [int(c) for c in counts]

    # ACF
    acf_lags, acf_values = _acf(residuals)

    # Subsample for scatter plot if too many points
    max_scatter = 500
    if n > max_scatter:
        idx = np.random.default_rng(42).choice(n, max_scatter, replace=False)
        idx.sort()
        scatter_residuals = residuals[idx].tolist()
        scatter_fitted = fitted[idx].tolist()
    else:
        scatter_residuals = residuals.tolist()
        scatter_fitted = fitted.tolist()

    return DiagnosticsResult(
        model_name=model_name,
        residual_mean=round(mean, 6),
        residual_std=round(std, 6),
        residual_skew=round(skew, 4),
        residual_kurtosis=round(kurtosis, 4),
        ljung_box_p=round(lb_p, 4),
        histogram_bins=histogram_bins,
        histogram_counts=histogram_counts,
        acf_lags=acf_lags,
        acf_values=[round(v, 4) for v in acf_values],
        residuals=scatter_residuals,
        fitted=scatter_fitted,
    )
