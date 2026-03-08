"""Anomaly detection module for time series.

Provides multiple detection methods:
- Statistical: Z-score, IQR, rolling statistics
- Model-based: forecast residual thresholding
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ts_autopilot.logging_config import get_logger

logger = get_logger("anomaly")


@dataclass
class Anomaly:
    """A single detected anomaly point."""

    unique_id: str
    ds: str  # ISO 8601
    value: float
    score: float  # anomaly score (higher = more anomalous)
    method: str  # detection method name
    threshold: float  # threshold used


@dataclass
class AnomalyReport:
    """Anomaly detection results for a dataset."""

    anomalies: list[Anomaly]
    n_series: int
    n_points_scanned: int
    method: str
    threshold: float
    anomaly_ratio: float  # fraction of points that are anomalies

    def summary(self) -> str:
        lines = [
            f"Anomaly Detection Report ({self.method}):",
            f"  Series scanned: {self.n_series}",
            f"  Points scanned: {self.n_points_scanned}",
            f"  Anomalies found: {len(self.anomalies)}",
            f"  Anomaly ratio: {self.anomaly_ratio:.2%}",
            f"  Threshold: {self.threshold}",
        ]
        if self.anomalies:
            top = self.anomalies[:5]
            lines.append("  Top anomalies:")
            for a in top:
                lines.append(
                    f"    {a.unique_id} @ {a.ds}: "
                    f"value={a.value:.4f}, score={a.score:.4f}"
                )
        return "\n".join(lines)


def detect_zscore(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> AnomalyReport:
    """Detect anomalies using Z-score method.

    Points with |z-score| > threshold are flagged as anomalies.

    Args:
        df: Canonical long-format DataFrame (unique_id, ds, y).
        threshold: Z-score threshold (default 3.0).

    Returns:
        AnomalyReport with detected anomalies.
    """
    anomalies: list[Anomaly] = []
    n_points = 0

    for uid, group in df.groupby("unique_id"):
        values = group["y"].values.astype(float)
        dates = group["ds"].astype(str).values
        n_points += len(values)

        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            continue

        z_scores = np.abs((values - mean) / std)
        for z, val, ds in zip(z_scores, values, dates, strict=False):
            if z > threshold:
                anomalies.append(
                    Anomaly(
                        unique_id=str(uid),
                        ds=str(ds),
                        value=float(val),
                        score=float(z),
                        method="zscore",
                        threshold=threshold,
                    )
                )

    # Sort by score descending
    anomalies.sort(key=lambda a: a.score, reverse=True)
    n_series = int(df["unique_id"].nunique())

    return AnomalyReport(
        anomalies=anomalies,
        n_series=n_series,
        n_points_scanned=n_points,
        method="zscore",
        threshold=threshold,
        anomaly_ratio=len(anomalies) / n_points if n_points > 0 else 0.0,
    )


def detect_iqr(
    df: pd.DataFrame,
    factor: float = 1.5,
) -> AnomalyReport:
    """Detect anomalies using IQR (Interquartile Range) method.

    Points outside [Q1 - factor*IQR, Q3 + factor*IQR] are flagged.

    Args:
        df: Canonical long-format DataFrame.
        factor: IQR multiplier (default 1.5).

    Returns:
        AnomalyReport with detected anomalies.
    """
    anomalies: list[Anomaly] = []
    n_points = 0

    for uid, group in df.groupby("unique_id"):
        values = group["y"].values.astype(float)
        dates = group["ds"].astype(str).values
        n_points += len(values)

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        for val, ds in zip(values, dates, strict=False):
            if val < lower or val > upper:
                # Score = distance from nearest bound, normalized by IQR
                score = max(lower - val, val - upper) / iqr
                anomalies.append(
                    Anomaly(
                        unique_id=str(uid),
                        ds=str(ds),
                        value=float(val),
                        score=float(score),
                        method="iqr",
                        threshold=factor,
                    )
                )

    anomalies.sort(key=lambda a: a.score, reverse=True)
    n_series = int(df["unique_id"].nunique())

    return AnomalyReport(
        anomalies=anomalies,
        n_series=n_series,
        n_points_scanned=n_points,
        method="iqr",
        threshold=factor,
        anomaly_ratio=len(anomalies) / n_points if n_points > 0 else 0.0,
    )


def detect_rolling(
    df: pd.DataFrame,
    window: int = 7,
    threshold: float = 3.0,
) -> AnomalyReport:
    """Detect anomalies using rolling mean/std.

    Computes rolling mean and std, flags points where deviation
    from rolling mean exceeds threshold * rolling std.

    Args:
        df: Canonical long-format DataFrame.
        window: Rolling window size (default 7).
        threshold: Number of rolling stds for threshold.

    Returns:
        AnomalyReport with detected anomalies.
    """
    anomalies: list[Anomaly] = []
    n_points = 0

    for uid, group in df.groupby("unique_id"):
        sorted_group = group.sort_values("ds")
        values = sorted_group["y"].astype(float)
        dates = sorted_group["ds"].astype(str).values
        n_points += len(values)

        if len(values) < window:
            continue

        rolling_mean = values.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = values.rolling(window=window, center=True, min_periods=1).std()

        for i in range(len(values)):
            rm = rolling_mean.iloc[i]
            rs = rolling_std.iloc[i]
            val = values.iloc[i]
            if rs > 0 and abs(val - rm) > threshold * rs:
                score = abs(val - rm) / rs
                anomalies.append(
                    Anomaly(
                        unique_id=str(uid),
                        ds=str(dates[i]),
                        value=float(val),
                        score=float(score),
                        method="rolling",
                        threshold=threshold,
                    )
                )

    anomalies.sort(key=lambda a: a.score, reverse=True)
    n_series = int(df["unique_id"].nunique())

    return AnomalyReport(
        anomalies=anomalies,
        n_series=n_series,
        n_points_scanned=n_points,
        method="rolling",
        threshold=threshold,
        anomaly_ratio=len(anomalies) / n_points if n_points > 0 else 0.0,
    )


def detect_residual(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    model_col: str,
    threshold: float = 3.0,
) -> AnomalyReport:
    """Detect anomalies from model forecast residuals.

    Points where |residual| > threshold * std(residuals) are flagged.

    Args:
        actuals: DataFrame with (unique_id, ds, y).
        forecasts: DataFrame with (unique_id, ds, model_col).
        model_col: Column name containing forecast values.
        threshold: Residual std multiplier.

    Returns:
        AnomalyReport with detected anomalies.
    """
    merged = actuals.merge(
        forecasts[["unique_id", "ds", model_col]],
        on=["unique_id", "ds"],
        how="inner",
    )

    anomalies: list[Anomaly] = []
    n_points = 0

    for uid, group in merged.groupby("unique_id"):
        residuals = (group["y"] - group[model_col]).values.astype(float)
        dates = group["ds"].astype(str).values
        values = group["y"].values.astype(float)
        n_points += len(residuals)

        std_r = np.std(residuals)
        if std_r == 0:
            continue

        z_scores = np.abs(residuals) / std_r
        for z, val, ds in zip(z_scores, values, dates, strict=False):
            if z > threshold:
                anomalies.append(
                    Anomaly(
                        unique_id=str(uid),
                        ds=str(ds),
                        value=float(val),
                        score=float(z),
                        method="residual",
                        threshold=threshold,
                    )
                )

    anomalies.sort(key=lambda a: a.score, reverse=True)
    n_series = int(merged["unique_id"].nunique())

    return AnomalyReport(
        anomalies=anomalies,
        n_series=n_series,
        n_points_scanned=n_points,
        method="residual",
        threshold=threshold,
        anomaly_ratio=len(anomalies) / n_points if n_points > 0 else 0.0,
    )


def run_all_detectors(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> list[AnomalyReport]:
    """Run all statistical anomaly detectors on a dataset.

    Args:
        df: Canonical long-format DataFrame.
        threshold: Detection threshold.

    Returns:
        List of AnomalyReports from each method.
    """
    return [
        detect_zscore(df, threshold=threshold),
        detect_iqr(df, factor=1.5),
        detect_rolling(df, window=7, threshold=threshold),
    ]
