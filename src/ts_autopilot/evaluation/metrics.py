"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_length: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    MASE = mean(|y_true - y_pred|) / scale
    where scale = mean(|y_train[m:] - y_train[:-m]|), m = season_length.

    Args:
        y_true: Actual values in the forecast horizon.
        y_pred: Predicted values.
        y_train: Training series for scaling denominator.
        season_length: Seasonal period (1 = non-seasonal naive scaling).

    Returns:
        MASE score. Lower is better. 1.0 = seasonal naive baseline.

    Raises:
        ValueError: if y_train has fewer than season_length + 1 observations.
        ZeroDivisionError: if scale is 0 (constant training series).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) < season_length + 1:
        raise ValueError(
            f"y_train has {len(y_train)} observations, needs at least "
            f"{season_length + 1} for season_length={season_length}."
        )

    mae = np.mean(np.abs(y_true - y_pred))
    diffs = np.abs(y_train[season_length:] - y_train[:-season_length])
    scale = np.mean(diffs)

    if scale == 0:
        raise ZeroDivisionError(
            "MASE denominator is zero: training series has no variation."
        )

    return float(mae / scale)


def per_series_mase(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    train_df: pd.DataFrame,
    season_length: int,
    model_col: str,
) -> dict[str, float]:
    """Compute MASE for each series in a single fold.

    Returns:
        Dict mapping unique_id to MASE score.
    """
    scores: dict[str, float] = {}

    for uid in actuals_df["unique_id"].unique():
        actual_series = (
            actuals_df.loc[actuals_df["unique_id"] == uid]
            .sort_values("ds")["y"]
            .values
        )
        pred_series = (
            forecast_df.loc[forecast_df["unique_id"] == uid]
            .sort_values("ds")[model_col]
            .values
        )
        train_series = (
            train_df.loc[train_df["unique_id"] == uid]
            .sort_values("ds")["y"]
            .values
        )

        scores[str(uid)] = mase(
            actual_series, pred_series, train_series, season_length
        )

    return scores


def mean_mase_per_fold(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    train_df: pd.DataFrame,
    season_length: int,
    model_col: str,
) -> float:
    """Average MASE across all series in a single fold."""
    scores = per_series_mase(
        forecast_df, actuals_df, train_df, season_length, model_col
    )
    return float(np.mean(list(scores.values())))
