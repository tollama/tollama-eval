"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Symmetric Mean Absolute Percentage Error.

    SMAPE = mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|)) * 100

    Returns a value between 0 and 200. Lower is better.

    Args:
        y_true: Actual values in the forecast horizon.
        y_pred: Predicted values.

    Returns:
        SMAPE score as a percentage. 0 = perfect, 200 = worst.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero: when both actual and predicted are 0, error is 0
    mask = denominator > 0
    result = np.zeros_like(y_true)
    result[mask] = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return float(np.mean(result) * 100)


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_length: int = 1,
) -> float:
    """Root Mean Squared Scaled Error.

    RMSSE = sqrt(mean((y_true - y_pred)^2)) / scale
    where scale = sqrt(mean((y_train[m:] - y_train[:-m])^2)), m = season_length.

    Args:
        y_true: Actual values in the forecast horizon.
        y_pred: Predicted values.
        y_train: Training series for scaling denominator.
        season_length: Seasonal period (1 = non-seasonal scaling).

    Returns:
        RMSSE score. Lower is better. 1.0 = seasonal naive baseline.

    Raises:
        ValueError: if y_train has fewer than season_length + 1 observations.
        ZeroDivisionError: if scale is 0 (constant training series).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if season_length < 1:
        raise ValueError(f"season_length must be >= 1, got {season_length}.")

    if len(y_train) < season_length + 1:
        raise ValueError(
            f"y_train has {len(y_train)} observations, needs at least "
            f"{season_length + 1} for season_length={season_length}."
        )

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    diffs_sq = (y_train[season_length:] - y_train[:-season_length]) ** 2
    scale = float(np.sqrt(np.mean(diffs_sq)))

    if scale == 0:
        raise ZeroDivisionError(
            "RMSSE denominator is zero: training series has no variation."
        )

    return rmse / scale


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAE score. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


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

    if season_length < 1:
        raise ValueError(f"season_length must be >= 1, got {season_length}.")

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
            actuals_df.loc[actuals_df["unique_id"] == uid].sort_values("ds")["y"].values
        )
        pred_series = (
            forecast_df.loc[forecast_df["unique_id"] == uid]
            .sort_values("ds")[model_col]
            .values
        )
        train_series = (
            train_df.loc[train_df["unique_id"] == uid].sort_values("ds")["y"].values
        )

        scores[str(uid)] = mase(actual_series, pred_series, train_series, season_length)

    return scores


def per_series_smape(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    model_col: str,
) -> dict[str, float]:
    """Compute SMAPE for each series in a single fold."""
    scores: dict[str, float] = {}
    for uid in actuals_df["unique_id"].unique():
        actual_series = (
            actuals_df.loc[actuals_df["unique_id"] == uid].sort_values("ds")["y"].values
        )
        pred_series = (
            forecast_df.loc[forecast_df["unique_id"] == uid]
            .sort_values("ds")[model_col]
            .values
        )
        scores[str(uid)] = smape(actual_series, pred_series)
    return scores


def per_series_rmsse(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    train_df: pd.DataFrame,
    season_length: int,
    model_col: str,
) -> dict[str, float]:
    """Compute RMSSE for each series in a single fold."""
    scores: dict[str, float] = {}
    for uid in actuals_df["unique_id"].unique():
        actual_series = (
            actuals_df.loc[actuals_df["unique_id"] == uid].sort_values("ds")["y"].values
        )
        pred_series = (
            forecast_df.loc[forecast_df["unique_id"] == uid]
            .sort_values("ds")[model_col]
            .values
        )
        train_series = (
            train_df.loc[train_df["unique_id"] == uid].sort_values("ds")["y"].values
        )
        scores[str(uid)] = rmsse(
            actual_series, pred_series, train_series, season_length
        )
    return scores


def per_series_mae(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    model_col: str,
) -> dict[str, float]:
    """Compute MAE for each series in a single fold."""
    scores: dict[str, float] = {}
    for uid in actuals_df["unique_id"].unique():
        actual_series = (
            actuals_df.loc[actuals_df["unique_id"] == uid].sort_values("ds")["y"].values
        )
        pred_series = (
            forecast_df.loc[forecast_df["unique_id"] == uid]
            .sort_values("ds")[model_col]
            .values
        )
        scores[str(uid)] = mae(actual_series, pred_series)
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
