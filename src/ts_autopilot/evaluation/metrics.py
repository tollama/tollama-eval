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


# --- Phase 3a: Probabilistic forecasting metrics ---


def msis(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_train: np.ndarray,
    season_length: int = 1,
    alpha: float = 0.1,
) -> float:
    """Mean Scaled Interval Score.

    MSIS penalizes wide intervals and intervals that don't contain the actual value.

    Args:
        y_true: Actual values in the forecast horizon.
        y_lower: Lower bound of prediction interval.
        y_upper: Upper bound of prediction interval.
        y_train: Training series for scaling.
        season_length: Seasonal period.
        alpha: Significance level (default 0.1 for 90% interval).

    Returns:
        MSIS score. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) < season_length + 1:
        raise ValueError(
            f"y_train has {len(y_train)} observations, needs at least "
            f"{season_length + 1} for season_length={season_length}."
        )

    # Interval score components
    width = y_upper - y_lower
    lower_penalty = (2.0 / alpha) * np.maximum(y_lower - y_true, 0)
    upper_penalty = (2.0 / alpha) * np.maximum(y_true - y_upper, 0)
    interval_score = width + lower_penalty + upper_penalty

    # Scale
    diffs = np.abs(y_train[season_length:] - y_train[:-season_length])
    scale = np.mean(diffs)

    if scale == 0:
        raise ZeroDivisionError(
            "MSIS denominator is zero: training series has no variation."
        )

    return float(np.mean(interval_score) / scale)


def coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Empirical coverage probability.

    Fraction of actual values that fall within the prediction interval.

    Args:
        y_true: Actual values.
        y_lower: Lower bounds.
        y_upper: Upper bounds.

    Returns:
        Coverage probability (0-1). Should be close to nominal level.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    within = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(within))


# --- Phase 8d: Multi-metric composite scoring ---


def composite_score(
    mase_val: float,
    smape_val: float,
    rmsse_val: float,
    mae_val: float,
    runtime_sec: float = 0.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted composite score across multiple metrics.

    All metrics are normalized before weighting. Lower composite score = better.

    Args:
        mase_val: MASE score.
        smape_val: SMAPE score (0-200 scale).
        rmsse_val: RMSSE score.
        mae_val: MAE score.
        runtime_sec: Model runtime in seconds.
        weights: Dict of metric_name -> weight. Default: equal weight on
                 MASE, SMAPE, RMSSE, MAE (no speed weight).

    Returns:
        Composite score (lower is better).
    """
    if weights is None:
        weights = {
            "mase": 0.4,
            "smape": 0.2,
            "rmsse": 0.2,
            "mae": 0.2,
            "speed": 0.0,
        }

    # Normalize SMAPE from [0, 200] to approximate [0, 2] for comparability
    smape_normalized = smape_val / 100.0

    score = (
        weights.get("mase", 0.0) * mase_val
        + weights.get("smape", 0.0) * smape_normalized
        + weights.get("rmsse", 0.0) * rmsse_val
        + weights.get("mae", 0.0) * mae_val
        + weights.get("speed", 0.0) * np.log1p(runtime_sec)
    )
    return float(score)


def parse_metric_weights(weights_str: str) -> dict[str, float]:
    """Parse a metric weights string like 'mase=0.5,smape=0.3,speed=0.2'.

    Args:
        weights_str: Comma-separated key=value pairs.

    Returns:
        Dict of metric name to weight.

    Raises:
        ValueError: if the string is malformed.
    """
    weights: dict[str, float] = {}
    for part in weights_str.split(","):
        part = part.strip()
        if "=" not in part:
            raise ValueError(
                f"Invalid weight format: '{part}'. Expected 'metric=value'."
            )
        key, val_str = part.split("=", 1)
        key = key.strip()
        try:
            weights[key] = float(val_str.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid weight value for '{key}': '{val_str}'") from exc

    valid_metrics = {"mase", "smape", "rmsse", "mae", "speed"}
    invalid = set(weights.keys()) - valid_metrics
    if invalid:
        raise ValueError(
            f"Unknown metric(s): {', '.join(sorted(invalid))}. "
            f"Valid: {', '.join(sorted(valid_metrics))}"
        )

    return weights
