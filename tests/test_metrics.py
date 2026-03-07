"""Tests for evaluation metrics — hand-checkable values."""

import numpy as np
import pytest

from ts_autopilot.evaluation.metrics import mae, mase, per_series_mase, rmsse, smape


def test_mase_perfect_forecast():
    """Perfect predictions → MASE = 0."""
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([6.0, 7.0])
    assert mase(y_true, y_pred, y_train) == 0.0


def test_mase_naive_baseline():
    """Naive forecast (repeat last value) on linear series → MASE = 1.0.

    y_train = [1, 2, 3, 4, 5], scale = mean(|diff|) = 1.0
    y_true = [6, 7], y_pred = [5, 5] → MAE = mean(|1, 2|) = 1.5
    Wait — naive(1-step) = repeat last = [5, 5], MAE = (1+2)/2 = 1.5
    That gives MASE = 1.5, not 1.0.

    For MASE = 1.0 exactly, we need MAE = scale = 1.0.
    y_pred = [5, 7] → |6-5|=1, |7-7|=0 → MAE=0.5 → MASE=0.5
    y_pred = [5, 6] → |6-5|=1, |7-6|=1 → MAE=1.0 → MASE=1.0 ✓
    """
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([5.0, 6.0])  # one-step-behind naive
    result = mase(y_true, y_pred, y_train)
    assert result == pytest.approx(1.0)


def test_mase_seasonal():
    """Seasonal MASE with season_length=2.

    y_train = [1, 3, 2, 4, 3, 5]
    seasonal diffs (m=2): |2-1|=1, |4-3|=1, |3-2|=1, |5-4|=1 → scale = 1.0
    y_true = [4, 6], y_pred = [3, 5] → MAE = 1.0 → MASE = 1.0
    """
    y_train = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    y_true = np.array([4.0, 6.0])
    y_pred = np.array([3.0, 5.0])
    result = mase(y_true, y_pred, y_train, season_length=2)
    assert result == pytest.approx(1.0)


def test_mase_known_value():
    """Concrete arithmetic check.

    y_train = [0, 1, 2, 3], scale (m=1) = mean([1,1,1]) = 1.0
    y_true = [4, 5], y_pred = [3, 6] → MAE = (1+1)/2 = 1.0 → MASE = 1.0
    """
    y_train = np.array([0.0, 1.0, 2.0, 3.0])
    y_true = np.array([4.0, 5.0])
    y_pred = np.array([3.0, 6.0])
    assert mase(y_true, y_pred, y_train) == pytest.approx(1.0)


def test_mase_zero_scale_raises():
    """Constant training series → ZeroDivisionError."""
    y_train = np.array([5.0, 5.0, 5.0, 5.0])
    y_true = np.array([6.0])
    y_pred = np.array([5.0])
    with pytest.raises(ZeroDivisionError):
        mase(y_true, y_pred, y_train)


def test_mase_short_train_raises():
    """Too-short training series raises ValueError."""
    y_train = np.array([1.0])
    y_true = np.array([2.0])
    y_pred = np.array([2.0])
    with pytest.raises(ValueError, match="needs at least"):
        mase(y_true, y_pred, y_train)


def test_per_series_mase_returns_dict():
    """per_series_mase returns dict keyed by unique_id."""
    import pandas as pd

    train = pd.DataFrame(
        {
            "unique_id": ["s1"] * 10 + ["s2"] * 10,
            "ds": list(pd.date_range("2020-01-01", periods=10, freq="D")) * 2,
            "y": list(range(10)) + list(range(0, 20, 2)),
        }
    )
    actuals = pd.DataFrame(
        {
            "unique_id": ["s1", "s2"],
            "ds": [pd.Timestamp("2020-01-11")] * 2,
            "y": [10.0, 20.0],
        }
    )
    forecast = pd.DataFrame(
        {
            "unique_id": ["s1", "s2"],
            "ds": [pd.Timestamp("2020-01-11")] * 2,
            "model": [10.0, 20.0],
        }
    )
    scores = per_series_mase(forecast, actuals, train, 1, "model")
    assert isinstance(scores, dict)
    assert set(scores.keys()) == {"s1", "s2"}
    assert scores["s1"] == pytest.approx(0.0)
    assert scores["s2"] == pytest.approx(0.0)


# --- SMAPE tests ---


def test_smape_perfect_forecast():
    """Perfect predictions → SMAPE = 0."""
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([6.0, 7.0])
    assert smape(y_true, y_pred) == 0.0


def test_smape_known_value():
    """Hand-check: y_true=[100], y_pred=[110].

    SMAPE = 2 * |100-110| / (100+110) * 100 = 2*10/210*100 ≈ 9.524%
    """
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    assert smape(y_true, y_pred) == pytest.approx(9.5238, rel=1e-3)


def test_smape_symmetric():
    """SMAPE is symmetric: swapping true/pred gives same result."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])
    assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))


def test_smape_both_zero():
    """Both actual and predicted are zero → SMAPE should be 0 (not NaN)."""
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])
    assert smape(y_true, y_pred) == 0.0


# --- RMSSE tests ---


def test_rmsse_perfect_forecast():
    """Perfect predictions → RMSSE = 0."""
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([6.0, 7.0])
    assert rmsse(y_true, y_pred, y_train) == 0.0


def test_rmsse_known_value():
    """Hand-check with linear series.

    y_train = [0, 1, 2, 3], diffs (m=1) = [1,1,1], scale = sqrt(mean([1,1,1])) = 1.0
    y_true = [4, 5], y_pred = [3, 6] → RMSE = sqrt((1+1)/2) = 1.0 → RMSSE = 1.0
    """
    y_train = np.array([0.0, 1.0, 2.0, 3.0])
    y_true = np.array([4.0, 5.0])
    y_pred = np.array([3.0, 6.0])
    assert rmsse(y_true, y_pred, y_train) == pytest.approx(1.0)


def test_rmsse_zero_scale_raises():
    """Constant training series → ZeroDivisionError."""
    y_train = np.array([5.0, 5.0, 5.0, 5.0])
    y_true = np.array([6.0])
    y_pred = np.array([5.0])
    with pytest.raises(ZeroDivisionError):
        rmsse(y_true, y_pred, y_train)


def test_rmsse_short_train_raises():
    """Too-short training series raises ValueError."""
    y_train = np.array([1.0])
    y_true = np.array([2.0])
    y_pred = np.array([2.0])
    with pytest.raises(ValueError, match="needs at least"):
        rmsse(y_true, y_pred, y_train)


# --- MAE tests ---


def test_mae_perfect_forecast():
    """Perfect predictions → MAE = 0."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mae(y_true, y_pred) == 0.0


def test_mae_known_value():
    """MAE([4,5], [3,6]) = (1+1)/2 = 1.0."""
    y_true = np.array([4.0, 5.0])
    y_pred = np.array([3.0, 6.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)
