"""Tests for probabilistic forecasting metrics (MSIS, coverage)."""

import numpy as np
import pytest

from ts_autopilot.evaluation.metrics import (
    composite_score,
    coverage,
    msis,
    parse_metric_weights,
)


class TestMSIS:
    def test_perfect_interval(self):
        """Tight interval containing all actuals → low score."""
        y_true = np.array([10.0, 11.0, 12.0])
        y_lower = np.array([9.0, 10.0, 11.0])
        y_upper = np.array([11.0, 12.0, 13.0])
        y_train = np.arange(20, dtype=float)
        score = msis(y_true, y_lower, y_upper, y_train, season_length=1)
        assert score > 0
        assert np.isfinite(score)

    def test_wide_interval_higher_score(self):
        """Wider interval → higher MSIS (penalizes width)."""
        y_true = np.array([10.0, 11.0, 12.0])
        y_train = np.arange(20, dtype=float)

        narrow_lower = np.array([9.5, 10.5, 11.5])
        narrow_upper = np.array([10.5, 11.5, 12.5])
        wide_lower = np.array([5.0, 6.0, 7.0])
        wide_upper = np.array([15.0, 16.0, 17.0])

        narrow_score = msis(y_true, narrow_lower, narrow_upper, y_train)
        wide_score = msis(y_true, wide_lower, wide_upper, y_train)
        assert wide_score > narrow_score

    def test_missing_values_penalized(self):
        """Interval that misses actuals → penalty applied."""
        y_true = np.array([10.0, 20.0, 30.0])  # 20 and 30 outside
        y_lower = np.array([9.0, 9.0, 9.0])
        y_upper = np.array([11.0, 11.0, 11.0])
        y_train = np.arange(20, dtype=float)
        score = msis(y_true, y_lower, y_upper, y_train)
        assert score > 10  # Should be high due to penalties

    def test_short_train_raises(self):
        with pytest.raises(ValueError, match="y_train has"):
            msis(
                np.array([1.0]),
                np.array([0.0]),
                np.array([2.0]),
                np.array([1.0]),  # too short for season_length=7
                season_length=7,
            )

    def test_constant_train_raises(self):
        with pytest.raises(ZeroDivisionError):
            msis(
                np.array([1.0]),
                np.array([0.0]),
                np.array([2.0]),
                np.array([5.0, 5.0, 5.0]),
            )


class TestCoverage:
    def test_full_coverage(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_lower = np.array([0.0, 1.0, 2.0])
        y_upper = np.array([2.0, 3.0, 4.0])
        assert coverage(y_true, y_lower, y_upper) == 1.0

    def test_zero_coverage(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_lower = np.array([0.0, 0.0, 0.0])
        y_upper = np.array([1.0, 1.0, 1.0])
        assert coverage(y_true, y_lower, y_upper) == 0.0

    def test_partial_coverage(self):
        y_true = np.array([1.0, 10.0])
        y_lower = np.array([0.0, 0.0])
        y_upper = np.array([2.0, 2.0])
        assert coverage(y_true, y_lower, y_upper) == 0.5


class TestCompositeScore:
    def test_default_weights(self):
        score = composite_score(
            mase_val=1.0, smape_val=10.0, rmsse_val=1.0, mae_val=5.0
        )
        assert score > 0
        assert np.isfinite(score)

    def test_custom_weights(self):
        score = composite_score(
            mase_val=1.0,
            smape_val=10.0,
            rmsse_val=1.0,
            mae_val=5.0,
            weights={"mase": 1.0, "smape": 0.0, "rmsse": 0.0, "mae": 0.0},
        )
        assert score == pytest.approx(1.0)

    def test_speed_weight(self):
        fast = composite_score(
            mase_val=1.0,
            smape_val=10.0,
            rmsse_val=1.0,
            mae_val=5.0,
            runtime_sec=1.0,
            weights={"mase": 0.5, "speed": 0.5},
        )
        slow = composite_score(
            mase_val=1.0,
            smape_val=10.0,
            rmsse_val=1.0,
            mae_val=5.0,
            runtime_sec=100.0,
            weights={"mase": 0.5, "speed": 0.5},
        )
        assert slow > fast


class TestParseMetricWeights:
    def test_valid_weights(self):
        w = parse_metric_weights("mase=0.5,smape=0.3,speed=0.2")
        assert w == {"mase": 0.5, "smape": 0.3, "speed": 0.2}

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            parse_metric_weights("mase=0.5,invalid=0.5")

    def test_bad_format(self):
        with pytest.raises(ValueError, match="Invalid weight format"):
            parse_metric_weights("mase")

    def test_bad_value(self):
        with pytest.raises(ValueError, match="Invalid weight value"):
            parse_metric_weights("mase=abc")
