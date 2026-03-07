"""Tests for residual diagnostics module."""

import numpy as np

from ts_autopilot.evaluation.diagnostics import (
    _acf,
    _ljung_box_p,
    compute_diagnostics,
)


def test_compute_diagnostics_basic():
    rng = np.random.default_rng(42)
    residuals = rng.normal(0, 1, 100)
    fitted = rng.normal(50, 10, 100)
    diag = compute_diagnostics("TestModel", residuals, fitted)

    assert diag.model_name == "TestModel"
    assert abs(diag.residual_mean) < 0.5
    assert diag.residual_std > 0
    assert len(diag.histogram_bins) > 0
    assert len(diag.histogram_counts) > 0
    assert len(diag.acf_lags) > 0
    assert len(diag.acf_values) > 0
    assert len(diag.residuals) == 100
    assert len(diag.fitted) == 100


def test_compute_diagnostics_white_noise_high_p():
    """White noise residuals should have high Ljung-Box p-value."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(0, 1, 500)
    fitted = np.ones(500) * 50
    diag = compute_diagnostics("WN", residuals, fitted)
    assert diag.ljung_box_p > 0.01


def test_compute_diagnostics_empty():
    diag = compute_diagnostics("Empty", np.array([]), np.array([]))
    assert diag.residual_mean == 0.0
    assert diag.residual_std == 0.0
    assert diag.ljung_box_p == 1.0


def test_compute_diagnostics_constant_residuals():
    residuals = np.zeros(50)
    fitted = np.ones(50) * 10
    diag = compute_diagnostics("Const", residuals, fitted)
    assert diag.residual_mean == 0.0
    assert diag.residual_std == 0.0


def test_compute_diagnostics_subsamples_large():
    """Large arrays should be subsampled for scatter data."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(0, 1, 1000)
    fitted = rng.normal(50, 10, 1000)
    diag = compute_diagnostics("Big", residuals, fitted)
    assert len(diag.residuals) == 500
    assert len(diag.fitted) == 500


def test_compute_diagnostics_to_dict():
    rng = np.random.default_rng(42)
    diag = compute_diagnostics("Test", rng.normal(0, 1, 50), rng.normal(0, 1, 50))
    d = diag.to_dict()
    assert d["model_name"] == "Test"
    assert "residual_mean" in d
    assert "histogram_bins" in d


def test_acf_short_series():
    lags, values = _acf(np.array([1.0]), max_lag=5)
    assert lags == []
    assert values == []


def test_ljung_box_short_series():
    p = _ljung_box_p(np.array([1.0, 2.0]), max_lag=10)
    assert p == 1.0
