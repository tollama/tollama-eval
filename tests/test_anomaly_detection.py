"""Tests for the anomaly detection module."""

import numpy as np
import pandas as pd
import pytest

from ts_autopilot.anomaly.detector import (
    detect_iqr,
    detect_rolling,
    detect_zscore,
    run_all_detectors,
)


@pytest.fixture
def normal_df():
    """DataFrame with normal data (no anomalies expected)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "unique_id": ["s1"] * 100,
            "ds": dates,
            "y": 10.0 + rng.normal(0, 1, 100),
        }
    )


@pytest.fixture
def anomalous_df():
    """DataFrame with obvious anomalies."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    values = 10.0 + rng.normal(0, 1, 100)
    # Insert obvious anomalies
    values[50] = 100.0  # huge spike
    values[75] = -50.0  # huge dip
    return pd.DataFrame(
        {
            "unique_id": ["s1"] * 100,
            "ds": dates,
            "y": values,
        }
    )


class TestZScoreDetector:
    def test_detects_anomalies(self, anomalous_df):
        report = detect_zscore(anomalous_df, threshold=3.0)
        assert len(report.anomalies) >= 2
        assert report.method == "zscore"
        assert report.n_points_scanned == 100

    def test_no_anomalies_in_normal_data(self, normal_df):
        report = detect_zscore(normal_df, threshold=4.0)
        assert len(report.anomalies) == 0 or report.anomaly_ratio < 0.05

    def test_anomalies_sorted_by_score(self, anomalous_df):
        report = detect_zscore(anomalous_df)
        if len(report.anomalies) > 1:
            scores = [a.score for a in report.anomalies]
            assert scores == sorted(scores, reverse=True)


class TestIQRDetector:
    def test_detects_outliers(self, anomalous_df):
        report = detect_iqr(anomalous_df, factor=1.5)
        assert len(report.anomalies) >= 2
        assert report.method == "iqr"

    def test_higher_factor_fewer_anomalies(self, anomalous_df):
        report_tight = detect_iqr(anomalous_df, factor=1.0)
        report_loose = detect_iqr(anomalous_df, factor=3.0)
        assert len(report_tight.anomalies) >= len(report_loose.anomalies)


class TestRollingDetector:
    def test_detects_anomalies(self, anomalous_df):
        report = detect_rolling(anomalous_df, window=7, threshold=2.0)
        assert len(report.anomalies) >= 1
        assert report.method == "rolling"


class TestRunAllDetectors:
    def test_returns_multiple_reports(self, anomalous_df):
        reports = run_all_detectors(anomalous_df)
        assert len(reports) == 3
        methods = {r.method for r in reports}
        assert methods == {"zscore", "iqr", "rolling"}

    def test_summary_output(self, anomalous_df):
        reports = run_all_detectors(anomalous_df)
        for report in reports:
            summary = report.summary()
            assert "Anomaly Detection Report" in summary
