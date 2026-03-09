"""Tests for ensemble construction (average, weighted, best-per-series)."""

import pytest

from ts_autopilot.contracts import ForecastData
from ts_autopilot.evaluation.ensemble import (
    EnsembleRecommendation,
    SeriesRecommendation,
    build_average_ensemble,
    build_best_per_series_ensemble,
    build_weighted_ensemble,
)


@pytest.fixture
def forecast_data():
    """Two models' forecasts for 2 series, 1 fold."""
    return [
        ForecastData(
            model_name="ModelA",
            fold=1,
            unique_id=["s1", "s1", "s2", "s2"],
            ds=["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
            y_hat=[10.0, 12.0, 20.0, 22.0],
            y_actual=[11.0, 13.0, 21.0, 23.0],
            ds_train_tail=[],
            y_train_tail=[],
        ),
        ForecastData(
            model_name="ModelB",
            fold=1,
            unique_id=["s1", "s1", "s2", "s2"],
            ds=["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
            y_hat=[14.0, 16.0, 24.0, 26.0],
            y_actual=[11.0, 13.0, 21.0, 23.0],
            ds_train_tail=[],
            y_train_tail=[],
        ),
    ]


def _zip_ens(ensemble):
    """Zip ensemble fields with strict=False."""
    return list(
        zip(
            ensemble.unique_id,
            ensemble.ds,
            ensemble.y_hat,
            strict=False,
        )
    )


class TestAverageEnsemble:
    def test_averages_predictions(self, forecast_data):
        ensemble = build_average_ensemble(forecast_data, fold=1)
        assert ensemble.method == "average"
        assert len(ensemble.y_hat) == 4
        # First point: avg(10, 14) = 12
        pairs = list(
            zip(
                ensemble.unique_id,
                ensemble.ds,
                strict=False,
            )
        )
        idx = pairs.index(("s1", "2020-01-01"))
        assert ensemble.y_hat[idx] == pytest.approx(12.0)

    def test_empty_input(self):
        ensemble = build_average_ensemble([], fold=1)
        assert ensemble.y_hat == []
        assert ensemble.component_models == []

    def test_equal_weights(self, forecast_data):
        ensemble = build_average_ensemble(forecast_data)
        for w in ensemble.weights.values():
            assert w == pytest.approx(0.5)


class TestWeightedEnsemble:
    def test_weighted_predictions(self, forecast_data):
        scores = {"ModelA": 0.5, "ModelB": 1.5}
        ensemble = build_weighted_ensemble(
            forecast_data,
            scores,
            fold=1,
        )
        assert ensemble.method == "weighted"
        assert ensemble.weights["ModelA"] > ensemble.weights["ModelB"]

    def test_empty_scores_falls_back_to_average(self, forecast_data):
        ensemble = build_weighted_ensemble(forecast_data, {}, fold=1)
        assert ensemble.method == "average"


class TestBestPerSeriesEnsemble:
    def test_selects_best_per_series(self, forecast_data):
        recommendation = EnsembleRecommendation(
            series_recommendations=[
                SeriesRecommendation(
                    series_id="s1",
                    best_model="ModelA",
                    best_mase=0.5,
                    all_scores={"ModelA": 0.5, "ModelB": 1.0},
                ),
                SeriesRecommendation(
                    series_id="s2",
                    best_model="ModelB",
                    best_mase=0.8,
                    all_scores={"ModelA": 1.2, "ModelB": 0.8},
                ),
            ],
            n_series=2,
            n_models=2,
            model_win_counts={"ModelA": 1, "ModelB": 1},
            avg_ensemble_mase=0.65,
        )
        ensemble = build_best_per_series_ensemble(
            forecast_data,
            recommendation,
            fold=1,
        )
        assert ensemble.method == "best_per_series"
        assert len(ensemble.y_hat) == 4

        # s1 should use ModelA predictions (10, 12)
        all_points = _zip_ens(ensemble)
        s1_points = [p for p in all_points if p[0] == "s1"]
        assert s1_points[0][2] == pytest.approx(10.0)

        # s2 should use ModelB predictions (24, 26)
        s2_points = [p for p in all_points if p[0] == "s2"]
        assert s2_points[0][2] == pytest.approx(24.0)
