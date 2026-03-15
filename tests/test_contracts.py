"""Tests for core data contracts."""

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataCharacteristics,
    DataProfile,
    FoldResult,
    ForecastOutput,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.runners.optional import OptionalRunnerStatus


def _make_benchmark_result():
    return BenchmarkResult(
        profile=DataProfile(
            n_series=3,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=100,
            max_length=200,
            total_rows=450,
        ),
        config=BenchmarkConfig(horizon=14, n_folds=3),
        models=[
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.12,
                folds=[
                    FoldResult(fold=1, cutoff="2020-06-01", mase=1.02),
                    FoldResult(fold=2, cutoff="2020-07-01", mase=0.98),
                ],
                mean_mase=1.00,
                std_mase=0.02,
            ),
            ModelResult(
                name="AutoETS",
                runtime_sec=0.45,
                folds=[
                    FoldResult(fold=1, cutoff="2020-06-01", mase=0.91),
                    FoldResult(fold=2, cutoff="2020-07-01", mase=0.95),
                ],
                mean_mase=0.93,
                std_mase=0.02,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.93),
            LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.00),
        ],
    )


def test_data_profile_round_trip():
    profile = DataProfile(
        n_series=3,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=100,
        max_length=200,
        total_rows=450,
    )
    assert DataProfile.from_dict(profile.to_dict()) == profile


def test_forecast_output_round_trip():
    fo = ForecastOutput(
        unique_id=["s1", "s1"],
        ds=["2020-01-01", "2020-01-02"],
        y_hat=[1.5, 2.5],
        model_name="SeasonalNaive",
        runtime_sec=0.05,
    )
    assert ForecastOutput.from_dict(fo.to_dict()) == fo


def test_fold_result_round_trip():
    fr = FoldResult(fold=1, cutoff="2020-06-01", mase=1.02)
    assert FoldResult.from_dict(fr.to_dict()) == fr


def test_model_result_round_trip():
    mr = ModelResult(
        name="AutoETS",
        runtime_sec=0.45,
        folds=[
            FoldResult(fold=1, cutoff="2020-06-01", mase=0.91),
            FoldResult(fold=2, cutoff="2020-07-01", mase=0.95),
        ],
        mean_mase=0.93,
        std_mase=0.02,
    )
    assert ModelResult.from_dict(mr.to_dict()) == mr


def test_benchmark_result_round_trip():
    result = _make_benchmark_result()
    assert BenchmarkResult.from_dict(result.to_dict()) == result


def test_benchmark_result_json_round_trip():
    result = _make_benchmark_result()
    json_str = result.to_json()
    restored = BenchmarkResult.from_json(json_str)
    assert restored == result


def test_benchmark_result_schema_keys():
    result = _make_benchmark_result()
    d = result.to_dict()
    assert set(d.keys()) == {"profile", "config", "models", "leaderboard"}
    assert set(d["models"][0].keys()) == {
        "name",
        "runtime_sec",
        "folds",
        "mean_mase",
        "std_mase",
        "mean_smape",
        "mean_rmsse",
        "mean_mae",
    }
    assert set(d["leaderboard"][0].keys()) == {
        "rank",
        "name",
        "mean_mase",
        "mean_smape",
        "mean_rmsse",
        "mean_mae",
    }
    assert set(d["config"].keys()) == {"horizon", "n_folds"}


def test_data_characteristics_round_trip():
    dc = DataCharacteristics(
        y_mean=10.5,
        y_std=3.2,
        y_min=1.0,
        y_max=25.0,
        y_median=10.0,
        y_skewness=0.5,
        y_kurtosis=-0.3,
        mean_cv=0.3,
        trend_strength=0.7,
        seasonality_strength=0.4,
        series_heterogeneity=0.8,
    )
    assert DataCharacteristics.from_dict(dc.to_dict()) == dc


def test_benchmark_result_details_with_data_chars():
    """BenchmarkResult includes data_characteristics in details dict."""
    result = _make_benchmark_result()
    result.data_characteristics = DataCharacteristics(
        y_mean=10.0,
        trend_strength=0.5,
    )
    details = result.to_details_dict()
    assert "data_characteristics" in details
    assert details["data_characteristics"]["y_mean"] == 10.0


def test_benchmark_result_from_dict_without_data_chars():
    """Old results.json without data_characteristics still loads."""
    result = _make_benchmark_result()
    d = result.to_dict()
    restored = BenchmarkResult.from_dict(d)
    assert restored.data_characteristics is None


def test_benchmark_result_details_with_optional_model_environment():
    result = _make_benchmark_result()
    result._optional_runner_statuses = [
        OptionalRunnerStatus(
            label="Prophet",
            available=True,
            reason="available",
            runner_names=["Prophet"],
        ),
        OptionalRunnerStatus(
            label="NeuralForecast",
            available=False,
            reason="failed health check",
            runner_names=["NHITS", "NBEATS"],
        ),
    ]

    details = result.to_details_dict()

    assert "optional_model_environment" in details
    assert details["optional_model_environment"][0]["label"] == "Prophet"
    assert details["optional_model_environment"][1]["reason"] == "failed health check"


def test_benchmark_result_from_dict_with_optional_model_environment():
    result = _make_benchmark_result()
    combined = result.to_dict()
    combined["optional_model_environment"] = [
        {
            "label": "LightGBM",
            "available": False,
            "reason": "missing dependency: lightgbm",
            "runner_names": ["LightGBM"],
        }
    ]

    restored = BenchmarkResult.from_dict(combined)

    assert restored._optional_runner_statuses[0]["label"] == "LightGBM"
