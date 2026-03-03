"""Tests for core data contracts."""

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    ForecastOutput,
    LeaderboardEntry,
    ModelResult,
)


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
    }
    assert set(d["leaderboard"][0].keys()) == {"rank", "name", "mean_mase"}
    assert set(d["config"].keys()) == {"horizon", "n_folds"}
