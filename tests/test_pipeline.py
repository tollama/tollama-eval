"""Tests for benchmark pipeline."""

import json

from ts_autopilot.contracts import BenchmarkResult, DataProfile
from ts_autopilot.pipeline import generate_warnings, run_benchmark, run_from_csv


def test_run_benchmark_returns_result(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    assert isinstance(result, BenchmarkResult)
    assert len(result.models) >= 5  # 5 core + optional runners
    assert len(result.leaderboard) >= 5


def test_leaderboard_sorted_by_mase(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    mase_values = [e.mean_mase for e in result.leaderboard]
    assert mase_values == sorted(mase_values)


def test_leaderboard_ranks_correct(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    ranks = [e.rank for e in result.leaderboard]
    assert ranks == list(range(1, len(ranks) + 1))


def test_model_names_are_known(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    names = {m.name for m in result.models}
    assert {"SeasonalNaive", "AutoETS", "AutoARIMA", "AutoTheta", "CES"} <= names


def test_each_model_has_correct_folds(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    for model in result.models:
        assert len(model.folds) == 2
        assert [f.fold for f in model.folds] == [1, 2]


def test_run_from_csv_writes_files(tmp_path):
    """Integration test: CSV → results.json + report.html."""
    import pandas as pd

    # Create a small CSV
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": str(d.date()), "y": float(i)})
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    run_from_csv(csv_path, horizon=7, n_folds=2, output_dir=out_dir)

    assert (out_dir / "results.json").exists()
    assert (out_dir / "report.html").exists()


def test_results_json_schema(tmp_path):
    """Verify the exact keys in results.json."""
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": str(d.date()), "y": float(i)})
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    run_from_csv(csv_path, horizon=7, n_folds=2, output_dir=out_dir)

    data = json.loads((out_dir / "results.json").read_text())
    assert {"profile", "config", "models", "leaderboard"} <= set(data.keys())
    assert data["config"]["horizon"] == 7
    assert data["config"]["n_folds"] == 2
    assert len(data["models"]) >= 5
    assert len(data["leaderboard"]) >= 5


def test_model_names_filter(tiny_long_df):
    """Only run selected models when model_names is provided."""
    result = run_benchmark(
        tiny_long_df, horizon=7, n_folds=2, model_names=["SeasonalNaive"]
    )
    assert len(result.models) == 1
    assert result.models[0].name == "SeasonalNaive"
    assert len(result.leaderboard) == 1


def test_model_names_unknown_raises(tiny_long_df):
    """Unknown model name raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown model"):
        run_benchmark(tiny_long_df, horizon=7, n_folds=2, model_names=["FakeModel"])


def test_progress_callback_called(tiny_long_df):
    """Progress callback is invoked for models and folds."""
    calls = []

    def cb(step, current, total):
        calls.append((step, current, total))

    run_benchmark(tiny_long_df, horizon=7, n_folds=2, progress_callback=cb)
    model_calls = [c for c in calls if c[0] == "model"]
    fold_calls = [c for c in calls if c[0] == "fold"]
    assert len(model_calls) >= 5  # 5+ runners
    assert len(fold_calls) >= 10  # 2 folds x 5+ runners


def test_generate_warnings_short_series():
    profile = DataProfile(
        n_series=2,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=10,
        max_length=20,
        total_rows=30,
    )
    warnings = generate_warnings(profile, horizon=7, n_folds=3)
    assert any("rows" in w and "unreliable" in w for w in warnings)


def test_generate_warnings_high_missing():
    profile = DataProfile(
        n_series=2,
        frequency="D",
        missing_ratio=0.25,
        season_length_guess=7,
        min_length=100,
        max_length=100,
        total_rows=200,
    )
    warnings = generate_warnings(profile, horizon=7, n_folds=3)
    assert any("Missing ratio" in w for w in warnings)


def test_generate_warnings_single_series():
    profile = DataProfile(
        n_series=1,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=100,
        max_length=100,
        total_rows=100,
    )
    warnings = generate_warnings(profile, horizon=7, n_folds=3)
    assert any("1 series" in w for w in warnings)


def test_generate_warnings_clean_data():
    """No warnings for well-formed data."""
    profile = DataProfile(
        n_series=5,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=200,
        max_length=200,
        total_rows=1000,
    )
    warnings = generate_warnings(profile, horizon=7, n_folds=3)
    assert warnings == []


def test_result_includes_warnings(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    assert isinstance(result.warnings, list)


def test_default_runners_is_immutable():
    """DEFAULT_RUNNERS should be a tuple to prevent accidental mutation."""
    from ts_autopilot.pipeline import DEFAULT_RUNNERS

    assert isinstance(DEFAULT_RUNNERS, tuple)


def test_fold_results_contain_series_scores(tiny_long_df):
    """Per-series MASE scores are captured in fold results."""
    result = run_benchmark(
        tiny_long_df, horizon=7, n_folds=2, model_names=["SeasonalNaive"]
    )
    for model in result.models:
        for fold in model.folds:
            assert isinstance(fold.series_scores, dict)
            assert len(fold.series_scores) > 0
            input_uids = set(tiny_long_df["unique_id"].unique())
            assert set(fold.series_scores.keys()) == input_uids


def test_warnings_are_deduplicated(tiny_long_df):
    """Duplicate warnings should be removed from benchmark results."""
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    assert len(result.warnings) == len(set(result.warnings))
