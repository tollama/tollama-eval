"""Tests for benchmark pipeline."""

import json

from ts_autopilot.contracts import BenchmarkResult
from ts_autopilot.pipeline import run_benchmark, run_from_csv


def test_run_benchmark_returns_result(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    assert isinstance(result, BenchmarkResult)
    assert len(result.models) == 2
    assert len(result.leaderboard) == 2


def test_leaderboard_sorted_by_mase(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    mase_values = [e.mean_mase for e in result.leaderboard]
    assert mase_values == sorted(mase_values)


def test_leaderboard_ranks_correct(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    ranks = [e.rank for e in result.leaderboard]
    assert ranks == [1, 2]


def test_model_names_are_known(tiny_long_df):
    result = run_benchmark(tiny_long_df, horizon=7, n_folds=2)
    names = {m.name for m in result.models}
    assert names == {"SeasonalNaive", "AutoETS"}


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
    assert set(data.keys()) == {"profile", "config", "models", "leaderboard"}
    assert data["config"]["horizon"] == 7
    assert data["config"]["n_folds"] == 2
    assert len(data["models"]) == 2
    assert len(data["leaderboard"]) == 2
