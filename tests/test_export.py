"""Tests for data export utilities."""

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.reporting.export import (
    export_fold_details_csv,
    export_leaderboard_csv,
    export_per_series_csv,
)


def _make_result():
    """Create a minimal BenchmarkResult for testing."""
    profile = DataProfile(
        n_series=2,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=60,
        max_length=60,
        total_rows=120,
    )
    config = BenchmarkConfig(horizon=14, n_folds=2)
    folds_a = [
        FoldResult(
            fold=1,
            cutoff="2020-02-15",
            mase=0.8,
            smape=10.0,
            series_scores={"s1": 0.7, "s2": 0.9},
        ),
        FoldResult(
            fold=2,
            cutoff="2020-02-29",
            mase=0.85,
            smape=11.0,
            series_scores={"s1": 0.75, "s2": 0.95},
        ),
    ]
    folds_b = [
        FoldResult(
            fold=1,
            cutoff="2020-02-15",
            mase=1.1,
            smape=15.0,
            series_scores={"s1": 1.0, "s2": 1.2},
        ),
        FoldResult(
            fold=2,
            cutoff="2020-02-29",
            mase=1.2,
            smape=16.0,
            series_scores={"s1": 1.1, "s2": 1.3},
        ),
    ]
    models = [
        ModelResult(
            name="ModelA",
            runtime_sec=1.0,
            folds=folds_a,
            mean_mase=0.825,
            std_mase=0.025,
            mean_smape=10.5,
        ),
        ModelResult(
            name="ModelB",
            runtime_sec=2.0,
            folds=folds_b,
            mean_mase=1.15,
            std_mase=0.05,
            mean_smape=15.5,
        ),
    ]
    leaderboard = [
        LeaderboardEntry(rank=1, name="ModelA", mean_mase=0.825, mean_smape=10.5),
        LeaderboardEntry(rank=2, name="ModelB", mean_mase=1.15, mean_smape=15.5),
    ]
    return BenchmarkResult(
        profile=profile,
        config=config,
        models=models,
        leaderboard=leaderboard,
    )


def test_export_leaderboard_csv(tmp_path):
    result = _make_result()
    path = export_leaderboard_csv(result, tmp_path / "lb.csv")
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3  # header + 2 models
    assert "ModelA" in lines[1]
    assert "rank" in lines[0]


def test_export_fold_details_csv(tmp_path):
    result = _make_result()
    path = export_fold_details_csv(result, tmp_path / "folds.csv")
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    # header + 2 models * 2 folds = 5 lines
    assert len(lines) == 5
    assert "model" in lines[0]
    assert "fold" in lines[0]


def test_export_per_series_csv(tmp_path):
    result = _make_result()
    path = export_per_series_csv(result, tmp_path / "series.csv")
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    # header + 2 models * 2 folds * 2 series = 9 lines
    assert len(lines) == 9
    assert "series" in lines[0]


def test_export_empty_leaderboard(tmp_path):
    result = _make_result()
    result.leaderboard = []
    path = export_leaderboard_csv(result, tmp_path / "empty.csv")
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    # Just header row (pandas writes header even with empty df)
    assert len(lines) >= 1


def test_export_creates_parent_dirs(tmp_path):
    result = _make_result()
    path = export_leaderboard_csv(result, tmp_path / "sub" / "dir" / "lb.csv")
    assert path.exists()
