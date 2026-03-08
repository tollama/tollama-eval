"""Tests for ensemble recommendation engine."""

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.evaluation.ensemble import recommend_ensemble


def _make_result(series_scores_a=None, series_scores_b=None):
    """Create a BenchmarkResult with per-series scores."""
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

    sa = series_scores_a or {"s1": 0.7, "s2": 0.9}
    sb = series_scores_b or {"s1": 1.0, "s2": 0.6}

    folds_a = [
        FoldResult(fold=1, cutoff="2020-02-15", mase=0.8, series_scores=sa),
        FoldResult(fold=2, cutoff="2020-02-29", mase=0.85, series_scores=sa),
    ]
    folds_b = [
        FoldResult(fold=1, cutoff="2020-02-15", mase=0.8, series_scores=sb),
        FoldResult(fold=2, cutoff="2020-02-29", mase=0.8, series_scores=sb),
    ]
    models = [
        ModelResult(
            name="ModelA",
            runtime_sec=1.0,
            folds=folds_a,
            mean_mase=0.825,
            std_mase=0.025,
        ),
        ModelResult(
            name="ModelB", runtime_sec=2.0, folds=folds_b, mean_mase=0.8, std_mase=0.0
        ),
    ]
    leaderboard = [
        LeaderboardEntry(rank=1, name="ModelB", mean_mase=0.8),
        LeaderboardEntry(rank=2, name="ModelA", mean_mase=0.825),
    ]
    return BenchmarkResult(
        profile=profile,
        config=config,
        models=models,
        leaderboard=leaderboard,
    )


def test_recommend_ensemble_basic():
    result = _make_result()
    rec = recommend_ensemble(result)
    assert rec.n_series == 2
    assert rec.n_models == 2
    assert len(rec.series_recommendations) == 2


def test_ensemble_picks_best_per_series():
    # ModelA is better for s1 (0.7 < 1.0), ModelB for s2 (0.6 < 0.9)
    result = _make_result()
    rec = recommend_ensemble(result)
    by_series = {r.series_id: r for r in rec.series_recommendations}
    assert by_series["s1"].best_model == "ModelA"
    assert by_series["s2"].best_model == "ModelB"


def test_ensemble_win_counts():
    result = _make_result()
    rec = recommend_ensemble(result)
    assert rec.model_win_counts.get("ModelA", 0) == 1
    assert rec.model_win_counts.get("ModelB", 0) == 1


def test_ensemble_avg_mase():
    # Best per series: s1=0.7, s2=0.6 → avg=0.65
    result = _make_result()
    rec = recommend_ensemble(result)
    assert abs(rec.avg_ensemble_mase - 0.65) < 0.01


def test_ensemble_summary_text():
    result = _make_result()
    rec = recommend_ensemble(result)
    text = rec.summary()
    assert "Ensemble recommendation" in text
    assert "ModelA" in text
    assert "ModelB" in text
    assert "Virtual ensemble MASE" in text


def test_ensemble_no_series_scores():
    """When no per-series scores exist, return empty recommendation."""
    profile = DataProfile(
        n_series=2,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=60,
        max_length=60,
        total_rows=120,
    )
    config = BenchmarkConfig(horizon=14, n_folds=1)
    folds = [FoldResult(fold=1, cutoff="2020-02-15", mase=0.8)]
    models = [
        ModelResult(
            name="ModelA", runtime_sec=1.0, folds=folds, mean_mase=0.8, std_mase=0.0
        ),
    ]
    leaderboard = [
        LeaderboardEntry(rank=1, name="ModelA", mean_mase=0.8),
    ]
    result = BenchmarkResult(
        profile=profile,
        config=config,
        models=models,
        leaderboard=leaderboard,
    )
    rec = recommend_ensemble(result)
    assert rec.n_series == 0
    assert rec.avg_ensemble_mase == 0.0


def test_ensemble_single_model():
    """With one model, it wins all series."""
    result = _make_result()
    result.models = [result.models[0]]  # keep only ModelA
    rec = recommend_ensemble(result)
    assert rec.n_models == 1
    for r in rec.series_recommendations:
        assert r.best_model == "ModelA"
