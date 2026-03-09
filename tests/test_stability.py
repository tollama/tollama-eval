"""Tests for model stability analysis."""

import numpy as np

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.evaluation.stability import compute_stability


def _make_result(models, leaderboard=None):
    """Create a minimal BenchmarkResult for testing."""
    profile = DataProfile(
        n_series=2,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=100,
        max_length=100,
        total_rows=200,
    )
    config = BenchmarkConfig(horizon=7, n_folds=3)
    if leaderboard is None:
        sorted_models = sorted(
            [m for m in models if not np.isnan(m.mean_mase)],
            key=lambda m: m.mean_mase,
        )
        leaderboard = [
            LeaderboardEntry(rank=i + 1, name=m.name, mean_mase=m.mean_mase)
            for i, m in enumerate(sorted_models)
        ]
    return BenchmarkResult(
        profile=profile,
        config=config,
        models=models,
        leaderboard=leaderboard,
    )


def _scores(s1, s2):
    return {"s1": s1, "s2": s2}


class TestStability:
    def test_stable_model_high_score(self):
        """Consistent fold scores → high stability."""
        stable = ModelResult(
            name="Stable",
            runtime_sec=1.0,
            folds=[
                FoldResult(
                    fold=1,
                    cutoff="2020-01-01",
                    mase=1.0,
                    series_scores=_scores(1.0, 1.0),
                ),
                FoldResult(
                    fold=2,
                    cutoff="2020-01-08",
                    mase=1.0,
                    series_scores=_scores(1.0, 1.0),
                ),
                FoldResult(
                    fold=3,
                    cutoff="2020-01-15",
                    mase=1.0,
                    series_scores=_scores(1.0, 1.0),
                ),
            ],
            mean_mase=1.0,
            std_mase=0.0,
        )
        result = _make_result([stable])
        report = compute_stability(result)
        assert len(report.scores) == 1
        assert report.scores[0].stability_score > 0.8

    def test_unstable_model_low_score(self):
        """Wildly varying fold scores → lower stability."""
        unstable = ModelResult(
            name="Unstable",
            runtime_sec=1.0,
            folds=[
                FoldResult(
                    fold=1,
                    cutoff="2020-01-01",
                    mase=0.5,
                    series_scores=_scores(0.1, 5.0),
                ),
                FoldResult(
                    fold=2,
                    cutoff="2020-01-08",
                    mase=3.0,
                    series_scores=_scores(0.5, 3.0),
                ),
                FoldResult(
                    fold=3,
                    cutoff="2020-01-15",
                    mase=0.1,
                    series_scores=_scores(4.0, 0.1),
                ),
            ],
            mean_mase=1.2,
            std_mase=1.5,
        )
        result = _make_result([unstable])
        report = compute_stability(result)
        assert report.scores[0].stability_score < 0.7

    def test_model_without_folds(self):
        """Model with no folds → zero stability."""
        failed = ModelResult(
            name="Failed",
            runtime_sec=0.0,
            folds=[],
            mean_mase=float("nan"),
            std_mase=float("nan"),
        )
        result = _make_result([failed])
        report = compute_stability(result)
        assert report.scores[0].stability_score == 0.0

    def test_summary_output(self):
        model = ModelResult(
            name="Test",
            runtime_sec=1.0,
            folds=[
                FoldResult(
                    fold=1,
                    cutoff="2020-01-01",
                    mase=1.0,
                    series_scores={"s1": 1.0},
                ),
            ],
            mean_mase=1.0,
            std_mase=0.0,
        )
        result = _make_result([model])
        report = compute_stability(result)
        summary = report.summary()
        assert "Model Stability Analysis" in summary
