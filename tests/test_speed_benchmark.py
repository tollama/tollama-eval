"""Tests for speed benchmarking and Pareto analysis."""


from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.evaluation.speed_benchmark import (
    compute_speed_report,
)


def _make_result():
    """Create a BenchmarkResult with varying speed/accuracy models."""
    profile = DataProfile(
        n_series=10,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=100,
        max_length=100,
        total_rows=1000,
    )
    config = BenchmarkConfig(horizon=7, n_folds=3)

    models = [
        ModelResult(
            name="FastBad",
            runtime_sec=1.0,
            folds=[FoldResult(fold=1, cutoff="2020-01-01", mase=2.0)],
            mean_mase=2.0,
            std_mase=0.1,
        ),
        ModelResult(
            name="SlowGood",
            runtime_sec=100.0,
            folds=[FoldResult(fold=1, cutoff="2020-01-01", mase=0.5)],
            mean_mase=0.5,
            std_mase=0.05,
        ),
        ModelResult(
            name="MediumMedium",
            runtime_sec=10.0,
            folds=[FoldResult(fold=1, cutoff="2020-01-01", mase=1.0)],
            mean_mase=1.0,
            std_mase=0.1,
        ),
    ]

    leaderboard = [
        LeaderboardEntry(rank=1, name="SlowGood", mean_mase=0.5),
        LeaderboardEntry(rank=2, name="MediumMedium", mean_mase=1.0),
        LeaderboardEntry(rank=3, name="FastBad", mean_mase=2.0),
    ]

    return BenchmarkResult(
        profile=profile,
        config=config,
        models=models,
        leaderboard=leaderboard,
    )


class TestSpeedBenchmark:
    def test_speed_profiles_computed(self):
        result = _make_result()
        report = compute_speed_report(result)
        assert len(report.profiles) == 3
        assert report.fastest_model == "FastBad"

    def test_pareto_frontier(self):
        result = _make_result()
        report = compute_speed_report(result)
        pareto = [p for p in report.pareto_points if p.is_pareto_optimal]
        # FastBad and SlowGood should be Pareto-optimal
        # (FastBad: fastest; SlowGood: most accurate)
        pareto_names = {p.model_name for p in pareto}
        assert "SlowGood" in pareto_names
        assert "FastBad" in pareto_names

    def test_throughput_positive(self):
        result = _make_result()
        report = compute_speed_report(result)
        for profile in report.profiles:
            assert profile.throughput_series_per_sec > 0

    def test_summary_output(self):
        result = _make_result()
        report = compute_speed_report(result)
        summary = report.summary()
        assert "Speed Benchmark" in summary
        assert "Fastest" in summary
