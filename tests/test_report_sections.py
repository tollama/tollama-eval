"""Tests for enhanced report sections (Pareto chart, etc.)."""

from __future__ import annotations

import pytest

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)


@pytest.fixture()
def mock_result() -> BenchmarkResult:
    """Create a mock BenchmarkResult for report testing."""
    profile = DataProfile(
        n_series=2,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=100,
        max_length=100,
        total_rows=200,
    )
    models = [
        ModelResult(
            name="SeasonalNaive",
            runtime_sec=0.5,
            folds=[
                FoldResult(fold=1, cutoff="2020-03-01", mase=1.0, smape=20.0),
                FoldResult(fold=2, cutoff="2020-04-01", mase=1.1, smape=22.0),
            ],
            mean_mase=1.05,
            std_mase=0.05,
            mean_smape=21.0,
            mean_rmsse=1.0,
            mean_mae=5.0,
        ),
        ModelResult(
            name="AutoETS",
            runtime_sec=2.0,
            folds=[
                FoldResult(fold=1, cutoff="2020-03-01", mase=0.8, smape=15.0),
                FoldResult(fold=2, cutoff="2020-04-01", mase=0.9, smape=18.0),
            ],
            mean_mase=0.85,
            std_mase=0.05,
            mean_smape=16.5,
            mean_rmsse=0.8,
            mean_mae=3.5,
        ),
    ]
    leaderboard = [
        LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.85, mean_smape=16.5),
        LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.05, mean_smape=21.0),
    ]
    return BenchmarkResult(
        profile=profile,
        config=BenchmarkConfig(horizon=14, n_folds=2),
        models=models,
        leaderboard=leaderboard,
    )


def test_pareto_chart_data(mock_result: BenchmarkResult) -> None:
    """Pareto chart data should contain model points and frontier."""
    from ts_autopilot.reporting.html_report import _build_pareto_chart_data

    data = _build_pareto_chart_data(mock_result)
    assert "points" in data
    assert "frontier" in data
    assert len(data["points"]) == 2

    # At least one point should be Pareto optimal
    pareto_optimal = [p for p in data["points"] if p["is_pareto"]]
    assert len(pareto_optimal) >= 1


def test_pareto_chart_empty() -> None:
    """Pareto chart with no models should return empty data."""
    from ts_autopilot.reporting.html_report import _build_pareto_chart_data

    empty_result = BenchmarkResult(
        profile=DataProfile(
            n_series=0,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=0,
            max_length=0,
            total_rows=0,
        ),
        config=BenchmarkConfig(horizon=14, n_folds=2),
        models=[],
        leaderboard=[],
    )
    data = _build_pareto_chart_data(empty_result)
    assert data["points"] == []
    assert data["frontier"] == []


def test_report_renders_with_pareto(mock_result: BenchmarkResult) -> None:
    """render_report should succeed with Pareto data."""
    from ts_autopilot.reporting.html_report import render_report

    html = render_report(mock_result)
    assert isinstance(html, str)
    assert len(html) > 0


def test_pareto_frontier_sorted(mock_result: BenchmarkResult) -> None:
    """Pareto frontier points should be sorted by runtime."""
    from ts_autopilot.reporting.html_report import _build_pareto_chart_data

    data = _build_pareto_chart_data(mock_result)
    frontier = data["frontier"]
    if len(frontier) > 1:
        runtimes = [p["runtime"] for p in frontier]
        assert runtimes == sorted(runtimes)
