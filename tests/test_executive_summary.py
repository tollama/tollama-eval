"""Tests for executive summary generation."""

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.reporting.executive_summary import generate_executive_summary


def _make_result(
    mean_mase=0.85,
    std_mase=0.02,
    n_series=5,
    n_models=2,
    missing_ratio=0.0,
):
    models = [
        ModelResult(
            name="AutoETS",
            runtime_sec=1.0,
            folds=[
                FoldResult(
                    fold=1,
                    cutoff="2020-06-01",
                    mase=mean_mase,
                    series_scores={"s1": mean_mase - 0.1, "s2": mean_mase + 0.1},
                ),
            ],
            mean_mase=mean_mase,
            std_mase=std_mase,
        ),
    ]
    if n_models >= 2:
        models.append(
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.5,
                folds=[FoldResult(fold=1, cutoff="2020-06-01", mase=1.0)],
                mean_mase=1.0,
                std_mase=0.01,
            )
        )

    leaderboard = sorted(models, key=lambda m: m.mean_mase)
    lb_entries = [
        LeaderboardEntry(rank=i + 1, name=m.name, mean_mase=m.mean_mase)
        for i, m in enumerate(leaderboard)
    ]

    return BenchmarkResult(
        profile=DataProfile(
            n_series=n_series,
            frequency="D",
            missing_ratio=missing_ratio,
            season_length_guess=7,
            min_length=100,
            max_length=100,
            total_rows=n_series * 100,
        ),
        config=BenchmarkConfig(horizon=14, n_folds=3),
        models=models,
        leaderboard=lb_entries,
    )


def test_summary_mentions_models_count():
    result = _make_result()
    summary = generate_executive_summary(result)
    assert "2 forecasting models" in summary


def test_summary_mentions_winner():
    result = _make_result(mean_mase=0.85)
    summary = generate_executive_summary(result)
    assert "AutoETS" in summary
    assert "0.8500" in summary


def test_summary_reports_naive_improvement():
    result = _make_result(mean_mase=0.85)
    summary = generate_executive_summary(result)
    assert "15.0%" in summary


def test_summary_warns_all_above_naive():
    """When all models score > 1.0, summary should flag it."""
    result = _make_result(mean_mase=1.2, n_models=1)
    summary = generate_executive_summary(result)
    assert "no model beat" in summary


def test_summary_empty_leaderboard():
    result = BenchmarkResult(
        profile=DataProfile(
            n_series=1,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=10,
            max_length=10,
            total_rows=10,
        ),
        config=BenchmarkConfig(horizon=7, n_folds=2),
        models=[],
        leaderboard=[],
    )
    summary = generate_executive_summary(result)
    assert "No models produced valid results" in summary


def test_summary_stability_assessment():
    result = _make_result(mean_mase=0.85, std_mase=0.01)
    summary = generate_executive_summary(result)
    assert "stability" in summary


def test_summary_missing_data_warning():
    result = _make_result(missing_ratio=0.15)
    summary = generate_executive_summary(result)
    assert "Missing data ratio" in summary


def test_summary_runner_up_close():
    """Close runner-up should be mentioned."""
    result = _make_result(mean_mase=0.995)
    summary = generate_executive_summary(result)
    assert "SeasonalNaive" in summary
