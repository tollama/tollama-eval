"""Tests for executive summary generation."""

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.reporting.executive_summary import (
    ExecutiveSummary,
    generate_executive_summary,
)


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
                    series_scores={
                        "s1": mean_mase - 0.1,
                        "s2": mean_mase + 0.1,
                    },
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


def test_summary_returns_dataclass():
    result = _make_result()
    summary = generate_executive_summary(result)
    assert isinstance(summary, ExecutiveSummary)
    assert summary.to_flat_text()


def test_summary_mentions_models_count():
    result = _make_result()
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "2 forecasting models" in text


def test_summary_mentions_winner():
    result = _make_result(mean_mase=0.85)
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "AutoETS" in text
    assert "0.8500" in text


def test_summary_reports_naive_improvement():
    result = _make_result(mean_mase=0.85)
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "15.0%" in text


def test_summary_warns_all_above_naive():
    """When all models score > 1.0, summary should flag it."""
    result = _make_result(mean_mase=1.2, n_models=1)
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "no model beat" in text


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
    text = summary.to_flat_text()
    assert "No models produced valid results" in text


def test_summary_stability_assessment():
    result = _make_result(mean_mase=0.85, std_mase=0.01)
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "stability" in text


def test_summary_missing_data_warning():
    result = _make_result(missing_ratio=0.15)
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "Missing data ratio" in text


def test_summary_runner_up_close():
    """Close runner-up should be mentioned."""
    result = _make_result(mean_mase=0.995)
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "SeasonalNaive" in text


def test_summary_high_variability():
    result = _make_result(mean_mase=0.85, std_mase=0.2)
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "high variability" in text


def test_summary_smape_included():
    result = _make_result()
    # Default smape is 0, so it won't show
    result.leaderboard[0] = LeaderboardEntry(
        rank=1,
        name="AutoETS",
        mean_mase=0.85,
        mean_smape=12.5,
    )
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "SMAPE" in text


def test_summary_worst_series_flagged():
    """Series with MASE > 2.0 should be flagged."""
    result = _make_result(mean_mase=0.85)
    result.models[0].folds[0].series_scores = {
        "s1": 0.5,
        "s2": 3.0,
    }
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "s2" in text


def test_summary_model_comparison_3_models():
    """With 3+ models, comparison narrative should appear."""
    result = _make_result(n_models=2)
    result.models.append(
        ModelResult(
            name="AutoARIMA",
            runtime_sec=3.0,
            folds=[FoldResult(fold=1, cutoff="2020-06-01", mase=0.9)],
            mean_mase=0.9,
            std_mase=0.03,
        )
    )
    result.leaderboard.append(LeaderboardEntry(rank=3, name="AutoARIMA", mean_mase=0.9))
    summary = generate_executive_summary(result)
    text = summary.to_flat_text().lower()
    assert "beat" in text or "naive" in text


def test_summary_single_model():
    result = _make_result(n_models=1)
    summary = generate_executive_summary(result)
    text = summary.to_flat_text()
    assert "1 forecasting model" in text


def test_summary_recommendations_present():
    """Summary should include actionable recommendations."""
    result = _make_result(mean_mase=0.75)
    summary = generate_executive_summary(result)
    assert len(summary.recommendations) > 0


def test_summary_risks_missing_data():
    """High missing ratio should appear in risks."""
    result = _make_result(missing_ratio=0.15)
    summary = generate_executive_summary(result)
    assert any("Missing" in r for r in summary.risks)
