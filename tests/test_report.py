"""Tests for HTML report generation."""

import pytest

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.reporting.html_report import render_report


@pytest.fixture
def benchmark_result():
    return BenchmarkResult(
        profile=DataProfile(
            n_series=2,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=60,
            max_length=60,
            total_rows=120,
        ),
        config=BenchmarkConfig(horizon=14, n_folds=3),
        models=[
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
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.93),
            LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.00),
        ],
    )


def test_report_contains_dataset_summary(benchmark_result):
    html = render_report(benchmark_result)
    assert "Dataset Summary" in html
    assert "Number of Series" in html


def test_report_contains_leaderboard(benchmark_result):
    html = render_report(benchmark_result)
    assert "Leaderboard" in html
    assert "AutoETS" in html
    assert "SeasonalNaive" in html


def test_report_contains_model_details(benchmark_result):
    html = render_report(benchmark_result)
    assert "Model Details" in html
    assert "Mean MASE" in html


def test_report_is_valid_html(benchmark_result):
    html = render_report(benchmark_result)
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_report_rank1_highlighted(benchmark_result):
    html = render_report(benchmark_result)
    assert 'class="rank-1"' in html
