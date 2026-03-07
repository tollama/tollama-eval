"""Tests for enhanced report features."""

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    DiagnosticsResult,
    FoldResult,
    ForecastData,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.reporting.html_report import render_report


def _make_enriched_result():
    """Create a BenchmarkResult with forecast data and diagnostics."""
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
        config=BenchmarkConfig(horizon=7, n_folds=2),
        models=[
            ModelResult(
                name="AutoETS",
                runtime_sec=0.5,
                folds=[
                    FoldResult(
                        fold=1,
                        cutoff="2020-06-01",
                        mase=0.85,
                        series_scores={"s1": 0.75, "s2": 0.95},
                    ),
                    FoldResult(
                        fold=2,
                        cutoff="2020-07-01",
                        mase=0.90,
                        series_scores={"s1": 0.80, "s2": 1.00},
                    ),
                ],
                mean_mase=0.875,
                std_mase=0.025,
                mean_smape=5.0,
                mean_rmsse=0.9,
                mean_mae=1.5,
            ),
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.1,
                folds=[
                    FoldResult(fold=1, cutoff="2020-06-01", mase=1.0),
                    FoldResult(fold=2, cutoff="2020-07-01", mase=1.0),
                ],
                mean_mase=1.0,
                std_mase=0.0,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(
                rank=1,
                name="AutoETS",
                mean_mase=0.875,
                mean_smape=5.0,
                mean_rmsse=0.9,
                mean_mae=1.5,
            ),
            LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.0),
        ],
        forecast_data=[
            ForecastData(
                model_name="AutoETS",
                fold=2,
                unique_id=["s1", "s1", "s2", "s2"],
                ds=["2020-07-02", "2020-07-03", "2020-07-02", "2020-07-03"],
                y_hat=[10.0, 11.0, 20.0, 21.0],
                y_actual=[10.5, 10.8, 20.3, 21.2],
                ds_train_tail=["2020-06-30", "2020-07-01"],
                y_train_tail=[9.0, 9.5],
            ),
        ],
        diagnostics=[
            DiagnosticsResult(
                model_name="AutoETS",
                residual_mean=0.1,
                residual_std=0.5,
                residual_skew=0.1,
                residual_kurtosis=-0.2,
                ljung_box_p=0.42,
                histogram_bins=[0.0, 0.5, 1.0],
                histogram_counts=[3, 2],
                acf_lags=[1, 2, 3],
                acf_values=[0.1, -0.05, 0.02],
                residuals=[0.1, -0.2, 0.3, 0.0],
                fitted=[10.0, 11.0, 20.0, 21.0],
            ),
        ],
    )


def test_report_has_executive_summary():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Executive Summary" in html
    assert "AutoETS" in html


def test_report_has_toc():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Navigate" in html
    assert "#leaderboard" in html


def test_report_has_methodology():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Methodology" in html
    assert "Cross-Validation Strategy" in html
    assert "MASE" in html


def test_report_has_diagnostics_section():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Residual Diagnostics" in html
    assert "Ljung-Box" in html


def test_report_has_forecast_section():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Forecast vs Actual" in html


def test_report_has_model_comparison():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Model Comparison" in html
    assert "chart-radar" in html
    assert "chart-fold-stability" in html


def test_report_has_stats_grid():
    result = _make_enriched_result()
    html = render_report(result)
    assert "stats-grid" in html
    assert "stat-card" in html


def test_report_has_professional_header():
    result = _make_enriched_result()
    html = render_report(result)
    assert "report-header" in html
    assert "Benchmark Report" in html


def test_report_no_diagnostics_without_data():
    """Report should not show diagnostics/forecast sections when no data."""
    result = BenchmarkResult(
        profile=DataProfile(
            n_series=1,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=60,
            max_length=60,
            total_rows=60,
        ),
        config=BenchmarkConfig(horizon=7, n_folds=2),
        models=[],
        leaderboard=[],
    )
    html = render_report(result)
    assert 'id="diagnostics"' not in html
    assert 'id="forecasts"' not in html
