"""Tests for enhanced report features."""

from tests.artifact_test_utils import make_rich_result
from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataCharacteristics,
    DataProfile,
)
from ts_autopilot.reporting.html_report import render_report
from ts_autopilot.runners.optional import OptionalRunnerStatus


def _make_enriched_result():
    """Create a BenchmarkResult with forecast data and diagnostics."""
    result = make_rich_result()
    result.data_characteristics = None
    if hasattr(result, "_optional_runner_statuses"):
        delattr(result, "_optional_runner_statuses")
    return result


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
    assert "Fold 2 forecasts for every model." in html
    assert "chart-forecast-1-1" in html
    assert "chart-forecast-2-1" in html


def test_report_has_data_overview_section():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Data Overview" in html
    assert "chart-data-overview-1" in html
    assert "Recent training history plus holdout actuals from fold 2." in html


def test_report_has_model_comparison():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Model Comparison" in html
    assert "chart-radar" in html
    assert "chart-fold-stability" in html


def test_report_has_per_series_winner_section():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Per-Series Winners" in html
    assert "chart-per-series-wins" in html
    assert "chart-per-series-heatmap" in html
    assert "Hardest Series Snapshot" in html


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


def test_report_has_optional_model_environment_section():
    result = _make_enriched_result()
    result._optional_runner_statuses = [
        OptionalRunnerStatus(
            label="Prophet",
            available=True,
            reason="available",
            runner_names=["Prophet"],
        ),
        OptionalRunnerStatus(
            label="NeuralForecast",
            available=False,
            reason="failed health check",
            runner_names=["NHITS", "NBEATS", "TiDE", "DeepAR", "PatchTST", "TFT"],
        ),
    ]
    html = render_report(result)
    assert "Optional Model Environment" in html
    assert "#optional-model-environment" in html
    assert "Dependency stack available" in html
    assert "failed health check" in html


def test_report_omits_optional_model_environment_without_context():
    result = _make_enriched_result()
    html = render_report(result)
    assert "Optional Model Environment" not in html


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


def test_report_has_data_characteristics():
    result = _make_enriched_result()
    result.data_characteristics = DataCharacteristics(
        y_mean=10.0,
        y_std=2.5,
        y_min=1.0,
        y_max=25.0,
        y_median=9.5,
        y_skewness=0.3,
        y_kurtosis=-0.1,
        mean_cv=0.25,
        trend_strength=0.6,
        seasonality_strength=0.4,
        series_heterogeneity=0.8,
    )
    html = render_report(result)
    assert "Data Characteristics" in html
    assert "Trend Strength" in html
    assert "Seasonality Strength" in html
    assert "Series Heterogeneity" in html
    assert "Avg CV" in html


def test_report_no_data_characteristics_without_data():
    """Report omits data characteristics section when data is None."""
    result = _make_enriched_result()
    result.data_characteristics = None
    html = render_report(result)
    assert "Data Characteristics" not in html
