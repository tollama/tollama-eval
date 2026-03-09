"""Tests for statistical significance testing."""

from ts_autopilot.evaluation.significance import (
    SignificanceReport,
    friedman_test,
    render_critical_difference_svg,
)


def _make_scores(n_series: int = 10) -> dict[str, dict[str, float]]:
    """Create synthetic per-series scores with clear ranking: A < B < C."""
    import numpy as np

    rng = np.random.default_rng(42)
    series_ids = [f"s{i}" for i in range(n_series)]
    return {
        "ModelA": {s: 0.5 + rng.normal(0, 0.05) for s in series_ids},
        "ModelB": {s: 0.8 + rng.normal(0, 0.05) for s in series_ids},
        "ModelC": {s: 1.2 + rng.normal(0, 0.1) for s in series_ids},
    }


def test_friedman_returns_report():
    scores = _make_scores(20)
    report = friedman_test(scores)
    assert report is not None
    assert isinstance(report, SignificanceReport)
    assert report.n_models == 3
    assert report.n_series == 20


def test_friedman_detects_significant_difference():
    scores = _make_scores(30)
    report = friedman_test(scores)
    assert report is not None
    assert report.friedman_p_value < 0.05
    assert report.critical_difference > 0


def test_friedman_pairwise_comparisons():
    scores = _make_scores(30)
    report = friedman_test(scores)
    assert report is not None
    assert len(report.pairwise) == 3  # 3 choose 2
    # A vs C should be significantly different (large gap)
    a_vs_c = next(
        p for p in report.pairwise if {p.model_a, p.model_b} == {"ModelA", "ModelC"}
    )
    assert a_vs_c.significant


def test_friedman_mean_ranks_order():
    scores = _make_scores(30)
    report = friedman_test(scores)
    assert report is not None
    # ModelA should have best (lowest) rank
    assert report.mean_ranks["ModelA"] < report.mean_ranks["ModelC"]


def test_friedman_returns_none_insufficient_models():
    scores = {"OnlyModel": {"s1": 0.5, "s2": 0.6, "s3": 0.7}}
    report = friedman_test(scores)
    assert report is None


def test_friedman_returns_none_insufficient_series():
    scores = {
        "ModelA": {"s1": 0.5, "s2": 0.6},
        "ModelB": {"s1": 0.8, "s2": 0.7},
    }
    report = friedman_test(scores)
    assert report is None


def test_friedman_handles_tied_scores():
    # All models have identical scores -> no significant difference
    scores = {
        "ModelA": {f"s{i}": 1.0 for i in range(10)},
        "ModelB": {f"s{i}": 1.0 for i in range(10)},
        "ModelC": {f"s{i}": 1.0 for i in range(10)},
    }
    report = friedman_test(scores)
    assert report is not None
    assert report.friedman_p_value > 0.05


def test_friedman_handles_mismatched_series():
    """Only common series across all models are used."""
    scores = {
        "ModelA": {"s1": 0.5, "s2": 0.6, "s3": 0.7, "s_extra": 0.4},
        "ModelB": {"s1": 0.8, "s2": 0.7, "s3": 0.9},
        "ModelC": {"s1": 1.2, "s2": 1.1, "s3": 1.3},
    }
    report = friedman_test(scores)
    assert report is not None
    assert report.n_series == 3


def test_cd_svg_renders():
    mean_ranks = {"ModelA": 1.2, "ModelB": 2.0, "ModelC": 2.8}
    svg = render_critical_difference_svg(mean_ranks, cd=0.9)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "ModelA" in svg
    assert "CD = 0.90" in svg


def test_cd_svg_empty_on_invalid_input():
    assert render_critical_difference_svg({}, cd=0.5) == ""
    assert render_critical_difference_svg({"A": 1.0}, cd=0.0) == ""


def test_cd_svg_clique_bars():
    """Models within CD should be connected by a bar."""
    mean_ranks = {"A": 1.0, "B": 1.3, "C": 3.0}
    svg = render_critical_difference_svg(mean_ranks, cd=0.5)
    assert "<svg" in svg
    # A and B are within CD=0.5, so there should be a clique bar
    assert 'stroke-linecap="round"' in svg
