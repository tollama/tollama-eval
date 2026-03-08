"""Tests for the auto-model recommendation engine."""


from ts_autopilot.automl.selector import (
    AutoSelector,
    detect_intermittency,
)
from ts_autopilot.contracts import DataProfile


def _make_profile(**kwargs):
    """Create a DataProfile with sensible defaults."""
    defaults = {
        "n_series": 10,
        "frequency": "D",
        "missing_ratio": 0.0,
        "season_length_guess": 7,
        "min_length": 100,
        "max_length": 365,
        "total_rows": 1000,
    }
    defaults.update(kwargs)
    return DataProfile(**defaults)


class TestAutoSelector:
    def test_always_includes_baseline(self):
        profile = _make_profile()
        selector = AutoSelector(profile=profile)
        names = selector.recommended_model_names()
        assert "SeasonalNaive" in names
        assert "AutoETS" in names

    def test_short_series_uses_simple_models(self):
        profile = _make_profile(min_length=20)
        selector = AutoSelector(profile=profile)
        names = selector.recommended_model_names()
        assert "Naive" in names
        assert "HistoricAverage" in names
        # Should NOT recommend ARIMA for very short series
        assert "AutoARIMA" not in names

    def test_long_series_includes_arima(self):
        profile = _make_profile(min_length=200)
        selector = AutoSelector(profile=profile)
        names = selector.recommended_model_names()
        assert "AutoARIMA" in names

    def test_seasonal_data_includes_holtwinters(self):
        profile = _make_profile(season_length_guess=7, min_length=100)
        selector = AutoSelector(profile=profile)
        names = selector.recommended_model_names()
        assert "HoltWinters" in names

    def test_intermittent_includes_croston(self):
        profile = _make_profile(missing_ratio=0.5)
        selector = AutoSelector(profile=profile)
        names = selector.recommended_model_names()
        assert "CrostonSBA" in names
        assert "IMAPA" in names

    def test_max_models_limit(self):
        profile = _make_profile()
        selector = AutoSelector(profile=profile, max_models=3)
        names = selector.recommended_model_names()
        assert len(names) <= 3

    def test_summary_output(self):
        profile = _make_profile()
        selector = AutoSelector(profile=profile)
        summary = selector.summary()
        assert "Auto-selected" in summary
        assert "MUST-RUN" in summary

    def test_neural_models_when_enabled(self):
        profile = _make_profile(n_series=5, min_length=200)
        selector = AutoSelector(profile=profile, include_neural=True)
        names = selector.recommended_model_names()
        assert "NHITS" in names or "NBEATS" in names


class TestIntermittencyDetection:
    def test_smooth_series(self):
        values = [10.0, 11.0, 10.5, 12.0, 11.5, 10.8, 11.2, 10.9]
        result = detect_intermittency(values)
        assert result.classification == "smooth"
        assert not result.is_intermittent
        assert result.zero_ratio == 0.0

    def test_intermittent_series(self):
        values = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0]
        result = detect_intermittency(values)
        assert result.is_intermittent
        assert result.zero_ratio > 0.5

    def test_all_zeros(self):
        values = [0.0] * 10
        result = detect_intermittency(values)
        assert result.is_intermittent
        assert result.zero_ratio == 1.0

    def test_empty_series(self):
        result = detect_intermittency([])
        assert result.is_intermittent
        assert result.zero_ratio == 1.0
