"""Statistical model runners using StatsForecast."""

from __future__ import annotations

from statsforecast.models import (
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoTheta,
    SeasonalNaive,
)

from ts_autopilot.runners.base import StatsForecastRunner


class SeasonalNaiveRunner(StatsForecastRunner):
    @property
    def name(self) -> str:
        return "SeasonalNaive"

    def _make_model(self, season_length: int) -> object:
        return SeasonalNaive(season_length=season_length)


class AutoETSRunner(StatsForecastRunner):
    @property
    def name(self) -> str:
        return "AutoETS"

    def _make_model(self, season_length: int) -> object:
        return AutoETS(season_length=season_length)


class AutoARIMARunner(StatsForecastRunner):
    @property
    def name(self) -> str:
        return "AutoARIMA"

    def _make_model(self, season_length: int) -> object:
        return AutoARIMA(season_length=season_length)


class AutoThetaRunner(StatsForecastRunner):
    @property
    def name(self) -> str:
        return "AutoTheta"

    def _make_model(self, season_length: int) -> object:
        return AutoTheta(season_length=season_length)


class AutoCESRunner(StatsForecastRunner):
    @property
    def name(self) -> str:
        # StatsForecast outputs the column as "CES" for AutoCES
        return "CES"

    def _make_model(self, season_length: int) -> object:
        return AutoCES(season_length=season_length)


# --- Phase 1a: Additional statistical models ---


class MSTLRunner(StatsForecastRunner):
    """Multiple Seasonal-Trend decomposition using LOESS."""

    @property
    def name(self) -> str:
        return "MSTL"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import MSTL as _MSTL

        return _MSTL(season_length=season_length)


class DynamicOptimizedThetaRunner(StatsForecastRunner):
    """Dynamic Optimized Theta method."""

    @property
    def name(self) -> str:
        return "DynamicOptimizedTheta"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import DynamicOptimizedTheta

        return DynamicOptimizedTheta(season_length=season_length)


class HoltRunner(StatsForecastRunner):
    """Holt's linear trend method (double exponential smoothing)."""

    @property
    def name(self) -> str:
        return "Holt"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import Holt

        return Holt()


class HoltWintersRunner(StatsForecastRunner):
    """Holt-Winters (triple exponential smoothing)."""

    @property
    def name(self) -> str:
        return "HoltWinters"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import HoltWinters

        return HoltWinters(season_length=season_length)


class HistoricAverageRunner(StatsForecastRunner):
    """Simple historic average (mean of all training data)."""

    @property
    def name(self) -> str:
        return "HistoricAverage"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import HistoricAverage

        return HistoricAverage()


class NaiveRunner(StatsForecastRunner):
    """Naive forecast (repeat last observation)."""

    @property
    def name(self) -> str:
        return "Naive"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import Naive

        return Naive()


class RandomWalkWithDriftRunner(StatsForecastRunner):
    """Random walk with drift."""

    @property
    def name(self) -> str:
        return "RandomWalkWithDrift"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import RandomWalkWithDrift

        return RandomWalkWithDrift()


class WindowAverageRunner(StatsForecastRunner):
    """Simple moving average over a window."""

    @property
    def name(self) -> str:
        return "WindowAverage"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import WindowAverage

        return WindowAverage(window_size=season_length)


class SeasonalWindowAverageRunner(StatsForecastRunner):
    """Seasonal moving average."""

    @property
    def name(self) -> str:
        return "SeasonalWindowAverage"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import SeasonalWindowAverage

        return SeasonalWindowAverage(
            season_length=season_length, window_size=season_length
        )


# --- Phase 1b: Intermittent demand models ---


class CrostonClassicRunner(StatsForecastRunner):
    """Croston's classic method for intermittent demand."""

    @property
    def name(self) -> str:
        return "CrostonClassic"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import CrostonClassic

        return CrostonClassic()


class CrostonOptimizedRunner(StatsForecastRunner):
    """Croston's optimized method for intermittent demand."""

    @property
    def name(self) -> str:
        return "CrostonOptimized"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import CrostonOptimized

        return CrostonOptimized()


class CrostonSBARunner(StatsForecastRunner):
    """Syntetos-Boylan Approximation (bias-corrected Croston)."""

    @property
    def name(self) -> str:
        return "CrostonSBA"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import CrostonSBA

        return CrostonSBA()


class ADIDARunner(StatsForecastRunner):
    """Aggregate-Disaggregate Intermittent Demand Approach."""

    @property
    def name(self) -> str:
        return "ADIDA"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import ADIDA

        return ADIDA()


class IMAPARunner(StatsForecastRunner):
    """Intermittent Multiple Aggregation Prediction Algorithm."""

    @property
    def name(self) -> str:
        return "IMAPA"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import IMAPA

        return IMAPA()


class TSBRunner(StatsForecastRunner):
    """Teunter-Syntetos-Babai method for intermittent demand."""

    @property
    def name(self) -> str:
        return "TSB"

    def _make_model(self, season_length: int) -> object:
        from statsforecast.models import TSB

        return TSB(alpha_d=0.2, alpha_p=0.2)


# --- Registry of all statistical runners ---

# Core runners (always available)
CORE_RUNNERS: tuple[StatsForecastRunner, ...] = (
    SeasonalNaiveRunner(),
    AutoETSRunner(),
    AutoARIMARunner(),
    AutoThetaRunner(),
    AutoCESRunner(),
)

# Extended statistical runners
EXTENDED_RUNNERS: tuple[StatsForecastRunner, ...] = (
    MSTLRunner(),
    DynamicOptimizedThetaRunner(),
    HoltRunner(),
    HoltWintersRunner(),
    HistoricAverageRunner(),
    NaiveRunner(),
    RandomWalkWithDriftRunner(),
    WindowAverageRunner(),
    SeasonalWindowAverageRunner(),
)

# Intermittent demand runners
INTERMITTENT_RUNNERS: tuple[StatsForecastRunner, ...] = (
    CrostonClassicRunner(),
    CrostonOptimizedRunner(),
    CrostonSBARunner(),
    ADIDARunner(),
    IMAPARunner(),
    TSBRunner(),
)

# All statistical runners
ALL_STATISTICAL_RUNNERS: tuple[StatsForecastRunner, ...] = (
    *CORE_RUNNERS,
    *EXTENDED_RUNNERS,
    *INTERMITTENT_RUNNERS,
)
