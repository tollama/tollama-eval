"""Statistical model runners using StatsForecast."""

from __future__ import annotations

from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta, SeasonalNaive

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
