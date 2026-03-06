"""Statistical model runners using StatsForecast."""

from __future__ import annotations

from statsforecast.models import AutoETS, SeasonalNaive

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
