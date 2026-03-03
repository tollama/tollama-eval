"""Statistical model runners using StatsForecast."""

from __future__ import annotations

import time

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from ts_autopilot.contracts import ForecastOutput
from ts_autopilot.runners.base import BaseRunner


class SeasonalNaiveRunner(BaseRunner):
    @property
    def name(self) -> str:
        return "SeasonalNaive"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
    ) -> ForecastOutput:
        t0 = time.perf_counter()
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=season_length)],
            freq=freq,
            n_jobs=1,
        )
        sf.fit(train)
        preds = sf.predict(h=horizon)
        elapsed = time.perf_counter() - t0

        # StatsForecast returns (unique_id, ds, <ModelName>) columns
        preds = preds.reset_index() if "unique_id" not in preds.columns else preds
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["SeasonalNaive"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


class AutoETSRunner(BaseRunner):
    @property
    def name(self) -> str:
        return "AutoETS"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
    ) -> ForecastOutput:
        t0 = time.perf_counter()
        sf = StatsForecast(
            models=[AutoETS(season_length=season_length)],
            freq=freq,
            n_jobs=1,
        )
        sf.fit(train)
        preds = sf.predict(h=horizon)
        elapsed = time.perf_counter() - t0

        preds = preds.reset_index() if "unique_id" not in preds.columns else preds
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["AutoETS"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )
