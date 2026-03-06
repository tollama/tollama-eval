"""Statistical model runners using StatsForecast."""

from __future__ import annotations

import time

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from ts_autopilot.contracts import ForecastOutput
from ts_autopilot.logging_config import get_logger
from ts_autopilot.runners.base import BaseRunner

logger = get_logger("runners")


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
        logger.debug(
            "SeasonalNaive.fit_predict: %d rows, horizon=%d, freq=%s, season=%d",
            len(train),
            horizon,
            freq,
            season_length,
        )
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
        logger.debug("SeasonalNaive completed in %.4fs", elapsed)
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
        logger.debug(
            "AutoETS.fit_predict: %d rows, horizon=%d, freq=%s, season=%d",
            len(train),
            horizon,
            freq,
            season_length,
        )
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
        logger.debug("AutoETS completed in %.4fs", elapsed)
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["AutoETS"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )
