"""Abstract base class for forecast model runners."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

# Adopt new statsforecast behavior where ID is a column, not an index.
# Must be set before importing statsforecast.
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

import pandas as pd
from statsforecast import StatsForecast

from ts_autopilot.contracts import ForecastOutput
from ts_autopilot.logging_config import get_logger

logger = get_logger("runners")


class BaseRunner(ABC):
    """Abstract base for all forecast model runners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier used in results.json."""
        ...

    @abstractmethod
    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
    ) -> ForecastOutput:
        """Fit on train, predict horizon steps ahead.

        Args:
            train: Canonical long-format DataFrame (unique_id, ds, y).
            horizon: Steps to forecast.
            freq: pandas freq string (e.g. 'D', 'W', 'ME').
            season_length: Seasonal period.
            n_jobs: Number of parallel workers.

        Returns:
            ForecastOutput with y_hat for all series.
        """
        ...


class StatsForecastRunner(BaseRunner):
    """Base class for runners backed by a single StatsForecast model.

    Subclasses only need to implement ``name`` and ``_make_model``.
    """

    @abstractmethod
    def _make_model(self, season_length: int) -> object:
        """Return an instantiated StatsForecast model."""
        ...

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
    ) -> ForecastOutput:
        logger.debug(
            "%s.fit_predict: %d rows, horizon=%d, freq=%s, season=%d",
            self.name,
            len(train),
            horizon,
            freq,
            season_length,
        )
        t0 = time.perf_counter()
        model = self._make_model(season_length)
        sf = StatsForecast(models=[model], freq=freq, n_jobs=n_jobs)
        sf.fit(train)
        preds = sf.predict(h=horizon)
        elapsed = time.perf_counter() - t0

        preds = preds.reset_index() if "unique_id" not in preds.columns else preds
        logger.debug("%s completed in %.4fs", self.name, elapsed)
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds[self.name].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )
