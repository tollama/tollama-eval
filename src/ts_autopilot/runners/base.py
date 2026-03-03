"""Abstract base class for forecast model runners."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ts_autopilot.contracts import ForecastOutput


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
    ) -> ForecastOutput:
        """Fit on train, predict horizon steps ahead.

        Args:
            train: Canonical long-format DataFrame (unique_id, ds, y).
            horizon: Steps to forecast.
            freq: pandas freq string (e.g. 'D', 'W', 'ME').
            season_length: Seasonal period.

        Returns:
            ForecastOutput with y_hat for all series.
        """
        ...
