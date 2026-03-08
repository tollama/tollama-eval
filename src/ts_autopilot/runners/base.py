"""Abstract base class for forecast model runners."""

from __future__ import annotations

import importlib.metadata
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

    @property
    def supports_exog(self) -> bool:
        """Whether this runner can use exogenous variables."""
        return False

    @abstractmethod
    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        """Fit on train, predict horizon steps ahead.

        Args:
            train: Canonical long-format DataFrame (unique_id, ds, y).
            horizon: Steps to forecast.
            freq: pandas freq string (e.g. 'D', 'W', 'ME').
            season_length: Seasonal period.
            n_jobs: Number of parallel workers.
            exog: Optional exogenous variables DataFrame with (unique_id, ds, ...).

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
        exog: pd.DataFrame | None = None,
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


def discover_plugins() -> list[BaseRunner]:
    """Discover model runners registered via entry points.

    Third-party packages can register runners using the
    ``ts_autopilot.runners`` entry point group::

        [project.entry-points."ts_autopilot.runners"]
        my_model = "my_package:MyModelRunner"

    Each entry point should resolve to a callable that returns a
    ``BaseRunner`` instance (either a class or a factory function).

    Returns:
        List of discovered BaseRunner instances.
    """
    runners: list[BaseRunner] = []
    try:
        eps = importlib.metadata.entry_points()
        # Python 3.12+ returns a SelectableGroups, 3.10/3.11 returns dict
        if hasattr(eps, "select"):
            group_eps = eps.select(group="ts_autopilot.runners")
        else:
            group_eps = eps.get("ts_autopilot.runners", [])  # type: ignore[assignment]

        for ep in group_eps:
            try:
                obj = ep.load()
                instance = obj() if isinstance(obj, type) else obj
                if isinstance(instance, BaseRunner):
                    runners.append(instance)
                    logger.info("Discovered plugin runner: %s", instance.name)
                else:
                    logger.warning(
                        "Plugin entry point '%s' did not produce a BaseRunner",
                        ep.name,
                    )
            except Exception:
                logger.warning(
                    "Failed to load plugin entry point '%s'",
                    ep.name,
                    exc_info=True,
                )
    except Exception:
        logger.debug("No plugin entry points found", exc_info=True)

    return runners
