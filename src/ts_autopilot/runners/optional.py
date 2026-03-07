"""Optional model runners requiring extra dependencies.

These runners gracefully degrade if their dependencies are not installed.
Install extras: pip install ts-autopilot[prophet] or ts-autopilot[lightgbm]
"""

from __future__ import annotations

import time

import pandas as pd

from ts_autopilot.contracts import ForecastOutput
from ts_autopilot.logging_config import get_logger
from ts_autopilot.runners.base import BaseRunner

logger = get_logger("runners.optional")


class ProphetRunner(BaseRunner):
    """Facebook Prophet model runner.

    Requires: pip install prophet
    """

    @property
    def name(self) -> str:
        return "Prophet"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
    ) -> ForecastOutput:
        from prophet import Prophet

        t0 = time.perf_counter()
        all_uids: list[str] = []
        all_ds: list[str] = []
        all_yhat: list[float] = []

        for uid, group in train.groupby("unique_id"):
            prophet_df = group[["ds", "y"]].copy()
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

            m = Prophet(weekly_seasonality=season_length == 7)
            m.fit(prophet_df)

            future = m.make_future_dataframe(periods=horizon, freq=freq)
            forecast = m.predict(future)
            preds = forecast.tail(horizon)

            all_uids.extend([str(uid)] * horizon)
            all_ds.extend(preds["ds"].astype(str).tolist())
            all_yhat.extend(preds["yhat"].tolist())

        elapsed = time.perf_counter() - t0
        return ForecastOutput(
            unique_id=all_uids,
            ds=all_ds,
            y_hat=all_yhat,
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


class LightGBMRunner(BaseRunner):
    """LightGBM model runner via mlforecast.

    Requires: pip install lightgbm mlforecast
    """

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
    ) -> ForecastOutput:
        import lightgbm as lgb
        from mlforecast import MLForecast

        t0 = time.perf_counter()

        models = [lgb.LGBMRegressor(n_estimators=100, verbosity=-1, n_jobs=n_jobs)]
        mlf = MLForecast(
            models=models,
            freq=freq,
            lags=[1, season_length],
        )
        mlf.fit(train)
        preds = mlf.predict(h=horizon)

        elapsed = time.perf_counter() - t0
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["LGBMRegressor"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


def get_optional_runners() -> list[BaseRunner]:
    """Return list of available optional runners (only if deps are installed)."""
    runners: list[BaseRunner] = []

    try:
        import prophet  # noqa: F401

        runners.append(ProphetRunner())
        logger.debug("Prophet runner available")
    except ImportError:
        logger.debug("Prophet not installed, skipping")

    try:
        import lightgbm  # noqa: F401
        import mlforecast  # noqa: F401

        runners.append(LightGBMRunner())
        logger.debug("LightGBM runner available")
    except ImportError:
        logger.debug("LightGBM/mlforecast not installed, skipping")

    return runners
