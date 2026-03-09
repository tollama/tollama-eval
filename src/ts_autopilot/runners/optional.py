"""Optional model runners requiring extra dependencies.

These runners gracefully degrade if their dependencies are not installed.
Install extras: pip install ts-autopilot[prophet], [lightgbm], or [neural]
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
        exog: pd.DataFrame | None = None,
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
        exog: pd.DataFrame | None = None,
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


class XGBoostRunner(BaseRunner):
    """XGBoost model runner via mlforecast.

    Requires: pip install xgboost mlforecast
    """

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        import xgboost as xgb
        from mlforecast import MLForecast

        t0 = time.perf_counter()

        models = [
            xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                verbosity=0,
                n_jobs=n_jobs,
            )
        ]
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
            y_hat=preds["XGBRegressor"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


def get_best_accelerator() -> str:
    """Detect best available hardware accelerator."""
    try:
        import torch

        if torch.backends.mps.is_available():
            # MPS can be unstable with float64, usually requires float32
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class NHITSRunner(BaseRunner):
    """N-HiTS neural model runner via neuralforecast.

    Requires: pip install neuralforecast
    """

    @property
    def name(self) -> str:
        return "NHITS"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS

        t0 = time.perf_counter()
        acc = get_best_accelerator()

        # Neural models on MPS/GPU prefer float32
        train = train.copy()
        train["y"] = train["y"].astype("float32")

        model = NHITS(
            input_size=2 * horizon,
            h=horizon,
            max_steps=100,
            accelerator=acc,
            start_padding_enabled=True,
            enable_progress_bar=False,
        )
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=train)
        preds = nf.predict()
        preds = preds.reset_index() if "unique_id" not in preds.columns else preds

        elapsed = time.perf_counter() - t0
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["NHITS"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


class NBEATSRunner(BaseRunner):
    """N-BEATS neural model runner via neuralforecast.

    Requires: pip install neuralforecast
    """

    @property
    def name(self) -> str:
        return "NBEATS"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS

        t0 = time.perf_counter()
        acc = get_best_accelerator()

        train = train.copy()
        train["y"] = train["y"].astype("float32")

        model = NBEATS(
            input_size=2 * horizon,
            h=horizon,
            max_steps=100,
            accelerator=acc,
            start_padding_enabled=True,
            enable_progress_bar=False,
        )
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=train)
        preds = nf.predict()
        preds = preds.reset_index() if "unique_id" not in preds.columns else preds

        elapsed = time.perf_counter() - t0
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["NBEATS"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


class TiDERunner(BaseRunner):
    """TiDE (Time-series Dense Encoder) neural model.

    Requires: pip install neuralforecast
    """

    @property
    def name(self) -> str:
        return "TiDE"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TiDE

        t0 = time.perf_counter()
        acc = get_best_accelerator()

        train = train.copy()
        train["y"] = train["y"].astype("float32")

        model = TiDE(
            input_size=2 * horizon,
            h=horizon,
            max_steps=100,
            accelerator=acc,
            enable_progress_bar=False,
        )
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=train)
        preds = nf.predict()
        preds = preds.reset_index() if "unique_id" not in preds.columns else preds

        elapsed = time.perf_counter() - t0
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["TiDE"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


class DeepARRunner(BaseRunner):
    """DeepAR probabilistic neural model.

    Requires: pip install neuralforecast
    """

    @property
    def name(self) -> str:
        return "DeepAR"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DeepAR

        t0 = time.perf_counter()
        acc = get_best_accelerator()

        train = train.copy()
        train["y"] = train["y"].astype("float32")

        model = DeepAR(
            input_size=2 * horizon,
            h=horizon,
            max_steps=100,
            accelerator=acc,
            enable_progress_bar=False,
        )
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=train)
        preds = nf.predict()
        preds = preds.reset_index() if "unique_id" not in preds.columns else preds

        elapsed = time.perf_counter() - t0
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["DeepAR"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


class PatchTSTRunner(BaseRunner):
    """PatchTST transformer model.

    Requires: pip install neuralforecast
    """

    @property
    def name(self) -> str:
        return "PatchTST"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import PatchTST

        t0 = time.perf_counter()
        acc = get_best_accelerator()

        train = train.copy()
        train["y"] = train["y"].astype("float32")

        model = PatchTST(
            input_size=2 * horizon,
            h=horizon,
            max_steps=100,
            accelerator=acc,
            enable_progress_bar=False,
        )
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=train)
        preds = nf.predict()
        preds = preds.reset_index() if "unique_id" not in preds.columns else preds

        elapsed = time.perf_counter() - t0
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["PatchTST"].tolist(),
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


class TFTRunner(BaseRunner):
    """Temporal Fusion Transformer model.

    Requires: pip install neuralforecast
    """

    @property
    def name(self) -> str:
        return "TFT"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TFT

        t0 = time.perf_counter()
        acc = get_best_accelerator()

        train = train.copy()
        train["y"] = train["y"].astype("float32")

        model = TFT(
            input_size=2 * horizon,
            h=horizon,
            max_steps=100,
            accelerator=acc,
            enable_progress_bar=False,
        )
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=train)
        preds = nf.predict()
        preds = preds.reset_index() if "unique_id" not in preds.columns else preds

        elapsed = time.perf_counter() - t0
        return ForecastOutput(
            unique_id=preds["unique_id"].astype(str).tolist(),
            ds=preds["ds"].astype(str).tolist(),
            y_hat=preds["TFT"].tolist(),
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
        import mlforecast

        runners.append(LightGBMRunner())
        logger.debug("LightGBM runner available")
    except ImportError:
        logger.debug("LightGBM/mlforecast not installed, skipping")

    try:
        import mlforecast  # noqa: F401
        import xgboost  # noqa: F401

        runners.append(XGBoostRunner())
        logger.debug("XGBoost runner available")
    except ImportError:
        logger.debug("XGBoost/mlforecast not installed, skipping")

    try:
        import neuralforecast  # noqa: F401

        runners.append(NHITSRunner())
        runners.append(NBEATSRunner())
        runners.append(TiDERunner())
        runners.append(DeepARRunner())
        runners.append(PatchTSTRunner())
        runners.append(TFTRunner())
        logger.debug(
            "NeuralForecast runners available"
            " (NHITS, NBEATS, TiDE, DeepAR, PatchTST, TFT)"
        )
    except ImportError:
        logger.debug("neuralforecast not installed, skipping")

    return runners
