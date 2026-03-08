"""Tollama TSFM model runner.

Delegates forecasting to a running tollama server, enabling benchmarking of
Time Series Foundation Models (Chronos-2, TimesFM, Moirai, etc.) alongside
traditional statistical models.

Requires a tollama server: https://github.com/tollama/tollama
"""

from __future__ import annotations

import time

import pandas as pd

from ts_autopilot.contracts import ForecastOutput
from ts_autopilot.logging_config import get_logger
from ts_autopilot.runners.base import BaseRunner
from ts_autopilot.tollama.client import TollamaError, forecast

logger = get_logger("runners.tollama")

# Tollama model identifiers users can select via --tollama-models
AVAILABLE_MODELS = (
    "chronos2",
    "timesfm",
    "moirai",
    "granite-ttm",
    "lag-llama",
    "patchtst",
    "tide",
)


class TollamaRunner(BaseRunner):
    """Runner that delegates forecasting to a tollama TSFM server.

    Each instance wraps a single tollama model (e.g. chronos2, timesfm).
    The runner sends each series independently to the tollama API and
    collects predictions.
    """

    def __init__(self, model: str, tollama_url: str) -> None:
        self._model = model
        self._tollama_url = tollama_url

    @property
    def name(self) -> str:
        return f"tollama/{self._model}"

    def fit_predict(
        self,
        train: pd.DataFrame,
        horizon: int,
        freq: str,
        season_length: int,
        n_jobs: int = 1,
        exog: pd.DataFrame | None = None,
    ) -> ForecastOutput:
        """Send each series to tollama and collect forecasts.

        Tollama models are pre-trained (zero-shot), so there is no fit step.
        Each series is sent independently to the /v1/forecast endpoint.
        """
        t0 = time.perf_counter()
        _ = (season_length, n_jobs, exog)  # interface compatibility

        all_uids: list[str] = []
        all_ds: list[str] = []
        all_yhat: list[float] = []

        grouped = train.groupby("unique_id")
        n_series = len(grouped)

        for idx, (uid, group) in enumerate(grouped):
            group_sorted = group.sort_values("ds")
            target = group_sorted["y"].tolist()
            last_date = pd.Timestamp(group_sorted["ds"].iloc[-1])

            logger.debug(
                "  %s series %d/%d (%s): %d history points",
                self.name,
                idx + 1,
                n_series,
                uid,
                len(target),
            )

            try:
                response = forecast(
                    target=target,
                    freq=freq,
                    horizon=horizon,
                    model=self._model,
                    tollama_url=self._tollama_url,
                )
                y_hat = response.mean
            except TollamaError as exc:
                logger.warning(
                    "Tollama forecast failed for series %s: %s. Using NaN.",
                    uid,
                    exc,
                )
                y_hat = [float("nan")] * horizon

            # Generate future dates
            future_dates = pd.date_range(
                start=last_date, periods=horizon + 1, freq=freq
            )[1:]

            all_uids.extend([str(uid)] * horizon)
            all_ds.extend(future_dates.astype(str).tolist())
            all_yhat.extend(y_hat)

        elapsed = time.perf_counter() - t0
        logger.debug("%s completed in %.4fs", self.name, elapsed)

        return ForecastOutput(
            unique_id=all_uids,
            ds=all_ds,
            y_hat=all_yhat,
            model_name=self.name,
            runtime_sec=round(elapsed, 4),
        )


def get_tollama_runners(
    tollama_url: str,
    model_names: list[str] | None = None,
) -> list[TollamaRunner]:
    """Create TollamaRunner instances for the requested models.

    Args:
        tollama_url: Base URL of the tollama server.
        model_names: List of tollama model identifiers. If None, no runners
            are created (tollama is opt-in).

    Returns:
        List of TollamaRunner instances.
    """
    if not model_names:
        return []

    runners: list[TollamaRunner] = []
    for model in model_names:
        runners.append(TollamaRunner(model=model, tollama_url=tollama_url))
        logger.info("Registered tollama runner: %s", model)
    return runners
