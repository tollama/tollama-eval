"""Tollama client for Time Series Foundation Model forecasting.

Tollama is a unified TSFM platform (https://github.com/tollama/tollama)
that provides access to models like Chronos-2, TimesFM, Moirai 2.0, etc.
via a standard HTTP API.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ts_autopilot.logging_config import get_logger

logger = get_logger("tollama")

_DEFAULT_TIMEOUT_SEC = 120


@dataclass
class TollamaForecastResponse:
    """Forecast response from the tollama service."""

    mean: list[float]
    model: str
    quantiles: dict[str, list[float]] = field(default_factory=dict)


def forecast(
    target: list[float],
    freq: str,
    horizon: int,
    model: str,
    tollama_url: str,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
) -> TollamaForecastResponse:
    """Send time series data to tollama and return TSFM forecast.

    Args:
        target: Historical values for a single series.
        freq: Pandas-style frequency string (e.g. 'D', 'W', 'ME').
        horizon: Number of steps to forecast.
        model: Tollama model identifier (e.g. 'chronos2', 'timesfm').
        tollama_url: Base URL of the tollama server.
        timeout: Request timeout in seconds.

    Returns:
        TollamaForecastResponse with forecasted values.

    Raises:
        TollamaError: On any failure (network, timeout, bad response).
    """
    url = tollama_url.rstrip("/") + "/v1/forecast"
    payload = json.dumps(
        {
            "model": model,
            "series": {"target": target, "freq": freq},
            "horizon": horizon,
        }
    ).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        logger.debug(
            "Requesting forecast from %s (model=%s, h=%d)",
            url,
            model,
            horizon,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            mean = body["mean"]
            if not isinstance(mean, list) or len(mean) != horizon:
                got = len(mean) if isinstance(mean, list) else type(mean)
                raise TollamaError(f"Expected {horizon} forecast values, got {got}")
            quantiles = {
                k: [float(x) for x in v] for k, v in body.get("quantiles", {}).items()
            }
            return TollamaForecastResponse(
                mean=[float(v) for v in mean],
                model=body.get("model", model),
                quantiles=quantiles,
            )
    except urllib.error.URLError as exc:
        raise TollamaError(f"Tollama unavailable at {url}: {exc}") from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TollamaError(f"Invalid tollama response: {exc}") from exc
    except TimeoutError as exc:
        raise TollamaError(f"Tollama request timed out after {timeout:.0f}s") from exc


class TollamaError(Exception):
    """Raised when tollama communication fails."""
