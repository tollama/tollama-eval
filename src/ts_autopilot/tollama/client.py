"""Tollama client for Time Series Foundation Model forecasting.

Tollama is a unified TSFM platform (https://github.com/tollama/tollama)
that provides access to models like Chronos-2, TimesFM, Moirai 2.0, etc.
via a standard HTTP API.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ts_autopilot.exceptions import URLValidationError
from ts_autopilot.logging_config import get_logger

logger = get_logger("tollama")

_DEFAULT_TIMEOUT_SEC = 120

# Private/internal IP ranges that are blocked by default (SSRF prevention)
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def validate_tollama_url(
    url: str,
    allow_private: bool = False,
) -> str:
    """Validate a tollama URL for scheme and SSRF safety.

    Args:
        url: The URL to validate.
        allow_private: If True, allow private/internal IP addresses.

    Returns:
        The validated URL string.

    Raises:
        URLValidationError: If the URL is invalid or targets a private network.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise URLValidationError(
            f"Tollama URL must use http:// or https://, got '{parsed.scheme}://'."
        )

    if not parsed.hostname:
        raise URLValidationError(f"Tollama URL has no hostname: '{url}'.")

    if not allow_private:
        try:
            # Resolve hostname to IP for SSRF check
            resolved_ips = socket.getaddrinfo(
                parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP
            )
            for _family, _type, _proto, _canonname, sockaddr in resolved_ips:
                ip = ipaddress.ip_address(sockaddr[0])
                for network in _PRIVATE_NETWORKS:
                    if ip in network:
                        raise URLValidationError(
                            f"Tollama URL '{url}' resolves to private address "
                            f"{ip}. Set allow_private_urls=true in config to allow."
                        )
        except socket.gaierror:
            # DNS resolution failed — let the actual request fail later
            logger.warning(
                "Could not resolve hostname '%s' for SSRF check",
                parsed.hostname,
            )

    return url


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
