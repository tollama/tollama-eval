"""Tollama client for LLM-powered benchmark interpretation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from ts_autopilot.contracts import BenchmarkResult
from ts_autopilot.logging_config import get_logger

logger = get_logger("tollama")

_DEFAULT_TIMEOUT_SEC = 30


@dataclass
class TollamaResponse:
    """Interpretation returned by the tollama service."""

    interpretation: str
    model_used: str


def interpret(
    result: BenchmarkResult,
    tollama_url: str,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
) -> TollamaResponse | None:
    """Send benchmark result to tollama and return an LLM interpretation.

    Returns None on any failure (network, timeout, bad response) so the
    pipeline never fails due to tollama unavailability.
    """
    url = tollama_url.rstrip("/") + "/interpret"
    payload = json.dumps({"benchmark": result.to_dict()}).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        logger.info("Requesting interpretation from %s", url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return TollamaResponse(
                interpretation=body["interpretation"],
                model_used=body.get("model", "unknown"),
            )
    except urllib.error.URLError as exc:
        logger.warning("Tollama unavailable: %s", exc)
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Invalid tollama response: %s", exc)
        return None
    except TimeoutError:
        logger.warning("Tollama request timed out after %.0fs", timeout)
        return None
