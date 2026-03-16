"""Webhook delivery for benchmark job completion notifications.

Sends an HMAC-signed POST request to a configured callback URL
when a benchmark job completes or fails.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import httpx

from ts_autopilot.logging_config import get_logger

logger = get_logger("server.webhooks")

# Default HMAC secret — override via config or env
_DEFAULT_SECRET = "tollama-eval-webhook-secret"


def compute_signature(
    payload: bytes, secret: str = _DEFAULT_SECRET
) -> str:
    """Compute HMAC-SHA256 signature for webhook payload."""
    return hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()


def deliver_webhook(
    callback_url: str,
    *,
    run_id: str,
    status: str,
    details: dict[str, Any] | None = None,
    secret: str = _DEFAULT_SECRET,
    max_retries: int = 3,
    timeout_sec: float = 10,
) -> bool:
    """Deliver a webhook notification with retries.

    Args:
        callback_url: URL to POST the notification to.
        run_id: Benchmark run ID.
        status: Job status (completed, failed).
        details: Additional payload data.
        secret: HMAC signing secret.
        max_retries: Number of retry attempts.
        timeout_sec: HTTP request timeout.

    Returns:
        True if delivery succeeded, False otherwise.
    """
    payload = json.dumps(
        {
            "event": "benchmark_completed",
            "run_id": run_id,
            "status": status,
            "timestamp": time.time(),
            "details": details or {},
        },
        default=str,
    ).encode()

    signature = compute_signature(payload, secret)
    headers = {
        "Content-Type": "application/json",
        "X-Tollama-Signature": f"sha256={signature}",
        "X-Tollama-Event": "benchmark_completed",
    }

    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                callback_url,
                content=payload,
                headers=headers,
                timeout=timeout_sec,
            )
            if resp.status_code < 400:
                logger.info(
                    "Webhook delivered to %s (status=%d, run=%s)",
                    callback_url,
                    resp.status_code,
                    run_id,
                )
                return True
            logger.warning(
                "Webhook to %s returned %d (attempt %d/%d)",
                callback_url,
                resp.status_code,
                attempt + 1,
                max_retries,
            )
        except httpx.RequestError as exc:
            logger.warning(
                "Webhook delivery failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                exc,
            )

        # Exponential backoff
        if attempt < max_retries - 1:
            backoff = 2**attempt
            time.sleep(backoff)

    logger.error(
        "Webhook delivery to %s failed after %d attempts (run=%s)",
        callback_url,
        max_retries,
        run_id,
    )
    return False
