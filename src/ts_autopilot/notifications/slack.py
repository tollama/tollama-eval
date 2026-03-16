"""Slack notification support for benchmark results.

Formats benchmark results as Slack Block Kit messages
and delivers them to a Slack webhook URL.
"""

from __future__ import annotations

from typing import Any

import httpx

from ts_autopilot.logging_config import get_logger

logger = get_logger("notifications.slack")


def format_benchmark_result(
    run_id: str,
    status: str,
    *,
    winner: str | None = None,
    winner_mase: float | None = None,
    total_runtime_sec: float | None = None,
    n_models: int | None = None,
) -> dict[str, Any]:
    """Format benchmark results as Slack Block Kit payload."""
    emoji = ":white_check_mark:" if status == "completed" else ":x:"
    title = f"{emoji} Benchmark `{run_id}` {status}"

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Benchmark {run_id}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": title,
            },
        },
    ]

    if status == "completed" and winner:
        fields = []
        fields.append(
            {"type": "mrkdwn", "text": f"*Winner:* {winner}"}
        )
        if winner_mase is not None:
            fields.append(
                {
                    "type": "mrkdwn",
                    "text": f"*MASE:* {winner_mase:.4f}",
                }
            )
        if total_runtime_sec is not None:
            fields.append(
                {
                    "type": "mrkdwn",
                    "text": f"*Runtime:* {total_runtime_sec:.1f}s",
                }
            )
        if n_models is not None:
            fields.append(
                {
                    "type": "mrkdwn",
                    "text": f"*Models:* {n_models}",
                }
            )
        blocks.append({"type": "section", "fields": fields})

    return {"blocks": blocks}


def send_notification(
    webhook_url: str,
    run_id: str,
    status: str,
    **kwargs: Any,
) -> bool:
    """Send a Slack notification.

    Returns True if delivery succeeded.
    """
    payload = format_benchmark_result(
        run_id, status, **kwargs
    )
    try:
        resp = httpx.post(
            webhook_url,
            json=payload,
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("Slack notification sent for run %s", run_id)
            return True
        logger.warning(
            "Slack notification failed: status=%d", resp.status_code
        )
    except httpx.RequestError as exc:
        logger.error("Slack notification error: %s", exc)
    return False
