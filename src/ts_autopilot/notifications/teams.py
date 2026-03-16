"""Microsoft Teams notification support for benchmark results.

Formats benchmark results as Teams Adaptive Card messages
and delivers them to a Teams webhook URL.
"""

from __future__ import annotations

from typing import Any

import httpx

from ts_autopilot.logging_config import get_logger

logger = get_logger("notifications.teams")


def format_benchmark_result(
    run_id: str,
    status: str,
    *,
    winner: str | None = None,
    winner_mase: float | None = None,
    total_runtime_sec: float | None = None,
    n_models: int | None = None,
) -> dict[str, Any]:
    """Format benchmark results as Teams Adaptive Card."""
    color = "good" if status == "completed" else "attention"

    facts: list[dict[str, str]] = [
        {"title": "Status", "value": status},
    ]
    if winner:
        facts.append({"title": "Winner", "value": winner})
    if winner_mase is not None:
        facts.append(
            {"title": "MASE", "value": f"{winner_mase:.4f}"}
        )
    if total_runtime_sec is not None:
        facts.append(
            {
                "title": "Runtime",
                "value": f"{total_runtime_sec:.1f}s",
            }
        )
    if n_models is not None:
        facts.append(
            {"title": "Models", "value": str(n_models)}
        )

    return {
        "type": "message",
        "attachments": [
            {
                "contentType": (
                    "application/vnd.microsoft.card.adaptive"
                ),
                "content": {
                    "$schema": (
                        "http://adaptivecards.io/schemas/"
                        "adaptive-card.json"
                    ),
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "size": "Large",
                            "weight": "Bolder",
                            "text": f"Benchmark {run_id}",
                            "color": color,
                        },
                        {
                            "type": "FactSet",
                            "facts": facts,
                        },
                    ],
                },
            }
        ],
    }


def send_notification(
    webhook_url: str,
    run_id: str,
    status: str,
    **kwargs: Any,
) -> bool:
    """Send a Teams notification.

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
            logger.info(
                "Teams notification sent for run %s", run_id
            )
            return True
        logger.warning(
            "Teams notification failed: status=%d",
            resp.status_code,
        )
    except httpx.RequestError as exc:
        logger.error("Teams notification error: %s", exc)
    return False
