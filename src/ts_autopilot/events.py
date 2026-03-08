"""Structured event emission for external monitoring integration.

Events are emitted as structured JSON log lines at INFO level.
External systems can parse these from stderr (when --log-json is enabled)
to build dashboards, alerts, and audit trails.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ts_autopilot.logging_config import get_logger

logger = get_logger("events")


@dataclass(frozen=True)
class Event:
    """Base structured event."""

    event_type: str
    run_id: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


def emit(event: Event) -> None:
    """Emit a structured event via the logging system."""
    logger.info("EVENT %s", event.to_json())


def benchmark_started(
    run_id: str,
    n_series: int,
    horizon: int,
    n_folds: int,
    models: list[str],
) -> None:
    """Emit when a benchmark run begins."""
    emit(
        Event(
            event_type="benchmark_started",
            run_id=run_id,
            payload={
                "n_series": n_series,
                "horizon": horizon,
                "n_folds": n_folds,
                "models": models,
            },
        )
    )


def model_started(run_id: str, model_name: str, index: int, total: int) -> None:
    """Emit when a model begins fitting."""
    emit(
        Event(
            event_type="model_started",
            run_id=run_id,
            payload={
                "model_name": model_name,
                "index": index,
                "total": total,
            },
        )
    )


def model_completed(
    run_id: str,
    model_name: str,
    mean_mase: float,
    runtime_sec: float,
) -> None:
    """Emit when a model finishes successfully."""
    emit(
        Event(
            event_type="model_completed",
            run_id=run_id,
            payload={
                "model_name": model_name,
                "mean_mase": mean_mase,
                "runtime_sec": runtime_sec,
            },
        )
    )


def model_failed(run_id: str, model_name: str, error: str) -> None:
    """Emit when a model fails."""
    emit(
        Event(
            event_type="model_failed",
            run_id=run_id,
            payload={
                "model_name": model_name,
                "error": error,
            },
        )
    )


def benchmark_completed(
    run_id: str,
    winner: str,
    winner_mase: float,
    total_runtime_sec: float,
    n_models_succeeded: int,
    n_models_failed: int,
) -> None:
    """Emit when a benchmark run completes."""
    emit(
        Event(
            event_type="benchmark_completed",
            run_id=run_id,
            payload={
                "winner": winner,
                "winner_mase": winner_mase,
                "total_runtime_sec": total_runtime_sec,
                "n_models_succeeded": n_models_succeeded,
                "n_models_failed": n_models_failed,
            },
        )
    )
