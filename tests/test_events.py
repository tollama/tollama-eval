"""Tests for structured event emission."""

from __future__ import annotations

import json
import logging

from ts_autopilot.events import (
    Event,
    benchmark_completed,
    benchmark_started,
    model_completed,
    model_failed,
    model_started,
)


def test_event_to_dict():
    event = Event(event_type="test", run_id="abc123", payload={"key": "val"})
    d = event.to_dict()
    assert d["event_type"] == "test"
    assert d["run_id"] == "abc123"
    assert d["payload"]["key"] == "val"
    assert "timestamp" in d


def test_event_to_json():
    event = Event(event_type="test", run_id="abc123")
    j = event.to_json()
    parsed = json.loads(j)
    assert parsed["event_type"] == "test"


class _LogCapture(logging.Handler):
    """A simple handler that captures log records for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(self.format(r) for r in self.records)


def _capture_event(fn, event_name):
    """Helper to capture log output from event functions."""
    events_logger = logging.getLogger("ts_autopilot.events")
    handler = _LogCapture()
    handler.setLevel(logging.DEBUG)
    events_logger.addHandler(handler)
    old_level = events_logger.level
    events_logger.setLevel(logging.DEBUG)
    try:
        fn()
        assert event_name in handler.text
    finally:
        events_logger.setLevel(old_level)
        events_logger.removeHandler(handler)


def test_benchmark_started_emits():
    _capture_event(
        lambda: benchmark_started(
            run_id="r1",
            n_series=5,
            horizon=14,
            n_folds=3,
            models=["SeasonalNaive"],
        ),
        "benchmark_started",
    )


def test_model_started_emits():
    _capture_event(
        lambda: model_started(run_id="r1", model_name="AutoETS", index=1, total=5),
        "model_started",
    )


def test_model_completed_emits():
    _capture_event(
        lambda: model_completed(
            run_id="r1", model_name="AutoETS", mean_mase=0.85, runtime_sec=1.2
        ),
        "model_completed",
    )


def test_model_failed_emits():
    _capture_event(
        lambda: model_failed(
            run_id="r1", model_name="BadModel", error="convergence failed"
        ),
        "model_failed",
    )


def test_benchmark_completed_emits():
    _capture_event(
        lambda: benchmark_completed(
            run_id="r1",
            winner="AutoETS",
            winner_mase=0.85,
            total_runtime_sec=10.5,
            n_models_succeeded=4,
            n_models_failed=1,
        ),
        "benchmark_completed",
    )
