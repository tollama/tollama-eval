"""Tests for optional OpenTelemetry tracing integration."""

from __future__ import annotations

from ts_autopilot.tracing import is_available, span


def test_span_noop_without_otel():
    """span() should be a no-op context manager when otel is not installed."""
    with span("test_span", attributes={"key": "val"}) as s:
        # Without otel installed, should yield None
        assert s is None


def test_is_available_returns_bool():
    result = is_available()
    assert isinstance(result, bool)


def test_span_no_attributes():
    with span("test") as _s:
        pass  # Should not raise
