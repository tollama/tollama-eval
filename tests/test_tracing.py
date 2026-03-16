"""Tests for optional OpenTelemetry tracing integration."""

from __future__ import annotations

from ts_autopilot.tracing import is_available, span


def test_span_context_manager():
    """span() should be a usable context manager regardless of otel."""
    with span("test_span", attributes={"key": "val"}) as s:
        if is_available():
            # When otel is installed, s is a real Span object
            assert s is not None
        else:
            # Without otel installed, should yield None
            assert s is None


def test_is_available_returns_bool():
    result = is_available()
    assert isinstance(result, bool)


def test_span_no_attributes():
    with span("test") as _s:
        pass  # Should not raise
