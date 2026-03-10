"""Optional OpenTelemetry tracing integration.

This module provides tracing spans for pipeline operations.
If opentelemetry-api is not installed, all operations are no-ops.

Install with: pip install "tollama-eval[tracing]"
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any

try:
    from opentelemetry import trace

    _TRACER = trace.get_tracer("ts_autopilot")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    _TRACER = None  # type: ignore[assignment]


@contextlib.contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Create a tracing span. No-op if OpenTelemetry is not installed."""
    if _HAS_OTEL and _TRACER is not None:
        with _TRACER.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    s.set_attribute(k, v)
            yield s
    else:
        yield None


def is_available() -> bool:
    """Check if OpenTelemetry tracing is available."""
    return _HAS_OTEL
