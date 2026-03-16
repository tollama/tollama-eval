"""Prometheus metrics for the tollama-eval REST API.

Exposes counters, histograms, and gauges for monitoring
benchmarks, models, and server health.

Requires: ``pip install prometheus-client``
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from ts_autopilot.logging_config import get_logger

logger = get_logger("server.metrics")

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class MetricsCollector:
    """Prometheus metrics for tollama-eval."""

    def __init__(
        self, *, registry: Any | None = None
    ) -> None:
        if not HAS_PROMETHEUS:
            self._enabled = False
            logger.info(
                "prometheus-client not installed — metrics disabled"
            )
            return

        self._enabled = True
        reg = registry or CollectorRegistry()
        self._registry = reg

        self.benchmarks_submitted = Counter(
            "tollama_benchmarks_submitted_total",
            "Total benchmark submissions",
            registry=reg,
        )
        self.benchmarks_completed = Counter(
            "tollama_benchmarks_completed_total",
            "Total completed benchmarks",
            registry=reg,
        )
        self.benchmarks_failed = Counter(
            "tollama_benchmarks_failed_total",
            "Total failed benchmarks",
            registry=reg,
        )
        self.models_run = Counter(
            "tollama_models_run_total",
            "Total model runs",
            ["model_name"],
            registry=reg,
        )

        self.benchmark_duration = Histogram(
            "tollama_benchmark_duration_seconds",
            "Benchmark execution time",
            buckets=[10, 30, 60, 120, 300, 600, 1800],
            registry=reg,
        )
        self.model_fit_duration = Histogram(
            "tollama_model_fit_duration_seconds",
            "Model fit/predict time",
            ["model_name"],
            buckets=[1, 5, 10, 30, 60, 120, 300],
            registry=reg,
        )
        self.csv_load_duration = Histogram(
            "tollama_csv_load_duration_seconds",
            "CSV loading time",
            buckets=[0.1, 0.5, 1, 5, 10, 30],
            registry=reg,
        )

        self.active_benchmarks = Gauge(
            "tollama_active_benchmarks",
            "Currently running benchmarks",
            registry=reg,
        )
        self.queue_depth = Gauge(
            "tollama_queue_depth",
            "Jobs waiting in queue",
            registry=reg,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def generate(self) -> bytes:
        """Generate Prometheus text format output."""
        if not self._enabled:
            return b""
        return generate_latest(self._registry)

    @contextmanager
    def track_benchmark(self) -> Generator[None, None, None]:
        """Context manager to track benchmark duration."""
        if not self._enabled:
            yield
            return
        self.active_benchmarks.inc()
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self.benchmark_duration.observe(elapsed)
            self.active_benchmarks.dec()

    def record_model_run(
        self, model_name: str, duration_sec: float
    ) -> None:
        """Record a model fit/predict execution."""
        if not self._enabled:
            return
        self.models_run.labels(model_name=model_name).inc()
        self.model_fit_duration.labels(
            model_name=model_name
        ).observe(duration_sec)

    def record_benchmark_submitted(self) -> None:
        if self._enabled:
            self.benchmarks_submitted.inc()

    def record_benchmark_completed(self) -> None:
        if self._enabled:
            self.benchmarks_completed.inc()

    def record_benchmark_failed(self) -> None:
        if self._enabled:
            self.benchmarks_failed.inc()


# Module-level singleton (lazy init)
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
