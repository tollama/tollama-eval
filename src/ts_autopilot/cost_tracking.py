"""Cost tracking for compute resource usage.

Tracks CPU-seconds and memory-seconds per benchmark,
attributable to tenants/users for chargeback.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

from ts_autopilot.logging_config import get_logger

logger = get_logger("cost_tracking")


@dataclass
class ResourceUsage:
    """Measured resource consumption for a benchmark."""

    run_id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    wall_time_sec: float = 0.0
    cpu_time_sec: float = 0.0
    peak_memory_mb: float = 0.0
    n_models: int = 0
    n_series: int = 0
    n_folds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def cost_units(self) -> float:
        """Approximate cost in abstract units (CPU-seconds)."""
        return self.cpu_time_sec


class CostTracker:
    """Tracks resource usage per benchmark run."""

    def __init__(self) -> None:
        self._records: list[ResourceUsage] = []

    @contextmanager
    def track(
        self,
        run_id: str,
        *,
        tenant_id: str = "",
        user_id: str = "",
    ) -> Generator[ResourceUsage, None, None]:
        """Context manager to track resource usage."""
        usage = ResourceUsage(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        start_wall = time.monotonic()
        start_cpu = time.process_time()
        start_mem = _get_rss_mb()
        try:
            yield usage
        finally:
            usage.wall_time_sec = time.monotonic() - start_wall
            usage.cpu_time_sec = time.process_time() - start_cpu
            usage.peak_memory_mb = max(
                _get_rss_mb(), start_mem
            )
            self._records.append(usage)
            logger.info(
                "Run %s: wall=%.1fs cpu=%.1fs mem=%.0fMB",
                run_id,
                usage.wall_time_sec,
                usage.cpu_time_sec,
                usage.peak_memory_mb,
            )

    def get_records(
        self, *, tenant_id: str | None = None
    ) -> list[ResourceUsage]:
        """Get all tracked records, optionally filtered."""
        if tenant_id is None:
            return list(self._records)
        return [
            r for r in self._records if r.tenant_id == tenant_id
        ]

    def total_cpu_seconds(
        self, *, tenant_id: str | None = None
    ) -> float:
        """Sum CPU seconds across all tracked runs."""
        records = self.get_records(tenant_id=tenant_id)
        return sum(r.cpu_time_sec for r in records)

    def summary(
        self, *, tenant_id: str | None = None
    ) -> dict[str, Any]:
        """Return a usage summary."""
        records = self.get_records(tenant_id=tenant_id)
        if not records:
            return {
                "total_runs": 0,
                "total_cpu_sec": 0,
                "total_wall_sec": 0,
            }
        return {
            "total_runs": len(records),
            "total_cpu_sec": round(
                sum(r.cpu_time_sec for r in records), 2
            ),
            "total_wall_sec": round(
                sum(r.wall_time_sec for r in records), 2
            ),
            "peak_memory_mb": round(
                max(r.peak_memory_mb for r in records), 1
            ),
        }


def _get_rss_mb() -> float:
    """Get current resident set size in MB."""
    import sys

    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024
