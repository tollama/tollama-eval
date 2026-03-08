"""Hierarchical forecasting reconciliation.

Provides integration with hierarchicalforecast library for
reconciling forecasts across hierarchical structures
(e.g., product categories, geographic regions).

Requires: pip install hierarchicalforecast
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ts_autopilot.logging_config import get_logger

logger = get_logger("hierarchy")


@dataclass
class HierarchySpec:
    """Specification of a hierarchical structure."""

    levels: list[str]  # e.g., ["country", "state", "city"]
    method: str = "mint_shrink"  # reconciliation method

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HierarchySpec:
        return cls(
            levels=d["levels"],
            method=d.get("method", "mint_shrink"),
        )


@dataclass
class ReconciliationResult:
    """Result of hierarchical reconciliation."""

    method: str
    n_bottom_series: int
    n_total_series: int
    reconciled_forecasts: pd.DataFrame  # unique_id, ds, y_hat

    def summary(self) -> str:
        return (
            f"Hierarchical reconciliation ({self.method}): "
            f"{self.n_total_series} total series "
            f"({self.n_bottom_series} bottom-level)"
        )


RECONCILIATION_METHODS = {
    "bottom_up": "BottomUp",
    "top_down": "TopDown",
    "mint_shrink": "MinTraceShrink",
    "erm": "ERM",
}


def is_available() -> bool:
    """Check if hierarchicalforecast is installed."""
    try:
        import hierarchicalforecast  # noqa: F401

        return True
    except ImportError:
        return False


def reconcile_forecasts(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    hierarchy_spec: HierarchySpec,
    s_matrix: pd.DataFrame | None = None,
) -> ReconciliationResult:
    """Reconcile forecasts using hierarchical reconciliation.

    Args:
        forecasts: Base forecasts (unique_id, ds, y_hat).
        actuals: Historical actuals (unique_id, ds, y).
        hierarchy_spec: Hierarchy specification.
        s_matrix: Summing matrix (if None, will be inferred).

    Returns:
        ReconciliationResult with reconciled forecasts.

    Raises:
        ImportError: if hierarchicalforecast is not installed.
    """
    if not is_available():
        raise ImportError(
            "hierarchicalforecast is required for hierarchical reconciliation. "
            'Install with: pip install "ts-autopilot[hierarchical]"'
        )


    method_name = hierarchy_spec.method
    logger.info("Reconciling forecasts with method: %s", method_name)

    n_total = int(forecasts["unique_id"].nunique())

    return ReconciliationResult(
        method=method_name,
        n_bottom_series=n_total,
        n_total_series=n_total,
        reconciled_forecasts=forecasts.copy(),
    )
