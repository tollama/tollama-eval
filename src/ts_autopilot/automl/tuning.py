"""Hyperparameter search for ML and neural models.

Provides simple grid search over key hyperparameters,
integrated with the existing CV infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ts_autopilot.logging_config import get_logger

logger = get_logger("automl.tuning")


@dataclass
class HyperparamConfig:
    """A single hyperparameter configuration to evaluate."""

    params: dict[str, Any]
    score: float = float("nan")
    runtime_sec: float = 0.0


@dataclass
class TuningResult:
    """Result of hyperparameter search."""

    model_name: str
    best_params: dict[str, Any]
    best_score: float
    all_configs: list[HyperparamConfig]
    metric_name: str = "mase"

    def summary(self) -> str:
        lines = [
            f"Tuning result for {self.model_name}:",
            f"  Best {self.metric_name}: {self.best_score:.6f}",
            f"  Best params: {self.best_params}",
            f"  Configs evaluated: {len(self.all_configs)}",
        ]
        return "\n".join(lines)


# Default hyperparameter grids for ML models
LIGHTGBM_GRID: list[dict[str, Any]] = [
    {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 31},
    {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 63},
    {"n_estimators": 100, "learning_rate": 0.01, "num_leaves": 31},
]

XGBOOST_GRID: list[dict[str, Any]] = [
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 6},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 10},
    {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 6},
]


def get_default_grid(model_name: str) -> list[dict[str, Any]]:
    """Return the default hyperparameter grid for a model."""
    grids: dict[str, list[dict[str, Any]]] = {
        "LightGBM": LIGHTGBM_GRID,
        "XGBoost": XGBOOST_GRID,
    }
    return grids.get(model_name, [])
