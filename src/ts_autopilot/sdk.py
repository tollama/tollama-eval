"""Fluent Python SDK for tollama-eval.

Usage::

    from ts_autopilot import TSAutopilot

    result = (
        TSAutopilot(df)
        .with_models(["SeasonalNaive", "AutoETS"])
        .with_horizon(14)
        .with_folds(3)
        .run()
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ts_autopilot.contracts import BenchmarkResult


class TSAutopilot:
    """Fluent builder for time series benchmarking."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with a canonical or raw DataFrame.

        Args:
            df: DataFrame with at least (unique_id, ds, y) columns,
                or (ds, y) for a single series.
        """
        self._df = df
        self._horizon: int = 14
        self._n_folds: int = 3
        self._model_names: list[str] | None = None
        self._n_jobs: int = 1
        self._auto_select: bool = False
        self._metric_weights: dict[str, float] | None = None
        self._parallel_models: bool = False

    def with_models(self, models: list[str]) -> TSAutopilot:
        """Specify which models to benchmark."""
        self._model_names = models
        return self

    def with_horizon(self, horizon: int) -> TSAutopilot:
        """Set forecast horizon."""
        self._horizon = horizon
        return self

    def with_folds(self, n_folds: int) -> TSAutopilot:
        """Set number of cross-validation folds."""
        self._n_folds = n_folds
        return self

    def with_n_jobs(self, n_jobs: int) -> TSAutopilot:
        """Set number of parallel workers."""
        self._n_jobs = n_jobs
        return self

    def with_auto_select(self) -> TSAutopilot:
        """Enable automatic model selection based on data profile."""
        self._auto_select = True
        return self

    def with_metric_weights(self, weights: dict[str, float]) -> TSAutopilot:
        """Set custom metric weights for composite scoring."""
        self._metric_weights = weights
        return self

    def with_parallel_models(self) -> TSAutopilot:
        """Enable parallel model execution."""
        self._parallel_models = True
        return self

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return results.

        Returns:
            BenchmarkResult with models, leaderboard, forecasts, etc.
        """
        from ts_autopilot.pipeline import EXTENDED_DEFAULT_RUNNERS, run_benchmark

        df = self._df.copy()

        # Ensure canonical columns
        if "unique_id" not in df.columns:
            df["unique_id"] = "series_1"

        model_names = self._model_names
        runners = None

        # Auto-select models if requested — use extended runner set
        if self._auto_select and model_names is None:
            from ts_autopilot.automl.selector import AutoSelector
            from ts_autopilot.ingestion.profiler import profile_dataframe

            profile = profile_dataframe(df)
            selector = AutoSelector(profile=profile)
            model_names = selector.recommended_model_names()
            runners = list(EXTENDED_DEFAULT_RUNNERS)

        return run_benchmark(
            df=df,
            horizon=self._horizon,
            n_folds=self._n_folds,
            runners=runners,
            model_names=model_names,
            n_jobs=self._n_jobs,
            parallel_models=self._parallel_models,
        )

    def save(self, output_dir: str | Path) -> Path:
        """Run the benchmark and save results to disk.

        Args:
            output_dir: Directory to write standard benchmark artifacts.

        Returns:
            Path to the output directory.
        """
        from ts_autopilot.pipeline import write_output_artifacts

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.run()
        write_output_artifacts(result, output_dir)

        return output_dir
