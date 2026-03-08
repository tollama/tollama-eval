"""Speed benchmarking and accuracy-speed tradeoff analysis.

Tracks model fitting speed per series and provides
Pareto frontier analysis for accuracy vs speed tradeoffs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_autopilot.contracts import BenchmarkResult


@dataclass
class SpeedProfile:
    """Speed profile for a single model."""

    model_name: str
    total_runtime_sec: float
    avg_sec_per_series: float
    avg_sec_per_fold: float
    throughput_series_per_sec: float  # series/second


@dataclass
class ParetoPoint:
    """A point on the accuracy-speed tradeoff space."""

    model_name: str
    mean_mase: float
    total_runtime_sec: float
    is_pareto_optimal: bool  # on the Pareto frontier?


@dataclass
class SpeedReport:
    """Speed analysis across all models."""

    profiles: list[SpeedProfile]
    pareto_points: list[ParetoPoint]
    fastest_model: str
    most_efficient_model: str  # best MASE per runtime second

    def summary(self) -> str:
        lines = ["Speed Benchmark:"]
        for p in sorted(self.profiles, key=lambda x: x.total_runtime_sec):
            lines.append(
                f"  {p.model_name}: {p.total_runtime_sec:.2f}s total, "
                f"{p.avg_sec_per_series:.4f}s/series, "
                f"{p.throughput_series_per_sec:.1f} series/s"
            )
        lines.append(f"\nFastest: {self.fastest_model}")
        lines.append(f"Most efficient (MASE/sec): {self.most_efficient_model}")
        pareto = [p for p in self.pareto_points if p.is_pareto_optimal]
        if pareto:
            lines.append("Pareto-optimal models:")
            for p in pareto:
                lines.append(
                    f"  {p.model_name}: MASE={p.mean_mase:.4f}, "
                    f"time={p.total_runtime_sec:.2f}s"
                )
        return "\n".join(lines)


def compute_speed_report(result: BenchmarkResult) -> SpeedReport:
    """Compute speed profiles and Pareto frontier from benchmark results.

    Args:
        result: Completed benchmark result.

    Returns:
        SpeedReport with per-model speed profiles and Pareto analysis.
    """
    profiles: list[SpeedProfile] = []
    pareto_candidates: list[ParetoPoint] = []

    n_series = result.profile.n_series
    n_folds = result.config.n_folds

    for model in result.models:
        runtime = model.runtime_sec
        avg_per_series = runtime / max(n_series * n_folds, 1)
        avg_per_fold = runtime / max(n_folds, 1)
        throughput = (n_series * n_folds) / max(runtime, 0.001)

        profiles.append(
            SpeedProfile(
                model_name=model.name,
                total_runtime_sec=round(runtime, 4),
                avg_sec_per_series=round(avg_per_series, 6),
                avg_sec_per_fold=round(avg_per_fold, 4),
                throughput_series_per_sec=round(throughput, 2),
            )
        )

        if not np.isnan(model.mean_mase):
            pareto_candidates.append(
                ParetoPoint(
                    model_name=model.name,
                    mean_mase=model.mean_mase,
                    total_runtime_sec=runtime,
                    is_pareto_optimal=False,
                )
            )

    # Compute Pareto frontier
    # A point is Pareto-optimal if no other point has both lower MASE and lower runtime
    for i, p in enumerate(pareto_candidates):
        dominated = False
        for j, q in enumerate(pareto_candidates):
            if (
                i != j
                and q.mean_mase <= p.mean_mase
                and q.total_runtime_sec <= p.total_runtime_sec
                and (
                    q.mean_mase < p.mean_mase
                    or q.total_runtime_sec < p.total_runtime_sec
                )
            ):
                dominated = True
                break
        p.is_pareto_optimal = not dominated

    # Find fastest and most efficient
    if profiles:
        fastest = min(profiles, key=lambda p: p.total_runtime_sec).model_name
    else:
        fastest = "N/A"
    valid_models = [
        m for m in result.models if not np.isnan(m.mean_mase) and m.runtime_sec > 0
    ]
    if valid_models:
        most_efficient = min(
            valid_models, key=lambda m: m.mean_mase * m.runtime_sec
        ).name
    else:
        most_efficient = "N/A"

    return SpeedReport(
        profiles=profiles,
        pareto_points=pareto_candidates,
        fastest_model=fastest,
        most_efficient_model=most_efficient,
    )
