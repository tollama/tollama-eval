"""Auto-model recommendation engine.

Analyzes data profile to recommend the most suitable subset of models
for a given dataset, avoiding wasting time on models unlikely to perform well.
"""

from __future__ import annotations

from dataclasses import dataclass

from ts_autopilot.contracts import DataProfile


@dataclass
class ModelRecommendation:
    """A recommended model with reasoning."""

    model_name: str
    reason: str
    priority: int  # 1 = must-run, 2 = recommended, 3 = optional


@dataclass
class AutoSelector:
    """Recommends models based on data characteristics."""

    profile: DataProfile
    include_intermittent: bool = False
    include_extended: bool = True
    include_neural: bool = False
    max_models: int = 10

    def recommend(self) -> list[ModelRecommendation]:
        """Analyze data profile and recommend models."""
        recs: list[ModelRecommendation] = []

        # Always include baseline
        recs.append(
            ModelRecommendation(
                "SeasonalNaive",
                "Baseline model for comparison",
                priority=1,
            )
        )

        # Core statistical models — always recommended
        recs.append(
            ModelRecommendation(
                "AutoETS",
                "Robust exponential smoothing — works well on most data",
                priority=1,
            )
        )

        # Short series → simpler models
        if self.profile.min_length < 50:
            recs.append(
                ModelRecommendation(
                    "Naive",
                    "Short series — simple models are safer",
                    priority=2,
                )
            )
            recs.append(
                ModelRecommendation(
                    "HistoricAverage",
                    "Short series — mean baseline",
                    priority=2,
                )
            )
        else:
            # Longer series → complex models
            recs.append(
                ModelRecommendation(
                    "AutoARIMA",
                    "Sufficient data for ARIMA model selection",
                    priority=1,
                )
            )
            recs.append(
                ModelRecommendation(
                    "AutoTheta",
                    "Theta method — strong on M3/M4 competition data",
                    priority=2,
                )
            )

        # Seasonal data → seasonal models
        if self.profile.season_length_guess > 1:
            recs.append(
                ModelRecommendation(
                    "HoltWinters",
                    f"Seasonal data (period={self.profile.season_length_guess})",
                    priority=2,
                )
            )
            if self.profile.min_length >= 2 * self.profile.season_length_guess:
                recs.append(
                    ModelRecommendation(
                        "MSTL",
                        "Multiple seasonal decomposition",
                        priority=2,
                    )
                )

        # Trend detection heuristic: if frequency is daily+
        if self.profile.frequency in ("D", "W", "ME", "MS", "h", "H"):
            recs.append(
                ModelRecommendation(
                    "DynamicOptimizedTheta",
                    "Dynamic theta for trending data",
                    priority=3,
                )
            )

        # Intermittent demand detection
        if self.include_intermittent or self.profile.missing_ratio > 0.3:
            recs.extend(
                [
                    ModelRecommendation(
                        "CrostonSBA",
                        "Intermittent demand — bias-corrected Croston",
                        priority=2,
                    ),
                    ModelRecommendation(
                        "IMAPA",
                        "Intermittent demand — multi-aggregation",
                        priority=2,
                    ),
                    ModelRecommendation(
                        "TSB",
                        "Intermittent demand — Teunter-Syntetos-Babai",
                        priority=3,
                    ),
                ]
            )

        # Many series → fast models preferred; few series → can afford complex models
        if self.profile.n_series > 100 and self.include_neural:
            recs.append(
                ModelRecommendation(
                    "LightGBM",
                    "Many series — gradient boosting scales well",
                    priority=2,
                )
            )
        elif self.include_neural and self.profile.min_length >= 100:
            recs.append(
                ModelRecommendation(
                    "NHITS",
                    "Sufficient data for neural forecasting",
                    priority=3,
                )
            )
            recs.append(
                ModelRecommendation(
                    "NBEATS",
                    "N-BEATS for long series",
                    priority=3,
                )
            )

        # Sort by priority, then truncate
        recs.sort(key=lambda r: r.priority)

        # Deduplicate by model name, keep first (highest priority)
        seen: set[str] = set()
        unique_recs: list[ModelRecommendation] = []
        for r in recs:
            if r.model_name not in seen:
                seen.add(r.model_name)
                unique_recs.append(r)

        return unique_recs[: self.max_models]

    def recommended_model_names(self) -> list[str]:
        """Return just the model names in priority order."""
        return [r.model_name for r in self.recommend()]

    def summary(self) -> str:
        """Human-readable summary of recommendations."""
        recs = self.recommend()
        lines = [f"Auto-selected {len(recs)} models for this dataset:"]
        for r in recs:
            priority_label = {1: "MUST-RUN", 2: "RECOMMENDED", 3: "OPTIONAL"}[
                r.priority
            ]
            lines.append(f"  [{priority_label}] {r.model_name}: {r.reason}")
        return "\n".join(lines)


@dataclass
class IntermittencyProfile:
    """Profile of intermittent demand characteristics."""

    zero_ratio: float  # fraction of zero values
    avg_demand_interval: float  # average inter-demand interval
    cv_squared: float  # squared coefficient of variation of non-zero demand
    is_intermittent: bool  # True if zero_ratio > 0.3
    classification: str  # 'smooth', 'erratic', 'intermittent', 'lumpy'


def detect_intermittency(
    series_values: list[float],
) -> IntermittencyProfile:
    """Classify a time series using the SBC (Syntetos-Boylan Classification).

    Categories:
    - smooth: low CV^2, low demand interval
    - erratic: high CV^2, low demand interval
    - intermittent: low CV^2, high demand interval
    - lumpy: high CV^2, high demand interval
    """
    import numpy as np

    values = np.array(series_values, dtype=float)
    n = len(values)
    if n == 0:
        return IntermittencyProfile(
            zero_ratio=1.0,
            avg_demand_interval=float("inf"),
            cv_squared=0.0,
            is_intermittent=True,
            classification="intermittent",
        )

    zero_ratio = float(np.mean(values == 0))
    nonzero = values[values > 0]

    if len(nonzero) == 0:
        return IntermittencyProfile(
            zero_ratio=1.0,
            avg_demand_interval=float("inf"),
            cv_squared=0.0,
            is_intermittent=True,
            classification="intermittent",
        )

    # Average demand interval
    demand_indices = np.where(values > 0)[0]
    if len(demand_indices) > 1:
        intervals = np.diff(demand_indices)
        adi = float(np.mean(intervals))
    else:
        adi = float(n)

    # CV^2 of non-zero demand
    mean_nz = float(np.mean(nonzero))
    std_nz = float(np.std(nonzero))
    cv_sq = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0.0

    # SBC classification thresholds
    adi_threshold = 1.32
    cv_threshold = 0.49

    if adi < adi_threshold and cv_sq < cv_threshold:
        classification = "smooth"
    elif adi < adi_threshold and cv_sq >= cv_threshold:
        classification = "erratic"
    elif adi >= adi_threshold and cv_sq < cv_threshold:
        classification = "intermittent"
    else:
        classification = "lumpy"

    return IntermittencyProfile(
        zero_ratio=zero_ratio,
        avg_demand_interval=adi,
        cv_squared=cv_sq,
        is_intermittent=zero_ratio > 0.3,
        classification=classification,
    )
