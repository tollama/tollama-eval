"""Ensemble recommendation engine.

Analyzes per-series scores to recommend the best model for each series,
enabling per-series model selection (virtual ensemble).
"""

from __future__ import annotations

from dataclasses import dataclass

from ts_autopilot.contracts import BenchmarkResult


@dataclass
class SeriesRecommendation:
    """Best model recommendation for a single series."""

    series_id: str
    best_model: str
    best_mase: float
    all_scores: dict[str, float]


@dataclass
class EnsembleRecommendation:
    """Per-series best-model recommendation across all models."""

    series_recommendations: list[SeriesRecommendation]
    n_series: int
    n_models: int
    model_win_counts: dict[str, int]
    avg_ensemble_mase: float

    def summary(self) -> str:
        """Human-readable summary of the ensemble recommendation."""
        lines = [
            f"Ensemble recommendation across {self.n_series} series "
            f"and {self.n_models} models:"
        ]
        for model, wins in sorted(self.model_win_counts.items(), key=lambda x: -x[1]):
            pct = 100 * wins / self.n_series if self.n_series > 0 else 0
            lines.append(f"  {model}: best for {wins} series ({pct:.0f}%)")
        lines.append(
            f"Virtual ensemble MASE: {self.avg_ensemble_mase:.4f} "
            f"(selecting best model per series)"
        )
        return "\n".join(lines)


def recommend_ensemble(result: BenchmarkResult) -> EnsembleRecommendation:
    """Analyze benchmark results and recommend per-series best models.

    For each series, finds which model achieved the lowest average MASE
    across folds, enabling a "virtual ensemble" that picks the best model
    per series.

    Args:
        result: Completed benchmark result with per-series scores.

    Returns:
        EnsembleRecommendation with per-series best model picks.
    """
    # Collect per-series average MASE for each model
    # model_name -> series_id -> list of fold scores
    model_series_scores: dict[str, dict[str, list[float]]] = {}

    for model in result.models:
        model_series_scores[model.name] = {}
        for fold in model.folds:
            for sid, score in fold.series_scores.items():
                model_series_scores[model.name].setdefault(sid, []).append(score)

    # Average across folds
    model_series_avg: dict[str, dict[str, float]] = {}
    for model_name, series_dict in model_series_scores.items():
        model_series_avg[model_name] = {
            sid: sum(scores) / len(scores) for sid, scores in series_dict.items()
        }

    # Find all unique series
    all_series: set[str] = set()
    for series_dict in model_series_avg.values():
        all_series.update(series_dict.keys())

    # For each series, find the best model
    recommendations: list[SeriesRecommendation] = []
    model_wins: dict[str, int] = {m.name: 0 for m in result.models}
    ensemble_scores: list[float] = []

    for sid in sorted(all_series):
        scores: dict[str, float] = {}
        for model_name, series_dict in model_series_avg.items():
            if sid in series_dict:
                scores[model_name] = series_dict[sid]

        if not scores:
            continue

        best_model = min(scores, key=lambda m: scores[m])
        best_mase = scores[best_model]
        model_wins[best_model] = model_wins.get(best_model, 0) + 1
        ensemble_scores.append(best_mase)

        recommendations.append(
            SeriesRecommendation(
                series_id=sid,
                best_model=best_model,
                best_mase=round(best_mase, 6),
                all_scores={k: round(v, 6) for k, v in scores.items()},
            )
        )

    avg_ensemble = (
        sum(ensemble_scores) / len(ensemble_scores) if ensemble_scores else 0.0
    )

    # Only include models with wins > 0
    win_counts = {m: c for m, c in model_wins.items() if c > 0}

    return EnsembleRecommendation(
        series_recommendations=recommendations,
        n_series=len(all_series),
        n_models=len(result.models),
        model_win_counts=win_counts,
        avg_ensemble_mase=round(avg_ensemble, 6),
    )
