"""Ensemble recommendation and construction engine.

Analyzes per-series scores to recommend the best model for each series,
enabling per-series model selection (virtual ensemble).

Also provides actual ensemble construction methods:
- Simple average
- Inverse-MASE weighted average
- Best-per-series selection
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ts_autopilot.contracts import BenchmarkResult, ForecastData


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
    for avg_dict in model_series_avg.values():
        all_series.update(avg_dict.keys())

    # For each series, find the best model
    recommendations: list[SeriesRecommendation] = []
    model_wins: dict[str, int] = {m.name: 0 for m in result.models}
    ensemble_scores: list[float] = []

    for sid in sorted(all_series):
        scores: dict[str, float] = {}
        for model_name, avg_dict in model_series_avg.items():
            if sid in avg_dict:
                scores[model_name] = avg_dict[sid]

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


# --- Phase 2b: Ensemble Construction ---


@dataclass
class EnsembleForecast:
    """Constructed ensemble forecast."""

    method: str  # 'average', 'weighted', 'best_per_series'
    unique_id: list[str]
    ds: list[str]
    y_hat: list[float]
    component_models: list[str]
    weights: dict[str, float] = field(default_factory=dict)


def build_average_ensemble(
    forecast_data: list[ForecastData],
    fold: int | None = None,
) -> EnsembleForecast:
    """Build a simple average ensemble from model forecasts.

    Averages predictions across all models for each (unique_id, ds) pair.

    Args:
        forecast_data: List of ForecastData from benchmark run.
        fold: If specified, only use forecasts from this fold.

    Returns:
        EnsembleForecast with averaged predictions.
    """
    if fold is not None:
        forecast_data = [fd for fd in forecast_data if fd.fold == fold]

    if not forecast_data:
        return EnsembleForecast(
            method="average",
            unique_id=[],
            ds=[],
            y_hat=[],
            component_models=[],
        )

    # Group predictions by (unique_id, ds) → list of y_hat values
    point_predictions: dict[tuple[str, str], list[float]] = {}
    component_models: set[str] = set()

    for fd in forecast_data:
        component_models.add(fd.model_name)
        for uid, ds, yhat in zip(fd.unique_id, fd.ds, fd.y_hat, strict=False):
            key = (uid, ds)
            point_predictions.setdefault(key, []).append(yhat)

    # Average
    unique_ids: list[str] = []
    dss: list[str] = []
    yhats: list[float] = []

    for (uid, ds), preds in sorted(point_predictions.items()):
        unique_ids.append(uid)
        dss.append(ds)
        yhats.append(float(np.mean(preds)))

    models_list = sorted(component_models)
    return EnsembleForecast(
        method="average",
        unique_id=unique_ids,
        ds=dss,
        y_hat=yhats,
        component_models=models_list,
        weights={m: 1.0 / len(models_list) for m in models_list},
    )


def build_weighted_ensemble(
    forecast_data: list[ForecastData],
    model_scores: dict[str, float],
    fold: int | None = None,
) -> EnsembleForecast:
    """Build an inverse-MASE weighted ensemble.

    Models with lower MASE get higher weight: w_i = (1/MASE_i) / sum(1/MASE_j).

    Args:
        forecast_data: List of ForecastData from benchmark run.
        model_scores: Dict of model_name → mean_mase.
        fold: If specified, only use forecasts from this fold.

    Returns:
        EnsembleForecast with weighted predictions.
    """
    if fold is not None:
        forecast_data = [fd for fd in forecast_data if fd.fold == fold]

    if not forecast_data:
        return EnsembleForecast(
            method="weighted",
            unique_id=[],
            ds=[],
            y_hat=[],
            component_models=[],
        )

    # Compute inverse-MASE weights
    valid_models = {
        name: score
        for name, score in model_scores.items()
        if score > 0 and not np.isnan(score)
    }
    if not valid_models:
        return build_average_ensemble(forecast_data, fold)

    inv_scores = {name: 1.0 / score for name, score in valid_models.items()}
    total_inv = sum(inv_scores.values())
    weights = {name: inv / total_inv for name, inv in inv_scores.items()}

    # Group and weight predictions
    point_predictions: dict[tuple[str, str], float] = {}
    component_models: set[str] = set()

    for fd in forecast_data:
        if fd.model_name not in weights:
            continue
        w = weights[fd.model_name]
        component_models.add(fd.model_name)
        for uid, ds, yhat in zip(fd.unique_id, fd.ds, fd.y_hat, strict=False):
            key = (uid, ds)
            point_predictions[key] = point_predictions.get(key, 0.0) + w * yhat

    unique_ids: list[str] = []
    dss: list[str] = []
    yhats: list[float] = []

    for (uid, ds), pred in sorted(point_predictions.items()):
        unique_ids.append(uid)
        dss.append(ds)
        yhats.append(float(pred))

    return EnsembleForecast(
        method="weighted",
        unique_id=unique_ids,
        ds=dss,
        y_hat=yhats,
        component_models=sorted(component_models),
        weights={k: round(v, 4) for k, v in weights.items()},
    )


def build_best_per_series_ensemble(
    forecast_data: list[ForecastData],
    recommendation: EnsembleRecommendation,
    fold: int | None = None,
) -> EnsembleForecast:
    """Build an ensemble that picks the best model for each series.

    Uses the recommendation engine to select the best model per series,
    then constructs a forecast using each series' best model.

    Args:
        forecast_data: List of ForecastData from benchmark run.
        recommendation: EnsembleRecommendation with per-series best models.
        fold: If specified, only use forecasts from this fold.

    Returns:
        EnsembleForecast with per-series best model predictions.
    """
    if fold is not None:
        forecast_data = [fd for fd in forecast_data if fd.fold == fold]

    # Build lookup: series_id → best_model_name
    best_model_map = {
        rec.series_id: rec.best_model for rec in recommendation.series_recommendations
    }

    # Build lookup: (model_name, unique_id, ds) → y_hat
    predictions: dict[tuple[str, str, str], float] = {}
    for fd in forecast_data:
        for uid, ds, yhat in zip(fd.unique_id, fd.ds, fd.y_hat, strict=False):
            predictions[(fd.model_name, uid, ds)] = yhat

    unique_ids: list[str] = []
    dss: list[str] = []
    yhats: list[float] = []
    component_models: set[str] = set()

    # For each point, use the best model's prediction
    seen_points: set[tuple[str, str]] = set()
    for fd in forecast_data:
        for uid, ds in zip(fd.unique_id, fd.ds, strict=False):
            point_key = (uid, ds)
            if point_key in seen_points:
                continue
            seen_points.add(point_key)

            best_model = best_model_map.get(uid)
            if best_model and (best_model, uid, ds) in predictions:
                unique_ids.append(uid)
                dss.append(ds)
                yhats.append(predictions[(best_model, uid, ds)])
                component_models.add(best_model)

    return EnsembleForecast(
        method="best_per_series",
        unique_id=unique_ids,
        ds=dss,
        y_hat=yhats,
        component_models=sorted(component_models),
        weights=dict(recommendation.model_win_counts),
    )
