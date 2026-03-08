"""Model stability analysis.

Measures how stable a model's performance is across CV folds,
series, and data perturbations. Provides a stability score
that can be used as a tie-breaker in the leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_autopilot.contracts import BenchmarkResult


@dataclass
class StabilityScore:
    """Stability metrics for a single model."""

    model_name: str
    fold_cv: float  # coefficient of variation across folds
    series_cv: float  # coefficient of variation across series
    stability_score: float  # composite score (0-1, higher = more stable)
    rank_consistency: float  # how often does it maintain its rank across folds


@dataclass
class StabilityReport:
    """Stability analysis for all models in a benchmark."""

    scores: list[StabilityScore]

    def summary(self) -> str:
        lines = ["Model Stability Analysis:"]
        sorted_scores = sorted(self.scores, key=lambda s: -s.stability_score)
        for s in sorted_scores:
            lines.append(
                f"  {s.model_name}: stability={s.stability_score:.3f} "
                f"(fold_cv={s.fold_cv:.3f}, series_cv={s.series_cv:.3f})"
            )
        return "\n".join(lines)


def compute_stability(result: BenchmarkResult) -> StabilityReport:
    """Compute stability scores for all models in a benchmark result.

    Stability is measured by:
    1. Fold CV: how much does MASE vary across CV folds (lower = more stable)
    2. Series CV: how much does MASE vary across series (lower = more stable)
    3. Rank consistency: how often does the model maintain its relative rank

    Returns:
        StabilityReport with per-model stability scores.
    """
    scores: list[StabilityScore] = []

    # Collect per-fold rankings for rank consistency
    n_folds = max(
        (len(m.folds) for m in result.models if m.folds),
        default=0,
    )
    fold_rankings: dict[int, dict[str, float]] = {}
    for fold_idx in range(n_folds):
        fold_mases: dict[str, float] = {}
        for model in result.models:
            if fold_idx < len(model.folds):
                fold_mases[model.name] = model.folds[fold_idx].mase
        fold_rankings[fold_idx] = fold_mases

    for model in result.models:
        if not model.folds:
            scores.append(
                StabilityScore(
                    model_name=model.name,
                    fold_cv=float("nan"),
                    series_cv=float("nan"),
                    stability_score=0.0,
                    rank_consistency=0.0,
                )
            )
            continue

        # Fold CV
        fold_mases = [f.mase for f in model.folds if not np.isnan(f.mase)]
        if len(fold_mases) > 1 and np.mean(fold_mases) > 0:
            fold_cv = float(np.std(fold_mases) / np.mean(fold_mases))
        else:
            fold_cv = 0.0

        # Series CV (from last fold's per-series scores)
        all_series_scores: list[float] = []
        for fold in model.folds:
            all_series_scores.extend(fold.series_scores.values())
        if len(all_series_scores) > 1 and np.mean(all_series_scores) > 0:
            series_cv = float(np.std(all_series_scores) / np.mean(all_series_scores))
        else:
            series_cv = 0.0

        # Rank consistency: what fraction of folds does this model
        # maintain the same relative rank as its overall rank?
        overall_rank = None
        for entry in result.leaderboard:
            if entry.name == model.name:
                overall_rank = entry.rank
                break

        consistent_folds = 0
        total_folds_ranked = 0
        if overall_rank is not None:
            for _fold_idx, fold_mases_dict in fold_rankings.items():
                if model.name not in fold_mases_dict:
                    continue
                ranked = sorted(fold_mases_dict.items(), key=lambda x: x[1])
                fold_rank = next(
                    (i + 1 for i, (name, _) in enumerate(ranked) if name == model.name),
                    None,
                )
                if fold_rank is not None:
                    total_folds_ranked += 1
                    if fold_rank == overall_rank:
                        consistent_folds += 1

        rank_consistency = (
            consistent_folds / total_folds_ranked if total_folds_ranked > 0 else 0.0
        )

        # Composite stability score (0-1)
        # Lower CV = more stable. Transform: stability = 1 / (1 + cv)
        fold_stability = 1.0 / (1.0 + fold_cv)
        series_stability = 1.0 / (1.0 + series_cv)
        stability_score = (
            0.4 * fold_stability + 0.3 * series_stability + 0.3 * rank_consistency
        )

        scores.append(
            StabilityScore(
                model_name=model.name,
                fold_cv=round(fold_cv, 4),
                series_cv=round(series_cv, 4),
                stability_score=round(stability_score, 4),
                rank_consistency=round(rank_consistency, 4),
            )
        )

    return StabilityReport(scores=scores)
