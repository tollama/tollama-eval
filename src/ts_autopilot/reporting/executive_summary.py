"""Auto-generated executive summary — no LLM required."""

from __future__ import annotations

from dataclasses import dataclass, field

from ts_autopilot.contracts import BenchmarkResult, ModelResult


@dataclass
class ExecutiveSummary:
    """Structured executive summary with actionable insights."""

    overview: str = ""
    winner: str = ""
    key_findings: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_flat_text(self) -> str:
        """Backward-compatible flat text representation."""
        parts = [self.overview]
        if self.winner:
            parts.append(self.winner)
        parts.extend(self.key_findings)
        if self.risks:
            parts.append("Key risks: " + " ".join(self.risks))
        if self.recommendations:
            parts.append("Recommendations: " + " ".join(self.recommendations))
        return " ".join(parts)


def generate_executive_summary(result: BenchmarkResult) -> ExecutiveSummary:
    """Generate a structured executive summary from benchmark results.

    Pure template logic with conditional branches — no LLM dependency.
    """
    p = result.profile
    c = result.config
    lb = result.leaderboard
    models = result.models
    summary = ExecutiveSummary()

    # Opening statement
    n_models = len(models)
    summary.overview = (
        f"We evaluated {n_models} forecasting model{'s' if n_models != 1 else ''} "
        f"on {p.n_series} time series ({p.total_rows:,} total observations) "
        f"using {c.n_folds}-fold expanding-window cross-validation "
        f"with a {c.horizon}-step forecast horizon."
    )

    if not lb:
        summary.overview += " No models produced valid results."
        return summary

    # Winner announcement
    winner = lb[0]
    naive_improvement = (1.0 - winner.mean_mase) * 100
    if naive_improvement > 0:
        summary.winner = (
            f"The best-performing model is {winner.name} with a mean MASE of "
            f"{winner.mean_mase:.4f}, outperforming the seasonal naive baseline "
            f"by {naive_improvement:.1f}%."
        )
    elif naive_improvement == 0:
        summary.winner = (
            f"The best-performing model is {winner.name} with a mean MASE of "
            f"{winner.mean_mase:.4f}, matching the seasonal naive baseline."
        )
    else:
        summary.winner = (
            f"The best-performing model is {winner.name} with a mean MASE of "
            f"{winner.mean_mase:.4f}. Note: no model beat the seasonal naive "
            f"baseline (MASE < 1.0)."
        )

    # Key findings
    findings: list[str] = []

    # Runner-up comparison
    if len(lb) >= 2:
        runner_up = lb[1]
        gap = runner_up.mean_mase - winner.mean_mase
        if gap < 0.01:
            findings.append(
                f"{runner_up.name} is a close second (MASE {runner_up.mean_mase:.4f}), "
                f"within {gap:.4f} of the winner."
            )
        else:
            findings.append(
                f"The runner-up is {runner_up.name} (MASE {runner_up.mean_mase:.4f})."
            )

    # Stability assessment
    winner_model = next((m for m in models if m.name == winner.name), None)
    cv = 0.0
    if winner_model and winner_model.std_mase > 0 and winner_model.mean_mase > 0:
        cv = winner_model.std_mase / winner_model.mean_mase
        if cv < 0.05:
            findings.append(
                f"{winner.name} shows excellent stability across folds (CV = {cv:.1%})."
            )
        elif cv < 0.15:
            findings.append(
                f"{winner.name} shows reasonable stability "
                f"across folds (CV = {cv:.1%})."
            )
        else:
            findings.append(
                f"Caution: {winner.name} shows high variability across folds "
                f"(CV = {cv:.1%}), suggesting sensitivity to the training window."
            )

    # SMAPE summary
    if winner.mean_smape > 0:
        findings.append(
            f"In terms of percentage accuracy, {winner.name} achieves "
            f"a SMAPE of {winner.mean_smape:.2f}%."
        )

    # Model comparison narrative
    if len(lb) >= 3:
        narrative = _model_comparison_narrative(result)
        if narrative:
            findings.append(narrative)

    # Ensemble recommendation hint
    if len(models) >= 2 and _has_per_series_scores(result):
        ensemble_hint = _ensemble_hint(result)
        if ensemble_hint:
            findings.append(ensemble_hint)

    summary.key_findings = findings

    # Risk flags
    risks: list[str] = []

    if winner_model:
        worst_series = _find_worst_series(winner_model)
        if worst_series:
            sid, score = worst_series
            if score > 2.0:
                risks.append(
                    f"Series '{sid}' has MASE={score:.2f}, indicating very poor "
                    f"forecastability for this particular series."
                )

    all_above_one = all(e.mean_mase > 1.0 for e in lb)
    if all_above_one:
        risks.append(
            "All models scored above 1.0 (worse than naive). Consider "
            "revisiting feature engineering or data preprocessing."
        )

    if p.missing_ratio > 0.05:
        risks.append(
            f"Missing data ratio of {p.missing_ratio:.1%} may affect "
            f"forecast reliability."
        )

    summary.risks = risks

    # Actionable recommendations
    recommendations: list[str] = []

    if winner.mean_mase < 0.8:
        recommendations.append(
            f"{winner.name} is strongly recommended for production deployment "
            f"with a {naive_improvement:.1f}% improvement over the naive baseline."
        )
    elif winner.mean_mase < 1.0:
        recommendations.append(
            f"{winner.name} outperforms the naive baseline and is a reasonable "
            f"candidate for production use."
        )

    if len(lb) >= 2:
        gap = lb[1].mean_mase - winner.mean_mase
        if gap < 0.01:
            recommendations.append(
                "The top models are very close in performance. Consider running "
                "additional folds or using a larger dataset to confirm the ranking."
            )

    if cv > 0.15 and winner_model and len(models) >= 2:
        runner = lb[1].name if len(lb) >= 2 else "another model"
        recommendations.append(
            f"{winner.name} shows instability across folds. Consider "
            f"ensembling with {runner} for more robust predictions."
        )

    if all_above_one:
        recommendations.append(
            "No model outperforms the naive baseline. Investigate additional "
            "feature engineering, exogenous variables, or alternative model classes."
        )

    if len(models) >= 2:
        fastest = min(models, key=lambda m: m.runtime_sec)
        if fastest.name != winner.name and fastest.runtime_sec > 0:
            speedup = (
                winner_model.runtime_sec / fastest.runtime_sec if winner_model else 0
            )
            if speedup > 5:
                fastest_entry = next(
                    (entry for entry in lb if entry.name == fastest.name),
                    None,
                )
                if fastest_entry is not None:
                    recommendations.append(
                        f"If latency is critical, {fastest.name} runs "
                        f"{speedup:.0f}x faster with MASE "
                        f"{fastest_entry.mean_mase:.4f}."
                    )

    if not recommendations:
        recommendations.append(
            "Review the detailed per-series breakdown to identify "
            "series that may benefit from specialized treatment."
        )

    summary.recommendations = recommendations

    return summary


def _has_per_series_scores(result: BenchmarkResult) -> bool:
    """Check if any model has per-series breakdown scores."""
    for model in result.models:
        for fold in model.folds:
            if fold.series_scores:
                return True
    return False


def _model_comparison_narrative(result: BenchmarkResult) -> str:
    """Generate a comparative narrative when 3+ models are present."""
    lb = result.leaderboard
    models = result.models

    beats_naive = [e for e in lb if e.mean_mase < 1.0]
    worse_than = [e for e in lb if e.mean_mase > 1.0]

    parts: list[str] = []
    if beats_naive:
        names = ", ".join(e.name for e in beats_naive)
        parts.append(
            f"{len(beats_naive)} model{'s' if len(beats_naive) != 1 else ''} "
            f"({names}) beat the naive baseline."
        )
    if worse_than:
        parts.append(
            f"{len(worse_than)} model{'s' if len(worse_than) != 1 else ''} "
            f"scored worse than naive."
        )

    if len(models) >= 2:
        fastest = min(models, key=lambda m: m.runtime_sec)
        best = next((m for m in models if m.name == lb[0].name), None)
        if best and fastest.name != best.name:
            speedup = (
                best.runtime_sec / fastest.runtime_sec if fastest.runtime_sec > 0 else 0
            )
            if speedup > 2:
                parts.append(
                    f"{fastest.name} runs {speedup:.1f}x faster than "
                    f"{best.name}, offering a speed-accuracy trade-off."
                )

    return " ".join(parts) if parts else ""


def _ensemble_hint(result: BenchmarkResult) -> str:
    """Hint about potential ensemble improvement."""
    from ts_autopilot.evaluation.ensemble import recommend_ensemble

    rec = recommend_ensemble(result)
    if rec.n_series < 2 or len(rec.model_win_counts) < 2:
        return ""

    winner_mase = result.leaderboard[0].mean_mase if result.leaderboard else 0
    if winner_mase > 0 and rec.avg_ensemble_mase < winner_mase * 0.95:
        improvement = (1 - rec.avg_ensemble_mase / winner_mase) * 100
        return (
            f"A per-series ensemble (selecting the best model for each "
            f"series) could improve MASE by ~{improvement:.1f}% "
            f"to {rec.avg_ensemble_mase:.4f}."
        )
    return ""


def _find_worst_series(model: ModelResult) -> tuple[str, float] | None:
    """Find the worst-performing series across all folds."""
    series_totals: dict[str, list[float]] = {}
    for fold in model.folds:
        for sid, score in fold.series_scores.items():
            series_totals.setdefault(sid, []).append(score)

    if not series_totals:
        return None

    avg_scores = {
        sid: sum(scores) / len(scores) for sid, scores in series_totals.items()
    }
    worst_sid = max(avg_scores, key=lambda sid: avg_scores[sid])
    return worst_sid, avg_scores[worst_sid]
