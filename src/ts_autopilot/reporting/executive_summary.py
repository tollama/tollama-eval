"""Auto-generated executive summary — no LLM required."""

from __future__ import annotations

from ts_autopilot.contracts import BenchmarkResult


def generate_executive_summary(result: BenchmarkResult) -> str:
    """Generate a natural-language executive summary from benchmark results.

    Pure template logic with conditional branches — no LLM dependency.
    """
    lines: list[str] = []
    p = result.profile
    c = result.config
    lb = result.leaderboard
    models = result.models

    # Opening statement
    n_models = len(models)
    lines.append(
        f"We evaluated {n_models} forecasting model{'s' if n_models != 1 else ''} "
        f"on {p.n_series} time series ({p.total_rows:,} total observations) "
        f"using {c.n_folds}-fold expanding-window cross-validation "
        f"with a {c.horizon}-step forecast horizon."
    )

    if not lb:
        lines.append("No models produced valid results.")
        return " ".join(lines)

    # Winner announcement
    winner = lb[0]
    naive_improvement = (1.0 - winner.mean_mase) * 100
    if naive_improvement > 0:
        lines.append(
            f"The best-performing model is {winner.name} with a mean MASE of "
            f"{winner.mean_mase:.4f}, outperforming the seasonal naive baseline "
            f"by {naive_improvement:.1f}%."
        )
    elif naive_improvement == 0:
        lines.append(
            f"The best-performing model is {winner.name} with a mean MASE of "
            f"{winner.mean_mase:.4f}, matching the seasonal naive baseline."
        )
    else:
        lines.append(
            f"The best-performing model is {winner.name} with a mean MASE of "
            f"{winner.mean_mase:.4f}. Note: no model beat the seasonal naive "
            f"baseline (MASE < 1.0), suggesting the data may be difficult "
            f"to forecast or additional feature engineering is needed."
        )

    # Runner-up comparison
    if len(lb) >= 2:
        runner_up = lb[1]
        gap = runner_up.mean_mase - winner.mean_mase
        if gap < 0.01:
            lines.append(
                f"{runner_up.name} is a close second (MASE {runner_up.mean_mase:.4f}), "
                f"within {gap:.4f} of the winner."
            )
        else:
            lines.append(
                f"The runner-up is {runner_up.name} (MASE {runner_up.mean_mase:.4f})."
            )

    # Stability assessment
    winner_model = next((m for m in models if m.name == winner.name), None)
    if winner_model and winner_model.std_mase > 0:
        if winner_model.mean_mase > 0:
            cv = winner_model.std_mase / winner_model.mean_mase
        else:
            cv = 0
        if cv < 0.05:
            lines.append(
                f"{winner.name} shows excellent stability across folds (CV = {cv:.1%})."
            )
        elif cv < 0.15:
            lines.append(
                f"{winner.name} shows reasonable stability across folds "
                f"(CV = {cv:.1%})."
            )
        else:
            lines.append(
                f"Caution: {winner.name} shows high variability across folds "
                f"(CV = {cv:.1%}), suggesting sensitivity to the training window."
            )

    # Risk flags
    risks: list[str] = []

    # Check for poor-performing series
    if winner_model:
        worst_series = _find_worst_series(winner_model)
        if worst_series:
            sid, score = worst_series
            if score > 2.0:
                risks.append(
                    f"Series '{sid}' has MASE={score:.2f}, indicating very poor "
                    f"forecastability for this particular series."
                )

    # All models worse than naive
    all_above_one = all(e.mean_mase > 1.0 for e in lb)
    if all_above_one:
        risks.append(
            "All models scored above 1.0 (worse than naive). Consider "
            "revisiting feature engineering or data preprocessing."
        )

    # Data quality concerns
    if p.missing_ratio > 0.05:
        risks.append(
            f"Missing data ratio of {p.missing_ratio:.1%} may affect "
            f"forecast reliability."
        )

    if risks:
        lines.append("Key risks: " + " ".join(risks))

    # SMAPE summary
    if winner.mean_smape > 0:
        lines.append(
            f"In terms of percentage accuracy, {winner.name} achieves "
            f"a SMAPE of {winner.mean_smape:.2f}%."
        )

    return " ".join(lines)


def _find_worst_series(model):
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
    worst_sid = max(avg_scores, key=avg_scores.get)
    return worst_sid, avg_scores[worst_sid]
