"""HTML report generator."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ts_autopilot import __version__
from ts_autopilot.contracts import BenchmarkResult
from ts_autopilot.logging_config import get_logger
from ts_autopilot.reporting.executive_summary import generate_executive_summary

logger = get_logger("html_report")

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class _ParetoPointDict(TypedDict):
    name: str
    mase: float
    runtime: float
    is_pareto: bool


def render_report(
    result: BenchmarkResult,
    report_title: str | None = None,
    report_lang: str | None = None,
    report_logo_url: str | None = None,
    report_company: str | None = None,
    report_confidential: bool = False,
) -> str:
    """Render an HTML report from a BenchmarkResult.

    Args:
        result: Fully populated BenchmarkResult.
        report_title: Custom title for the report header.
        report_lang: Language code for the HTML lang attribute (e.g. 'ko', 'ja').
        report_logo_url: URL or base64 data URI for a company logo.
        report_company: Company name for header/footer branding.
        report_confidential: Show confidentiality notice in footer.

    Returns:
        HTML string.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")

    max_mase = (
        max(e.mean_mase for e in result.leaderboard) if result.leaderboard else 1.0
    )

    chart_data = _build_chart_data(result)
    pareto_data = _build_pareto_chart_data(result)
    chart_data["pareto"] = pareto_data
    executive_summary = generate_executive_summary(result)

    # Significance testing
    significance_data = _build_significance_data(result)

    # Confidence intervals for leaderboard
    ci_data = _build_confidence_intervals(result)

    # Runtime comparison table
    runtime_data = _build_runtime_table_data(result)

    # Detect if any tollama TSFM models are in the results
    tollama_models = [m for m in result.models if m.name.startswith("tollama/")]
    has_tsfm = len(tollama_models) > 0

    # Report traceability
    run_id = getattr(result.metadata, "run_id", None) if result.metadata else None

    return str(
        template.render(
            profile=result.profile,
            config=result.config,
            leaderboard=result.leaderboard,
            models=result.models,
            warnings=result.warnings,
            max_mase=max_mase,
            chart_data=chart_data,
            version=__version__,
            generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            executive_summary=executive_summary,
            diagnostics=result.diagnostics,
            has_forecast_data=len(result.forecast_data) > 0,
            has_diagnostics=len(result.diagnostics) > 0,
            has_tsfm=has_tsfm,
            tollama_models=tollama_models,
            report_title=report_title,
            lang=report_lang or "en",
            report_logo_url=report_logo_url,
            report_company=report_company,
            report_confidential=report_confidential,
            significance=significance_data,
            ci_data=ci_data,
            runtime_data=runtime_data,
            run_id=run_id,
            data_chars=result.data_characteristics,
        )
    )


def _build_chart_data(result: BenchmarkResult) -> dict:
    """Prepare data structures for Plotly charts."""
    # Bar chart data
    bar_names = [e.name for e in result.leaderboard]
    bar_mase = [e.mean_mase for e in result.leaderboard]
    bar_smape = [e.mean_smape for e in result.leaderboard]
    bar_colors = ["#2563eb" if v < 1.0 else "#dc2626" for v in bar_mase]

    # Heatmap data: models x folds
    heatmap_models = [m.name for m in result.models]
    heatmap_folds: list[str] = []
    heatmap_z: list[list[float]] = []
    if result.models and result.models[0].folds:
        heatmap_folds = [f"Fold {f.fold}" for f in result.models[0].folds]
        for model in result.models:
            heatmap_z.append([f.mase for f in model.folds])

    # Fold stability: MASE per fold per model (line chart)
    fold_stability: dict[str, list[float]] = {}
    for model in result.models:
        fold_stability[model.name] = [f.mase for f in model.folds]
    fold_labels = heatmap_folds

    # Radar chart: normalized metrics per model
    radar_data = _build_radar_data(result)

    # Error distribution: per-series MASE box plot data
    box_plot_data = _build_box_plot_data(result)

    # Data overview snapshots from the evaluation window
    data_overview = _build_data_overview_chart_data(result)

    # Forecast vs actual (all models, last fold)
    forecast_chart = _build_forecast_chart_data(result)

    # Diagnostics charts (best model)
    diagnostics_chart = _build_diagnostics_chart_data(result)

    return {
        "bar_names": bar_names,
        "bar_mase": bar_mase,
        "bar_smape": bar_smape,
        "bar_colors": bar_colors,
        "heatmap_models": heatmap_models,
        "heatmap_folds": heatmap_folds,
        "heatmap_z": heatmap_z,
        "fold_stability": fold_stability,
        "fold_labels": fold_labels,
        "radar": radar_data,
        "box_plot": box_plot_data,
        "data_overview": data_overview,
        "forecast": forecast_chart,
        "diagnostics": diagnostics_chart,
    }


def _build_radar_data(result: BenchmarkResult) -> dict:
    """Build normalized radar chart data for multi-metric comparison."""
    if not result.models:
        return {"categories": [], "models": {}}

    categories = ["MASE", "SMAPE", "RMSSE", "MAE"]

    # Find max values for normalization
    max_mase = max(m.mean_mase for m in result.models) or 1.0
    max_smape = max(m.mean_smape for m in result.models) or 1.0
    max_rmsse = max(m.mean_rmsse for m in result.models) or 1.0
    max_mae = max(m.mean_mae for m in result.models) or 1.0

    models: dict[str, list[float]] = {}
    for m in result.models:
        models[m.name] = [
            round(m.mean_mase / max_mase, 4) if max_mase else 0,
            round(m.mean_smape / max_smape, 4) if max_smape else 0,
            round(m.mean_rmsse / max_rmsse, 4) if max_rmsse else 0,
            round(m.mean_mae / max_mae, 4) if max_mae else 0,
        ]

    return {"categories": categories, "models": models}


def _build_box_plot_data(result: BenchmarkResult) -> dict:
    """Build per-series MASE distribution data for box plots."""
    data: dict[str, list[float]] = {}
    for model in result.models:
        all_scores: list[float] = []
        for fold in model.folds:
            all_scores.extend(fold.series_scores.values())
        if all_scores:
            data[model.name] = [round(v, 6) for v in all_scores]
    return data


def _build_series_score_lookup(result: BenchmarkResult) -> dict[str, dict[str, float]]:
    """Average per-series MASE across folds for each model."""
    lookup: dict[str, dict[str, float]] = {}
    for model in result.models:
        totals: dict[str, list[float]] = {}
        for fold in model.folds:
            for sid, score in fold.series_scores.items():
                totals.setdefault(sid, []).append(score)
        if totals:
            lookup[model.name] = {
                sid: round(sum(scores) / len(scores), 4)
                for sid, scores in totals.items()
            }
    return lookup


def _get_reference_forecast_data(result: BenchmarkResult):
    """Pick one forecast payload to represent the evaluation window."""
    if not result.forecast_data:
        return None

    last_fold = max(fd.fold for fd in result.forecast_data)
    best_model = result.leaderboard[0].name if result.leaderboard else None
    if best_model is not None:
        for fd in result.forecast_data:
            if fd.model_name == best_model and fd.fold == last_fold:
                return fd

    for fd in result.forecast_data:
        if fd.fold == last_fold:
            return fd
    return result.forecast_data[0]


def _group_forecast_series(fd) -> dict[str, dict]:
    """Group train/evaluation points by unique_id for chart rendering."""
    series: dict[str, dict] = {}

    for i, uid in enumerate(fd.unique_id):
        if uid not in series:
            series[uid] = {
                "ds_history": [],
                "y_history": [],
                "ds_actual": [],
                "y_actual": [],
                "ds_forecast": [],
                "y_hat": [],
            }
        if i < len(fd.ds):
            series[uid]["ds_actual"].append(fd.ds[i])
            series[uid]["ds_forecast"].append(fd.ds[i])
        if i < len(fd.y_actual):
            series[uid]["y_actual"].append(fd.y_actual[i])
        if i < len(fd.y_hat):
            series[uid]["y_hat"].append(fd.y_hat[i])

    train_ids = fd.train_unique_id
    if not train_ids and len(set(fd.unique_id)) == 1:
        train_ids = [fd.unique_id[0]] * len(fd.ds_train_tail)

    for i, uid in enumerate(train_ids):
        if uid not in series:
            series[uid] = {
                "ds_history": [],
                "y_history": [],
                "ds_actual": [],
                "y_actual": [],
                "ds_forecast": [],
                "y_hat": [],
            }
        if i < len(fd.ds_train_tail):
            series[uid]["ds_history"].append(fd.ds_train_tail[i])
        if i < len(fd.y_train_tail):
            series[uid]["y_history"].append(fd.y_train_tail[i])

    return series


def _select_representative_series(
    result: BenchmarkResult,
    available_series: list[str],
    scores_by_model: dict[str, dict[str, float]],
    max_series: int = 3,
) -> list[str]:
    """Pick best, median, and worst series to keep charts focused."""
    if not available_series:
        return []

    best_model = result.leaderboard[0].name if result.leaderboard else None
    score_map = scores_by_model.get(best_model or "", {})
    ranked = [
        (sid, score_map[sid]) for sid in available_series if sid in score_map
    ]
    ranked.sort(key=lambda item: item[1])

    selected: list[str] = []
    if ranked:
        candidate_positions = [0, len(ranked) // 2, len(ranked) - 1]
        for pos in candidate_positions:
            sid = ranked[pos][0]
            if sid not in selected:
                selected.append(sid)
        for sid, _ in ranked:
            if len(selected) >= min(max_series, len(available_series)):
                break
            if sid not in selected:
                selected.append(sid)

    if len(selected) < min(max_series, len(available_series)):
        for sid in sorted(available_series):
            if sid not in selected:
                selected.append(sid)
            if len(selected) >= min(max_series, len(available_series)):
                break

    return selected


def _build_data_overview_insights(result: BenchmarkResult) -> list[str]:
    """Generate short chart-adjacent data interpretation notes."""
    insights: list[str] = []
    profile = result.profile
    data_chars = result.data_characteristics

    if profile.missing_ratio == 0:
        insights.append("No missing values were detected in the input data.")
    elif profile.missing_ratio < 0.05:
        insights.append(
            f"Missingness is low at {profile.missing_ratio:.1%}, "
            "so ranking noise from gaps should be limited."
        )
    else:
        insights.append(
            f"Missingness is {profile.missing_ratio:.1%}; "
            "treat model differences cautiously on sparse regions."
        )

    if data_chars is not None:
        if data_chars.seasonality_strength >= 0.5:
            insights.append(
                f"Seasonality is strong "
                f"({data_chars.seasonality_strength:.2f}), "
                "so seasonal models should have an advantage."
            )
        elif data_chars.trend_strength >= 0.6:
            insights.append(
                f"Trend strength is elevated ({data_chars.trend_strength:.2f}), "
                "so trend-tracking behavior matters more than seasonal fit."
            )

        if data_chars.series_heterogeneity >= 0.7:
            insights.append(
                f"Series heterogeneity is high "
                f"({data_chars.series_heterogeneity:.2f}), "
                "which increases the chance that one global winner "
                "hides per-series losers."
            )

    if profile.min_length < profile.season_length_guess * 2:
        insights.append(
            "The shortest series are relatively short for the "
            "inferred seasonality, which can reduce forecast stability."
        )

    return insights[:3]


def _build_data_overview_chart_data(result: BenchmarkResult) -> dict:
    """Build recent-history charts to explain the benchmarked data window."""
    reference_fd = _get_reference_forecast_data(result)
    if reference_fd is None:
        return {"series": [], "insights": []}

    scores_by_model = _build_series_score_lookup(result)
    grouped = _group_forecast_series(reference_fd)
    selected = _select_representative_series(
        result=result,
        available_series=list(grouped.keys()),
        scores_by_model=scores_by_model,
    )

    best_model = result.leaderboard[0].name if result.leaderboard else ""
    best_scores = scores_by_model.get(best_model, {})
    series_payload = []
    for idx, sid in enumerate(selected, start=1):
        payload = grouped[sid]
        score = best_scores.get(sid)
        series_payload.append(
            {
                "name": sid,
                "chart_id": f"chart-data-overview-{idx}",
                "ds_history": payload["ds_history"],
                "y_history": payload["y_history"],
                "ds_actual": payload["ds_actual"],
                "y_actual": payload["y_actual"],
                "mase": score,
                "summary": _describe_series_snapshot(
                    sid=sid,
                    mase=score,
                    history_points=len(payload["y_history"]),
                    horizon_points=len(payload["y_actual"]),
                ),
            }
        )

    return {
        "series": series_payload,
        "fold": reference_fd.fold,
        "reference_model": reference_fd.model_name,
        "insights": _build_data_overview_insights(result),
    }


def _describe_series_snapshot(
    sid: str,
    mase: float | None,
    history_points: int,
    horizon_points: int,
) -> str:
    """Describe the data window shown for one representative series."""
    parts = [
        f"{sid} shows {history_points} recent training points "
        f"and {horizon_points} holdout observations."
    ]
    if mase is not None:
        if mase < 1.0:
            parts.append(
                f"The winning model beats naive on this series "
                f"(MASE {mase:.4f})."
            )
        else:
            parts.append(
                "This series remains difficult for the winning model "
                f"(MASE {mase:.4f})."
            )
    return " ".join(parts)


def _describe_model_forecast(
    model_name: str,
    model_score: float,
    leader_name: str | None,
    leader_score: float | None,
    selected_scores: list[float],
) -> str:
    """Summarize what a model's forecast charts mean."""
    parts = [f"Mean MASE is {model_score:.4f}."]
    if model_score < 1.0:
        parts.append(
            f"That is {(1.0 - model_score) * 100:.1f}% better "
            "than the naive baseline."
        )
    elif model_score > 1.0:
        parts.append(
            f"That is {(model_score - 1.0) * 100:.1f}% worse "
            "than the naive baseline."
        )
    else:
        parts.append("That matches the naive baseline.")

    if leader_name and leader_score is not None and model_name != leader_name:
        parts.append(
            f"It trails {leader_name} "
            f"by {model_score - leader_score:.4f} MASE."
        )

    if selected_scores:
        beats_naive = sum(score < 1.0 for score in selected_scores)
        parts.append(
            f"Among the highlighted series, "
            f"{beats_naive}/{len(selected_scores)} beat the naive baseline."
        )

    return " ".join(parts)


def _build_forecast_chart_data(result: BenchmarkResult) -> dict:
    """Build forecast vs actual chart data for every model on the last fold."""
    if not result.forecast_data:
        return {"models": [], "selected_series": []}

    last_fold = max(fd.fold for fd in result.forecast_data)
    scores_by_model = _build_series_score_lookup(result)

    reference_fd = _get_reference_forecast_data(result)
    if reference_fd is None:
        return {"models": [], "selected_series": []}

    selected_series = _select_representative_series(
        result=result,
        available_series=list(_group_forecast_series(reference_fd).keys()),
        scores_by_model=scores_by_model,
    )

    leaderboard_map = {entry.name: entry.mean_mase for entry in result.leaderboard}
    leader_name = result.leaderboard[0].name if result.leaderboard else None
    leader_score = result.leaderboard[0].mean_mase if result.leaderboard else None

    models_payload = []
    for model_idx, model in enumerate(result.models, start=1):
        fd = next(
            (
                item
                for item in result.forecast_data
                if item.model_name == model.name and item.fold == last_fold
            ),
            None,
        )
        if fd is None:
            continue

        grouped = _group_forecast_series(fd)
        per_series_scores = scores_by_model.get(model.name, {})
        series_payload = []
        selected_scores: list[float] = []
        for series_idx, sid in enumerate(selected_series, start=1):
            if sid not in grouped:
                continue
            payload = grouped[sid]
            score = per_series_scores.get(sid)
            if score is not None:
                selected_scores.append(score)
            series_payload.append(
                {
                    "name": sid,
                    "chart_id": f"chart-forecast-{model_idx}-{series_idx}",
                    "mase": score,
                    "note": _describe_series_forecast(score),
                    "ds_history": payload["ds_history"],
                    "y_history": payload["y_history"],
                    "ds_actual": payload["ds_actual"],
                    "y_actual": payload["y_actual"],
                    "ds_forecast": payload["ds_forecast"],
                    "y_hat": payload["y_hat"],
                }
            )

        if not series_payload:
            continue

        models_payload.append(
            {
                "name": model.name,
                "rank": next(
                    (
                        entry.rank
                        for entry in result.leaderboard
                        if entry.name == model.name
                    ),
                    None,
                ),
                "mean_mase": leaderboard_map.get(model.name, model.mean_mase),
                "summary": _describe_model_forecast(
                    model_name=model.name,
                    model_score=leaderboard_map.get(model.name, model.mean_mase),
                    leader_name=leader_name,
                    leader_score=leader_score,
                    selected_scores=selected_scores,
                ),
                "series": series_payload,
            }
        )

    return {
        "models": models_payload,
        "selected_series": selected_series,
        "fold": last_fold,
    }


def _describe_series_forecast(mase: float | None) -> str:
    """Explain one per-series forecast panel in plain language."""
    if mase is None:
        return "Per-series MASE is unavailable for this panel."
    if mase < 1.0:
        return (
            f"MASE {mase:.4f}; this model beats the naive baseline "
            "on the highlighted series."
        )
    if mase > 1.0:
        return (
            f"MASE {mase:.4f}; this model underperforms the naive baseline "
            "on the highlighted series."
        )
    return (
        "MASE 1.0000; this model matches the naive baseline "
        "on the highlighted series."
    )


def _build_diagnostics_chart_data(result: BenchmarkResult) -> dict:
    """Build diagnostics chart data for the best model."""
    if not result.diagnostics or not result.leaderboard:
        return {}

    best_model = result.leaderboard[0].name
    diag = next(
        (d for d in result.diagnostics if d.model_name == best_model),
        None,
    )

    if not diag:
        return {}

    return {
        "model_name": diag.model_name,
        "residual_mean": diag.residual_mean,
        "residual_std": diag.residual_std,
        "residual_skew": diag.residual_skew,
        "residual_kurtosis": diag.residual_kurtosis,
        "ljung_box_p": diag.ljung_box_p,
        "histogram_bins": diag.histogram_bins,
        "histogram_counts": diag.histogram_counts,
        "acf_lags": diag.acf_lags,
        "acf_values": diag.acf_values,
        "residuals": diag.residuals,
        "fitted": diag.fitted,
    }


def _build_pareto_chart_data(result: BenchmarkResult) -> dict:
    """Build Pareto frontier (accuracy vs speed) chart data."""
    from ts_autopilot.evaluation.speed_benchmark import compute_speed_report

    if not result.models:
        return {"points": [], "frontier": []}

    report = compute_speed_report(result)

    points: list[_ParetoPointDict] = []
    frontier: list[_ParetoPointDict] = []
    for pp in report.pareto_points:
        point: _ParetoPointDict = {
            "name": pp.model_name,
            "mase": round(pp.mean_mase, 4),
            "runtime": round(pp.total_runtime_sec, 4),
            "is_pareto": pp.is_pareto_optimal,
        }
        points.append(point)
        if pp.is_pareto_optimal:
            frontier.append(point)

    # Sort frontier by runtime for line drawing
    frontier.sort(key=lambda p: p["runtime"])

    return {"points": points, "frontier": frontier}


def _build_significance_data(result: BenchmarkResult) -> dict:
    """Build statistical significance data (Friedman + Nemenyi)."""
    from ts_autopilot.evaluation.significance import (
        friedman_test,
        render_critical_difference_svg,
    )

    if len(result.models) < 2:
        return {}

    # Collect per-series MASE scores averaged across folds
    per_series_scores: dict[str, dict[str, float]] = {}
    for model in result.models:
        series_totals: dict[str, list[float]] = {}
        for fold in model.folds:
            for sid, score in fold.series_scores.items():
                series_totals.setdefault(sid, []).append(score)
        if series_totals:
            per_series_scores[model.name] = {
                sid: sum(scores) / len(scores) for sid, scores in series_totals.items()
            }

    if len(per_series_scores) < 2:
        return {}

    report = friedman_test(per_series_scores)
    if report is None:
        return {}

    cd_svg = ""
    if report.critical_difference > 0:
        cd_svg = render_critical_difference_svg(
            report.mean_ranks, report.critical_difference
        )

    # Build pairwise matrix for template
    pairwise_matrix: list[dict] = []
    for p in report.pairwise:
        pairwise_matrix.append(
            {
                "model_a": p.model_a,
                "model_b": p.model_b,
                "rank_diff": round(p.rank_diff, 3),
                "significant": p.significant,
            }
        )

    # Sort mean ranks by rank value
    sorted_ranks = sorted(report.mean_ranks.items(), key=lambda x: x[1])

    return {
        "friedman_statistic": round(report.friedman_statistic, 4),
        "friedman_p_value": report.friedman_p_value,
        "n_models": report.n_models,
        "n_series": report.n_series,
        "mean_ranks": sorted_ranks,
        "pairwise": pairwise_matrix,
        "cd": round(report.critical_difference, 4),
        "cd_svg": cd_svg,
        "is_significant": report.friedman_p_value < 0.05,
    }


def _build_confidence_intervals(result: BenchmarkResult) -> dict[str, dict]:
    """Compute 95% confidence intervals on MASE for each model."""
    ci: dict[str, dict] = {}

    for model in result.models:
        if not model.folds:
            continue
        n = len(model.folds)
        if n < 2:
            ci[model.name] = {
                "lower": model.mean_mase,
                "upper": model.mean_mase,
            }
            continue

        # t-distribution critical value for 95% CI
        # For small n, use approximate t values
        t_values = {
            2: 12.706,
            3: 4.303,
            4: 3.182,
            5: 2.776,
            6: 2.571,
            7: 2.447,
            8: 2.365,
            9: 2.306,
            10: 2.262,
        }
        t_val = t_values.get(n, 1.96)  # fallback to z for large n

        margin = t_val * model.std_mase / math.sqrt(n)
        ci[model.name] = {
            "lower": round(model.mean_mase - margin, 4),
            "upper": round(model.mean_mase + margin, 4),
        }

    return ci


def _build_runtime_table_data(result: BenchmarkResult) -> list[dict]:
    """Build runtime comparison table data."""
    from ts_autopilot.evaluation.speed_benchmark import compute_speed_report

    if not result.models:
        return []

    report = compute_speed_report(result)
    rows = []
    for p in sorted(report.profiles, key=lambda x: x.total_runtime_sec):
        rows.append(
            {
                "name": p.model_name,
                "total_sec": round(p.total_runtime_sec, 2),
                "avg_per_series": round(p.avg_sec_per_series, 3),
                "throughput": round(p.throughput_series_per_sec, 1),
            }
        )
    return rows
