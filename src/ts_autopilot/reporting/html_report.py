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

    # Forecast vs actual (best model, last fold)
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


def _build_forecast_chart_data(result: BenchmarkResult) -> dict:
    """Build forecast vs actual chart data for best model."""
    if not result.forecast_data or not result.leaderboard:
        return {"series": []}

    best_model = result.leaderboard[0].name
    # Use last fold for visualization
    last_fold = max(fd.fold for fd in result.forecast_data)
    best_fd = next(
        (
            fd
            for fd in result.forecast_data
            if fd.model_name == best_model and fd.fold == last_fold
        ),
        None,
    )

    if not best_fd:
        return {"series": []}

    # Group by series
    series_data: dict[str, dict] = {}
    for i, uid in enumerate(best_fd.unique_id):
        if uid not in series_data:
            series_data[uid] = {
                "ds_forecast": [],
                "y_hat": [],
                "y_actual": [],
            }
        series_data[uid]["ds_forecast"].append(best_fd.ds[i])
        series_data[uid]["y_hat"].append(best_fd.y_hat[i])
        if i < len(best_fd.y_actual):
            series_data[uid]["y_actual"].append(best_fd.y_actual[i])

    # Find best and worst series by MASE
    winner_model = next((m for m in result.models if m.name == best_model), None)
    if not winner_model:
        return {"series": []}

    scores_by_series: dict[str, list[float]] = {}
    for fold in winner_model.folds:
        for sid, score in fold.series_scores.items():
            scores_by_series.setdefault(sid, []).append(score)
    avg_scores: dict[str, float] = {
        sid: sum(scores) / len(scores) for sid, scores in scores_by_series.items()
    }

    if not avg_scores:
        return {"series": []}

    sorted_series = sorted(avg_scores.items(), key=lambda x: x[1])
    # Top 3 best + top 3 worst (unique)
    n_show = min(3, len(sorted_series))
    selected = []
    for sid, _ in sorted_series[:n_show]:
        selected.append(sid)
    for sid, _ in sorted_series[-n_show:]:
        if sid not in selected:
            selected.append(sid)

    output_series = []
    for sid in selected:
        if sid in series_data:
            sd = series_data[sid]
            output_series.append(
                {
                    "name": sid,
                    "mase": round(avg_scores.get(sid, 0), 4),
                    "ds": sd["ds_forecast"],
                    "y_hat": sd["y_hat"],
                    "y_actual": sd["y_actual"],
                }
            )

    return {
        "series": output_series,
        "model_name": best_model,
        "fold": last_fold,
    }


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
