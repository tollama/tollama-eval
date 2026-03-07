"""HTML report generator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ts_autopilot import __version__
from ts_autopilot.contracts import BenchmarkResult
from ts_autopilot.reporting.executive_summary import generate_executive_summary

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_report(
    result: BenchmarkResult,
    tollama_interpretation: str | None = None,
) -> str:
    """Render an HTML report from a BenchmarkResult.

    Args:
        result: Fully populated BenchmarkResult.
        tollama_interpretation: Optional LLM-generated interpretation text.

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
    executive_summary = generate_executive_summary(result)

    return template.render(
        profile=result.profile,
        config=result.config,
        leaderboard=result.leaderboard,
        models=result.models,
        warnings=result.warnings,
        max_mase=max_mase,
        chart_data=chart_data,
        version=__version__,
        generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        tollama_interpretation=tollama_interpretation,
        executive_summary=executive_summary,
        diagnostics=result.diagnostics,
        has_forecast_data=len(result.forecast_data) > 0,
        has_diagnostics=len(result.diagnostics) > 0,
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
    winner_model = next(
        (m for m in result.models if m.name == best_model), None
    )
    if not winner_model:
        return {"series": []}

    avg_scores: dict[str, float] = {}
    for fold in winner_model.folds:
        for sid, score in fold.series_scores.items():
            avg_scores.setdefault(sid, [])
            avg_scores[sid].append(score)
    avg_scores = {
        sid: sum(scores) / len(scores) for sid, scores in avg_scores.items()
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
            output_series.append({
                "name": sid,
                "mase": round(avg_scores.get(sid, 0), 4),
                "ds": sd["ds_forecast"],
                "y_hat": sd["y_hat"],
                "y_actual": sd["y_actual"],
            })

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
