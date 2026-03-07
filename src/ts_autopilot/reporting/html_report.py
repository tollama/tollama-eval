"""HTML report generator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ts_autopilot import __version__
from ts_autopilot.contracts import BenchmarkResult

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

    # Build chart data for Plotly visualizations
    chart_data = _build_chart_data(result)

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
    )


def _build_chart_data(result: BenchmarkResult) -> dict:
    """Prepare data structures for Plotly charts."""
    # Bar chart data: model names + MASE values (sorted by leaderboard rank)
    bar_names = [e.name for e in result.leaderboard]
    bar_mase = [e.mean_mase for e in result.leaderboard]
    bar_smape = [e.mean_smape for e in result.leaderboard]
    bar_colors = ["#28a745" if v < 1.0 else "#dc3545" for v in bar_mase]

    # Heatmap data: models x folds
    heatmap_models = [m.name for m in result.models]
    heatmap_folds = []
    heatmap_z = []
    if result.models and result.models[0].folds:
        heatmap_folds = [f"Fold {f.fold}" for f in result.models[0].folds]
        for model in result.models:
            heatmap_z.append([f.mase for f in model.folds])

    return {
        "bar_names": bar_names,
        "bar_mase": bar_mase,
        "bar_smape": bar_smape,
        "bar_colors": bar_colors,
        "heatmap_models": heatmap_models,
        "heatmap_folds": heatmap_folds,
        "heatmap_z": heatmap_z,
    }
