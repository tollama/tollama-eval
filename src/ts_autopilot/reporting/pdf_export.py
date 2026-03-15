"""Optional PDF export from HTML report."""

from __future__ import annotations

import base64
import re
from pathlib import Path

from ts_autopilot.logging_config import get_logger

logger = get_logger("pdf_export")

# CSS injected into PDF for better print quality
_PDF_EXTRA_CSS = """
@page {
    size: A4 landscape;
    margin: 1.5cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #6b7280;
    }
}
body {
    font-size: 10pt;
    max-width: 100%;
}
.report-header {
    page-break-after: avoid;
}
.card {
    page-break-inside: avoid;
}
.chart-container {
    page-break-inside: avoid;
}
.toc, .skip-link, .theme-toggle, .header-controls, .download-btn, .model-filter {
    display: none !important;
}
/* Show PDF cover page */
.pdf-cover {
    display: flex !important;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 90vh;
    text-align: center;
    page-break-after: always;
}
.pdf-cover h1 { font-size: 2rem; color: #1e40af; margin-bottom: 0.5rem; }
.pdf-cover .cover-meta { color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0; }
.pdf-cover .cover-logo { max-height: 80px; margin-bottom: 1.5rem; }
/* Hide interactive chart divs, show static images */
.chart-container .js-plotly-plot {
    display: none !important;
}
.static-chart-img {
    display: block !important;
    max-width: 100%;
    margin: 0.5rem auto;
}
"""


def is_available() -> bool:
    """Check if weasyprint is installed."""
    try:
        import weasyprint  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def _kaleido_available() -> bool:
    """Check if kaleido is available for static chart rendering."""
    try:
        import plotly.io  # noqa: F401

        # Test kaleido by checking the engine
        from plotly.io._kaleido import scope  # noqa: F401

        return True
    except (ImportError, Exception):
        return False


def _inject_static_charts(html_content: str) -> str:
    """Replace Plotly chart divs with static PNG images for PDF rendering.

    Requires kaleido. If not available, returns HTML unchanged.
    """
    if not _kaleido_available():
        logger.info(
            "kaleido not available for static chart rendering. "
            'Install with: pip install "tollama-eval[pdf]"'
        )
        return html_content

    try:
        import json as json_mod

        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return html_content

    # Extract chart data from the script section
    chart_data_match = re.search(
        r"var chartData = ({.*?});\s*var fontFamily",
        html_content,
        re.DOTALL,
    )
    if not chart_data_match:
        return html_content

    try:
        chart_data = json_mod.loads(chart_data_match.group(1))
    except (json_mod.JSONDecodeError, ValueError):
        logger.warning("Could not parse chart data for static rendering")
        return html_content

    static_images: dict[str, str] = {}

    def _figure_to_img_html(fig, alt: str) -> str:
        img_bytes = pio.to_image(fig, format="png", scale=2)
        b64 = base64.b64encode(img_bytes).decode()
        return (
            f'<img class="static-chart-img" src="data:image/png;base64,{b64}" '
            f'alt="{alt}">'
        )

    def _build_series_line_chart(
        title: str,
        history_x: list[str] | None,
        history_y: list[float] | None,
        actual_x: list[str] | None,
        actual_y: list[float] | None,
        forecast_x: list[str] | None = None,
        forecast_y: list[float] | None = None,
    ):
        fig = go.Figure()
        if history_x and history_y:
            fig.add_trace(
                go.Scatter(
                    x=history_x,
                    y=history_y,
                    mode="lines+markers",
                    name="Recent history",
                    line={"color": "#6b7280", "width": 2},
                )
            )
        if actual_x and actual_y:
            fig.add_trace(
                go.Scatter(
                    x=actual_x,
                    y=actual_y,
                    mode="lines+markers",
                    name="Actual",
                    line={"color": "#2563eb", "width": 2},
                )
            )
        if forecast_x and forecast_y:
            fig.add_trace(
                go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode="lines+markers",
                    name="Forecast",
                    line={"color": "#dc2626", "width": 2, "dash": "dash"},
                )
            )
        fig.update_layout(
            title=title,
            width=900,
            height=320,
            showlegend=True,
            margin={"l": 60, "r": 30, "t": 50, "b": 50},
        )
        return fig

    # Generate static images for key charts
    try:
        # MASE bar chart
        if chart_data.get("bar_names"):
            fig = go.Figure(
                go.Bar(
                    y=chart_data["bar_names"][::-1],
                    x=chart_data["bar_mase"][::-1],
                    orientation="h",
                    marker_color=chart_data["bar_colors"][::-1],
                )
            )
            fig.update_layout(
                title="Model Comparison (MASE)",
                xaxis_title="MASE",
                height=max(250, len(chart_data["bar_names"]) * 45 + 100),
                width=900,
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="#9ca3af")
            img_bytes = pio.to_image(fig, format="png", scale=2)
            b64 = base64.b64encode(img_bytes).decode()
            static_images["chart-mase"] = (
                f'<img class="static-chart-img" src="data:image/png;base64,{b64}" '
                f'alt="MASE comparison bar chart">'
            )

        # Heatmap
        if chart_data.get("heatmap_z"):
            fig = go.Figure(
                go.Heatmap(
                    z=chart_data["heatmap_z"],
                    x=chart_data["heatmap_folds"],
                    y=chart_data["heatmap_models"],
                    colorscale=[[0, "#059669"], [0.5, "#fbbf24"], [1, "#dc2626"]],
                )
            )
            fig.update_layout(title="MASE Heatmap by Fold", width=900, height=400)
            img_bytes = pio.to_image(fig, format="png", scale=2)
            b64 = base64.b64encode(img_bytes).decode()
            static_images["chart-heatmap"] = (
                f'<img class="static-chart-img" src="data:image/png;base64,{b64}" '
                f'alt="MASE heatmap">'
            )

        # Per-series winner charts
        per_series = chart_data.get("per_series") or {}
        if per_series.get("winner_summary"):
            fig = go.Figure(
                go.Bar(
                    x=[item["name"] for item in per_series["winner_summary"]],
                    y=[item["count"] for item in per_series["winner_summary"]],
                )
            )
            fig.update_layout(
                title="Series Wins by Model",
                width=800,
                height=320,
                yaxis_title="Series won",
            )
            static_images["chart-per-series-wins"] = _figure_to_img_html(
                fig,
                "Per-series winner counts",
            )

        if per_series.get("heatmap_z"):
            fig = go.Figure(
                go.Heatmap(
                    z=per_series["heatmap_z"],
                    x=per_series["models"],
                    y=per_series["heatmap_series"],
                    colorscale=[[0, "#059669"], [0.5, "#fbbf24"], [1, "#dc2626"]],
                )
            )
            fig.update_layout(
                title="Per-Series Difficulty by Model",
                width=900,
                height=max(400, len(per_series["heatmap_series"]) * 28 + 120),
            )
            static_images["chart-per-series-heatmap"] = _figure_to_img_html(
                fig,
                "Per-series MASE heatmap",
            )

        # Data overview charts
        data_overview = chart_data.get("data_overview") or {}
        for series in data_overview.get("series", []):
            title = series["name"]
            mase = series.get("mase")
            if mase is not None:
                title += f" (Winner MASE: {mase:.4f})"
            fig = _build_series_line_chart(
                title=title,
                history_x=series.get("ds_history"),
                history_y=series.get("y_history"),
                actual_x=series.get("ds_actual"),
                actual_y=series.get("y_actual"),
            )
            static_images[series["chart_id"]] = _figure_to_img_html(
                fig,
                f"Data overview for {series['name']}",
            )

        # Forecast charts
        forecast_data = chart_data.get("forecast") or {}
        for model in forecast_data.get("models", []):
            for series in model.get("series", []):
                title = f"{model['name']} - {series['name']}"
                mase = series.get("mase")
                if mase is not None:
                    title += f" (MASE: {mase:.4f})"
                fig = _build_series_line_chart(
                    title=title,
                    history_x=series.get("ds_history"),
                    history_y=series.get("y_history"),
                    actual_x=series.get("ds_actual"),
                    actual_y=series.get("y_actual"),
                    forecast_x=series.get("ds_forecast"),
                    forecast_y=series.get("y_hat"),
                )
                static_images[series["chart_id"]] = _figure_to_img_html(
                    fig,
                    f"Forecast chart for {model['name']} on {series['name']}",
                )

    except Exception as exc:
        logger.warning("Static chart generation failed: %s", exc)

    # Inject static images after their corresponding chart divs
    for chart_id, img_html in static_images.items():
        pattern = f'<div id="{chart_id}"></div>'
        replacement = f"{pattern}\n{img_html}"
        html_content = html_content.replace(pattern, replacement)

    return html_content


def generate_pdf(html_path: str | Path, output_path: str | Path) -> bool:
    """Convert an HTML report to PDF.

    If kaleido is available, injects static chart images before rendering.
    WeasyPrint cannot execute JavaScript, so Plotly charts would otherwise
    appear blank in the PDF.

    Args:
        html_path: Path to the HTML report file.
        output_path: Path for the output PDF file.

    Returns:
        True if PDF was generated successfully, False otherwise.
    """
    try:
        from weasyprint import CSS, HTML
    except (ImportError, OSError):
        logger.info(
            'weasyprint not installed. Install with: pip install "tollama-eval[pdf]"'
        )
        return False

    try:
        html_path = Path(html_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_content = html_path.read_text(encoding="utf-8")

        # Inject static chart images for PDF
        html_content = _inject_static_charts(html_content)

        html = HTML(string=html_content, base_url=str(html_path.parent))
        extra_css = CSS(string=_PDF_EXTRA_CSS)
        html.write_pdf(str(output_path), stylesheets=[extra_css])
        logger.info("Generated PDF: %s", output_path)
        return True
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)
        return False
