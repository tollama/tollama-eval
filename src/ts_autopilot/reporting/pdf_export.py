"""Optional PDF export from HTML report."""

from __future__ import annotations

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
.toc, .skip-link, .theme-toggle, .header-controls {
    display: none !important;
}
"""


def is_available() -> bool:
    """Check if weasyprint is installed."""
    try:
        import weasyprint  # noqa: F401

        return True
    except ImportError:
        return False


def generate_pdf(html_path: str | Path, output_path: str | Path) -> bool:
    """Convert an HTML report to PDF.

    Args:
        html_path: Path to the HTML report file.
        output_path: Path for the output PDF file.

    Returns:
        True if PDF was generated successfully, False otherwise.
    """
    try:
        from weasyprint import CSS, HTML
    except ImportError:
        logger.info(
            'weasyprint not installed. Install with: pip install "ts-autopilot[pdf]"'
        )
        return False

    try:
        html_path = Path(html_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = HTML(filename=str(html_path))
        extra_css = CSS(string=_PDF_EXTRA_CSS)
        html.write_pdf(str(output_path), stylesheets=[extra_css])
        logger.info("Generated PDF: %s", output_path)
        return True
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)
        return False
