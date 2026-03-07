"""Optional PDF export from HTML report."""

from __future__ import annotations

from pathlib import Path

from ts_autopilot.logging_config import get_logger

logger = get_logger("pdf_export")


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
        from weasyprint import HTML
    except ImportError:
        logger.info(
            'weasyprint not installed. Install with: pip install "ts-autopilot[pdf]"'
        )
        return False

    try:
        html_path = Path(html_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        HTML(filename=str(html_path)).write_pdf(str(output_path))
        logger.info("Generated PDF: %s", output_path)
        return True
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)
        return False
