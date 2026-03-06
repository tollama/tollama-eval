"""HTML report generator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ts_autopilot import __version__
from ts_autopilot.contracts import BenchmarkResult

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_report(result: BenchmarkResult) -> str:
    """Render an HTML report from a BenchmarkResult.

    Args:
        result: Fully populated BenchmarkResult.

    Returns:
        HTML string.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")

    max_mase = (
        max(e.mean_mase for e in result.leaderboard)
        if result.leaderboard
        else 1.0
    )

    return template.render(
        profile=result.profile,
        config=result.config,
        leaderboard=result.leaderboard,
        models=result.models,
        warnings=result.warnings,
        max_mase=max_mase,
        version=__version__,
        generated_at=datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        ),
    )
