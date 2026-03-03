"""Minimal HTML report generator."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ts_autopilot.contracts import BenchmarkResult

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_report(result: BenchmarkResult) -> str:
    """Render a minimal HTML report from a BenchmarkResult.

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
    return template.render(
        profile=result.profile,
        config=result.config,
        leaderboard=result.leaderboard,
        models=result.models,
    )
