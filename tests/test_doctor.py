"""Tests for the doctor CLI command."""

from __future__ import annotations

from typer.testing import CliRunner

from ts_autopilot.cli import app

runner = CliRunner()


def test_doctor_runs_successfully():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "tollama-eval doctor" in result.output
    assert "PASS" in result.output
    assert "Python version" in result.output


def test_doctor_checks_core_deps():
    result = runner.invoke(app, ["doctor"])
    assert "Core: pandas" in result.output
    assert "Core: numpy" in result.output


def test_doctor_checks_output_dir():
    result = runner.invoke(app, ["doctor"])
    assert "Output dir writable" in result.output


def test_doctor_shows_summary():
    result = runner.invoke(app, ["doctor"])
    assert "passed" in result.output
