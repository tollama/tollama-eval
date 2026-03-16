"""Tests for the doctor CLI command."""

from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner

import ts_autopilot.cli as cli_module
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


def test_doctor_does_not_import_optional_modules_directly(
    monkeypatch: pytest.MonkeyPatch,
):
    real_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name == "neuralforecast":
            raise AssertionError("doctor should not import neuralforecast directly")
        return real_import_module(name, package)

    monkeypatch.setattr(cli_module.importlib, "import_module", guarded_import)
    monkeypatch.setattr(
        cli_module,
        "_module_available",
        lambda module_name: module_name == "neuralforecast" or bool(module_name),
    )
    monkeypatch.setattr(
        cli_module,
        "_probe_hardware_acceleration",
        lambda timeout_sec=10.0: "cpu",
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Optional: NeuralForecast" in result.output
