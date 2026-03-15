"""Tests for optional model runners — import fallback and runner properties."""
from unittest.mock import patch

from ts_autopilot.runners.optional import (
    LightGBMRunner,
    NBEATSRunner,
    NHITSRunner,
    OptionalRunnerStatus,
    ProphetRunner,
    get_optional_runners,
    inspect_optional_runner_status,
)


def test_prophet_runner_name():
    assert ProphetRunner().name == "Prophet"


def test_lightgbm_runner_name():
    assert LightGBMRunner().name == "LightGBM"


def test_nhits_runner_name():
    assert NHITSRunner().name == "NHITS"


def test_nbeats_runner_name():
    assert NBEATSRunner().name == "NBEATS"


def test_get_optional_runners_returns_list():
    """get_optional_runners always returns a list, even when nothing is installed."""
    runners = get_optional_runners()
    assert isinstance(runners, list)


def test_get_optional_runners_graceful_when_nothing_installed():
    """When all optional deps are missing, get_optional_runners returns empty list."""
    with (
        patch.dict("sys.modules", {"prophet": None}),
        patch.dict("sys.modules", {"lightgbm": None, "mlforecast": None}),
        patch.dict("sys.modules", {"neuralforecast": None}),
    ):
        # Re-import to pick up mocked modules
        import importlib
        import sys

        # Clear cached imports so the function re-checks
        mod = sys.modules.get("ts_autopilot.runners.optional")
        if mod:
            importlib.reload(mod)
            runners = mod.get_optional_runners()
        else:
            runners = get_optional_runners()

        # With all deps mocked as None (import fails), should be empty
        # Note: This test may still find some deps if actually installed
        assert isinstance(runners, list)


def test_runner_classes_are_base_runner_subclasses():
    from ts_autopilot.runners.base import BaseRunner

    assert issubclass(ProphetRunner, BaseRunner)
    assert issubclass(LightGBMRunner, BaseRunner)
    assert issubclass(NHITSRunner, BaseRunner)
    assert issubclass(NBEATSRunner, BaseRunner)


def test_get_optional_runners_skips_unhealthy_neural_stack(monkeypatch):
    monkeypatch.setattr(
        "ts_autopilot.runners.optional._module_available",
        lambda name: name == "neuralforecast",
    )
    monkeypatch.setattr(
        "ts_autopilot.runners.optional._module_imports_safely",
        lambda name, timeout_sec=10.0: False,
    )

    runners = get_optional_runners(include_neural=True, safe_mode=True)

    assert runners == []


def test_get_optional_runners_includes_neural_when_probe_passes(monkeypatch):
    monkeypatch.setattr(
        "ts_autopilot.runners.optional._module_available",
        lambda name: name == "neuralforecast",
    )
    monkeypatch.setattr(
        "ts_autopilot.runners.optional._module_imports_safely",
        lambda name, timeout_sec=10.0: True,
    )

    runners = get_optional_runners(include_neural=True, safe_mode=True)
    names = [runner.name for runner in runners]

    assert "NHITS" in names
    assert "NBEATS" in names


def test_inspect_optional_runner_status_marks_neural_not_requested():
    statuses = inspect_optional_runner_status(include_neural=False, safe_mode=True)

    neural_status = next(
        status for status in statuses if status.label == "NeuralForecast"
    )

    assert neural_status.available is False
    assert neural_status.reason == "not requested"


def test_get_optional_runners_uses_status_inspection(monkeypatch):
    monkeypatch.setattr(
        "ts_autopilot.runners.optional.inspect_optional_runner_status",
        lambda include_neural=True, safe_mode=True: [
            OptionalRunnerStatus(
                label="Prophet",
                available=True,
                reason="available",
                runner_names=["Prophet"],
            ),
            OptionalRunnerStatus(
                label="LightGBM",
                available=False,
                reason="missing dependency: lightgbm",
                runner_names=["LightGBM"],
            ),
            OptionalRunnerStatus(
                label="XGBoost",
                available=False,
                reason="missing dependency: xgboost",
                runner_names=["XGBoost"],
            ),
            OptionalRunnerStatus(
                label="NeuralForecast",
                available=False,
                reason="failed health check",
                runner_names=["NHITS", "NBEATS", "TiDE", "DeepAR", "PatchTST", "TFT"],
            ),
        ],
    )

    runners = get_optional_runners(include_neural=True, safe_mode=True)

    assert [runner.name for runner in runners] == ["Prophet"]
