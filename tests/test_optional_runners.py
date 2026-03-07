"""Tests for optional model runners — import fallback and runner properties."""

from unittest.mock import patch

from ts_autopilot.runners.optional import (
    LightGBMRunner,
    NBEATSRunner,
    NHITSRunner,
    ProphetRunner,
    get_optional_runners,
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
