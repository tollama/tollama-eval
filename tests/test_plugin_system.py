"""Tests for the custom model plugin system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_discover_plugins_empty() -> None:
    """discover_plugins returns empty list when no plugins are registered."""
    from ts_autopilot.runners.base import discover_plugins

    # In test environment, there should be no registered plugins
    plugins = discover_plugins()
    assert isinstance(plugins, list)


def test_discover_plugins_with_mock() -> None:
    """discover_plugins should load valid entry points."""
    from ts_autopilot.contracts import ForecastOutput
    from ts_autopilot.runners.base import BaseRunner, discover_plugins

    class FakeRunner(BaseRunner):
        @property
        def name(self) -> str:
            return "FakePlugin"

        def fit_predict(self, train, horizon, freq, season_length, n_jobs=1, exog=None):
            return ForecastOutput(
                unique_id=["s1"],
                ds=["2020-01-01"],
                y_hat=[1.0],
                model_name="FakePlugin",
                runtime_sec=0.01,
            )

    mock_ep = MagicMock()
    mock_ep.name = "fake_plugin"
    mock_ep.load.return_value = FakeRunner

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]

    with patch("ts_autopilot.runners.base.importlib.metadata.entry_points") as mock_fn:
        mock_fn.return_value = mock_eps
        plugins = discover_plugins()

    assert len(plugins) == 1
    assert plugins[0].name == "FakePlugin"


def test_discover_plugins_invalid_entry_point() -> None:
    """discover_plugins should skip invalid entry points gracefully."""
    from ts_autopilot.runners.base import discover_plugins

    mock_ep = MagicMock()
    mock_ep.name = "bad_plugin"
    mock_ep.load.side_effect = ImportError("No such module")

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]

    with patch("ts_autopilot.runners.base.importlib.metadata.entry_points") as mock_fn:
        mock_fn.return_value = mock_eps
        plugins = discover_plugins()

    assert len(plugins) == 0


def test_discover_plugins_non_runner() -> None:
    """discover_plugins should skip objects that aren't BaseRunner."""
    from ts_autopilot.runners.base import discover_plugins

    mock_ep = MagicMock()
    mock_ep.name = "not_a_runner"
    mock_ep.load.return_value = lambda: "not a runner"

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]

    with patch("ts_autopilot.runners.base.importlib.metadata.entry_points") as mock_fn:
        mock_fn.return_value = mock_eps
        plugins = discover_plugins()

    assert len(plugins) == 0
