"""Tests for extended statistical and intermittent demand runners."""

import numpy as np
import pandas as pd
import pytest

from ts_autopilot.runners.statistical import (
    ALL_STATISTICAL_RUNNERS,
    CORE_RUNNERS,
    EXTENDED_RUNNERS,
    INTERMITTENT_RUNNERS,
    ADIDARunner,
    CrostonClassicRunner,
    CrostonSBARunner,
    HistoricAverageRunner,
    HoltRunner,
    IMAPARunner,
    NaiveRunner,
    WindowAverageRunner,
)


@pytest.fixture
def sample_df():
    """DataFrame with enough data points for model fitting."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    rows = []
    for uid in ["s1", "s2"]:
        for idx, d in enumerate(dates):
            rows.append(
                {"unique_id": uid, "ds": d, "y": 10.0 + idx * 0.1 + rng.normal(0, 0.5)}
            )
    return pd.DataFrame(rows)


@pytest.fixture
def intermittent_df():
    """DataFrame with intermittent demand (many zeros)."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    rows = []
    for uid in ["s1"]:
        for _i, d in enumerate(dates):
            # ~60% zeros (intermittent demand)
            val = rng.choice([0.0, 0.0, 0.0, 5.0, 10.0])
            rows.append({"unique_id": uid, "ds": d, "y": val})
    return pd.DataFrame(rows)


def test_runner_registries():
    """Verify registry tuples contain expected count."""
    assert len(CORE_RUNNERS) == 5
    assert len(EXTENDED_RUNNERS) == 9
    assert len(INTERMITTENT_RUNNERS) == 6
    assert len(ALL_STATISTICAL_RUNNERS) == 20


def test_all_runners_have_unique_names():
    """No two runners should share the same name."""
    names = [r.name for r in ALL_STATISTICAL_RUNNERS]
    assert len(names) == len(set(names)), f"Duplicate names: {names}"


@pytest.mark.parametrize(
    "runner_cls",
    [NaiveRunner, HistoricAverageRunner, HoltRunner, WindowAverageRunner],
)
def test_simple_extended_runners(runner_cls, sample_df):
    """Extended runners produce forecasts."""
    runner = runner_cls()
    output = runner.fit_predict(
        train=sample_df, horizon=7, freq="D", season_length=7, n_jobs=1
    )
    assert len(output.y_hat) == 2 * 7  # 2 series x 7 horizon
    assert all(not np.isnan(v) for v in output.y_hat)


@pytest.mark.parametrize(
    "runner_cls",
    [CrostonClassicRunner, CrostonSBARunner, ADIDARunner, IMAPARunner],
)
def test_intermittent_runners(runner_cls, intermittent_df):
    """Intermittent demand runners produce non-negative forecasts."""
    runner = runner_cls()
    output = runner.fit_predict(
        train=intermittent_df, horizon=7, freq="D", season_length=7, n_jobs=1
    )
    assert len(output.y_hat) == 7  # 1 series x 7 horizon
    # Intermittent demand models should produce non-negative values
    assert all(v >= 0 for v in output.y_hat)
