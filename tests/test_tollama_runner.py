"""Tests for TollamaRunner (TSFM model runner via tollama server)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ts_autopilot.runners.tollama as tollama_runner_module
from ts_autopilot.runners.tollama import TollamaRunner, get_tollama_runners
from ts_autopilot.tollama.client import TollamaError, TollamaForecastResponse


@pytest.fixture()
def train_df():
    """Canonical long-format DataFrame: 2 series x 30 daily rows."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for d in dates:
            rows.append({"unique_id": uid, "ds": d, "y": 10.0 + np.random.random()})
    return pd.DataFrame(rows)


def test_tollama_runner_name():
    runner = TollamaRunner(model="chronos2", tollama_url="http://tollama.test")
    assert runner.name == "tollama/chronos2"


def test_tollama_runner_fit_predict(
    train_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict] = []

    def fake_forecast(*, target, freq, horizon, model, tollama_url, timeout=120):
        calls.append(
            {
                "target": list(target),
                "freq": freq,
                "horizon": horizon,
                "model": model,
                "tollama_url": tollama_url,
            }
        )
        return TollamaForecastResponse(
            mean=[float(i + 100) for i in range(horizon)],
            model=model,
        )

    monkeypatch.setattr(tollama_runner_module, "forecast", fake_forecast)

    runner = TollamaRunner(model="chronos2", tollama_url="http://tollama.test")
    output = runner.fit_predict(train=train_df, horizon=7, freq="D", season_length=7)

    assert output.model_name == "tollama/chronos2"
    assert len(output.y_hat) == 14
    assert len(output.unique_id) == 14
    assert len(output.ds) == 14
    assert output.runtime_sec > 0
    assert set(output.unique_id) == {"s1", "s2"}
    assert len(calls) == 2
    assert all(call["horizon"] == 7 for call in calls)
    assert all(call["tollama_url"] == "http://tollama.test" for call in calls)


def test_tollama_runner_forecast_dates(
    train_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        tollama_runner_module,
        "forecast",
        lambda *, target, freq, horizon, model, tollama_url, timeout=120: (
            TollamaForecastResponse(
                mean=[float(i + 100) for i in range(horizon)],
                model=model,
            )
        ),
    )

    runner = TollamaRunner(model="timesfm", tollama_url="http://tollama.test")
    output = runner.fit_predict(train=train_df, horizon=3, freq="D", season_length=7)

    last_train = train_df.groupby("unique_id")["ds"].max()
    for uid in ["s1", "s2"]:
        uid_ds = [
            pd.Timestamp(d)
            for d, u in zip(output.ds, output.unique_id, strict=False)
            if u == uid
        ]
        assert len(uid_ds) == 3
        assert uid_ds[0] > last_train[uid]


def test_tollama_runner_unreachable(
    train_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_unreachable(*, target, freq, horizon, model, tollama_url, timeout=120):
        raise TollamaError("unavailable")

    monkeypatch.setattr(tollama_runner_module, "forecast", raise_unreachable)

    runner = TollamaRunner(model="chronos2", tollama_url="http://tollama.test")
    output = runner.fit_predict(train=train_df, horizon=3, freq="D", season_length=7)

    assert len(output.y_hat) == 6
    assert all(np.isnan(v) for v in output.y_hat)


def test_get_tollama_runners_empty():
    assert get_tollama_runners("http://localhost:8000", None) == []
    assert get_tollama_runners("http://localhost:8000", []) == []


def test_get_tollama_runners_creates_instances():
    runners = get_tollama_runners("http://localhost:8000", ["chronos2", "timesfm"])
    assert len(runners) == 2
    assert runners[0].name == "tollama/chronos2"
    assert runners[1].name == "tollama/timesfm"
