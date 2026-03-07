"""Tests for TollamaRunner (TSFM model runner via tollama server)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from ts_autopilot.runners.tollama import TollamaRunner, get_tollama_runners


class _MockForecastHandler(BaseHTTPRequestHandler):
    """Mock tollama /v1/forecast handler."""

    horizon: ClassVar[int] = 7

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        h = body.get("horizon", self.horizon)
        # Return incrementing values as mock forecast
        mean = [float(i + 100) for i in range(h)]
        resp = {"mean": mean, "model": body.get("model", "test")}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def tollama_server():
    server = HTTPServer(("127.0.0.1", 0), _MockForecastHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture()
def train_df():
    """Canonical long-format DataFrame: 2 series x 30 daily rows."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for d in dates:
            rows.append({"unique_id": uid, "ds": d, "y": 10.0 + np.random.random()})
    return pd.DataFrame(rows)


def test_tollama_runner_name(tollama_server):
    runner = TollamaRunner(model="chronos2", tollama_url=tollama_server)
    assert runner.name == "tollama/chronos2"


def test_tollama_runner_fit_predict(tollama_server, train_df):
    runner = TollamaRunner(model="chronos2", tollama_url=tollama_server)
    output = runner.fit_predict(train=train_df, horizon=7, freq="D", season_length=7)
    assert output.model_name == "tollama/chronos2"
    # 2 series x 7 horizon = 14 predictions
    assert len(output.y_hat) == 14
    assert len(output.unique_id) == 14
    assert len(output.ds) == 14
    assert output.runtime_sec > 0
    # Check both series are present
    assert set(output.unique_id) == {"s1", "s2"}


def test_tollama_runner_forecast_dates(tollama_server, train_df):
    runner = TollamaRunner(model="timesfm", tollama_url=tollama_server)
    output = runner.fit_predict(train=train_df, horizon=3, freq="D", season_length=7)
    # Verify forecast dates start after the last training date
    last_train = train_df.groupby("unique_id")["ds"].max()
    for uid in ["s1", "s2"]:
        uid_ds = [
            pd.Timestamp(d)
            for d, u in zip(output.ds, output.unique_id, strict=False)
            if u == uid
        ]
        assert len(uid_ds) == 3
        assert uid_ds[0] > last_train[uid]


def test_tollama_runner_unreachable(train_df):
    runner = TollamaRunner(model="chronos2", tollama_url="http://127.0.0.1:1")
    output = runner.fit_predict(train=train_df, horizon=3, freq="D", season_length=7)
    # Should produce NaN values but not crash
    assert len(output.y_hat) == 6  # 2 series x 3
    assert all(np.isnan(v) for v in output.y_hat)


def test_get_tollama_runners_empty():
    assert get_tollama_runners("http://localhost:8000", None) == []
    assert get_tollama_runners("http://localhost:8000", []) == []


def test_get_tollama_runners_creates_instances():
    runners = get_tollama_runners("http://localhost:8000", ["chronos2", "timesfm"])
    assert len(runners) == 2
    assert runners[0].name == "tollama/chronos2"
    assert runners[1].name == "tollama/timesfm"
