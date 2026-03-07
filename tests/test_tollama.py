"""Tests for tollama TSFM integration."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import ClassVar

import pytest

from ts_autopilot.tollama.client import TollamaError, TollamaForecastResponse, forecast


class _MockTollamaHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that returns a canned tollama forecast response."""

    response_body: ClassVar[dict] = {
        "mean": [10.0, 11.0, 12.0],
        "model": "chronos2",
        "quantiles": {"0.1": [9.0, 9.5, 10.0], "0.9": [11.0, 12.5, 14.0]},
    }
    status_code: ClassVar[int] = 200

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        # Validate the request payload structure
        data = json.loads(body)
        assert "model" in data
        assert "series" in data
        assert "horizon" in data
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.response_body).encode())

    def log_message(self, format, *args):
        pass  # suppress server logs in tests


@pytest.fixture()
def tollama_server():
    """Start a mock tollama server on a random port."""
    server = HTTPServer(("127.0.0.1", 0), _MockTollamaHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_forecast_success(tollama_server):
    resp = forecast(
        target=[1.0, 2.0, 3.0, 4.0, 5.0],
        freq="D",
        horizon=3,
        model="chronos2",
        tollama_url=tollama_server,
    )
    assert isinstance(resp, TollamaForecastResponse)
    assert resp.mean == [10.0, 11.0, 12.0]
    assert resp.model == "chronos2"
    assert "0.1" in resp.quantiles
    assert "0.9" in resp.quantiles


def test_forecast_unreachable():
    with pytest.raises(TollamaError, match="unavailable"):
        forecast(
            target=[1.0, 2.0, 3.0],
            freq="D",
            horizon=3,
            model="chronos2",
            tollama_url="http://127.0.0.1:1",
            timeout=1,
        )


def test_forecast_bad_json(tollama_server):
    _MockTollamaHandler.response_body = {"bad": "response"}
    try:
        with pytest.raises(TollamaError, match="Invalid"):
            forecast(
                target=[1.0, 2.0, 3.0],
                freq="D",
                horizon=3,
                model="chronos2",
                tollama_url=tollama_server,
            )
    finally:
        _MockTollamaHandler.response_body = {
            "mean": [10.0, 11.0, 12.0],
            "model": "chronos2",
            "quantiles": {"0.1": [9.0, 9.5, 10.0], "0.9": [11.0, 12.5, 14.0]},
        }


def test_forecast_url_trailing_slash(tollama_server):
    resp = forecast(
        target=[1.0, 2.0, 3.0, 4.0, 5.0],
        freq="D",
        horizon=3,
        model="chronos2",
        tollama_url=tollama_server + "/",
    )
    assert isinstance(resp, TollamaForecastResponse)
    assert len(resp.mean) == 3


def test_forecast_wrong_length(tollama_server):
    """Server returns wrong number of forecast values."""
    _MockTollamaHandler.response_body = {
        "mean": [10.0, 11.0],  # only 2, but horizon=3
        "model": "chronos2",
    }
    try:
        with pytest.raises(TollamaError, match="Expected 3"):
            forecast(
                target=[1.0, 2.0, 3.0],
                freq="D",
                horizon=3,
                model="chronos2",
                tollama_url=tollama_server,
            )
    finally:
        _MockTollamaHandler.response_body = {
            "mean": [10.0, 11.0, 12.0],
            "model": "chronos2",
            "quantiles": {"0.1": [9.0, 9.5, 10.0], "0.9": [11.0, 12.5, 14.0]},
        }
