"""Tests for tollama integration."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import ClassVar

import pytest

from ts_autopilot.tollama.client import TollamaResponse, interpret


@pytest.fixture()
def mock_benchmark_result(tiny_long_df):
    from ts_autopilot.pipeline import run_benchmark

    return run_benchmark(
        tiny_long_df, horizon=7, n_folds=2, model_names=["SeasonalNaive"]
    )


class _MockTollamaHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that returns a canned tollama response."""

    response_body: ClassVar[dict] = {
        "interpretation": "AutoETS performs best on this dataset.",
        "model": "test-llm",
    }
    status_code: ClassVar[int] = 200

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)
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


def test_interpret_success(mock_benchmark_result, tollama_server):
    resp = interpret(mock_benchmark_result, tollama_server)
    assert resp is not None
    assert isinstance(resp, TollamaResponse)
    assert resp.interpretation == "AutoETS performs best on this dataset."
    assert resp.model_used == "test-llm"


def test_interpret_unreachable(mock_benchmark_result):
    resp = interpret(mock_benchmark_result, "http://127.0.0.1:1", timeout=1)
    assert resp is None


def test_interpret_bad_json(mock_benchmark_result, tollama_server):
    _MockTollamaHandler.response_body = {"bad": "response"}
    try:
        resp = interpret(mock_benchmark_result, tollama_server)
        assert resp is None
    finally:
        _MockTollamaHandler.response_body = {
            "interpretation": "AutoETS performs best on this dataset.",
            "model": "test-llm",
        }


def test_interpret_url_trailing_slash(mock_benchmark_result, tollama_server):
    resp = interpret(mock_benchmark_result, tollama_server + "/")
    assert resp is not None
    assert resp.interpretation == "AutoETS performs best on this dataset."
