"""Tests for tollama TSFM integration."""

from __future__ import annotations

import json
import urllib.error
from dataclasses import dataclass, field

import pytest

import ts_autopilot.tollama.client as tollama_client
from ts_autopilot.tollama.client import TollamaError, TollamaForecastResponse, forecast


@dataclass
class _MockTollamaTransport:
    """Capture outgoing requests and return canned forecast responses."""

    response_body: dict = field(
        default_factory=lambda: {
            "mean": [10.0, 11.0, 12.0],
            "model": "chronos2",
            "quantiles": {"0.1": [9.0, 9.5, 10.0], "0.9": [11.0, 12.5, 14.0]},
        }
    )
    requests: list[dict] = field(default_factory=list)

    def urlopen(self, req, timeout=120):
        payload = json.loads(req.data.decode())
        self.requests.append(
            {
                "url": req.full_url,
                "timeout": timeout,
                "payload": payload,
            }
        )
        return _MockHTTPResponse(self.response_body)


class _MockHTTPResponse:
    """Minimal urllib response object for tollama client tests."""

    def __init__(self, body: dict) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._body).encode()


@pytest.fixture()
def mock_tollama(monkeypatch: pytest.MonkeyPatch) -> _MockTollamaTransport:
    """Patch urllib transport so tests do not require a local TCP server."""
    transport = _MockTollamaTransport()
    monkeypatch.setattr(tollama_client.urllib.request, "urlopen", transport.urlopen)
    return transport


def test_forecast_success(mock_tollama: _MockTollamaTransport):
    resp = forecast(
        target=[1.0, 2.0, 3.0, 4.0, 5.0],
        freq="D",
        horizon=3,
        model="chronos2",
        tollama_url="http://tollama.test",
    )
    assert isinstance(resp, TollamaForecastResponse)
    assert resp.mean == [10.0, 11.0, 12.0]
    assert resp.model == "chronos2"
    assert "0.1" in resp.quantiles
    assert "0.9" in resp.quantiles
    assert mock_tollama.requests[0]["url"] == "http://tollama.test/v1/forecast"
    assert mock_tollama.requests[0]["payload"] == {
        "model": "chronos2",
        "series": {"target": [1.0, 2.0, 3.0, 4.0, 5.0], "freq": "D"},
        "horizon": 3,
    }


def test_forecast_unreachable(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(req, timeout=120):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(tollama_client.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(TollamaError, match="unavailable"):
        forecast(
            target=[1.0, 2.0, 3.0],
            freq="D",
            horizon=3,
            model="chronos2",
            tollama_url="http://tollama.test",
            timeout=1,
        )


def test_forecast_bad_json(mock_tollama: _MockTollamaTransport):
    mock_tollama.response_body = {"bad": "response"}

    with pytest.raises(TollamaError, match="Invalid"):
        forecast(
            target=[1.0, 2.0, 3.0],
            freq="D",
            horizon=3,
            model="chronos2",
            tollama_url="http://tollama.test",
        )


def test_forecast_url_trailing_slash(mock_tollama: _MockTollamaTransport):
    resp = forecast(
        target=[1.0, 2.0, 3.0, 4.0, 5.0],
        freq="D",
        horizon=3,
        model="chronos2",
        tollama_url="http://tollama.test/",
    )
    assert isinstance(resp, TollamaForecastResponse)
    assert len(resp.mean) == 3
    assert mock_tollama.requests[0]["url"] == "http://tollama.test/v1/forecast"


def test_forecast_wrong_length(mock_tollama: _MockTollamaTransport):
    """Server returns wrong number of forecast values."""
    mock_tollama.response_body = {
        "mean": [10.0, 11.0],
        "model": "chronos2",
    }

    with pytest.raises(TollamaError, match="Expected 3"):
        forecast(
            target=[1.0, 2.0, 3.0],
            freq="D",
            horizon=3,
            model="chronos2",
            tollama_url="http://tollama.test",
        )
