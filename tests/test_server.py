"""Tests for the REST API server."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip all tests if fastapi is not installed
fastapi = pytest.importorskip("fastapi")


@pytest.fixture()
def client():
    """Create a TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient

    from ts_autopilot.server.app import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture()
def sample_csv_bytes() -> bytes:
    """Generate sample CSV content as bytes."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for sid in ["s1", "s2"]:
        for d in dates:
            rows.append({"unique_id": sid, "ds": d, "y": np.random.rand() * 100})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()


def test_health_endpoint(client) -> None:
    """GET /health should return ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_not_found(client) -> None:
    """GET /status/nonexistent should return 404."""
    response = client.get("/status/nonexistent")
    assert response.status_code == 404


def test_results_not_found(client) -> None:
    """GET /results/nonexistent should return 404."""
    response = client.get("/results/nonexistent")
    assert response.status_code == 404


def test_submit_benchmark(client, sample_csv_bytes: bytes) -> None:
    """POST /benchmark should accept CSV and return run_id."""
    response = client.post(
        "/benchmark",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        params={"horizon": 7, "n_folds": 2, "models": "SeasonalNaive"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "pending"


def test_submit_and_check_status(client, sample_csv_bytes: bytes) -> None:
    """Submit a benchmark and check its status."""
    response = client.post(
        "/benchmark",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        params={"horizon": 7, "n_folds": 2, "models": "SeasonalNaive"},
    )
    run_id = response.json()["run_id"]

    # Status should be available immediately
    status_response = client.get(f"/status/{run_id}")
    assert status_response.status_code == 200
    status = status_response.json()["status"]
    assert status in ("pending", "running", "completed", "failed")
