"""Tests for distributed execution via Ray."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Create a minimal DataFrame for distributed tests."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for sid in ["s1", "s2"]:
        for d in dates:
            rows.append({"unique_id": sid, "ds": d, "y": np.random.rand() * 100})
    return pd.DataFrame(rows)


def test_is_available() -> None:
    """is_available should return bool."""
    from ts_autopilot.distributed.ray_runner import is_available

    result = is_available()
    assert isinstance(result, bool)


def test_fallback_to_local(sample_df: pd.DataFrame) -> None:
    """Without Ray, run_benchmark_distributed should fall back to local."""
    from ts_autopilot.distributed.ray_runner import run_benchmark_distributed

    result = run_benchmark_distributed(
        df=sample_df,
        horizon=7,
        n_folds=2,
        model_names=["SeasonalNaive"],
    )
    assert len(result.leaderboard) == 1
    assert result.leaderboard[0].name == "SeasonalNaive"


def test_fallback_preserves_metrics(sample_df: pd.DataFrame) -> None:
    """Fallback execution should produce valid metrics."""
    from ts_autopilot.distributed.ray_runner import run_benchmark_distributed

    result = run_benchmark_distributed(
        df=sample_df,
        horizon=7,
        n_folds=2,
        model_names=["SeasonalNaive"],
    )
    model = result.models[0]
    assert model.mean_mase > 0
    assert model.mean_smape > 0
    assert len(model.folds) == 2


def test_fallback_on_ray_init_permission_error(
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray bootstrap failures should also fall back to local execution."""
    from ts_autopilot.distributed import ray_runner

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: (_ for _ in ()).throw(PermissionError("blocked")),
    )
    monkeypatch.setattr(ray_runner, "HAS_RAY", True)
    monkeypatch.setattr(ray_runner, "ray", fake_ray)

    result = ray_runner.run_benchmark_distributed(
        df=sample_df,
        horizon=7,
        n_folds=2,
        model_names=["SeasonalNaive"],
    )

    assert len(result.leaderboard) == 1
    assert result.leaderboard[0].name == "SeasonalNaive"
