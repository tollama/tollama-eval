"""Property-based tests using Hypothesis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    FoldResult,
    LeaderboardEntry,
    ModelResult,
    ResultMetadata,
)
from ts_autopilot.evaluation.metrics import (
    per_series_mae,
    per_series_mase,
    per_series_smape,
)

# --- Strategy helpers ---


@st.composite
def fold_results(draw: st.DrawFn) -> FoldResult:
    return FoldResult(
        fold=draw(st.integers(min_value=1, max_value=10)),
        cutoff="2024-01-01T00:00:00",
        mase=draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False)),
        smape=draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False)),
        rmsse=draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False)),
        mae=draw(st.floats(min_value=0.0, max_value=1e6, allow_nan=False)),
    )


@st.composite
def model_results(draw: st.DrawFn) -> ModelResult:
    folds = draw(st.lists(fold_results(), min_size=1, max_size=5))
    mase_vals = [f.mase for f in folds]
    return ModelResult(
        name=draw(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnop")),
        runtime_sec=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)),
        folds=folds,
        mean_mase=round(float(np.mean(mase_vals)), 6),
        std_mase=round(float(np.std(mase_vals)), 6),
        mean_smape=round(float(np.mean([f.smape for f in folds])), 4),
        mean_rmsse=round(float(np.mean([f.rmsse for f in folds])), 6),
        mean_mae=round(float(np.mean([f.mae for f in folds])), 6),
    )


# --- Contract round-trip tests ---


@given(
    n_series=st.integers(min_value=1, max_value=100),
    freq=st.sampled_from(["D", "W", "ME", "h"]),
    missing_ratio=st.floats(min_value=0.0, max_value=1.0),
    season_length=st.integers(min_value=1, max_value=365),
)
def test_data_profile_roundtrip(
    n_series: int, freq: str, missing_ratio: float, season_length: int
) -> None:
    profile = DataProfile(
        n_series=n_series,
        frequency=freq,
        missing_ratio=missing_ratio,
        season_length_guess=season_length,
        min_length=10,
        max_length=100,
        total_rows=n_series * 50,
    )
    d = profile.to_dict()
    restored = DataProfile.from_dict(d)
    assert restored.n_series == profile.n_series
    assert restored.frequency == profile.frequency


@given(mr=model_results())
@settings(max_examples=20)
def test_model_result_roundtrip(mr: ModelResult) -> None:
    d = mr.to_dict()
    restored = ModelResult.from_dict(d)
    assert restored.name == mr.name
    assert len(restored.folds) == len(mr.folds)
    assert restored.mean_mase == mr.mean_mase


@given(mr=model_results())
@settings(max_examples=20)
def test_benchmark_result_roundtrip(mr: ModelResult) -> None:
    profile = DataProfile(
        n_series=1,
        frequency="D",
        missing_ratio=0.0,
        season_length_guess=7,
        min_length=60,
        max_length=60,
        total_rows=60,
    )
    config = BenchmarkConfig(horizon=14, n_folds=3)
    leaderboard = [LeaderboardEntry(rank=1, name=mr.name, mean_mase=mr.mean_mase)]
    result = BenchmarkResult(
        profile=profile,
        config=config,
        models=[mr],
        leaderboard=leaderboard,
        metadata=ResultMetadata.create_now(),
    )
    json_str = result.to_json()
    restored = BenchmarkResult.from_json(json_str)
    assert restored.models[0].name == mr.name
    assert restored.leaderboard[0].mean_mase == mr.mean_mase


# --- Metric property tests ---


@given(
    n=st.integers(min_value=2, max_value=50),
    scale=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False),
)
@settings(max_examples=30)
def test_mae_nonnegative(n: int, scale: float) -> None:
    """MAE must always be non-negative."""
    rng = np.random.default_rng(42)
    actual = rng.normal(0, scale, n)
    forecast = actual + rng.normal(0, scale * 0.1, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    uid = ["s1"] * n

    actuals_df = pd.DataFrame({"unique_id": uid, "ds": dates, "y": actual})
    forecast_df = pd.DataFrame({"unique_id": uid, "ds": dates, "model": forecast})

    scores = per_series_mae(
        forecast_df=forecast_df, actuals_df=actuals_df, model_col="model"
    )
    for v in scores.values():
        assert v >= 0


@given(
    n=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=20)
def test_smape_nonnegative(n: int) -> None:
    """SMAPE must always be non-negative."""
    rng = np.random.default_rng(42)
    actual = rng.uniform(1, 100, n)
    forecast = actual * rng.uniform(0.5, 1.5, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    uid = ["s1"] * n

    actuals_df = pd.DataFrame({"unique_id": uid, "ds": dates, "y": actual})
    forecast_df = pd.DataFrame({"unique_id": uid, "ds": dates, "model": forecast})

    scores = per_series_smape(
        forecast_df=forecast_df, actuals_df=actuals_df, model_col="model"
    )
    for v in scores.values():
        assert v >= 0


@given(
    n=st.integers(min_value=10, max_value=50),
)
@settings(max_examples=20)
def test_perfect_forecast_mase_zero(n: int) -> None:
    """A perfect forecast should have MASE = 0."""
    rng = np.random.default_rng(42)
    values = rng.normal(100, 10, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    uid = ["s1"] * n

    train_n = n - 5
    train_df = pd.DataFrame(
        {
            "unique_id": uid[:train_n],
            "ds": dates[:train_n],
            "y": values[:train_n],
        }
    )
    actuals_df = pd.DataFrame(
        {
            "unique_id": uid[train_n:],
            "ds": dates[train_n:],
            "y": values[train_n:],
        }
    )
    # Perfect forecast = same as actual
    forecast_df = pd.DataFrame(
        {
            "unique_id": uid[train_n:],
            "ds": dates[train_n:],
            "model": values[train_n:],
        }
    )

    scores = per_series_mase(
        forecast_df=forecast_df,
        actuals_df=actuals_df,
        train_df=train_df,
        season_length=1,
        model_col="model",
    )
    for v in scores.values():
        assert math.isclose(v, 0.0, abs_tol=1e-10)
