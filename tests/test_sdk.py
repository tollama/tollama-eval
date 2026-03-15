"""Tests for the fluent Python SDK."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Create a minimal DataFrame for SDK testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for sid in ["s1", "s2"]:
        for d in dates:
            rows.append({"unique_id": sid, "ds": d, "y": np.random.rand() * 100})
    return pd.DataFrame(rows)


def test_sdk_basic_run(sample_df: pd.DataFrame) -> None:
    """TSAutopilot should run a benchmark and return results."""
    from ts_autopilot.sdk import TSAutopilot

    result = (
        TSAutopilot(sample_df)
        .with_models(["SeasonalNaive"])
        .with_horizon(7)
        .with_folds(2)
        .run()
    )
    assert len(result.leaderboard) == 1
    assert result.leaderboard[0].name == "SeasonalNaive"


def test_sdk_chaining(sample_df: pd.DataFrame) -> None:
    """Fluent chaining should return the same TSAutopilot instance."""
    from ts_autopilot.sdk import TSAutopilot

    ts = TSAutopilot(sample_df)
    result = ts.with_models(["SeasonalNaive"]).with_horizon(7).with_folds(2)
    assert result is ts


def test_sdk_with_optional_models_chains(sample_df: pd.DataFrame) -> None:
    from ts_autopilot.sdk import TSAutopilot

    ts = TSAutopilot(sample_df)
    result = ts.with_optional_models(include_neural=True)
    assert result is ts


def test_sdk_save(sample_df: pd.DataFrame, tmp_path: object) -> None:
    """save() should write the standard benchmark artifacts."""
    from pathlib import Path

    from ts_autopilot.sdk import TSAutopilot

    out = Path(str(tmp_path)) / "sdk_out"
    TSAutopilot(sample_df).with_models(["SeasonalNaive"]).with_horizon(7).with_folds(
        2
    ).save(out)

    assert (out / "results.json").exists()
    assert (out / "report.html").exists()
    assert (out / "leaderboard.csv").exists()
    assert (out / "fold_details.csv").exists()
    assert (out / "per_series_scores.csv").exists()
    assert (out / "per_series_winners.csv").exists()


def test_sdk_auto_select(sample_df: pd.DataFrame) -> None:
    """with_auto_select() should pick models automatically."""
    from ts_autopilot.sdk import TSAutopilot

    result = (
        TSAutopilot(sample_df).with_auto_select().with_horizon(7).with_folds(2).run()
    )
    assert len(result.leaderboard) >= 1


def test_sdk_import_from_package() -> None:
    """TSAutopilot should be importable from the package root."""
    from ts_autopilot import TSAutopilot

    assert TSAutopilot is not None
