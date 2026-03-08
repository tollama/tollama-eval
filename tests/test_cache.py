"""Tests for result caching."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_autopilot.cache import ResultCache, _compute_data_hash
from ts_autopilot.contracts import FoldResult


@pytest.fixture()
def sample_df():
    return pd.DataFrame(
        {
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
            "y": range(30),
        }
    )


@pytest.fixture()
def sample_fold_result():
    return FoldResult(
        fold=1,
        cutoff="2020-01-20",
        mase=0.85,
        smape=0.12,
        rmsse=0.9,
        mae=1.5,
        series_scores={"A": 0.85},
    )


def test_compute_data_hash_deterministic(sample_df):
    h1 = _compute_data_hash(sample_df, horizon=7, n_folds=3)
    h2 = _compute_data_hash(sample_df, horizon=7, n_folds=3)
    assert h1 == h2
    assert len(h1) == 16


def test_compute_data_hash_changes_with_horizon(sample_df):
    h1 = _compute_data_hash(sample_df, horizon=7, n_folds=3)
    h2 = _compute_data_hash(sample_df, horizon=14, n_folds=3)
    assert h1 != h2


def test_cache_miss_returns_none(tmp_path):
    cache = ResultCache(tmp_path / "cache")
    assert cache.get("abc", "Model", 1) is None


def test_cache_put_get_roundtrip(tmp_path, sample_fold_result):
    cache = ResultCache(tmp_path / "cache")
    cache.put("abc", "Model", 1, sample_fold_result)
    result = cache.get("abc", "Model", 1)
    assert result is not None
    assert result.fold == 1
    assert result.mase == 0.85


def test_cache_clear(tmp_path, sample_fold_result):
    cache = ResultCache(tmp_path / "cache")
    cache.put("abc", "Model", 1, sample_fold_result)
    cache.put("abc", "Model", 2, sample_fold_result)
    count = cache.clear()
    assert count == 2
    assert cache.get("abc", "Model", 1) is None


def test_cache_corrupt_entry_handled(tmp_path):
    cache = ResultCache(tmp_path / "cache")
    # Write corrupt JSON
    path = cache._key_path("abc", "Model", 1)
    path.write_text("not valid json{{{")
    result = cache.get("abc", "Model", 1)
    assert result is None
    # Corrupt file should be cleaned up
    assert not path.exists()


def test_cache_creates_directory(tmp_path):
    cache_dir = tmp_path / "deep" / "nested" / "cache"
    ResultCache(cache_dir)
    assert cache_dir.exists()


def test_cache_model_name_with_slash(tmp_path, sample_fold_result):
    cache = ResultCache(tmp_path / "cache")
    cache.put("abc", "org/model-name", 1, sample_fold_result)
    result = cache.get("abc", "org/model-name", 1)
    assert result is not None
    assert result.mase == 0.85
