"""Tests for CLI config merging logic."""

from __future__ import annotations

from pathlib import Path

from ts_autopilot.cli import _merge_config
from ts_autopilot.config import FileConfig


def test_merge_config_cli_defaults_use_file():
    """When CLI values are at their defaults, file config wins."""
    cfg = FileConfig(
        input="data.csv",
        output="results/",
        horizon=7,
        n_folds=5,
        n_jobs=4,
    )
    m = _merge_config(
        cfg,
        cli_input=None,
        cli_output=Path("out/"),
        cli_horizon=14,
        cli_n_folds=3,
        cli_models=None,
        cli_tollama_url=None,
        cli_tollama_models=None,
        cli_n_jobs=1,
        cli_no_cache=False,
        cli_cache_dir=None,
        cli_parallel_models=False,
        cli_include_optional_models=False,
        cli_include_neural_models=False,
        default_timeout=300.0,
    )
    assert m["input"] == Path("data.csv")
    assert m["output"] == Path("results/")
    assert m["horizon"] == 7
    assert m["n_folds"] == 5
    assert m["n_jobs"] == 4


def test_merge_config_cli_overrides_file():
    """When CLI values differ from defaults, CLI wins."""
    cfg = FileConfig(horizon=7, n_folds=5)
    m = _merge_config(
        cfg,
        cli_input=Path("custom.csv"),
        cli_output=Path("custom_out/"),
        cli_horizon=30,
        cli_n_folds=10,
        cli_models="ModelA,ModelB",
        cli_tollama_url="http://example.com",
        cli_tollama_models="chronos",
        cli_n_jobs=8,
        cli_no_cache=False,
        cli_cache_dir=None,
        cli_parallel_models=False,
        cli_include_optional_models=False,
        cli_include_neural_models=False,
        default_timeout=300.0,
    )
    assert m["input"] == Path("custom.csv")
    assert m["output"] == Path("custom_out/")
    assert m["horizon"] == 30
    assert m["n_folds"] == 10
    assert m["models"] == "ModelA,ModelB"
    assert m["tollama_url"] == "http://example.com"
    assert m["n_jobs"] == 8


def test_merge_config_cache_settings():
    """Cache settings merge correctly."""
    cfg = FileConfig(no_cache=True, cache_dir="/tmp/cache")
    m = _merge_config(
        cfg,
        cli_input=None,
        cli_output=Path("out/"),
        cli_horizon=14,
        cli_n_folds=3,
        cli_models=None,
        cli_tollama_url=None,
        cli_tollama_models=None,
        cli_n_jobs=1,
        cli_no_cache=False,
        cli_cache_dir=None,
        cli_parallel_models=False,
        cli_include_optional_models=False,
        cli_include_neural_models=False,
        default_timeout=300.0,
    )
    assert m["no_cache"] is True
    assert m["cache_dir"] == Path("/tmp/cache")


def test_merge_config_parallel_models():
    """Parallel models setting from file config."""
    cfg = FileConfig(parallel_models=True)
    m = _merge_config(
        cfg,
        cli_input=None,
        cli_output=Path("out/"),
        cli_horizon=14,
        cli_n_folds=3,
        cli_models=None,
        cli_tollama_url=None,
        cli_tollama_models=None,
        cli_n_jobs=1,
        cli_no_cache=False,
        cli_cache_dir=None,
        cli_parallel_models=False,
        cli_include_optional_models=False,
        cli_include_neural_models=False,
        default_timeout=300.0,
    )
    assert m["parallel_models"] is True


def test_merge_config_timeout_from_file():
    """Timeout from file config overrides default."""
    cfg = FileConfig(model_timeout_sec=60.0)
    m = _merge_config(
        cfg,
        cli_input=None,
        cli_output=Path("out/"),
        cli_horizon=14,
        cli_n_folds=3,
        cli_models=None,
        cli_tollama_url=None,
        cli_tollama_models=None,
        cli_n_jobs=1,
        cli_no_cache=False,
        cli_cache_dir=None,
        cli_parallel_models=False,
        cli_include_optional_models=False,
        cli_include_neural_models=False,
        default_timeout=300.0,
    )
    assert m["model_timeout_sec"] == 60.0


def test_merge_config_models_from_file():
    """Models list from file config when CLI is None."""
    cfg = FileConfig(models=["SeasonalNaive", "AutoETS"])
    m = _merge_config(
        cfg,
        cli_input=None,
        cli_output=Path("out/"),
        cli_horizon=14,
        cli_n_folds=3,
        cli_models=None,
        cli_tollama_url=None,
        cli_tollama_models=None,
        cli_n_jobs=1,
        cli_no_cache=False,
        cli_cache_dir=None,
        cli_parallel_models=False,
        cli_include_optional_models=False,
        cli_include_neural_models=False,
        default_timeout=300.0,
    )
    assert m["models"] == "SeasonalNaive,AutoETS"


def test_merge_config_optional_model_flags():
    """Optional model flags merge from config and CLI."""
    cfg = FileConfig(include_optional_models=True, include_neural_models=False)
    m = _merge_config(
        cfg,
        cli_input=None,
        cli_output=Path("out/"),
        cli_horizon=14,
        cli_n_folds=3,
        cli_models=None,
        cli_tollama_url=None,
        cli_tollama_models=None,
        cli_n_jobs=1,
        cli_no_cache=False,
        cli_cache_dir=None,
        cli_parallel_models=False,
        cli_include_optional_models=False,
        cli_include_neural_models=True,
        default_timeout=300.0,
    )
    assert m["include_optional_models"] is True
    assert m["include_neural_models"] is True
