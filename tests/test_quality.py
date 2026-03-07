"""Tests for enterprise quality improvements:
- Input validation (_validate_dataframe)
- Date gap detection (_detect_date_gaps)
- Atomic file writes (_atomic_write)
- Result metadata (ResultMetadata)
- Structured logging (setup_logging, get_logger)
- Retry logic (_fit_predict_with_retry)
- Distinct exit codes (ExitCode)
"""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from ts_autopilot.cli import ExitCode
from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    ResultMetadata,
)
from ts_autopilot.logging_config import get_logger, setup_logging
from ts_autopilot.pipeline import (
    _atomic_write,
    _detect_date_gaps,
    _validate_dataframe,
    run_from_csv,
)

# ---------------------------------------------------------------------------
# _validate_dataframe tests
# ---------------------------------------------------------------------------


def test_validate_empty_dataframe_raises():
    df = pd.DataFrame(columns=["unique_id", "ds", "y"])
    with pytest.raises(ValueError, match="empty"):
        _validate_dataframe(df)


def test_validate_nan_values_warns(tiny_long_df):
    df = tiny_long_df.copy()
    df.loc[0, "y"] = np.nan
    df.loc[1, "y"] = np.nan
    warnings = _validate_dataframe(df)
    assert any("NaN" in w for w in warnings)
    assert any("2" in w for w in warnings)


def test_validate_inf_values_raises(tiny_long_df):
    df = tiny_long_df.copy()
    df.loc[0, "y"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        _validate_dataframe(df)


def test_validate_neg_inf_raises(tiny_long_df):
    df = tiny_long_df.copy()
    df.loc[0, "y"] = -np.inf
    with pytest.raises(ValueError, match="infinite"):
        _validate_dataframe(df)


def test_validate_duplicate_timestamps_raises(tiny_long_df):
    df = tiny_long_df.copy()
    # Duplicate a row to create (unique_id, ds) collision
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _validate_dataframe(df)


def test_validate_negative_values_warns():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"unique_id": "s1", "ds": dates, "y": [-1.0, -2.0, 3.0, 4.0, 5.0]}
    )
    warnings = _validate_dataframe(df)
    assert any("negative" in w for w in warnings)


def test_validate_clean_data_no_warnings(tiny_long_df):
    warnings = _validate_dataframe(tiny_long_df)
    # tiny_long_df has no NaN, inf, duplicates, or negatives by default
    # It may have negatives from random data, so just check no NaN/inf/dup
    assert not any("NaN" in w for w in warnings)
    assert not any("infinite" in w for w in warnings)
    assert not any("duplicate" in w for w in warnings)


# ---------------------------------------------------------------------------
# _detect_date_gaps tests
# ---------------------------------------------------------------------------


def test_detect_date_gaps_with_gaps():
    # Create series with a gap (missing 2020-01-03)
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
    df = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": [1, 2, 4, 5]})
    warnings = _detect_date_gaps(df, "D")
    assert any("gap" in w.lower() for w in warnings)


def test_detect_date_gaps_no_gaps(tiny_long_df):
    warnings = _detect_date_gaps(tiny_long_df, "D")
    assert len(warnings) == 0


def test_detect_date_gaps_multiple_series():
    dates1 = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])
    dates2 = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    df = pd.concat(
        [
            pd.DataFrame({"unique_id": "s1", "ds": dates1, "y": [1, 2, 4]}),
            pd.DataFrame({"unique_id": "s2", "ds": dates2, "y": [1, 2, 3]}),
        ]
    )
    warnings = _detect_date_gaps(df, "D")
    assert any("1 series" in w for w in warnings)
    assert "s1" in warnings[0]


# ---------------------------------------------------------------------------
# _atomic_write tests
# ---------------------------------------------------------------------------


def test_atomic_write_creates_file(tmp_path):
    target = tmp_path / "output.json"
    _atomic_write(target, '{"key": "value"}')
    assert target.exists()
    assert json.loads(target.read_text()) == {"key": "value"}


def test_atomic_write_overwrites_existing(tmp_path):
    target = tmp_path / "output.json"
    target.write_text("old content")
    _atomic_write(target, "new content")
    assert target.read_text() == "new content"


def test_atomic_write_no_leftover_tmp(tmp_path):
    target = tmp_path / "output.json"
    _atomic_write(target, "content")
    # No .tmp files should remain
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


# ---------------------------------------------------------------------------
# ResultMetadata tests
# ---------------------------------------------------------------------------


def test_result_metadata_create_now():
    meta = ResultMetadata.create_now()
    assert meta.version == "0.2.0"
    assert "T" in meta.generated_at  # ISO format
    assert meta.total_runtime_sec == 0.0


def test_result_metadata_round_trip():
    meta = ResultMetadata(
        version="0.2.0",
        generated_at="2024-01-01T00:00:00+00:00",
        total_runtime_sec=1.234,
    )
    assert ResultMetadata.from_dict(meta.to_dict()) == meta


def test_benchmark_result_with_metadata_round_trip():
    result = BenchmarkResult(
        profile=DataProfile(
            n_series=2,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=60,
            max_length=60,
            total_rows=120,
        ),
        config=BenchmarkConfig(horizon=7, n_folds=2),
        models=[],
        leaderboard=[],
        metadata=ResultMetadata(
            version="0.2.0",
            generated_at="2024-01-01T00:00:00+00:00",
            total_runtime_sec=5.0,
        ),
    )
    d = result.to_dict()
    assert "metadata" in d
    assert d["metadata"]["version"] == "0.2.0"
    restored = BenchmarkResult.from_dict(d)
    assert restored.metadata.version == "0.2.0"
    assert restored.metadata.total_runtime_sec == 5.0


def test_benchmark_result_without_metadata():
    result = BenchmarkResult(
        profile=DataProfile(
            n_series=2,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=60,
            max_length=60,
            total_rows=120,
        ),
        config=BenchmarkConfig(horizon=7, n_folds=2),
        models=[],
        leaderboard=[],
    )
    d = result.to_dict()
    assert "metadata" not in d
    restored = BenchmarkResult.from_dict(d)
    assert restored.metadata is None


def test_run_from_csv_includes_metadata(tmp_path):
    """Integration: results.json should contain metadata after run_from_csv."""
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": str(d.date()), "y": float(i)})
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    run_from_csv(csv_path, horizon=7, n_folds=2, output_dir=out_dir)

    data = json.loads((out_dir / "results.json").read_text())
    assert "metadata" in data
    assert data["metadata"]["version"] == "0.2.0"
    assert data["metadata"]["total_runtime_sec"] > 0


# ---------------------------------------------------------------------------
# Structured logging tests
# ---------------------------------------------------------------------------


def test_setup_logging_default():
    setup_logging()
    logger = get_logger("test")
    assert logger.parent.level == logging.INFO


def test_setup_logging_verbose():
    setup_logging(verbose=True)
    logger = get_logger("test")
    assert logger.parent.level == logging.DEBUG


def test_setup_logging_quiet():
    setup_logging(quiet=True)
    logger = get_logger("test")
    assert logger.parent.level == logging.WARNING


def test_get_logger_namespace():
    logger = get_logger("mymodule")
    assert logger.name == "ts_autopilot.mymodule"


# ---------------------------------------------------------------------------
# ExitCode tests
# ---------------------------------------------------------------------------


def test_exit_codes_are_distinct():
    codes = [
        ExitCode.SUCCESS,
        ExitCode.SCHEMA_ERROR,
        ExitCode.DATA_ERROR,
        ExitCode.UNEXPECTED_ERROR,
    ]
    assert len(set(codes)) == 4


def test_exit_code_success_is_zero():
    assert ExitCode.SUCCESS == 0


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


def test_retry_on_runtime_error(tiny_long_df):
    """_fit_predict_with_retry retries on RuntimeError."""
    from unittest.mock import MagicMock

    from ts_autopilot.contracts import ForecastOutput
    from ts_autopilot.pipeline import _fit_predict_with_retry
    from ts_autopilot.runners.base import BaseRunner

    mock_runner = MagicMock(spec=BaseRunner)
    mock_runner.name = "MockModel"

    good_output = ForecastOutput(
        unique_id=["s1"],
        ds=["2020-01-01"],
        y_hat=[1.0],
        model_name="MockModel",
        runtime_sec=0.01,
    )

    # Fail first, succeed second
    mock_runner.fit_predict.side_effect = [RuntimeError("convergence"), good_output]

    result = _fit_predict_with_retry(
        runner=mock_runner,
        train=tiny_long_df,
        horizon=7,
        freq="D",
        season_length=7,
        retry_backoff=0.01,
    )

    assert result == good_output
    assert mock_runner.fit_predict.call_count == 2


def test_retry_exhausted_raises():
    """_fit_predict_with_retry raises after all retries exhausted."""
    from unittest.mock import MagicMock

    from ts_autopilot.pipeline import _fit_predict_with_retry
    from ts_autopilot.runners.base import BaseRunner

    mock_runner = MagicMock(spec=BaseRunner)
    mock_runner.name = "MockModel"
    mock_runner.fit_predict.side_effect = RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="failed after"):
        _fit_predict_with_retry(
            runner=mock_runner,
            train=pd.DataFrame(),
            horizon=7,
            freq="D",
            season_length=7,
            retry_backoff=0.01,
        )


def test_retry_does_not_catch_value_error():
    """ValueError is raised immediately without retry."""
    from unittest.mock import MagicMock

    from ts_autopilot.pipeline import _fit_predict_with_retry
    from ts_autopilot.runners.base import BaseRunner

    mock_runner = MagicMock(spec=BaseRunner)
    mock_runner.name = "MockModel"
    mock_runner.fit_predict.side_effect = ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        _fit_predict_with_retry(
            runner=mock_runner,
            train=pd.DataFrame(),
            horizon=7,
            freq="D",
            season_length=7,
        )
    assert mock_runner.fit_predict.call_count == 1


def test_retry_catches_linalg_error():
    """LinAlgError should be retried like RuntimeError."""
    from unittest.mock import MagicMock

    from ts_autopilot.contracts import ForecastOutput
    from ts_autopilot.pipeline import _fit_predict_with_retry
    from ts_autopilot.runners.base import BaseRunner

    mock_runner = MagicMock(spec=BaseRunner)
    mock_runner.name = "MockModel"

    good_output = ForecastOutput(
        unique_id=["s1"],
        ds=["2020-01-01"],
        y_hat=[1.0],
        model_name="MockModel",
        runtime_sec=0.01,
    )
    mock_runner.fit_predict.side_effect = [
        np.linalg.LinAlgError("singular matrix"),
        good_output,
    ]

    result = _fit_predict_with_retry(
        runner=mock_runner,
        train=pd.DataFrame(),
        horizon=7,
        freq="D",
        season_length=7,
        retry_backoff=0.01,
    )
    assert result == good_output
    assert mock_runner.fit_predict.call_count == 2


# ---------------------------------------------------------------------------
# CLI exit code integration tests
# ---------------------------------------------------------------------------


def test_cli_schema_error_exit_code(tmp_path):
    from typer.testing import CliRunner

    from ts_autopilot.cli import app

    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\nabc,def\n")
    result = CliRunner().invoke(
        app,
        ["run", "--input", str(bad), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == ExitCode.SCHEMA_ERROR


def test_cli_data_error_exit_code(tmp_path):
    """Series too short for requested config → DATA_ERROR."""
    from typer.testing import CliRunner

    from ts_autopilot.cli import app

    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": range(5)})
    csv_path = tmp_path / "short.csv"
    df.to_csv(csv_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(csv_path),
            "--horizon",
            "7",
            "--n-folds",
            "3",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == ExitCode.DATA_ERROR
