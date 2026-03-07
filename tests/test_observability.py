"""Tests for observability improvements:
- JSON structured logging
- details.json serialization
"""

import json
import logging

import pytest

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    DiagnosticsResult,
    FoldResult,
    ForecastData,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.logging_config import get_logger, setup_logging

# -- JSON logging tests --


class TestJsonLogging:
    def test_json_format_emits_valid_json(self, capfd):
        setup_logging(log_json=True)
        logger = get_logger("test_json")
        logger.warning("hello %s", "world")

        handler = logging.getLogger("ts_autopilot").handlers[0]
        # Flush handler
        handler.flush()

        # Read from the handler's stream directly
        record = logging.LogRecord(
            name="ts_autopilot.test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="test message %d",
            args=(42,),
            exc_info=None,
        )
        formatted = handler.formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["level"] == "WARNING"
        assert parsed["msg"] == "test message 42"
        assert "ts" in parsed
        assert "logger" in parsed

    def test_json_format_includes_exception(self):
        setup_logging(log_json=True)
        handler = logging.getLogger("ts_autopilot").handlers[0]

        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="ts_autopilot.test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )

        formatted = handler.formatter.format(record)
        parsed = json.loads(formatted)
        assert "exception" in parsed
        assert "boom" in parsed["exception"]

    def test_plain_format_still_works(self):
        setup_logging(log_json=False)
        handler = logging.getLogger("ts_autopilot").handlers[0]
        record = logging.LogRecord(
            name="ts_autopilot.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="plain message",
            args=(),
            exc_info=None,
        )
        formatted = handler.formatter.format(record)
        # Plain format should NOT be JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(formatted)
        assert "plain message" in formatted


# -- details.json serialization tests --


def _make_result_with_details():
    return BenchmarkResult(
        profile=DataProfile(
            n_series=1,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=60,
            max_length=60,
            total_rows=60,
        ),
        config=BenchmarkConfig(horizon=7, n_folds=2),
        models=[
            ModelResult(
                name="AutoETS",
                runtime_sec=1.0,
                folds=[
                    FoldResult(fold=1, cutoff="2020-06-01", mase=0.9),
                ],
                mean_mase=0.9,
                std_mase=0.0,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.9),
        ],
        forecast_data=[
            ForecastData(
                model_name="AutoETS",
                fold=1,
                unique_id=["s1", "s1"],
                ds=["2020-06-02", "2020-06-03"],
                y_hat=[10.0, 11.0],
                y_actual=[10.5, 11.2],
                ds_train_tail=["2020-05-30", "2020-05-31"],
                y_train_tail=[9.0, 9.5],
            ),
        ],
        diagnostics=[
            DiagnosticsResult(
                model_name="AutoETS",
                residual_mean=0.1,
                residual_std=0.5,
                residual_skew=0.2,
                residual_kurtosis=3.0,
                ljung_box_p=0.45,
                histogram_bins=[0.0, 0.5, 1.0],
                histogram_counts=[5, 10],
                acf_lags=[1, 2, 3],
                acf_values=[0.8, 0.5, 0.2],
                residuals=[0.5, -0.2, 0.1],
                fitted=[10.0, 11.0, 12.0],
            ),
        ],
    )


class TestDetailsJson:
    def test_to_details_dict_has_both_keys(self):
        result = _make_result_with_details()
        details = result.to_details_dict()
        assert "forecast_data" in details
        assert "diagnostics" in details
        assert len(details["forecast_data"]) == 1
        assert len(details["diagnostics"]) == 1

    def test_to_details_json_is_valid(self):
        result = _make_result_with_details()
        raw = result.to_details_json()
        parsed = json.loads(raw)
        assert parsed["forecast_data"][0]["model_name"] == "AutoETS"
        assert parsed["diagnostics"][0]["ljung_box_p"] == 0.45

    def test_details_dict_empty_when_no_data(self):
        result = BenchmarkResult(
            profile=DataProfile(
                n_series=1,
                frequency="D",
                missing_ratio=0.0,
                season_length_guess=7,
                min_length=60,
                max_length=60,
                total_rows=60,
            ),
            config=BenchmarkConfig(horizon=7, n_folds=2),
            models=[],
            leaderboard=[],
        )
        details = result.to_details_dict()
        assert details == {}

    def test_roundtrip_via_from_dict(self):
        """details fields survive a dict roundtrip through from_dict."""
        result = _make_result_with_details()
        # Combine results.json + details.json as one dict
        combined = result.to_dict()
        combined.update(result.to_details_dict())
        restored = BenchmarkResult.from_dict(combined)
        assert len(restored.forecast_data) == 1
        assert restored.forecast_data[0].model_name == "AutoETS"
        assert len(restored.diagnostics) == 1
        assert restored.diagnostics[0].ljung_box_p == 0.45

    def test_results_json_unchanged(self):
        """results.json must NOT contain forecast_data or diagnostics."""
        result = _make_result_with_details()
        results_dict = result.to_dict()
        assert "forecast_data" not in results_dict
        assert "diagnostics" not in results_dict
