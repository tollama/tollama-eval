"""Tests for custom exception hierarchy and retry config."""

from __future__ import annotations

import pytest

from ts_autopilot.exceptions import (
    AutopilotError,
    ConfigError,
    ModelFitError,
    SchemaError,
)


class TestExceptionHierarchy:
    def test_schema_error_is_autopilot_error(self):
        exc = SchemaError("bad schema")
        assert isinstance(exc, AutopilotError)
        assert isinstance(exc, ValueError)

    def test_config_error_is_autopilot_error(self):
        exc = ConfigError("bad config")
        assert isinstance(exc, AutopilotError)
        assert isinstance(exc, ValueError)

    def test_model_fit_error_is_autopilot_error(self):
        exc = ModelFitError("AutoETS", 3)
        assert isinstance(exc, AutopilotError)
        assert isinstance(exc, RuntimeError)

    def test_model_fit_error_attributes(self):
        cause = RuntimeError("convergence")
        exc = ModelFitError("AutoETS", 3, cause)
        assert exc.model_name == "AutoETS"
        assert exc.attempts == 3
        assert "AutoETS" in str(exc)
        assert "3 attempts" in str(exc)
        assert exc.__cause__ is cause

    def test_catch_all_autopilot_errors(self):
        """All custom exceptions can be caught with AutopilotError."""
        for exc_cls in (SchemaError, ConfigError, ModelFitError):
            with pytest.raises(AutopilotError):
                if exc_cls is ModelFitError:
                    raise exc_cls("model", 1)
                else:
                    raise exc_cls("test")


class TestSchemaErrorBackwardCompat:
    def test_import_from_loader(self):
        """SchemaError can still be imported from loader for backward compat."""
        from ts_autopilot.ingestion.loader import SchemaError as LoaderSchemaError

        assert LoaderSchemaError is SchemaError


class TestConfigErrorInConfig:
    def test_unknown_key_raises_config_error(self, tmp_path):
        from ts_autopilot.config import load_config

        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("bogus_key: true\n")
        with pytest.raises(ConfigError, match="Unknown config keys"):
            load_config(cfg_path)

    def test_unsupported_format_raises_config_error(self, tmp_path):
        from ts_autopilot.config import load_config

        cfg_path = tmp_path / "config.txt"
        cfg_path.write_text("input: data.csv\n")
        with pytest.raises(ConfigError, match="Unsupported config format"):
            load_config(cfg_path)


class TestRetryConfig:
    def test_retry_settings_in_yaml(self, tmp_path):
        from ts_autopilot.config import load_config

        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("max_retries: 5\nretry_backoff: 0.5\n")
        cfg = load_config(cfg_path)
        assert cfg.max_retries == 5
        assert cfg.retry_backoff == 0.5

    def test_retry_zero_retries_allowed(self, tmp_path):
        from ts_autopilot.config import load_config

        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("max_retries: 0\n")
        cfg = load_config(cfg_path)
        assert cfg.max_retries == 0

    def test_negative_retries_raises(self, tmp_path):
        from ts_autopilot.config import load_config

        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("max_retries: -1\n")
        with pytest.raises(ConfigError, match="non-negative integer"):
            load_config(cfg_path)

    def test_negative_backoff_raises(self, tmp_path):
        from ts_autopilot.config import load_config

        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("retry_backoff: -0.5\n")
        with pytest.raises(ConfigError, match="positive number"):
            load_config(cfg_path)


class TestModelFitErrorInPipeline:
    def test_retry_raises_model_fit_error(self):
        """_fit_predict_with_retry raises ModelFitError after exhausting retries."""
        from unittest.mock import MagicMock

        from ts_autopilot.pipeline import _fit_predict_with_retry

        runner = MagicMock()
        runner.name = "BadModel"
        runner.fit_predict.side_effect = RuntimeError("boom")

        import pandas as pd

        train = pd.DataFrame(
            {"unique_id": ["s1"], "ds": pd.Timestamp("2020-01-01"), "y": [1.0]}
        )

        with pytest.raises(ModelFitError) as exc_info:
            _fit_predict_with_retry(
                runner=runner,
                train=train,
                horizon=1,
                freq="D",
                season_length=7,
                max_retries=0,
                retry_backoff=0.01,
            )
        assert exc_info.value.model_name == "BadModel"
        assert exc_info.value.attempts == 1

    def test_custom_retry_count(self):
        """Custom max_retries is respected."""
        from unittest.mock import MagicMock

        from ts_autopilot.pipeline import _fit_predict_with_retry

        runner = MagicMock()
        runner.name = "BadModel"
        runner.fit_predict.side_effect = RuntimeError("boom")

        import pandas as pd

        train = pd.DataFrame(
            {"unique_id": ["s1"], "ds": pd.Timestamp("2020-01-01"), "y": [1.0]}
        )

        with pytest.raises(ModelFitError):
            _fit_predict_with_retry(
                runner=runner,
                train=train,
                horizon=1,
                freq="D",
                season_length=7,
                max_retries=3,
                retry_backoff=0.01,
            )
        # 1 initial + 3 retries = 4 total calls
        assert runner.fit_predict.call_count == 4
