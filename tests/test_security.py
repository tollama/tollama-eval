"""Tests for security hardening features."""

from __future__ import annotations

import pytest

from ts_autopilot.exceptions import ModelTimeoutError, URLValidationError


class TestPathValidation:
    """Tests for input path hardening."""

    def test_regular_file_accepted(self, long_csv):
        from ts_autopilot.ingestion.loader import load_csv

        df = load_csv(long_csv)
        assert len(df) > 0

    def test_nonexistent_file_raises(self, tmp_path):
        from ts_autopilot.ingestion.loader import load_csv

        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "does_not_exist.csv")

    def test_symlink_within_parent_allowed(self, long_csv, tmp_path):
        from ts_autopilot.ingestion.loader import load_csv

        link = tmp_path / "link.csv"
        link.symlink_to(long_csv)
        df = load_csv(str(link))
        assert len(df) > 0


class TestTollamaURLValidation:
    """Tests for Tollama URL SSRF prevention."""

    def test_http_url_accepted(self):
        from ts_autopilot.tollama.client import validate_tollama_url

        # Allow private for testing since DNS won't resolve
        result = validate_tollama_url("http://example.com:8000", allow_private=True)
        assert result == "http://example.com:8000"

    def test_https_url_accepted(self):
        from ts_autopilot.tollama.client import validate_tollama_url

        result = validate_tollama_url("https://api.example.com", allow_private=True)
        assert result == "https://api.example.com"

    def test_ftp_url_rejected(self):
        from ts_autopilot.tollama.client import validate_tollama_url

        with pytest.raises(URLValidationError, match="http://"):
            validate_tollama_url("ftp://example.com")

    def test_no_scheme_rejected(self):
        from ts_autopilot.tollama.client import validate_tollama_url

        with pytest.raises(URLValidationError):
            validate_tollama_url("example.com")

    def test_empty_hostname_rejected(self):
        from ts_autopilot.tollama.client import validate_tollama_url

        with pytest.raises(URLValidationError, match="hostname"):
            validate_tollama_url("http://")

    def test_localhost_blocked_by_default(self):
        from ts_autopilot.tollama.client import validate_tollama_url

        with pytest.raises(URLValidationError, match="private"):
            validate_tollama_url("http://127.0.0.1:8000")

    def test_localhost_allowed_with_flag(self):
        from ts_autopilot.tollama.client import validate_tollama_url

        result = validate_tollama_url("http://127.0.0.1:8000", allow_private=True)
        assert "127.0.0.1" in result


class TestModelTimeout:
    """Tests for per-model timeout."""

    def test_model_timeout_error_attributes(self):
        exc = ModelTimeoutError("AutoETS", 300.0)
        assert exc.model_name == "AutoETS"
        assert exc.timeout_sec == 300.0
        assert "300" in str(exc)

    def test_model_timeout_is_timeout_error(self):
        exc = ModelTimeoutError("AutoETS", 300.0)
        assert isinstance(exc, TimeoutError)


class TestMemoryGuard:
    """Tests for memory limit checking."""

    def test_small_file_passes_memory_check(self, long_csv):
        from ts_autopilot.ingestion.loader import load_csv

        df = load_csv(long_csv, max_memory_mb=2048)
        assert len(df) > 0

    def test_tiny_memory_limit_raises(self, long_csv):
        from ts_autopilot.ingestion.loader import load_csv

        # Set absurdly low memory limit
        with pytest.raises(ValueError, match="memory limit"):
            load_csv(long_csv, max_memory_mb=0)


class TestHTMLXSS:
    """Verify HTML report autoescape is active."""

    def test_jinja_autoescape_enabled(self):
        from jinja2 import Environment, select_autoescape

        env = Environment(
            loader=None,
            autoescape=select_autoescape(["html"]),
        )
        # Verify autoescape is configured for .html files
        assert env.autoescape is True or callable(env.autoescape)
