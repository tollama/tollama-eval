"""Centralized exception hierarchy for tollama-eval."""

from __future__ import annotations


class AutopilotError(Exception):
    """Base exception for all tollama-eval errors."""


class SchemaError(AutopilotError, ValueError):
    """Raised when a CSV cannot be coerced to canonical long format."""


class ConfigError(AutopilotError, ValueError):
    """Raised for invalid configuration (file or CLI)."""


class ModelFitError(AutopilotError, RuntimeError):
    """Raised when a model fails to fit/predict after all retries."""

    def __init__(self, model_name: str, attempts: int, cause: Exception | None = None):
        self.model_name = model_name
        self.attempts = attempts
        super().__init__(f"Model {model_name} failed after {attempts} attempts")
        if cause is not None:
            self.__cause__ = cause


class ModelTimeoutError(AutopilotError, TimeoutError):
    """Raised when a model exceeds the per-fold timeout."""

    def __init__(self, model_name: str, timeout_sec: float):
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        super().__init__(f"Model {model_name} timed out after {timeout_sec:.0f}s")


class URLValidationError(ConfigError):
    """Raised when a URL fails security validation (e.g. SSRF prevention)."""
