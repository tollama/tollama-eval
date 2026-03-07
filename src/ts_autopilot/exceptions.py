"""Centralized exception hierarchy for ts-autopilot."""

from __future__ import annotations


class AutopilotError(Exception):
    """Base exception for all ts-autopilot errors."""


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
