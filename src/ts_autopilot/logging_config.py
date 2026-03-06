"""Structured logging configuration for ts-autopilot."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Configure root logger for ts-autopilot.

    Args:
        verbose: Set DEBUG level and show detailed output.
        quiet: Set WARNING level, suppressing info messages.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))

    root = logging.getLogger("ts_autopilot")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ts_autopilot namespace."""
    return logging.getLogger(f"ts_autopilot.{name}")
