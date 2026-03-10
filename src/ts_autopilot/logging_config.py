"""Structured logging configuration for tollama-eval."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Thread-local storage for run_id injection
_current_run_id: str = ""


def set_run_id(run_id: str) -> None:
    """Set the current run_id for log correlation."""
    global _current_run_id
    _current_run_id = run_id


class _RunIdFilter(logging.Filter):
    """Inject run_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _current_run_id  # type: ignore[attr-defined]
        return True


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        run_id = getattr(record, "run_id", "")
        if run_id:
            entry["run_id"] = run_id
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    log_json: bool = False,
) -> None:
    """Configure root logger for tollama-eval.

    Args:
        verbose: Set DEBUG level and show detailed output.
        quiet: Set WARNING level, suppressing info messages.
        log_json: Emit structured JSON logs (one object per line).
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    if log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))

    root = logging.getLogger("ts_autopilot")
    root.handlers.clear()
    root.filters.clear()
    root.addHandler(handler)
    root.addFilter(_RunIdFilter())
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ts_autopilot namespace."""
    return logging.getLogger(f"ts_autopilot.{name}")
