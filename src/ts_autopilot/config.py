"""YAML/JSON config file loading for ts-autopilot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ts_autopilot.exceptions import ConfigError

_VALID_KEYS = {
    "input",
    "output",
    "horizon",
    "n_folds",
    "models",
    "tollama_url",
    "tollama_models",
    "n_jobs",
    "max_retries",
    "retry_backoff",
    "report_title",
    "report_lang",
    "model_timeout_sec",
    "memory_limit_mb",
    "allow_private_urls",
    "parallel_models",
    "cache_dir",
    "no_cache",
}


@dataclass
class FileConfig:
    """Parsed config from a YAML or JSON file."""

    input: str | None = None
    output: str | None = None
    horizon: int | None = None
    n_folds: int | None = None
    models: list[str] = field(default_factory=list)
    tollama_url: str | None = None
    tollama_models: list[str] = field(default_factory=list)
    n_jobs: int | None = None
    max_retries: int | None = None
    retry_backoff: float | None = None
    report_title: str | None = None
    report_lang: str | None = None
    model_timeout_sec: float | None = None
    memory_limit_mb: int | None = None
    allow_private_urls: bool = False
    parallel_models: bool = False
    cache_dir: str | None = None
    no_cache: bool = False


def load_config(path: str | Path) -> FileConfig:
    """Load and validate a YAML or JSON config file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If the file contains invalid keys or values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text()
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(
            f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json."
        )

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a mapping (key-value pairs).")

    unknown = set(data.keys()) - _VALID_KEYS
    if unknown:
        raise ConfigError(
            f"Unknown config keys: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(_VALID_KEYS))}"
        )

    config = FileConfig()

    if "input" in data:
        config.input = str(data["input"])

    if "output" in data:
        config.output = str(data["output"])

    if "horizon" in data:
        val = data["horizon"]
        if not isinstance(val, int) or val < 1:
            raise ConfigError(f"horizon must be a positive integer, got {val!r}")
        config.horizon = val

    if "n_folds" in data:
        val = data["n_folds"]
        if not isinstance(val, int) or val < 1:
            raise ConfigError(f"n_folds must be a positive integer, got {val!r}")
        config.n_folds = val

    if "models" in data:
        val = data["models"]
        if isinstance(val, str):
            config.models = [m.strip() for m in val.split(",") if m.strip()]
        elif isinstance(val, list):
            config.models = [str(m) for m in val]
        else:
            raise ConfigError(
                "models must be a list or comma-separated "
                f"string, got {type(val).__name__}"
            )

    if "tollama_url" in data:
        config.tollama_url = str(data["tollama_url"])

    if "tollama_models" in data:
        val = data["tollama_models"]
        if isinstance(val, str):
            config.tollama_models = [m.strip() for m in val.split(",") if m.strip()]
        elif isinstance(val, list):
            config.tollama_models = [str(m) for m in val]
        else:
            raise ConfigError(
                "tollama_models must be a list or comma-separated "
                f"string, got {type(val).__name__}"
            )

    if "n_jobs" in data:
        val = data["n_jobs"]
        if not isinstance(val, int) or val < 1:
            raise ConfigError(f"n_jobs must be a positive integer, got {val!r}")
        config.n_jobs = val

    if "max_retries" in data:
        val = data["max_retries"]
        if not isinstance(val, int) or val < 0:
            raise ConfigError(
                f"max_retries must be a non-negative integer, got {val!r}"
            )
        config.max_retries = val

    if "retry_backoff" in data:
        val = data["retry_backoff"]
        if not isinstance(val, int | float) or val <= 0:
            raise ConfigError(f"retry_backoff must be a positive number, got {val!r}")
        config.retry_backoff = float(val)

    if "report_title" in data:
        config.report_title = str(data["report_title"])

    if "report_lang" in data:
        config.report_lang = str(data["report_lang"])

    if "model_timeout_sec" in data:
        val = data["model_timeout_sec"]
        if not isinstance(val, int | float) or val <= 0:
            raise ConfigError(
                f"model_timeout_sec must be a positive number, got {val!r}"
            )
        config.model_timeout_sec = float(val)

    if "memory_limit_mb" in data:
        val = data["memory_limit_mb"]
        if not isinstance(val, int) or val < 1:
            raise ConfigError(
                f"memory_limit_mb must be a positive integer, got {val!r}"
            )
        config.memory_limit_mb = val

    if "allow_private_urls" in data:
        config.allow_private_urls = bool(data["allow_private_urls"])

    if "parallel_models" in data:
        config.parallel_models = bool(data["parallel_models"])

    if "cache_dir" in data:
        config.cache_dir = str(data["cache_dir"])

    if "no_cache" in data:
        config.no_cache = bool(data["no_cache"])

    return config
