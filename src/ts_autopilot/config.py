"""YAML/JSON config file loading for ts-autopilot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_VALID_KEYS = {
    "input", "output", "horizon", "n_folds",
    "models", "tollama_url", "n_jobs",
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
    n_jobs: int | None = None


def load_config(path: str | Path) -> FileConfig:
    """Load and validate a YAML or JSON config file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid keys or values.
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
        raise ValueError(
            f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json."
        )

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping (key-value pairs).")

    unknown = set(data.keys()) - _VALID_KEYS
    if unknown:
        raise ValueError(
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
            raise ValueError(f"horizon must be a positive integer, got {val!r}")
        config.horizon = val

    if "n_folds" in data:
        val = data["n_folds"]
        if not isinstance(val, int) or val < 1:
            raise ValueError(f"n_folds must be a positive integer, got {val!r}")
        config.n_folds = val

    if "models" in data:
        val = data["models"]
        if isinstance(val, str):
            config.models = [m.strip() for m in val.split(",") if m.strip()]
        elif isinstance(val, list):
            config.models = [str(m) for m in val]
        else:
            raise ValueError(
                "models must be a list or comma-separated "
                f"string, got {type(val).__name__}"
            )

    if "tollama_url" in data:
        config.tollama_url = str(data["tollama_url"])

    if "n_jobs" in data:
        val = data["n_jobs"]
        if not isinstance(val, int) or val < 1:
            raise ValueError(
                f"n_jobs must be a positive integer, got {val!r}"
            )
        config.n_jobs = val

    return config
