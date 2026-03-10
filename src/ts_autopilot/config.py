"""YAML/JSON config file loading for tollama-eval."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ts_autopilot.exceptions import ConfigError


class FileConfig(BaseModel):
    """Parsed config from a YAML or JSON file."""

    model_config = ConfigDict(extra="forbid")

    input: str | None = None
    output: str | None = None
    horizon: int | None = Field(None, ge=1)
    n_folds: int | None = Field(None, ge=1)
    models: list[str] = Field(default_factory=list)
    tollama_url: str | None = None
    tollama_models: list[str] = Field(default_factory=list)
    n_jobs: int | None = Field(None, ge=1)
    max_retries: int | None = Field(None, ge=0)
    retry_backoff: float | None = Field(None, gt=0)
    report_title: str | None = None
    report_lang: str | None = None
    report_logo_url: str | None = None
    report_company: str | None = None
    report_confidential: bool = False
    model_timeout_sec: float | None = Field(None, gt=0)
    memory_limit_mb: int | None = Field(None, ge=1)
    allow_private_urls: bool = False
    parallel_models: bool = False
    cache_dir: str | None = None
    no_cache: bool = False

    @field_validator("models", "tollama_models", mode="before")
    @classmethod
    def _parse_csv_list(cls, v: object) -> object:
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v


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

    try:
        return FileConfig(**data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
