"""Data versioning and lineage tracking.

Computes a full SHA-256 hash of input data and records
provenance information for reproducibility.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

import ts_autopilot


def compute_full_data_hash(df: pd.DataFrame) -> str:
    """Compute a full SHA-256 hash of the entire DataFrame."""
    csv_bytes = df.to_csv(index=False).encode()
    return hashlib.sha256(csv_bytes).hexdigest()


def compute_file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LineageRecord:
    """Provenance record for a benchmark run."""

    input_data_hash: str = ""
    input_file_name: str = ""
    input_row_count: int = 0
    input_series_count: int = 0
    pipeline_version: str = ""
    python_version: str = ""
    platform: str = ""
    dependency_versions: dict[str, str] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageRecord:
        return cls(**{
            k: v
            for k, v in data.items()
            if k in cls.__dataclass_fields__
        })


def build_lineage(
    df: pd.DataFrame,
    *,
    input_file_name: str = "",
) -> LineageRecord:
    """Build a lineage record from input data and environment."""
    dep_versions: dict[str, str] = {}
    for pkg in [
        "pandas",
        "numpy",
        "statsforecast",
        "pydantic",
        "httpx",
    ]:
        try:
            mod = __import__(pkg)
            dep_versions[pkg] = getattr(
                mod, "__version__", "unknown"
            )
        except ImportError:
            pass

    n_series = 0
    if "unique_id" in df.columns:
        n_series = df["unique_id"].nunique()

    return LineageRecord(
        input_data_hash=compute_full_data_hash(df),
        input_file_name=input_file_name,
        input_row_count=len(df),
        input_series_count=n_series,
        pipeline_version=ts_autopilot.__version__,
        python_version=sys.version,
        platform=platform.platform(),
        dependency_versions=dep_versions,
    )
