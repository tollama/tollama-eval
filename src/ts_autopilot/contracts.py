"""Core data contracts for ts-autopilot.

All modules import shared types from here. Do not rename fields —
the results.json schema is frozen.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ts_autopilot import __version__


@dataclass
class DataProfile:
    n_series: int
    frequency: str  # pandas freq string: 'D', 'W', 'ME', 'h', etc.
    missing_ratio: float  # 0.0-1.0
    season_length_guess: int
    min_length: int
    max_length: int
    total_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataProfile:
        return cls(**d)


@dataclass
class ForecastOutput:
    """Single model's predictions for all series across one fold."""

    unique_id: list[str]
    ds: list[str]  # ISO 8601 strings for JSON safety
    y_hat: list[float]
    model_name: str
    runtime_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ForecastOutput:
        return cls(**d)


@dataclass
class FoldResult:
    fold: int  # 1-indexed
    cutoff: str  # ISO 8601 string of last training date
    mase: float
    series_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not self.series_scores:
            d.pop("series_scores", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FoldResult:
        return cls(
            fold=d["fold"],
            cutoff=d["cutoff"],
            mase=d["mase"],
            series_scores=d.get("series_scores", {}),
        )


@dataclass
class ModelResult:
    name: str
    runtime_sec: float
    folds: list[FoldResult]
    mean_mase: float
    std_mase: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runtime_sec": self.runtime_sec,
            "folds": [f.to_dict() for f in self.folds],
            "mean_mase": self.mean_mase,
            "std_mase": self.std_mase,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelResult:
        d = dict(d)
        d["folds"] = [FoldResult.from_dict(f) for f in d["folds"]]
        return cls(**d)


@dataclass
class LeaderboardEntry:
    rank: int
    name: str
    mean_mase: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LeaderboardEntry:
        return cls(**d)


@dataclass
class BenchmarkConfig:
    horizon: int
    n_folds: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkConfig:
        return cls(**d)


@dataclass
class ResultMetadata:
    """Run metadata attached to every results.json output."""

    version: str
    generated_at: str  # ISO 8601 UTC timestamp
    total_runtime_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResultMetadata:
        return cls(**d)

    @classmethod
    def create_now(cls) -> ResultMetadata:
        return cls(
            version=__version__,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
        )


@dataclass
class BenchmarkResult:
    profile: DataProfile
    config: BenchmarkConfig
    models: list[ModelResult]
    leaderboard: list[LeaderboardEntry]
    warnings: list[str] = field(default_factory=list)
    metadata: ResultMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.metadata is not None:
            d["metadata"] = self.metadata.to_dict()
        d["profile"] = self.profile.to_dict()
        d["config"] = self.config.to_dict()
        d["models"] = [m.to_dict() for m in self.models]
        d["leaderboard"] = [e.to_dict() for e in self.leaderboard]
        if self.warnings:
            d["warnings"] = self.warnings
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkResult:
        metadata = None
        if "metadata" in d:
            metadata = ResultMetadata.from_dict(d["metadata"])
        return cls(
            profile=DataProfile.from_dict(d["profile"]),
            config=BenchmarkConfig.from_dict(d["config"]),
            models=[ModelResult.from_dict(m) for m in d["models"]],
            leaderboard=[LeaderboardEntry.from_dict(e) for e in d["leaderboard"]],
            warnings=d.get("warnings", []),
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, s: str) -> BenchmarkResult:
        return cls.from_dict(json.loads(s))
