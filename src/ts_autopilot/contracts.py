"""Core data contracts for ts-autopilot.

All modules import shared types from here. Do not rename fields —
the results.json schema is frozen.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DataProfile:
    n_series: int
    frequency: str  # pandas freq string: 'D', 'W', 'ME', 'h', etc.
    missing_ratio: float  # 0.0–1.0
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FoldResult:
        return cls(**d)


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
class BenchmarkResult:
    profile: DataProfile
    config: BenchmarkConfig
    models: list[ModelResult]
    leaderboard: list[LeaderboardEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "config": self.config.to_dict(),
            "models": [m.to_dict() for m in self.models],
            "leaderboard": [e.to_dict() for e in self.leaderboard],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkResult:
        return cls(
            profile=DataProfile.from_dict(d["profile"]),
            config=BenchmarkConfig.from_dict(d["config"]),
            models=[ModelResult.from_dict(m) for m in d["models"]],
            leaderboard=[LeaderboardEntry.from_dict(e) for e in d["leaderboard"]],
        )

    @classmethod
    def from_json(cls, s: str) -> BenchmarkResult:
        return cls.from_dict(json.loads(s))
