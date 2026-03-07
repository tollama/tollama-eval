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
    y_actual: list[float] = field(default_factory=list)
    ds_train_tail: list[str] = field(default_factory=list)
    y_train_tail: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Omit empty optional fields for backward compat
        for k in ("y_actual", "ds_train_tail", "y_train_tail"):
            if not d.get(k):
                d.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ForecastOutput:
        return cls(
            unique_id=d["unique_id"],
            ds=d["ds"],
            y_hat=d["y_hat"],
            model_name=d["model_name"],
            runtime_sec=d["runtime_sec"],
            y_actual=d.get("y_actual", []),
            ds_train_tail=d.get("ds_train_tail", []),
            y_train_tail=d.get("y_train_tail", []),
        )


@dataclass
class FoldResult:
    fold: int  # 1-indexed
    cutoff: str  # ISO 8601 string of last training date
    mase: float
    smape: float = 0.0
    rmsse: float = 0.0
    mae: float = 0.0
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
            smape=d.get("smape", 0.0),
            rmsse=d.get("rmsse", 0.0),
            mae=d.get("mae", 0.0),
            series_scores=d.get("series_scores", {}),
        )


@dataclass
class ModelResult:
    name: str
    runtime_sec: float
    folds: list[FoldResult]
    mean_mase: float
    std_mase: float
    mean_smape: float = 0.0
    mean_rmsse: float = 0.0
    mean_mae: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runtime_sec": self.runtime_sec,
            "folds": [f.to_dict() for f in self.folds],
            "mean_mase": self.mean_mase,
            "std_mase": self.std_mase,
            "mean_smape": self.mean_smape,
            "mean_rmsse": self.mean_rmsse,
            "mean_mae": self.mean_mae,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelResult:
        return cls(
            name=d["name"],
            runtime_sec=d["runtime_sec"],
            folds=[FoldResult.from_dict(f) for f in d["folds"]],
            mean_mase=d["mean_mase"],
            std_mase=d["std_mase"],
            mean_smape=d.get("mean_smape", 0.0),
            mean_rmsse=d.get("mean_rmsse", 0.0),
            mean_mae=d.get("mean_mae", 0.0),
        )


@dataclass
class LeaderboardEntry:
    rank: int
    name: str
    mean_mase: float
    mean_smape: float = 0.0
    mean_rmsse: float = 0.0
    mean_mae: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LeaderboardEntry:
        return cls(
            rank=d["rank"],
            name=d["name"],
            mean_mase=d["mean_mase"],
            mean_smape=d.get("mean_smape", 0.0),
            mean_rmsse=d.get("mean_rmsse", 0.0),
            mean_mae=d.get("mean_mae", 0.0),
        )


@dataclass
class ForecastData:
    """Captured forecast vs actual data for report visualizations."""

    model_name: str
    fold: int
    unique_id: list[str]
    ds: list[str]
    y_hat: list[float]
    y_actual: list[float]
    ds_train_tail: list[str]
    y_train_tail: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ForecastData:
        return cls(**d)


@dataclass
class DiagnosticsResult:
    """Residual diagnostics for a model."""

    model_name: str
    residual_mean: float
    residual_std: float
    residual_skew: float
    residual_kurtosis: float
    ljung_box_p: float  # p-value; > 0.05 = residuals are white noise
    histogram_bins: list[float] = field(default_factory=list)
    histogram_counts: list[int] = field(default_factory=list)
    acf_lags: list[int] = field(default_factory=list)
    acf_values: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    fitted: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiagnosticsResult:
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
    forecast_data: list[ForecastData] = field(default_factory=list)
    diagnostics: list[DiagnosticsResult] = field(default_factory=list)

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
        # forecast_data and diagnostics are NOT written to results.json
        # (they're for report rendering only, keeping schema frozen)
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
