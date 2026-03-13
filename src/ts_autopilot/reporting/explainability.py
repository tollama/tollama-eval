"""Machine-readable evaluation explanations for v3.8 trust-layer packaging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ts_autopilot.contracts import BenchmarkResult
from ts_autopilot.evaluation.stability import compute_stability
from ts_autopilot.reporting.executive_summary import generate_executive_summary


@dataclass
class EvalEvidenceItem:
    label: str
    value: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateExplanation:
    model: str
    rank: int
    mean_mase: float
    runtime_sec: float
    stability_score: float | None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSelectionExplanation:
    winner: str
    why_this_model: str
    selection_rationale: list[str] = field(default_factory=list)
    top_candidates: list[CandidateExplanation] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    evidence: list[EvalEvidenceItem] = field(default_factory=list)
    executive_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "why_this_model": self.why_this_model,
            "selection_rationale": self.selection_rationale,
            "top_candidates": [item.to_dict() for item in self.top_candidates],
            "risk_flags": self.risk_flags,
            "evidence": [item.to_dict() for item in self.evidence],
            "executive_summary": self.executive_summary,
        }


def build_model_selection_explanation(result: BenchmarkResult) -> ModelSelectionExplanation:
    if not result.leaderboard:
        return ModelSelectionExplanation(
            winner="",
            why_this_model="no leaderboard entries were available",
            risk_flags=["NO_VALID_MODELS"],
            executive_summary={"overview": "No models produced valid benchmark results."},
        )

    stability_report = compute_stability(result)
    stability_by_model = {item.model_name: item for item in stability_report.scores}
    model_by_name = {model.name: model for model in result.models}
    summary = generate_executive_summary(result)

    winner_entry = result.leaderboard[0]
    winner_model = model_by_name.get(winner_entry.name)
    winner_stability = stability_by_model.get(winner_entry.name)

    runner_up_gap = None
    if len(result.leaderboard) > 1:
        runner_up_gap = float(result.leaderboard[1].mean_mase) - float(winner_entry.mean_mase)

    rationale = [f"lowest mean_mase on the frozen leaderboard ({winner_entry.mean_mase:.4f})"]
    if runner_up_gap is not None:
        rationale.append(f"margin to runner-up={runner_up_gap:.4f}")
    if winner_model is not None:
        rationale.append(f"runtime_sec={winner_model.runtime_sec:.3f}")
        rationale.append(f"std_mase={winner_model.std_mase:.4f}")
    if winner_stability is not None:
        rationale.append(
            "stability_score="
            f"{winner_stability.stability_score:.4f} "
            f"(rank_consistency={winner_stability.rank_consistency:.4f})"
        )

    risk_flags: list[str] = []
    if float(winner_entry.mean_mase) >= 1.0:
        risk_flags.append("WINNER_NOT_BETTER_THAN_NAIVE")
    if winner_model is not None and winner_model.mean_mase > 0 and winner_model.std_mase / winner_model.mean_mase > 0.25:
        risk_flags.append("HIGH_FOLD_VARIANCE")
    if winner_stability is not None and winner_stability.rank_consistency < 0.5:
        risk_flags.append("LOW_RANK_CONSISTENCY")

    evidence: list[EvalEvidenceItem] = [
        EvalEvidenceItem(
            label="dataset_profile",
            value={
                "n_series": result.profile.n_series,
                "total_rows": result.profile.total_rows,
                "frequency": result.profile.frequency,
            },
            detail="input dataset coverage used during benchmark",
        ),
        EvalEvidenceItem(
            label="benchmark_config",
            value={
                "horizon": result.config.horizon,
                "n_folds": result.config.n_folds,
            },
            detail="benchmark configuration used to produce leaderboard",
        ),
    ]
    if winner_model is not None:
        evidence.append(
            EvalEvidenceItem(
                label="winner_runtime",
                value=winner_model.runtime_sec,
                detail=f"runtime_sec={winner_model.runtime_sec:.3f}",
            )
        )
    if winner_stability is not None:
        evidence.append(
            EvalEvidenceItem(
                label="winner_stability",
                value={
                    "fold_cv": winner_stability.fold_cv,
                    "series_cv": winner_stability.series_cv,
                    "stability_score": winner_stability.stability_score,
                    "rank_consistency": winner_stability.rank_consistency,
                },
                detail="stability analysis derived from fold and per-series dispersion",
            )
        )

    top_candidates: list[CandidateExplanation] = []
    for entry in result.leaderboard[: min(3, len(result.leaderboard))]:
        model = model_by_name.get(entry.name)
        stability = stability_by_model.get(entry.name)
        reasons = [f"mean_mase={entry.mean_mase:.4f}"]
        if model is not None:
            reasons.append(f"runtime_sec={model.runtime_sec:.3f}")
            reasons.append(f"std_mase={model.std_mase:.4f}")
        if stability is not None:
            reasons.append(f"stability_score={stability.stability_score:.4f}")
            reasons.append(f"rank_consistency={stability.rank_consistency:.4f}")
        top_candidates.append(
            CandidateExplanation(
                model=entry.name,
                rank=int(entry.rank),
                mean_mase=float(entry.mean_mase),
                runtime_sec=float(model.runtime_sec if model is not None else 0.0),
                stability_score=float(stability.stability_score) if stability is not None else None,
                reasons=reasons,
            )
        )

    why_this_model = (
        f"{winner_entry.name} ranked #1 because it combined the best mean MASE "
        "with the strongest available stability/runtime evidence in this run."
    )

    return ModelSelectionExplanation(
        winner=winner_entry.name,
        why_this_model=why_this_model,
        selection_rationale=rationale,
        top_candidates=top_candidates,
        risk_flags=risk_flags,
        evidence=evidence,
        executive_summary={
            "overview": summary.overview,
            "winner": summary.winner,
            "key_findings": summary.key_findings,
            "risks": summary.risks,
            "recommendations": summary.recommendations,
        },
    )
