"""Data export utilities for benchmark results."""

from __future__ import annotations

from pathlib import Path

from ts_autopilot.contracts import BenchmarkResult
from ts_autopilot.logging_config import get_logger

logger = get_logger("export")


def export_leaderboard_csv(result: BenchmarkResult, output_path: str | Path) -> Path:
    """Export leaderboard as a CSV file.

    Args:
        result: Benchmark result with leaderboard data.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in result.leaderboard:
        rows.append(
            {
                "rank": entry.rank,
                "model": entry.name,
                "mase": entry.mean_mase,
                "smape": entry.mean_smape,
                "rmsse": entry.mean_rmsse,
                "mae": entry.mean_mae,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Exported leaderboard CSV: %s", output_path)
    return output_path


def export_fold_details_csv(
    result: BenchmarkResult, output_path: str | Path
) -> Path:
    """Export per-model per-fold details as a CSV file.

    Args:
        result: Benchmark result with model data.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in result.models:
        for fold in model.folds:
            rows.append(
                {
                    "model": model.name,
                    "fold": fold.fold,
                    "cutoff": fold.cutoff,
                    "mase": fold.mase,
                    "smape": fold.smape,
                    "rmsse": fold.rmsse,
                    "mae": fold.mae,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Exported fold details CSV: %s", output_path)
    return output_path


def export_per_series_csv(
    result: BenchmarkResult, output_path: str | Path
) -> Path:
    """Export per-series scores across all models as a CSV file.

    Args:
        result: Benchmark result with per-series score data.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in result.models:
        for fold in model.folds:
            for sid, score in fold.series_scores.items():
                rows.append(
                    {
                        "model": model.name,
                        "fold": fold.fold,
                        "series": sid,
                        "mase": score,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Exported per-series CSV: %s", output_path)
    return output_path
