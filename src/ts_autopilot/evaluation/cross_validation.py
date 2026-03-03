"""Expanding-window cross-validation splitter."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CVSplit:
    """One fold of expanding-window cross-validation."""

    fold: int  # 1-indexed
    cutoff: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame


def make_expanding_splits(
    df: pd.DataFrame,
    horizon: int,
    n_folds: int,
) -> list[CVSplit]:
    """Generate expanding-window cross-validation splits.

    For each series (grouped by unique_id), cutoffs are spaced backward
    from the end by ``horizon`` steps. Fold 1 has the smallest training
    set; fold n_folds has the largest.

    Args:
        df: Canonical long-format DataFrame (unique_id, ds, y).
        horizon: Number of forecast steps per fold.
        n_folds: Number of folds.

    Returns:
        List of CVSplit, one per fold, ordered by increasing train size.

    Raises:
        ValueError: if any series has fewer than ``(n_folds + 1) * horizon``
            observations.
    """
    splits: list[CVSplit] = []

    # Validate minimum length per series
    for uid, group in df.groupby("unique_id"):
        if len(group) < (n_folds + 1) * horizon:
            raise ValueError(
                f"Series '{uid}' has {len(group)} rows but needs at least "
                f"{(n_folds + 1) * horizon} for {n_folds} folds with "
                f"horizon {horizon}."
            )

    # Build splits: fold_i goes from n_folds (oldest cutoff) to 1 (newest)
    for fold_idx in range(n_folds):
        # fold_idx=0 → fold 1 (smallest train), fold_idx=n_folds-1 → fold n_folds
        steps_from_end = (n_folds - fold_idx) * horizon

        train_parts = []
        test_parts = []
        cutoff_ts = None

        for uid, group in df.groupby("unique_id"):
            group = group.sort_values("ds").reset_index(drop=True)
            n = len(group)

            cutoff_idx = n - steps_from_end - 1
            train = group.iloc[: cutoff_idx + 1]
            test = group.iloc[cutoff_idx + 1 : cutoff_idx + 1 + horizon]

            train_parts.append(train)
            test_parts.append(test)

            if cutoff_ts is None:
                cutoff_ts = group.iloc[cutoff_idx]["ds"]

        splits.append(
            CVSplit(
                fold=fold_idx + 1,
                cutoff=pd.Timestamp(cutoff_ts),
                train=pd.concat(train_parts, ignore_index=True),
                test=pd.concat(test_parts, ignore_index=True),
            )
        )

    return splits
