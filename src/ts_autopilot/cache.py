"""Result caching for incremental benchmarks.

Caches per-model-per-fold results keyed by a hash of the input data + config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ts_autopilot.contracts import FoldResult
from ts_autopilot.logging_config import get_logger

logger = get_logger("cache")


def _compute_data_hash(df: pd.DataFrame, horizon: int, n_folds: int) -> str:
    """Compute a deterministic hash of the input data + config."""
    h = hashlib.sha256()
    # Hash the data shape and a sample of values
    h.update(f"shape={df.shape}".encode())
    h.update(f"horizon={horizon}".encode())
    h.update(f"n_folds={n_folds}".encode())
    # Hash column dtypes
    for col in sorted(df.columns):
        h.update(f"{col}:{df[col].dtype}".encode())
    # Hash first and last rows for fast fingerprinting
    if len(df) > 0:
        h.update(df.head(5).to_csv(index=False).encode())
        h.update(df.tail(5).to_csv(index=False).encode())
    return h.hexdigest()[:16]


class ResultCache:
    """File-based cache for benchmark fold results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, data_hash: str, model_name: str, fold: int) -> Path:
        safe_name = model_name.replace("/", "_")
        return self.cache_dir / f"{data_hash}_{safe_name}_fold{fold}.json"

    def get(self, data_hash: str, model_name: str, fold: int) -> FoldResult | None:
        """Retrieve a cached fold result, or None if not cached."""
        path = self._key_path(data_hash, model_name, fold)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            result = FoldResult.from_dict(data)
            logger.debug("Cache hit: %s fold %d", model_name, fold)
            return result
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupt cache entry: %s", path)
            path.unlink(missing_ok=True)
            return None

    def put(
        self, data_hash: str, model_name: str, fold: int, result: FoldResult
    ) -> None:
        """Store a fold result in the cache."""
        path = self._key_path(data_hash, model_name, fold)
        path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.debug("Cached: %s fold %d", model_name, fold)

    def clear(self) -> int:
        """Remove all cached results. Returns number of entries removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cache entries", count)
        return count
