"""CSV ingestion: load long or wide format CSV to canonical long format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ts_autopilot.exceptions import SchemaError
from ts_autopilot.logging_config import get_logger

REQUIRED_LONG_COLS = {"unique_id", "ds", "y"}

logger = get_logger("loader")

# Re-export for backward compatibility
__all__ = ["SchemaError", "load_csv"]


_MAX_CSV_BYTES = 500 * 1024 * 1024  # 500 MB
_MAX_MEMORY_MB_DEFAULT = 2048


def _validate_path(path: Path) -> Path:
    """Resolve and validate a file path against traversal and symlink attacks.

    Raises:
        ValueError: if the path is a symlink pointing outside its parent.
        FileNotFoundError: if the resolved path does not exist.
    """
    resolved = path.resolve()

    if path.is_symlink():
        link_target = resolved
        parent_dir = path.parent.resolve()
        if not str(link_target).startswith(str(parent_dir)):
            raise ValueError(
                f"Refusing to follow symlink '{path}' — it points outside "
                f"the parent directory to '{link_target}'."
            )

    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    return resolved


def load_csv(
    path: str | Path,
    max_memory_mb: int = _MAX_MEMORY_MB_DEFAULT,
) -> pd.DataFrame:
    """Load a CSV and return a canonical long-format DataFrame.

    Canonical columns:
      - unique_id: str
      - ds: datetime64[ns] (timezone-naive)
      - y: float64

    Supports long format (unique_id, ds, y) and wide format
    (dates as index/first column, series as remaining columns).

    Args:
        path: Path to the CSV file.
        max_memory_mb: Maximum in-memory DataFrame size in MB (default 2048).

    Raises:
        SchemaError: if the file cannot be parsed into canonical format.
        ValueError: if the file exceeds the size or memory limit.
    """
    path = Path(path)
    resolved = _validate_path(path)
    file_size = resolved.stat().st_size
    if file_size > _MAX_CSV_BYTES:
        raise ValueError(
            f"CSV file is {file_size / 1024 / 1024:.0f} MB, "
            f"exceeding the {_MAX_CSV_BYTES // 1024 // 1024} MB limit. "
            "Split large datasets before benchmarking."
        )
    logger.debug("Reading CSV: %s (%.1f KB)", resolved, file_size / 1024)
    df = pd.read_csv(resolved)

    cols = set(df.columns)

    # Long format: has ds and y columns
    if "ds" in cols and "y" in cols:
        logger.debug("Detected long format (columns: %s)", sorted(cols))
        result = _parse_long(df)
    else:
        # Wide format: try to parse first column as dates
        logger.debug("Attempting wide format parse (columns: %s)", sorted(cols))
        result = _parse_wide(df, resolved)

    # Memory guard
    mem_mb = result.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info("DataFrame memory usage: %.1f MB", mem_mb)
    if mem_mb > max_memory_mb:
        raise ValueError(
            f"Loaded DataFrame uses {mem_mb:.0f} MB, exceeding the "
            f"{max_memory_mb} MB memory limit. Reduce dataset size or "
            "increase max_memory_mb in config."
        )
    if mem_mb > max_memory_mb * 0.5:
        logger.warning(
            "DataFrame uses %.0f MB (>50%% of %d MB limit). "
            "Consider splitting the dataset.",
            mem_mb,
            max_memory_mb,
        )

    return result


def _parse_long(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a long-format DataFrame."""
    if "unique_id" not in df.columns:
        df["unique_id"] = "series_1"

    return _coerce_canonical(df)


def _parse_wide(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Melt a wide-format DataFrame to canonical long format."""
    # Re-read with first column as index
    df = pd.read_csv(path, index_col=0)

    # Try to parse index as dates
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError) as e:
        raise SchemaError(
            "Cannot parse CSV as long or wide format: "
            "no (ds, y) columns and first column is not parseable as dates."
        ) from e

    # Sanity-check parsed dates fall in a reasonable range
    min_year = df.index.min().year
    max_year = df.index.max().year
    if min_year < 1900 or max_year > 2100:
        raise SchemaError(
            f"Wide format dates span {min_year}-{max_year}, "
            "which looks like a misparse. Expected years in 1900-2100."
        )

    # All remaining columns must be numeric
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        raise SchemaError("Wide format CSV has no numeric columns.")

    df = df[numeric_cols]

    # Melt to long format
    long = df.reset_index().melt(
        id_vars=df.index.name or "index",
        var_name="unique_id",
        value_name="y",
    )
    date_col = df.index.name or "index"
    long = long.rename(columns={date_col: "ds"})

    return _coerce_canonical(long)


def _coerce_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Apply final type coercions and sorting."""
    df = df[["unique_id", "ds", "y"]].copy()

    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])

    # Strip timezone if present
    if hasattr(df["ds"].dtype, "tz") and df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_localize(None)

    df["y"] = df["y"].astype("float64")

    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    logger.debug(
        "Canonical DataFrame: %d rows, %d series",
        len(df),
        df["unique_id"].nunique(),
    )
    return df
