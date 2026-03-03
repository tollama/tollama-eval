"""Tests for CSV ingestion loader."""

import pandas as pd
import pytest

from ts_autopilot.ingestion.loader import SchemaError, load_csv


def test_long_format_loads_correctly(long_csv):
    df = load_csv(long_csv)
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert len(df) == 4
    assert set(df["unique_id"]) == {"s1", "s2"}


def test_wide_format_melts_correctly(wide_csv):
    df = load_csv(wide_csv)
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert len(df) == 4
    assert set(df["unique_id"]) == {"series_a", "series_b"}


def test_missing_unique_id_defaults_to_series_1(tmp_path):
    csv = tmp_path / "no_uid.csv"
    pd.DataFrame(
        {"ds": ["2020-01-01", "2020-01-02"], "y": [1.0, 2.0]}
    ).to_csv(csv, index=False)

    df = load_csv(csv)
    assert set(df["unique_id"]) == {"series_1"}


def test_timezone_stripped(tmp_path):
    csv = tmp_path / "tz.csv"
    pd.DataFrame(
        {
            "unique_id": ["s1", "s1"],
            "ds": ["2020-01-01T00:00:00+09:00", "2020-01-02T00:00:00+09:00"],
            "y": [1.0, 2.0],
        }
    ).to_csv(csv, index=False)

    df = load_csv(csv)
    assert df["ds"].dt.tz is None


def test_bad_schema_raises_schema_error(bad_csv):
    with pytest.raises(SchemaError):
        load_csv(bad_csv)


def test_canonical_types(long_csv):
    df = load_csv(long_csv)
    assert df["unique_id"].dtype == object  # str
    assert df["ds"].dtype == "datetime64[ns]"
    assert df["y"].dtype == "float64"


def test_sorted_by_unique_id_and_ds(long_csv):
    df = load_csv(long_csv)
    for uid, group in df.groupby("unique_id"):
        assert group["ds"].is_monotonic_increasing
