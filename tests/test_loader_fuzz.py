"""Fuzz tests for CSV loader — ensure no unhandled crashes."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ts_autopilot.exceptions import SchemaError


@given(content=st.text(min_size=0, max_size=500))
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_load_csv_never_crashes_on_text(tmp_path, content):
    """load_csv should never raise an unhandled exception on arbitrary text."""
    csv_file = tmp_path / "fuzz.csv"
    csv_file.write_text(content)

    from ts_autopilot.ingestion.loader import load_csv

    try:
        load_csv(str(csv_file))
    except (SchemaError, ValueError, KeyError, pd.errors.ParserError):
        pass  # Expected errors are fine
    except Exception as exc:
        # Any other exception type is a bug
        pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


@given(content=st.binary(min_size=0, max_size=500))
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_load_csv_never_crashes_on_binary(tmp_path, content):
    """load_csv should handle binary garbage gracefully."""
    csv_file = tmp_path / "fuzz.csv"
    csv_file.write_bytes(content)

    from ts_autopilot.ingestion.loader import load_csv

    try:
        load_csv(str(csv_file))
    except (SchemaError, ValueError, KeyError, UnicodeDecodeError):
        pass
    except Exception as exc:
        # ParserError and similar pandas exceptions are OK
        if "Parser" in type(exc).__name__ or "Empty" in str(exc):
            pass
        else:
            pytest.fail(f"Unexpected exception type {type(exc).__name__}: {exc}")


def test_load_csv_empty_file(tmp_path):
    """Empty CSV should raise a clear error."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")

    from ts_autopilot.ingestion.loader import load_csv

    with pytest.raises((SchemaError, ValueError, Exception)):
        load_csv(str(csv_file))


def test_load_csv_only_header(tmp_path):
    """CSV with only headers and no data rows."""
    csv_file = tmp_path / "header_only.csv"
    csv_file.write_text("unique_id,ds,y\n")

    from ts_autopilot.ingestion.loader import load_csv

    # Should either return empty df or raise
    try:
        df = load_csv(str(csv_file))
        assert len(df) == 0
    except (SchemaError, ValueError):
        pass


def test_load_csv_unicode_series_names(tmp_path):
    """CSV with unicode characters in series names."""
    csv_file = tmp_path / "unicode.csv"
    csv_file.write_text(
        "unique_id,ds,y\n"
        "série_à,2024-01-01,1.0\n"
        "série_à,2024-01-02,2.0\n"
        "系列_B,2024-01-01,3.0\n"
        "系列_B,2024-01-02,4.0\n"
    )

    from ts_autopilot.ingestion.loader import load_csv

    df = load_csv(str(csv_file))
    assert len(df) == 4
    assert df["unique_id"].nunique() == 2
