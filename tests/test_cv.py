"""Tests for expanding-window cross-validation splitter."""

import pandas as pd
import pytest

from ts_autopilot.evaluation.cross_validation import make_expanding_splits


@pytest.fixture
def cv_df():
    """2 series x 30 daily rows each — enough for 3 folds with horizon 7."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": d, "y": float(i)})
    return pd.DataFrame(rows)


def test_correct_fold_count(cv_df):
    splits = make_expanding_splits(cv_df, horizon=7, n_folds=3)
    assert len(splits) == 3


def test_folds_are_1_indexed(cv_df):
    splits = make_expanding_splits(cv_df, horizon=7, n_folds=3)
    assert [s.fold for s in splits] == [1, 2, 3]


def test_no_data_leakage(cv_df):
    splits = make_expanding_splits(cv_df, horizon=7, n_folds=3)
    for split in splits:
        assert split.test["ds"].min() > split.train["ds"].max()


def test_train_grows(cv_df):
    splits = make_expanding_splits(cv_df, horizon=7, n_folds=3)
    train_sizes = [len(s.train) for s in splits]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]


def test_test_length_per_series(cv_df):
    horizon = 7
    splits = make_expanding_splits(cv_df, horizon=horizon, n_folds=3)
    for split in splits:
        for _uid, g in split.test.groupby("unique_id"):
            assert len(g) == horizon


def test_series_too_short_raises():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": range(5)})
    with pytest.raises(ValueError, match="needs at least"):
        make_expanding_splits(df, horizon=7, n_folds=3)


def test_single_series():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {"unique_id": "only", "ds": dates, "y": [float(i) for i in range(40)]}
    )
    splits = make_expanding_splits(df, horizon=7, n_folds=3)
    assert len(splits) == 3
    for split in splits:
        assert set(split.test["unique_id"]) == {"only"}


def test_cutoff_uses_max_across_series():
    """When series have different date ranges, cutoff should be the max."""
    # s1 starts earlier, s2 starts later but both have 30 rows
    s1_dates = pd.date_range("2020-01-01", periods=30, freq="D")
    s2_dates = pd.date_range("2020-02-01", periods=30, freq="D")
    rows = []
    for d in s1_dates:
        rows.append({"unique_id": "s1", "ds": d, "y": 1.0})
    for d in s2_dates:
        rows.append({"unique_id": "s2", "ds": d, "y": 2.0})
    df = pd.DataFrame(rows)

    splits = make_expanding_splits(df, horizon=7, n_folds=3)
    s2_dates = set(df[df["unique_id"] == "s2"]["ds"])
    for split in splits:
        # Cutoff should be from the later series (s2)
        assert isinstance(split.cutoff, pd.Timestamp)
        assert split.cutoff in s2_dates
