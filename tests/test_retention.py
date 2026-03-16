"""Tests for result retention policies."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ts_autopilot.server.job_store import (
    JobRecord,
    JobStatus,
    SQLiteJobStore,
)
from ts_autopilot.server.retention import (
    RetentionPolicy,
    enforce_retention,
)


@pytest.fixture()
def setup(tmp_path: Path):
    """Create store and results dir with sample data."""
    store = SQLiteJobStore(db_path=tmp_path / "jobs.db")
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create 5 completed jobs with result dirs
    now = time.time()
    for i in range(5):
        run_id = f"run{i}"
        job = JobRecord(
            run_id=run_id,
            status=JobStatus.COMPLETED,
            created_at=now - (i * 86400),  # each 1 day older
            completed_at=now - (i * 86400),
        )
        store.create(job)
        # Create a result directory with a file
        rd = results_dir / run_id
        rd.mkdir()
        (rd / "results.json").write_text('{"test": true}')

    return store, results_dir


def test_age_retention(setup) -> None:
    store, results_dir = setup
    # Delete results older than 3 days
    policy = RetentionPolicy(max_age_days=3)
    deleted = enforce_retention(store, results_dir, policy)
    assert deleted >= 1
    # Newest results should still exist
    assert (results_dir / "run0").exists()


def test_count_retention(setup) -> None:
    store, results_dir = setup
    # Keep only 2 results
    policy = RetentionPolicy(max_results_count=2)
    deleted = enforce_retention(store, results_dir, policy)
    assert deleted == 3
    remaining = [d for d in results_dir.iterdir() if d.is_dir()]
    assert len(remaining) == 2


def test_dry_run(setup) -> None:
    store, results_dir = setup
    policy = RetentionPolicy(max_results_count=1)
    deleted = enforce_retention(
        store, results_dir, policy, dry_run=True
    )
    assert deleted == 0
    # All dirs should still exist
    remaining = [d for d in results_dir.iterdir() if d.is_dir()]
    assert len(remaining) == 5


def test_no_policy(setup) -> None:
    store, results_dir = setup
    policy = RetentionPolicy()  # All None
    deleted = enforce_retention(store, results_dir, policy)
    assert deleted == 0
