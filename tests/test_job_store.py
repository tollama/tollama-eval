"""Tests for the persistent job store."""

from __future__ import annotations

from pathlib import Path

import pytest

from ts_autopilot.server.job_store import (
    JobRecord,
    JobStatus,
    SQLiteJobStore,
)


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteJobStore:
    return SQLiteJobStore(db_path=tmp_path / "test_jobs.db")


def test_create_and_get(store: SQLiteJobStore) -> None:
    job = JobRecord(run_id="r1", csv_path="/tmp/test.csv", horizon=7, n_folds=2)
    store.create(job)
    retrieved = store.get("r1")
    assert retrieved is not None
    assert retrieved.run_id == "r1"
    assert retrieved.status == JobStatus.PENDING
    assert retrieved.horizon == 7


def test_get_nonexistent(store: SQLiteJobStore) -> None:
    assert store.get("nonexistent") is None


def test_update_status(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="r2"))
    store.update_status("r2", JobStatus.RUNNING)
    job = store.get("r2")
    assert job is not None
    assert job.status == JobStatus.RUNNING
    assert job.completed_at is None

    store.update_status("r2", JobStatus.COMPLETED)
    job = store.get("r2")
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.completed_at is not None


def test_update_status_with_error(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="r3"))
    store.update_status("r3", JobStatus.FAILED, error="OOM")
    job = store.get("r3")
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error == "OOM"


def test_list_jobs_empty(store: SQLiteJobStore) -> None:
    assert store.list_jobs() == []


def test_list_jobs_pagination(store: SQLiteJobStore) -> None:
    for i in range(5):
        store.create(JobRecord(run_id=f"r{i}"))
    all_jobs = store.list_jobs(limit=10)
    assert len(all_jobs) == 5

    page1 = store.list_jobs(limit=2, offset=0)
    assert len(page1) == 2
    page2 = store.list_jobs(limit=2, offset=2)
    assert len(page2) == 2
    page3 = store.list_jobs(limit=2, offset=4)
    assert len(page3) == 1


def test_list_jobs_filter_by_tenant(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="a", tenant_id="t1"))
    store.create(JobRecord(run_id="b", tenant_id="t2"))
    store.create(JobRecord(run_id="c", tenant_id="t1"))
    t1_jobs = store.list_jobs(tenant_id="t1")
    assert len(t1_jobs) == 2
    assert all(j.tenant_id == "t1" for j in t1_jobs)


def test_list_jobs_filter_by_status(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="a"))
    store.create(JobRecord(run_id="b"))
    store.update_status("b", JobStatus.COMPLETED)
    completed = store.list_jobs(status=JobStatus.COMPLETED)
    assert len(completed) == 1
    assert completed[0].run_id == "b"


def test_cancel_pending_job(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="r4"))
    assert store.cancel("r4") is True
    job = store.get("r4")
    assert job is not None
    assert job.status == JobStatus.CANCELLED


def test_cancel_completed_job_fails(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="r5"))
    store.update_status("r5", JobStatus.COMPLETED)
    assert store.cancel("r5") is False
    job = store.get("r5")
    assert job is not None
    assert job.status == JobStatus.COMPLETED


def test_recover_orphans(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="a"))
    store.update_status("a", JobStatus.RUNNING)
    store.create(JobRecord(run_id="b"))  # pending
    store.create(JobRecord(run_id="c"))
    store.update_status("c", JobStatus.COMPLETED)

    count = store.recover_orphans()
    assert count == 2  # a (running) + b (pending)

    a = store.get("a")
    assert a is not None
    assert a.status == JobStatus.FAILED
    assert "restarted" in (a.error or "").lower()

    c = store.get("c")
    assert c is not None
    assert c.status == JobStatus.COMPLETED


def test_count(store: SQLiteJobStore) -> None:
    assert store.count() == 0
    store.create(JobRecord(run_id="a", tenant_id="t1"))
    store.create(JobRecord(run_id="b", tenant_id="t2"))
    assert store.count() == 2
    assert store.count(tenant_id="t1") == 1


def test_active_count(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="a"))
    store.create(JobRecord(run_id="b"))
    store.update_status("b", JobStatus.RUNNING)
    store.create(JobRecord(run_id="c"))
    store.update_status("c", JobStatus.COMPLETED)
    assert store.active_count() == 2  # a (pending) + b (running)


def test_model_names_serialization(store: SQLiteJobStore) -> None:
    store.create(
        JobRecord(run_id="r6", model_names=["SeasonalNaive", "AutoETS"])
    )
    job = store.get("r6")
    assert job is not None
    assert job.model_names == ["SeasonalNaive", "AutoETS"]


def test_model_names_none(store: SQLiteJobStore) -> None:
    store.create(JobRecord(run_id="r7"))
    job = store.get("r7")
    assert job is not None
    assert job.model_names is None


def test_to_dict(store: SQLiteJobStore) -> None:
    job = JobRecord(run_id="r8", status=JobStatus.COMPLETED)
    d = job.to_dict()
    assert d["status"] == "completed"
    assert d["run_id"] == "r8"
