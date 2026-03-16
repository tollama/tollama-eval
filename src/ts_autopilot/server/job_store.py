"""Persistent job storage for the tollama-eval REST API.

Replaces the in-memory ``_JOBS`` dict with a durable backend so that
job state survives server restarts.

Two implementations:
- ``SQLiteJobStore``: Single-node, zero external dependencies (default).
- ``RedisJobStore``: Multi-node deployments (requires ``redis`` package).
"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ts_autopilot.logging_config import get_logger

logger = get_logger("server.job_store")


# --- Data types ---


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    """Represents a benchmark job."""

    run_id: str
    status: JobStatus = JobStatus.PENDING
    tenant_id: str = ""
    user_id: str = ""
    csv_path: str = ""
    horizon: int = 14
    n_folds: int = 3
    model_names: list[str] | None = None
    error: str | None = None
    callback_url: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# --- Abstract interface ---


class JobStore(ABC):
    """Abstract job store interface."""

    @abstractmethod
    def create(self, job: JobRecord) -> None:
        """Insert a new job record."""

    @abstractmethod
    def get(self, run_id: str) -> JobRecord | None:
        """Retrieve a job by run_id, or None if not found."""

    @abstractmethod
    def update_status(
        self,
        run_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
    ) -> None:
        """Update a job's status and optionally set error message."""

    @abstractmethod
    def list_jobs(
        self,
        *,
        tenant_id: str | None = None,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobRecord]:
        """List jobs with optional filtering."""

    @abstractmethod
    def cancel(self, run_id: str) -> bool:
        """Mark a job as cancelled. Returns True if it was cancellable."""

    @abstractmethod
    def recover_orphans(self) -> int:
        """Mark any RUNNING/PENDING jobs as FAILED on startup.

        Returns the number of recovered jobs.
        """

    @abstractmethod
    def count(self, *, tenant_id: str | None = None) -> int:
        """Count total jobs, optionally filtered by tenant."""


# --- SQLite implementation ---

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    tenant_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    csv_path TEXT NOT NULL DEFAULT '',
    horizon INTEGER NOT NULL DEFAULT 14,
    n_folds INTEGER NOT NULL DEFAULT 3,
    model_names TEXT,
    error TEXT,
    callback_url TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_jobs_tenant_status
    ON jobs (tenant_id, status);
"""


class SQLiteJobStore(JobStore):
    """SQLite-backed job store for single-node deployments."""

    def __init__(self, db_path: str | Path = "jobs.db") -> None:
        self._db_path = str(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            conn.commit()
        finally:
            conn.close()
        logger.info("SQLite job store initialized at %s", self._db_path)

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        model_names = json.loads(row["model_names"]) if row["model_names"] else None
        return JobRecord(
            run_id=row["run_id"],
            status=JobStatus(row["status"]),
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            csv_path=row["csv_path"],
            horizon=row["horizon"],
            n_folds=row["n_folds"],
            model_names=model_names,
            error=row["error"],
            callback_url=row["callback_url"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def create(self, job: JobRecord) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO jobs
                   (run_id, status, tenant_id, user_id,
                    csv_path, horizon, n_folds, model_names,
                    error, callback_url,
                    created_at, updated_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.run_id,
                    job.status.value,
                    job.tenant_id,
                    job.user_id,
                    job.csv_path,
                    job.horizon,
                    job.n_folds,
                    json.dumps(job.model_names) if job.model_names else None,
                    job.error,
                    job.callback_url,
                    job.created_at,
                    job.updated_at,
                    job.completed_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, run_id: str) -> JobRecord | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return self._row_to_job(row) if row else None
        finally:
            conn.close()

    def update_status(
        self,
        run_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
    ) -> None:
        now = time.time()
        is_terminal = status in (
            JobStatus.COMPLETED, JobStatus.FAILED,
        )
        completed_at = now if is_terminal else None
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE jobs SET status = ?, error = ?, updated_at = ?,
                   completed_at = COALESCE(?, completed_at)
                   WHERE run_id = ?""",
                (status.value, error, now, completed_at, run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def list_jobs(
        self,
        *,
        tenant_id: str | None = None,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if tenant_id is not None:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_conn()
        try:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_job(r) for r in rows]
        finally:
            conn.close()

    def cancel(self, run_id: str) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """UPDATE jobs SET status = ?, updated_at = ?
                   WHERE run_id = ? AND status IN (?, ?)""",
                (
                    JobStatus.CANCELLED.value,
                    time.time(),
                    run_id,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def recover_orphans(self) -> int:
        now = time.time()
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """UPDATE jobs
                   SET status = ?, error = ?,
                       updated_at = ?, completed_at = ?
                   WHERE status IN (?, ?)""",
                (
                    JobStatus.FAILED.value,
                    "Server restarted — job was interrupted",
                    now,
                    now,
                    JobStatus.RUNNING.value,
                    JobStatus.PENDING.value,
                ),
            )
            conn.commit()
            count = cursor.rowcount
            if count > 0:
                logger.warning("Recovered %d orphaned job(s) on startup", count)
            return count
        finally:
            conn.close()

    def count(self, *, tenant_id: str | None = None) -> int:
        conn = self._get_conn()
        try:
            if tenant_id is not None:
                row = conn.execute(
                    "SELECT COUNT(*) as c FROM jobs WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as c FROM jobs").fetchone()
            return row["c"] if row else 0
        finally:
            conn.close()

    def active_count(self, *, tenant_id: str | None = None) -> int:
        """Count jobs in PENDING or RUNNING state."""
        conn = self._get_conn()
        try:
            if tenant_id is not None:
                row = conn.execute(
                    "SELECT COUNT(*) as c FROM jobs "
                    "WHERE tenant_id = ? "
                    "AND status IN (?, ?)",
                    (tenant_id, JobStatus.PENDING.value, JobStatus.RUNNING.value),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) as c FROM jobs WHERE status IN (?, ?)",
                    (JobStatus.PENDING.value, JobStatus.RUNNING.value),
                ).fetchone()
            return row["c"] if row else 0
        finally:
            conn.close()
