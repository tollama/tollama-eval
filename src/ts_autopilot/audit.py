"""Audit trail for enterprise compliance.

Records structured audit events for who did what, when.
Events are stored in an append-only JSONL file and optionally
in a SQLite table for querying via the admin API.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ts_autopilot.logging_config import get_logger

logger = get_logger("audit")

_CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    tenant_id TEXT NOT NULL DEFAULT '',
    run_id TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    details TEXT,
    source_ip TEXT NOT NULL DEFAULT ''
);
"""

_CREATE_AUDIT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_audit_user
    ON audit_events (user_id, timestamp);
"""


@dataclass(frozen=True)
class AuditEvent:
    """A single audit trail entry."""

    event_type: str
    user_id: str = ""
    tenant_id: str = ""
    run_id: str = ""
    action: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    source_ip: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Persistent audit logger backed by JSONL file and SQLite."""

    def __init__(
        self,
        *,
        log_path: str | Path | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self._log_path = Path(log_path) if log_path else None
        self._db_path = str(db_path) if db_path else None
        if self._db_path:
            self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)  # type: ignore[arg-type]
        try:
            conn.execute(_CREATE_AUDIT_TABLE)
            conn.execute(_CREATE_AUDIT_INDEX)
            conn.commit()
        finally:
            conn.close()

    def log(self, event: AuditEvent) -> None:
        """Record an audit event."""
        logger.info("AUDIT %s", event.to_json())

        # Append to JSONL file
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a") as f:
                f.write(event.to_json() + "\n")

        # Insert into SQLite
        if self._db_path:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """INSERT INTO audit_events
                       (timestamp, event_type, user_id, tenant_id,
                        run_id, action, details, source_ip)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.timestamp,
                        event.event_type,
                        event.user_id,
                        event.tenant_id,
                        event.run_id,
                        event.action,
                        json.dumps(event.details),
                        event.source_ip,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def query(
        self,
        *,
        user_id: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events from SQLite."""
        if not self._db_path:
            return []

        conditions: list[str] = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = (
            f"WHERE {' AND '.join(conditions)}" if conditions else ""
        )
        query = (
            f"SELECT * FROM audit_events {where} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "event_type": r["event_type"],
                    "user_id": r["user_id"],
                    "tenant_id": r["tenant_id"],
                    "run_id": r["run_id"],
                    "action": r["action"],
                    "details": json.loads(r["details"])
                    if r["details"]
                    else {},
                    "source_ip": r["source_ip"],
                }
                for r in rows
            ]
        finally:
            conn.close()


# --- Convenience functions ---


def benchmark_submitted(
    audit: AuditLogger,
    *,
    user_id: str,
    tenant_id: str,
    run_id: str,
    source_ip: str = "",
    **details: Any,
) -> None:
    audit.log(
        AuditEvent(
            event_type="benchmark_submitted",
            user_id=user_id,
            tenant_id=tenant_id,
            run_id=run_id,
            action="submit",
            source_ip=source_ip,
            details=details,
        )
    )


def benchmark_completed(
    audit: AuditLogger,
    *,
    user_id: str,
    tenant_id: str,
    run_id: str,
    **details: Any,
) -> None:
    audit.log(
        AuditEvent(
            event_type="benchmark_completed",
            user_id=user_id,
            tenant_id=tenant_id,
            run_id=run_id,
            action="complete",
            details=details,
        )
    )


def results_accessed(
    audit: AuditLogger,
    *,
    user_id: str,
    tenant_id: str,
    run_id: str,
    source_ip: str = "",
) -> None:
    audit.log(
        AuditEvent(
            event_type="results_accessed",
            user_id=user_id,
            tenant_id=tenant_id,
            run_id=run_id,
            action="access",
            source_ip=source_ip,
        )
    )
