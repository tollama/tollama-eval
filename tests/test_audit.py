"""Tests for the audit trail system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ts_autopilot.audit import (
    AuditEvent,
    AuditLogger,
    benchmark_completed,
    benchmark_submitted,
    results_accessed,
)


@pytest.fixture()
def audit_logger(tmp_path: Path) -> AuditLogger:
    return AuditLogger(
        log_path=tmp_path / "audit.jsonl",
        db_path=tmp_path / "audit.db",
    )


def test_audit_event_to_json() -> None:
    event = AuditEvent(
        event_type="test",
        user_id="u1",
        run_id="r1",
        action="do_thing",
    )
    parsed = json.loads(event.to_json())
    assert parsed["event_type"] == "test"
    assert parsed["user_id"] == "u1"
    assert "timestamp" in parsed


def test_log_to_jsonl(audit_logger: AuditLogger, tmp_path: Path) -> None:
    audit_logger.log(
        AuditEvent(
            event_type="test", user_id="u1", action="create"
        )
    )
    log_path = tmp_path / "audit.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "test"


def test_log_to_sqlite(audit_logger: AuditLogger) -> None:
    audit_logger.log(
        AuditEvent(
            event_type="benchmark_submitted",
            user_id="alice",
            run_id="run1",
        )
    )
    results = audit_logger.query(user_id="alice")
    assert len(results) == 1
    assert results[0]["event_type"] == "benchmark_submitted"
    assert results[0]["run_id"] == "run1"


def test_query_filters(audit_logger: AuditLogger) -> None:
    for i in range(5):
        audit_logger.log(
            AuditEvent(
                event_type="submit" if i < 3 else "access",
                user_id="alice" if i < 2 else "bob",
                run_id=f"r{i}",
            )
        )
    assert len(audit_logger.query(user_id="alice")) == 2
    assert len(audit_logger.query(event_type="access")) == 2
    assert len(audit_logger.query(limit=3)) == 3


def test_convenience_functions(audit_logger: AuditLogger) -> None:
    benchmark_submitted(
        audit_logger,
        user_id="u1",
        tenant_id="t1",
        run_id="r1",
        horizon=14,
    )
    benchmark_completed(
        audit_logger,
        user_id="u1",
        tenant_id="t1",
        run_id="r1",
        winner="AutoETS",
    )
    results_accessed(
        audit_logger,
        user_id="u2",
        tenant_id="t1",
        run_id="r1",
    )
    events = audit_logger.query()
    assert len(events) == 3
    types = {e["event_type"] for e in events}
    assert types == {
        "benchmark_submitted",
        "benchmark_completed",
        "results_accessed",
    }


def test_no_db_query() -> None:
    logger = AuditLogger(log_path=None, db_path=None)
    logger.log(AuditEvent(event_type="test"))
    assert logger.query() == []
