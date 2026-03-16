# ADR-0005: Persistent Job Storage with SQLite

## Status

Accepted

## Context

The original REST API stored job state in a Python dict (`_JOBS`), which
was lost on every server restart. Production deployments need durable
job tracking, crash recovery, and job listing/filtering.

## Decision

Use SQLite as the default persistent job store (`SQLiteJobStore`):

- Zero external dependencies (SQLite is in Python's stdlib).
- WAL mode for concurrent read/write from the server + workers.
- On startup, orphaned RUNNING/PENDING jobs are marked FAILED with
  a recovery note.
- An abstract `JobStore` interface allows alternative backends
  (e.g., `RedisJobStore` for multi-node deployments).

## Consequences

- Single-node deployments work with zero infrastructure setup.
- Multi-node requires Redis or a shared database (not SQLite).
- Job history persists across restarts, enabling audit and analytics.
- SQLite file must be on persistent storage (PVC in Kubernetes).
