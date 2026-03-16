"""Pydantic request/response models for the REST API.

These models provide OpenAPI documentation, validation,
and type safety for all API endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# --- Health ---


class HealthResponse(BaseModel):
    status: str = Field(
        ..., examples=["ok"], description="Server status"
    )


class ReadinessResponse(BaseModel):
    status: str = Field(
        ..., examples=["ok"], description="Server status"
    )
    active_jobs: int = Field(
        ..., description="Number of currently running jobs"
    )
    total_jobs: int = Field(
        ..., description="Total number of jobs in the store"
    )
    memory_rss_mb: float = Field(
        ..., description="Process resident memory in MB"
    )


# --- Benchmark ---


class BenchmarkSubmitResponse(BaseModel):
    run_id: str = Field(
        ..., description="Unique identifier for this benchmark run"
    )
    status: str = Field(
        ..., examples=["pending"], description="Initial job status"
    )


class JobStatusResponse(BaseModel):
    run_id: str
    status: str = Field(
        ...,
        description="Job status: pending, running, completed, failed",
    )
    error: str | None = Field(
        None, description="Error message if job failed"
    )


class JobSummary(BaseModel):
    run_id: str
    status: str
    tenant_id: str = ""
    user_id: str = ""
    horizon: int = 14
    n_folds: int = 3
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float | None = None
    error: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobSummary]
    total: int = Field(..., description="Total count of jobs")
    limit: int
    offset: int


class CancelResponse(BaseModel):
    run_id: str
    status: str = Field(
        ..., examples=["cancelled"], description="New status"
    )


# --- Audit ---


class AuditEventResponse(BaseModel):
    id: int
    timestamp: float
    event_type: str
    user_id: str
    tenant_id: str
    run_id: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    source_ip: str = ""


class AuditListResponse(BaseModel):
    events: list[AuditEventResponse]
