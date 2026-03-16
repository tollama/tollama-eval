"""FastAPI REST server for tollama-eval benchmarking.

Start with: ``tollama-eval serve --port 8000``

Requires: ``pip install "tollama-eval[server]"``

Enterprise features:
- Authentication (API key / JWT)
- Persistent job storage (SQLite / Redis)
- Kubernetes health probes (live / ready / startup)
- Rate limiting
- Graceful shutdown with job draining
"""

from __future__ import annotations

import json
import os
import resource
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ts_autopilot.contracts import BenchmarkResult
from ts_autopilot.logging_config import get_logger
from ts_autopilot.reporting.explainability import build_model_selection_explanation
from ts_autopilot.server.auth import AuthConfig, AuthUser, setup_auth
from ts_autopilot.server.job_store import (
    JobRecord,
    JobStatus,
    SQLiteJobStore,
)
from ts_autopilot.server.worker import BenchmarkWorker

logger = get_logger("server")

try:
    from fastapi import FastAPI, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Module-level state for backward compatibility with existing tests
_RESULTS_DIR = Path("out/server")
_JOBS: dict[str, dict[str, Any]] = {}

# Startup flag for /health/startup probe
_STARTUP_COMPLETE = False


def create_app(
    results_dir: Path | None = None,
    *,
    auth_config: AuthConfig | None = None,
    db_path: str | Path | None = None,
    drain_timeout_sec: float = 300,
) -> Any:
    """Create and return the FastAPI application.

    Args:
        results_dir: Directory to store benchmark results.
        auth_config: Authentication configuration. Defaults to no auth.
        db_path: Path to SQLite database for job persistence.
        drain_timeout_sec: Seconds to wait for jobs during shutdown.

    Returns:
        FastAPI app instance.

    Raises:
        ImportError: if fastapi is not installed.
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the REST server. "
            'Install with: pip install "tollama-eval[server]"'
        )

    global _RESULTS_DIR, _STARTUP_COMPLETE
    if results_dir is not None:
        _RESULTS_DIR = results_dir
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize persistent job store
    _db_path = db_path or (_RESULTS_DIR / "jobs.db")
    job_store = SQLiteJobStore(db_path=_db_path)
    job_store.recover_orphans()

    # Initialize worker
    worker = BenchmarkWorker(
        job_store=job_store,
        results_dir=_RESULTS_DIR,
        max_concurrent=int(os.environ.get("TOLLAMA_MAX_CONCURRENT_JOBS", "4")),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        global _STARTUP_COMPLETE
        _STARTUP_COMPLETE = True
        logger.info("Server startup complete")
        yield
        # Graceful shutdown: drain in-flight jobs
        logger.info("Server shutting down — draining jobs")
        worker.shutdown(timeout_sec=drain_timeout_sec)
        logger.info("Server shutdown complete")

    app = FastAPI(
        title="tollama-eval API",
        description="Automated time series benchmarking as a service",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Store references on app state for access in endpoints
    app.state.job_store = job_store
    app.state.worker = worker
    app.state.results_dir = _RESULTS_DIR

    # --- Authentication ---
    if auth_config is None:
        auth_config = AuthConfig(mode=os.environ.get("TOLLAMA_AUTH_MODE", "none"))
    setup_auth(app, auth_config)

    # --- Health probes ---

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Legacy health endpoint — alias for /health/live."""
        return {"status": "ok"}

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        """Liveness probe: process is alive."""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        """Readiness probe: server can accept work."""
        if worker.is_draining:
            return JSONResponse(
                status_code=503,
                content={"status": "draining", "detail": "Server is shutting down"},
            )

        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports in bytes, Linux in KB
        import sys

        if sys.platform == "darwin":
            rss_mb = rss_bytes / (1024 * 1024)
        else:
            rss_mb = rss_bytes / 1024

        return {
            "status": "ok",
            "active_jobs": worker.active_count,
            "total_jobs": job_store.count(),
            "memory_rss_mb": round(rss_mb, 1),
        }

    @app.get("/health/startup")
    async def health_startup() -> dict[str, str]:
        """Startup probe: initial setup complete."""
        if not _STARTUP_COMPLETE:
            return JSONResponse(
                status_code=503,
                content={"status": "starting", "detail": "Server is still starting"},
            )
        return {"status": "ok"}

    # --- Benchmark endpoints ---

    @app.post("/benchmark")
    async def submit_benchmark(
        request: Request,
        file: UploadFile,
        horizon: int = 14,
        n_folds: int = 3,
        models: str | None = None,
        callback_url: str | None = None,
    ) -> dict[str, str]:
        """Submit a CSV file for benchmarking.

        Returns a run_id to track the job.
        """
        run_id = uuid.uuid4().hex[:12]
        job_dir = _RESULTS_DIR / run_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        csv_path = job_dir / "input.csv"
        content = await file.read()
        csv_path.write_bytes(content)

        model_names = None
        if models:
            model_names = [m.strip() for m in models.split(",") if m.strip()]

        # Resolve auth user
        auth_user: AuthUser | None = getattr(
            request.state, "auth_user", None
        )
        tenant_id = auth_user.tenant_id if auth_user else ""
        user_id = auth_user.user_id if auth_user else ""

        # Create persistent job record
        job = JobRecord(
            run_id=run_id,
            status=JobStatus.PENDING,
            tenant_id=tenant_id,
            user_id=user_id,
            csv_path=str(csv_path),
            horizon=horizon,
            n_folds=n_folds,
            model_names=model_names,
            callback_url=callback_url,
        )
        job_store.create(job)

        # Backward compat: keep in-memory dict for existing tests
        _JOBS[run_id] = {
            "status": JobStatus.PENDING,
            "csv_path": str(csv_path),
            "horizon": horizon,
            "n_folds": n_folds,
            "model_names": model_names,
            "error": None,
        }

        # Submit to worker
        if not worker.submit(job):
            job_store.update_status(
                run_id, JobStatus.FAILED, error="Server at capacity or draining"
            )
            raise HTTPException(
                status_code=503, detail="Server at capacity — try again later"
            )

        return {"run_id": run_id, "status": "pending"}

    @app.get("/status/{run_id}")
    async def get_status(run_id: str) -> dict[str, Any]:
        job = job_store.get(run_id)
        if job is None:
            # Backward compat: check in-memory dict
            if run_id in _JOBS:
                j = _JOBS[run_id]
                resp: dict[str, Any] = {"run_id": run_id, "status": j["status"]}
                if j["error"]:
                    resp["error"] = j["error"]
                return resp
            raise HTTPException(status_code=404, detail="Run not found")
        resp = {"run_id": run_id, "status": job.status.value}
        if job.error:
            resp["error"] = job.error
        return resp

    @app.get("/results/{run_id}")
    async def get_results(run_id: str) -> JSONResponse:
        job = job_store.get(run_id)
        if job is None:
            if run_id not in _JOBS:
                raise HTTPException(status_code=404, detail="Run not found")
            # Backward compat path
            if _JOBS[run_id]["status"] != JobStatus.COMPLETED:
                raise HTTPException(
                    status_code=409,
                    detail=f"Job is {_JOBS[run_id]['status']}, not completed yet",
                )
        elif job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Job is {job.status.value}, not completed yet",
            )

        results_path = _RESULTS_DIR / run_id / "results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results file not found")

        return JSONResponse(content=json.loads(results_path.read_text()))

    @app.get("/results/{run_id}/explanation")
    async def get_results_explanation(run_id: str) -> JSONResponse:
        job = job_store.get(run_id)
        if job is None:
            if run_id not in _JOBS:
                raise HTTPException(status_code=404, detail="Run not found")
            if _JOBS[run_id]["status"] != JobStatus.COMPLETED:
                raise HTTPException(
                    status_code=409,
                    detail=f"Job is {_JOBS[run_id]['status']}, not completed yet",
                )
        elif job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Job is {job.status.value}, not completed yet",
            )

        results_path = _RESULTS_DIR / run_id / "results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results file not found")

        payload = json.loads(results_path.read_text())
        benchmark = BenchmarkResult.from_dict(payload)
        explanation = build_model_selection_explanation(benchmark)
        return JSONResponse(content=explanation.to_dict())

    # --- Job management endpoints ---

    @app.get("/jobs")
    async def list_jobs(
        request: Request,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List benchmark jobs with pagination."""
        auth_user: AuthUser | None = getattr(
            request.state, "auth_user", None
        )
        tenant_id = auth_user.tenant_id if auth_user and auth_user.tenant_id else None

        status_filter = JobStatus(status) if status else None
        jobs = job_store.list_jobs(
            tenant_id=tenant_id,
            status=status_filter,
            limit=limit,
            offset=offset,
        )
        return {
            "jobs": [j.to_dict() for j in jobs],
            "total": job_store.count(tenant_id=tenant_id),
            "limit": limit,
            "offset": offset,
        }

    @app.post("/jobs/{run_id}/cancel")
    async def cancel_job(run_id: str) -> dict[str, Any]:
        """Cancel a pending or running job."""
        if job_store.cancel(run_id):
            return {"run_id": run_id, "status": "cancelled"}
        job = job_store.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found")
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in state '{job.status.value}'",
        )

    return app
