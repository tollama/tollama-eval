"""FastAPI REST server for tollama-eval benchmarking.

Start with: ``tollama-eval serve --port 8000``

Requires: ``pip install "tollama-eval[server]"``
"""

from __future__ import annotations

import json
import threading
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from ts_autopilot.contracts import BenchmarkResult
from ts_autopilot.reporting.explainability import build_model_selection_explanation

try:
    from fastapi import FastAPI, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

_JOBS: dict[str, dict[str, Any]] = {}
_RESULTS_DIR = Path("out/server")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def create_app(results_dir: Path | None = None) -> Any:
    """Create and return the FastAPI application.

    Args:
        results_dir: Directory to store benchmark results.

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

    global _RESULTS_DIR
    if results_dir is not None:
        _RESULTS_DIR = results_dir

    app = FastAPI(
        title="tollama-eval API",
        description="Automated time series benchmarking as a service",
        version="0.2.0",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/benchmark")
    async def submit_benchmark(
        file: UploadFile,
        horizon: int = 14,
        n_folds: int = 3,
        models: str | None = None,
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

        _JOBS[run_id] = {
            "status": JobStatus.PENDING,
            "csv_path": str(csv_path),
            "horizon": horizon,
            "n_folds": n_folds,
            "model_names": model_names,
            "error": None,
        }

        # Run in background thread
        thread = threading.Thread(
            target=_run_benchmark_job,
            args=(run_id, csv_path, horizon, n_folds, model_names, job_dir),
            daemon=True,
        )
        thread.start()

        return {"run_id": run_id, "status": "pending"}

    @app.get("/status/{run_id}")
    async def get_status(run_id: str) -> dict[str, Any]:
        if run_id not in _JOBS:
            raise HTTPException(status_code=404, detail="Run not found")
        job = _JOBS[run_id]
        resp: dict[str, Any] = {"run_id": run_id, "status": job["status"]}
        if job["error"]:
            resp["error"] = job["error"]
        return resp

    @app.get("/results/{run_id}")
    async def get_results(run_id: str) -> JSONResponse:
        if run_id not in _JOBS:
            raise HTTPException(status_code=404, detail="Run not found")
        job = _JOBS[run_id]
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Job is {job['status']}, not completed yet",
            )

        results_path = _RESULTS_DIR / run_id / "results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results file not found")

        return JSONResponse(content=json.loads(results_path.read_text()))

    @app.get("/results/{run_id}/explanation")
    async def get_results_explanation(run_id: str) -> JSONResponse:
        if run_id not in _JOBS:
            raise HTTPException(status_code=404, detail="Run not found")
        job = _JOBS[run_id]
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=f"Job is {job['status']}, not completed yet",
            )

        results_path = _RESULTS_DIR / run_id / "results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results file not found")

        payload = json.loads(results_path.read_text())
        benchmark = BenchmarkResult.from_dict(payload)
        explanation = build_model_selection_explanation(benchmark)
        return JSONResponse(content=explanation.to_dict())

    return app


def _run_benchmark_job(
    run_id: str,
    csv_path: Path,
    horizon: int,
    n_folds: int,
    model_names: list[str] | None,
    output_dir: Path,
) -> None:
    """Run benchmark in a background thread."""
    _JOBS[run_id]["status"] = JobStatus.RUNNING
    try:
        from ts_autopilot.pipeline import run_from_csv

        run_from_csv(
            csv_path=csv_path,
            horizon=horizon,
            n_folds=n_folds,
            output_dir=output_dir,
            model_names=model_names,
            run_id=run_id,
        )
        _JOBS[run_id]["status"] = JobStatus.COMPLETED
    except Exception as exc:
        _JOBS[run_id]["status"] = JobStatus.FAILED
        _JOBS[run_id]["error"] = str(exc)
