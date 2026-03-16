"""Background worker for benchmark job execution.

Manages the lifecycle of benchmark jobs: picks up pending jobs,
runs them in background threads, and updates the job store.
"""

from __future__ import annotations

import signal
import threading
from pathlib import Path
from typing import Any

from ts_autopilot.logging_config import get_logger
from ts_autopilot.server.job_store import JobRecord, JobStatus, JobStore

logger = get_logger("server.worker")


class BenchmarkWorker:
    """Runs benchmark jobs in background threads with lifecycle management."""

    def __init__(
        self,
        job_store: JobStore,
        results_dir: Path,
        *,
        max_concurrent: int = 4,
    ) -> None:
        self._store = job_store
        self._results_dir = results_dir
        self._max_concurrent = max_concurrent
        self._threads: dict[str, threading.Thread] = {}
        self._shutdown = threading.Event()
        self._lock = threading.Lock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._threads.values() if t.is_alive())

    @property
    def is_draining(self) -> bool:
        return self._shutdown.is_set()

    def submit(self, job: JobRecord) -> bool:
        """Submit a job for execution.

        Returns False if the worker is draining or at capacity.
        """
        if self._shutdown.is_set():
            logger.warning("Rejecting job %s — worker is draining", job.run_id)
            return False

        with self._lock:
            alive = sum(1 for t in self._threads.values() if t.is_alive())
            if alive >= self._max_concurrent:
                logger.warning(
                    "Rejecting job %s — at capacity (%d/%d)",
                    job.run_id,
                    alive,
                    self._max_concurrent,
                )
                return False

            thread = threading.Thread(
                target=self._run_job,
                args=(job,),
                name=f"worker-{job.run_id}",
                daemon=True,
            )
            self._threads[job.run_id] = thread
            thread.start()
            return True

    def _run_job(self, job: JobRecord) -> None:
        """Execute a benchmark job."""
        run_id = job.run_id
        self._store.update_status(run_id, JobStatus.RUNNING)
        output_dir = self._results_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from ts_autopilot.pipeline import run_from_csv

            run_from_csv(
                csv_path=Path(job.csv_path),
                horizon=job.horizon,
                n_folds=job.n_folds,
                output_dir=output_dir,
                model_names=job.model_names,
                run_id=run_id,
            )
            self._store.update_status(run_id, JobStatus.COMPLETED)
            logger.info("Job %s completed successfully", run_id)
        except Exception as exc:
            self._store.update_status(
                run_id, JobStatus.FAILED, error=str(exc)
            )
            logger.error("Job %s failed: %s", run_id, exc)

        # Clean up thread reference
        with self._lock:
            self._threads.pop(run_id, None)

    def drain(self, timeout_sec: float = 300) -> int:
        """Stop accepting new jobs and wait for in-flight jobs to finish.

        Args:
            timeout_sec: Max seconds to wait for jobs to complete.

        Returns:
            Number of jobs that were still running when timeout expired.
        """
        self._shutdown.set()
        logger.info("Worker draining — waiting up to %.0fs for in-flight jobs", timeout_sec)

        with self._lock:
            active = {k: t for k, t in self._threads.items() if t.is_alive()}

        for run_id, thread in active.items():
            thread.join(timeout=timeout_sec)
            if thread.is_alive():
                logger.warning("Job %s still running after drain timeout", run_id)

        with self._lock:
            remaining = sum(1 for t in self._threads.values() if t.is_alive())

        if remaining == 0:
            logger.info("All jobs drained successfully")
        else:
            logger.warning("%d job(s) still running after drain", remaining)

        return remaining

    def shutdown(self, timeout_sec: float = 30) -> None:
        """Gracefully shut down the worker."""
        remaining = self.drain(timeout_sec=timeout_sec)
        if remaining > 0:
            logger.warning(
                "Forcing shutdown with %d job(s) still in progress", remaining
            )
