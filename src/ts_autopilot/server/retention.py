"""Result retention policies for the tollama-eval REST API.

Automatically cleans up old benchmark results based on
configurable age, count, and size limits.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ts_autopilot.logging_config import get_logger
from ts_autopilot.server.job_store import JobStatus, JobStore

logger = get_logger("server.retention")


@dataclass
class RetentionPolicy:
    """Configuration for result retention."""

    max_age_days: int | None = None
    max_total_size_mb: int | None = None
    max_results_count: int | None = None


def enforce_retention(
    job_store: JobStore,
    results_dir: Path,
    policy: RetentionPolicy,
    *,
    dry_run: bool = False,
) -> int:
    """Apply retention policy and delete expired results.

    Args:
        job_store: Job store to query/update.
        results_dir: Directory containing result subdirectories.
        policy: Retention rules to enforce.
        dry_run: If True, log but don't delete.

    Returns:
        Number of results deleted.
    """
    deleted = 0

    # Age-based cleanup
    if policy.max_age_days is not None:
        cutoff = time.time() - (policy.max_age_days * 86400)
        old_jobs = job_store.list_jobs(
            status=JobStatus.COMPLETED, limit=10000
        )
        for job in old_jobs:
            if job.completed_at and job.completed_at < cutoff:
                result_dir = results_dir / job.run_id
                if result_dir.exists():
                    if dry_run:
                        logger.info(
                            "Would delete %s (age policy)",
                            job.run_id,
                        )
                    else:
                        shutil.rmtree(result_dir)
                        logger.info(
                            "Deleted %s (age: >%d days)",
                            job.run_id,
                            policy.max_age_days,
                        )
                        deleted += 1

    # Count-based cleanup
    if policy.max_results_count is not None:
        all_jobs = job_store.list_jobs(
            status=JobStatus.COMPLETED, limit=10000
        )
        if len(all_jobs) > policy.max_results_count:
            # Jobs are ordered by created_at DESC,
            # so trim from the end (oldest)
            excess = all_jobs[policy.max_results_count :]
            for job in excess:
                result_dir = results_dir / job.run_id
                if result_dir.exists():
                    if dry_run:
                        logger.info(
                            "Would delete %s (count policy)",
                            job.run_id,
                        )
                    else:
                        shutil.rmtree(result_dir)
                        logger.info(
                            "Deleted %s (count: >%d)",
                            job.run_id,
                            policy.max_results_count,
                        )
                        deleted += 1

    # Size-based cleanup
    if policy.max_total_size_mb is not None:
        total_bytes = _compute_total_size(results_dir)
        max_bytes = policy.max_total_size_mb * 1024 * 1024
        if total_bytes > max_bytes:
            all_jobs = job_store.list_jobs(
                status=JobStatus.COMPLETED, limit=10000
            )
            # Delete oldest first
            for job in reversed(all_jobs):
                if total_bytes <= max_bytes:
                    break
                result_dir = results_dir / job.run_id
                if result_dir.exists():
                    dir_size = _dir_size(result_dir)
                    if dry_run:
                        logger.info(
                            "Would delete %s (size policy)",
                            job.run_id,
                        )
                    else:
                        shutil.rmtree(result_dir)
                        total_bytes -= dir_size
                        logger.info(
                            "Deleted %s (size: total>%dMB)",
                            job.run_id,
                            policy.max_total_size_mb,
                        )
                        deleted += 1

    if deleted > 0:
        logger.info("Retention cleanup: deleted %d result(s)", deleted)

    return deleted


def _compute_total_size(results_dir: Path) -> int:
    """Compute total size of all files in results directory."""
    return sum(
        f.stat().st_size
        for f in results_dir.rglob("*")
        if f.is_file()
    )


def _dir_size(path: Path) -> int:
    """Compute total size of files in a directory."""
    return sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    )
