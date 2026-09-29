"""
sweeper.py — Timeout recovery and startup crash recovery.

Two operating modes
-------------------
startup_recovery():
  Called once at process start before the main loop.
  - Cleans up incomplete queue jobs (dir exists but no meta.json).
  - Moves ALL processing/ jobs to failed/ (they are orphaned because
    the previous process died). For single-worker MVP this is correct:
    if the process restarted, no job can legitimately be in processing/.

sweep():
  Called every scan cycle.
  - Moves processing/ jobs that have exceeded processing_timeout_seconds
    to failed/.

Why move to failed/ on timeout instead of back to queue/?
----------------------------------------------------------
A timed-out job may have partial output (a .srt.tmp file, incomplete
audio copy, etc.) in the job dir. Re-queuing it would cause the worker
to process a potentially corrupt bundle. Moving to failed/ forces an
operator to inspect and decide whether to re-ingest the source file.

Re-ingesting after a failed job:
  The runtime excludes failed/ jobs from FingerprintRegistry rebuild and
  unregisters fingerprints when jobs are moved to failed/, so operators can
  re-ingest without manual registry cleanup.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Config
from .fingerprint import FingerprintRegistry
from .log import get_logger
from .paths import write_json_atomic

log = get_logger()


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        s = s.rstrip("Z")
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return None


class Sweeper:
    """Handles crash recovery and processing timeout enforcement."""

    def __init__(self, config: Config, registry: Optional[FingerprintRegistry] = None) -> None:
        self._config = config
        self._registry = registry

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def startup_recovery(self) -> None:
        """
        Run immediately on process start.
        Cleans up crash remnants from the previous process.
        """
        log.info("Sweeper: startup recovery begin")
        cleaned = self._cleanup_incomplete_queue_jobs()
        recovered = self._recover_processing_jobs(force=True)
        log.info(
            f"Sweeper: startup recovery complete — "
            f"cleaned={cleaned} recovered={recovered}"
        )

    def sweep(self) -> None:
        """
        Periodic timeout check.
        Called every scan cycle from the main loop.
        """
        timed_out = self._recover_processing_jobs(force=False)
        if timed_out:
            log.warning(f"Sweeper: moved {timed_out} timed-out jobs to failed/")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cleanup_incomplete_queue_jobs(self) -> int:
        """
        Delete queue job directories that have no meta.json.

        These are artefacts of a crash during bundling.
        The audio copy or intermediate files may be partial — discard them.
        The source file in inbox/ is untouched and will be re-picked up
        on the next scan (after the stability window).
        """
        queue_dir = self._config.paths.queue_dir
        if not queue_dir.exists():
            return 0

        count = 0
        for job_dir in queue_dir.iterdir():
            if not job_dir.is_dir():
                continue
            if not (job_dir / "meta.json").exists():
                log.warning(
                    f"Removing incomplete queue job dir: {job_dir.name}",
                    extra={"job_id": job_dir.name},
                )
                shutil.rmtree(job_dir, ignore_errors=True)
                count += 1

        return count

    def _recover_processing_jobs(self, force: bool) -> int:
        """
        Move processing/ jobs to failed/.

        If force=True, move all of them (startup recovery).
        If force=False, move only those that have exceeded the timeout.
        """
        proc_dir = self._config.paths.processing_dir
        if not proc_dir.exists():
            return 0

        timeout_secs = self._config.runtime.processing_timeout_seconds
        now = datetime.now(timezone.utc)
        count = 0

        for job_dir in proc_dir.iterdir():
            if not job_dir.is_dir():
                continue

            meta_file = job_dir / "meta.json"
            if not meta_file.exists():
                # No meta.json in processing/ is unusual — treat as corrupt
                log.error(
                    f"processing/ job has no meta.json, moving to failed/: {job_dir.name}"
                )
                self._move_to_failed(job_dir, "no meta.json in processing dir")
                count += 1
                continue

            started_file = job_dir / "started.json"
            if not started_file.exists():
                if force:
                    # Claimed but never started — move back to queue
                    log.warning(
                        f"processing/ job has no started.json, requeuning: {job_dir.name}"
                    )
                    self._move_back_to_queue(job_dir)
                    count += 1
                continue

            # Parse started_at from started.json
            try:
                started_data = json.loads(started_file.read_text())
                started_at = _parse_iso(started_data.get("started_at", ""))
            except Exception as exc:
                log.error(
                    f"Cannot parse started.json for {job_dir.name}: {exc}"
                )
                if force:
                    self._move_to_failed(job_dir, f"cannot parse started.json: {exc}")
                    count += 1
                continue

            if started_at is None:
                if force:
                    self._move_to_failed(job_dir, "invalid started_at timestamp")
                    count += 1
                continue

            age_secs = (now - started_at).total_seconds()

            if force or age_secs > timeout_secs:
                reason = (
                    "startup recovery (orphaned process)"
                    if force
                    else f"timeout ({age_secs:.0f}s > {timeout_secs}s)"
                )
                log.warning(
                    f"Sweeping job to failed/: {job_dir.name} — {reason}",
                    extra={"job_id": job_dir.name},
                )
                self._move_to_failed(job_dir, reason)
                count += 1

        return count

    def _move_to_failed(self, job_dir: Path, reason: str) -> None:
        fingerprint = ""
        meta_file = job_dir / "meta.json"
        if self._registry and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                fingerprint = str(meta.get("fingerprint", "")).strip()
            except Exception:
                fingerprint = ""

        try:
            error_info = {
                "error": f"sweeper: {reason}",
                "failed_at": _utcnow(),
                "sweeper_pid": os.getpid(),
            }
            # Best-effort — don't block the move if this fails
            try:
                error_file = job_dir / "error.json"
                error_file.write_text(json.dumps(error_info, indent=2))
            except Exception:
                pass

            failed_dir = self._config.paths.failed_dir / job_dir.name
            job_dir.rename(failed_dir)
            if self._registry and fingerprint:
                self._registry.unregister(fingerprint)
        except OSError as exc:
            log.error(f"Sweeper could not move {job_dir.name} to failed/: {exc}")

    def _move_back_to_queue(self, job_dir: Path) -> None:
        try:
            queue_dir = self._config.paths.queue_dir / job_dir.name
            job_dir.rename(queue_dir)
        except OSError as exc:
            log.error(f"Sweeper could not requeue {job_dir.name}: {exc}")
