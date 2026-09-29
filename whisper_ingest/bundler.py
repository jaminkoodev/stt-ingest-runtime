"""
bundler.py — Job bundler: dedup, copy, and queue creation.

A "job bundle" is a directory under work_root/queue/{job_id}/ that contains:
  - input.audio   : copy of the source audio
  - policy.key    : plaintext policy key (one line)
  - meta.json     : job metadata — the idempotency sentinel

Invariant: meta.json is written LAST. A job dir without meta.json is
incomplete and will be cleaned up by the sweeper on next startup.

Copy vs Hardlink decision:
  - We use shutil.copy2 (full data copy) rather than hardlinks.
  - Reason: Samba shares may not support hardlinks, and cross-device
    hardlinks are impossible. A copy is always portable and safe.
  - The audio file is removed from inbox ONLY by the operator (manual ingest
    model). We intentionally do NOT delete the source after copy.

Dedup:
  - Fingerprint (sha256 of size + first 64 KiB) is checked against the
    in-memory FingerprintRegistry before creating a job.
  - Registry is populated at startup from queue/processing/done meta.json files.
    failed/ jobs are excluded so operator-triggered re-ingest can proceed.
  - Fingerprint is registered AFTER meta.json is written, so a crash
    between these two steps is safe: on restart, meta.json provides the
    fingerprint to the registry, or the incomplete dir is cleaned up.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .config import Config
from .fingerprint import FingerprintRegistry, compute as fp_compute
from .log import get_logger
from .models import Job, JobMeta
from .paths import write_json_atomic

log = get_logger()


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_job_id(audio_path: Path, policy_key: str, fingerprint: str) -> str:
    """
    Generate a sortable, human-readable job ID.

    Format: YYYYMMDD_HHMMSS_{policy_key}_{fp_short}
    The fingerprint prefix makes collisions within the same second impossible.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fp_short = fingerprint.split(":")[-1][:8]
    safe_policy = policy_key[:20].replace("/", "_")
    return f"{ts}_{safe_policy}_{fp_short}"


class JobBundler:
    """Creates job bundles in queue/ for stable, non-duplicate inbox files."""

    def __init__(self, config: Config, registry: FingerprintRegistry) -> None:
        self._config = config
        self._registry = registry

    def bundle_all(self, audio_files: List[Tuple[Path, str]]) -> List[Job]:
        """
        Attempt to create job bundles for each (audio_path, policy_key) pair.
        Returns the list of successfully created jobs.
        """
        jobs: List[Job] = []
        for audio_path, policy_key in audio_files:
            try:
                job = self._bundle_one(audio_path, policy_key)
                if job is not None:
                    jobs.append(job)
            except Exception as exc:
                log.error(
                    f"Failed to bundle {audio_path.name}: {exc}",
                    extra={"source_path": str(audio_path), "policy_key": policy_key},
                    exc_info=True,
                )
        return jobs

    def _bundle_one(self, audio_path: Path, policy_key: str) -> Optional[Job]:
        # --- 1. Stat the file (may have disappeared between scan and now) ---
        try:
            stat = audio_path.stat()
        except FileNotFoundError:
            log.warning(f"Source file vanished before bundling: {audio_path}")
            return None

        # Validate source path safety (scan layer already validates, but race/tamper hardening is done here too).
        if not self._is_safe_source(audio_path, policy=self._config.paths.inbox_root / policy_key):
            log.warning(f"Skipping unsafe source path: {audio_path}")
            return None

        max_bytes = self._config.runtime.max_audio_bytes
        if stat.st_size > max_bytes:
            log.warning(
                "File rejected by size policy",
                extra={
                    "source_path": str(audio_path),
                    "source_size": stat.st_size,
                    "max_bytes": max_bytes,
                    "policy_key": policy_key,
                },
            )
            return None

        # --- 2. Compute fingerprint ---
        fingerprint = fp_compute(audio_path)

        # --- 3. Dedup check ---
        if self._registry.is_known(fingerprint):
            log.info(
                f"Duplicate detected, skipping: {audio_path.name}",
                extra={
                    "source_path": str(audio_path),
                    "fingerprint": fingerprint,
                    "policy_key": policy_key,
                },
            )
            return None

        # --- 4. Validate policy ---
        policy_cfg = self._config.policies.get(policy_key)
        if policy_cfg is None:
            log.error(f"Policy '{policy_key}' not found in config, skipping")
            return None

        # --- 5. Generate job ID and create queue dir ---
        job_id = _make_job_id(audio_path, policy_key, fingerprint)
        job_dir = self._config.paths.queue_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            # --- 6. Copy audio into job dir ---
            dest_audio = job_dir / "input.audio"
            tmp_audio = job_dir / ".input.audio.tmp"
            shutil.copy2(audio_path, tmp_audio)
            tmp_audio.rename(dest_audio)

            # --- 7. Write policy key ---
            (job_dir / "policy.key").write_text(policy_key + "\n")

            # --- 8. Build metadata ---
            meta = JobMeta(
                job_id=job_id,
                source_path=str(audio_path),
                source_filename=audio_path.name,
                source_size=stat.st_size,
                fingerprint=fingerprint,
                policy_key=policy_key,
                policy=asdict(policy_cfg),
                created_at=_utcnow(),
            )

            # --- 9. Write meta.json (idempotency sentinel — must be last) ---
            write_json_atomic(job_dir / "meta.json", meta.to_dict())

        except Exception:
            # Roll back partial job dir to avoid orphaned dirs
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        # --- 10. Register fingerprint (AFTER meta.json on disk) ---
        self._registry.register(fingerprint)

        log.info(
            f"Job queued: {job_id}",
            extra={
                "job_id": job_id,
                "source_path": str(audio_path),
                "fingerprint": fingerprint,
                "policy_key": policy_key,
            },
        )

        return Job(job_id=job_id, job_dir=job_dir, meta=meta)

    def _is_safe_source(self, audio_path: Path, policy: Path) -> bool:
        """Reject symlink sources and ensure the source is inside the policy directory."""
        try:
            if audio_path.is_symlink() or not audio_path.is_file():
                return False
            policy_root = policy.resolve()
            source = audio_path.resolve()
        except OSError:
            return False
        return source.is_relative_to(policy_root)
