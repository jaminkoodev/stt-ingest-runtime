"""
models.py — Core data models for jobs and their metadata.

JobMeta is persisted to disk as meta.json inside each job directory.
It is the single source of truth for a job's identity and lifecycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class JobMeta:
    """Serialisable metadata for one STT job. Written as meta.json."""

    job_id: str
    source_path: str           # original path in inbox_root (informational)
    source_filename: str       # original basename
    source_size: int           # bytes, at time of pickup
    fingerprint: str           # sha256:<hex> of size+content prefix
    policy_key: str
    policy: Dict[str, Any]     # snapshot of PolicyConfig at queue time
    created_at: str            # ISO-8601 UTC

    # Set when job enters processing state
    started_at: Optional[str] = None
    started_pid: Optional[int] = None

    # Set on terminal states
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobMeta":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class Job:
    """Runtime handle for a job bundle on disk."""

    job_id: str
    job_dir: Path
    meta: JobMeta

    @property
    def input_audio(self) -> Path:
        return self.job_dir / "input.audio"

    @property
    def meta_file(self) -> Path:
        return self.job_dir / "meta.json"

    @property
    def started_file(self) -> Path:
        return self.job_dir / "started.json"

    @property
    def output_srt(self) -> Path:
        return self.job_dir / "output.srt"

    @property
    def error_file(self) -> Path:
        return self.job_dir / "error.json"
