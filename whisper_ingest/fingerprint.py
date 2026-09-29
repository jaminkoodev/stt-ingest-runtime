"""
fingerprint.py — Content-based deduplication fingerprint.

Algorithm
---------
SHA-256 over the concatenation of:
  - file size as decimal string + colon separator
  - first CHUNK_SIZE bytes of the file content

Rationale:
  - We do NOT hash the full file to avoid reading GBs of audio on every scan.
  - Including file size ensures that two files with identical first-chunk but
    different total lengths are not considered duplicates (e.g. truncated files).
  - We do NOT include the filename or mtime because the same audio may be
    renamed or re-copied. Content identity is what matters for dedup.
  - Same-size files with an identical first 64 KiB can collide. This is a
    fast accidental-duplicate guard, not a full content identity check.

FingerprintRegistry
-------------------
In-memory set, rebuilt on startup by scanning active/successful job directories.
No external DB or index file is required.

Crash safety: the fingerprint is committed to the registry only AFTER
meta.json is written to disk. On restart, we rebuild from meta.json files
so no fingerprint is lost.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Set

CHUNK_SIZE = 64 * 1024  # 64 KiB


def compute(path: Path) -> str:
    """Return a sha256:<hex> fingerprint for the file at path."""
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(f"{size}:".encode())
    with open(path, "rb") as fh:
        chunk = fh.read(CHUNK_SIZE)
    h.update(chunk)
    return f"sha256:{h.hexdigest()}"


class FingerprintRegistry:
    """
    Thread-unsafe, in-memory registry of known fingerprints.

    For the single-worker MVP this is sufficient. A multi-worker
    extension would require a file-based lock around is_known + register.
    """

    def __init__(self) -> None:
        self._known: Set[str] = set()

    def load_from_work_root(self, work_root: Path) -> int:
        """
        Rebuild registry by scanning meta.json files in active/success states.

        Failed jobs are intentionally excluded so operators can re-ingest
        the same source without manual cleanup/restart.

        Returns the number of fingerprints loaded.
        """
        loaded = 0
        for state in ("queue", "processing", "done"):
            state_dir = work_root / state
            if not state_dir.exists():
                continue
            for job_dir in state_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                meta_file = job_dir / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text())
                    fp = meta.get("fingerprint", "")
                    if fp:
                        self._known.add(fp)
                        loaded += 1
                except Exception:
                    pass  # malformed meta — sweeper will handle
        return loaded

    def is_known(self, fingerprint: str) -> bool:
        return fingerprint in self._known

    def register(self, fingerprint: str) -> None:
        self._known.add(fingerprint)

    def unregister(self, fingerprint: str) -> None:
        self._known.discard(fingerprint)

    def __len__(self) -> int:
        return len(self._known)
