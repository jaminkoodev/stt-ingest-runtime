"""
paths.py — Atomic file system helpers.

All writes that must be crash-safe use the write-to-temp + rename pattern.
rename(2) is atomic on POSIX for same-filesystem moves.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """
    Persist a dict as JSON atomically.

    Writes to a sibling .tmp file first, then renames to final path.
    Guarantees the reader never observes a partial write.
    """
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def copy_to_atomic(src: Path, dest_dir: Path, dest_name: str) -> Path:
    """
    Copy src into dest_dir as dest_name atomically.

    Uses a .tmp sibling to avoid readers seeing partial files.
    Returns the final destination path.
    """
    final = dest_dir / dest_name
    tmp = dest_dir / f".{dest_name}.tmp"
    try:
        shutil.copy2(src, tmp)
        tmp.rename(final)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return final


def atomic_rename(src: Path, dst: Path) -> None:
    """
    Rename src to dst.

    On Linux, os.rename / Path.rename is atomic for same-filesystem moves.
    Will raise OSError if dst already exists on some filesystems.
    """
    src.rename(dst)


def rmdir_safe(path: Path) -> None:
    """Remove directory tree, ignoring errors (best-effort cleanup)."""
    try:
        shutil.rmtree(path)
    except Exception:
        pass
