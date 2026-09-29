"""
scanner.py — Inbox scanner with file-stability detection.

Scan Cycle
----------
1. Iterate inbox_root subdirectories (each subdir is a policy_key).
2. For each audio file, track (size, first_seen_timestamp) in memory.
3. A file is considered "stable" when its size hasn't changed for
   file_stable_seconds — this guards against in-progress Samba uploads.
4. Return stable (path, policy_key) pairs to the caller.

inotify is intentionally NOT used:
  - polling is simpler, portable, and crash-safe.
  - eliminates edge cases around inotify event delivery order.

Stability cache is in-memory. On restart it resets, meaning files
will wait another stability window before pickup — an acceptable tradeoff.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .config import Config
from .log import get_logger

log = get_logger()

# Audio extensions we will accept. Whisper handles all common formats.
AUDIO_EXTENSIONS: Set[str] = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg",
    ".mp4", ".mkv", ".webm", ".aac", ".opus",
}


class InboxScanner:
    """
    Scans inbox_root for audio files that are stable and ready for pickup.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        # path_str → (last_size, first_seen_at)
        self._stability: Dict[str, Tuple[int, float]] = {}

    def scan(self) -> List[Tuple[Path, str]]:
        """
        Return a list of (audio_path, policy_key) pairs ready for processing.

        A file is "ready" when:
          - Its parent directory matches a known policy key, AND
          - Its extension is in AUDIO_EXTENSIONS, AND
          - Its size has been stable for >= file_stable_seconds.
        """
        inbox = self._config.paths.inbox_root
        if not inbox.exists():
            log.warning("inbox_root does not exist, skipping scan", extra={"source_path": str(inbox)})
            return []

        stable_entries: List[Tuple[Path, str]] = []
        now = time.monotonic()

        for policy_dir in sorted(inbox.iterdir()):
            if not policy_dir.is_dir():
                continue
            if policy_dir.is_symlink():
                log.warning(
                    f"Ignoring policy directory symlink: {policy_dir}",
                    extra={"policy_dir": str(policy_dir)},
                )
                continue
            policy_key = policy_dir.name

            if policy_key not in self._config.policies:
                log.debug(
                    f"Ignoring unknown policy directory: {policy_key}",
                    extra={"policy_key": policy_key},
                )
                continue

            for audio_file in sorted(policy_dir.iterdir()):
                if not self._is_file_in_policy_dir(audio_file, policy_dir):
                    continue
                if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue

                if self._is_stable(audio_file, now):
                    stable_entries.append((audio_file, policy_key))

        # Evict stale entries from stability cache (files that disappeared)
        self._evict_gone_files(inbox)

        return stable_entries

    def _is_stable(self, path: Path, now: float) -> bool:
        key = str(path)
        try:
            current_size = path.stat().st_size
        except OSError:
            self._stability.pop(key, None)
            return False

        threshold = self._config.runtime.file_stable_seconds

        if key in self._stability:
            last_size, first_seen = self._stability[key]
            if current_size == last_size:
                age = now - first_seen
                if age >= threshold:
                    return True
                # Same size but not old enough yet
                return False
            else:
                # Size changed — reset the timer
                self._stability[key] = (current_size, now)
                return False
        else:
            # First time we see this file
            self._stability[key] = (current_size, now)
            return False

    def _is_file_in_policy_dir(self, path: Path, policy_dir: Path) -> bool:
        """
        Validate that path is a plain file located directly under policy_dir.

        This blocks symlink-based path traversal from inbox to attacker-controlled paths.
        """
        try:
            if path.is_symlink() or not path.is_file():
                return False
            policy_root = policy_dir.resolve()
            source = path.resolve()
        except OSError:
            return False
        return source.is_relative_to(policy_root)

    def _evict_gone_files(self, inbox: Path) -> None:
        """Remove stability entries for files that no longer exist."""
        gone = [k for k in self._stability if not Path(k).exists()]
        for k in gone:
            del self._stability[k]
