"""
cli.py — Argument parsing and main runtime loop.

Entry points:
  - run.py (direct execution)
  - `stt-ingest-runtime` (installed console script)
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
import shutil

from .config import load_config
from .bundler import JobBundler
from .fingerprint import FingerprintRegistry
from .log import get_logger, setup_logging
from .scanner import InboxScanner
from .sweeper import Sweeper
from .worker import Worker


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="stt-ingest-runtime",
        description="Manual-drop Whisper STT runtime with file-based state machine.",
    )
    p.add_argument(
        "--config", "-c",
        default="config.yaml",
        metavar="FILE",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Override: run mock STT instead of real whisper",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Process all currently queued jobs and exit (no daemon loop)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    log = setup_logging(args.log_level)
    log.info("stt-ingest-runtime starting up")

    # Build config overrides from CLI flags
    overrides = {}
    if args.dry_run:
        overrides["runtime.dry_run"] = True

    try:
        config = load_config(args.config, overrides=overrides or None)
    except Exception as exc:
        log.error(f"Failed to load config: {exc}", exc_info=True)
        return 1

    if config.runtime.dry_run:
        log.info("DRY-RUN mode enabled — whisper will not be called")

    if not config.runtime.dry_run:
        _preflight_runtime_check(config, log)

    log.info(
        f"Config loaded: inbox={config.paths.inbox_root} "
        f"work={config.paths.work_root} "
        f"policies={list(config.policies.keys())}"
    )

    # Create work directories
    config.paths.ensure_dirs()

    # Build fingerprint registry from existing jobs
    registry = FingerprintRegistry()
    loaded = registry.load_from_work_root(config.paths.work_root)
    log.info(f"FingerprintRegistry loaded {loaded} known fingerprints")

    # Run startup crash recovery
    sweeper = Sweeper(config, registry)
    sweeper.startup_recovery()

    scanner = InboxScanner(config)
    bundler = JobBundler(config, registry)
    worker = Worker(config, registry)

    # ------------------------------------------------------------------
    # Graceful shutdown on SIGTERM / SIGINT
    # ------------------------------------------------------------------
    _shutdown = [False]

    def _handle_signal(signum, frame):
        log.info(f"Signal {signum} received — shutting down after current job")
        _shutdown[0] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    log.info(
        f"Main loop started "
        f"(scan_interval={config.runtime.scan_interval_seconds}s, "
        f"timeout={config.runtime.processing_timeout_seconds}s)"
    )

    try:
        while not _shutdown[0]:
            # 1. Periodic timeout sweep
            sweeper.sweep()

            # 2. Scan inbox for stable, new audio files
            ready_files = scanner.scan()
            if ready_files:
                log.debug(f"Scanner found {len(ready_files)} stable file(s)")
                bundler.bundle_all(ready_files)

            # 3. Drain the job queue (one job at a time, MVP)
            processed = worker.drain_queue()
            if processed:
                log.info(f"Processed {processed} job(s) this cycle")

            if args.once:
                log.info("--once flag: exiting after first cycle")
                break

            if not _shutdown[0]:
                log.debug(f"Sleeping {config.runtime.scan_interval_seconds}s")
                # Sleep in small increments to respond to signals quickly
                deadline = time.monotonic() + config.runtime.scan_interval_seconds
                while time.monotonic() < deadline and not _shutdown[0]:
                    time.sleep(min(1.0, deadline - time.monotonic()))

    except Exception as exc:
        log.error(f"Unhandled exception in main loop: {exc}", exc_info=True)
        return 1

    log.info("stt-ingest-runtime stopped cleanly")
    return 0


def _preflight_runtime_check(config, log) -> None:
    if config.runtime.whisper_mode == "local":
        binary = (config.runtime.whisper_binary or "").strip()
        if not binary:
            log.error("runtime.whisper_binary is required when whisper_mode=local")
            raise RuntimeError("runtime.whisper_binary is empty in config.yaml")

        binary_path = Path(binary)
        if binary_path.is_absolute():
            if not binary_path.exists():
                raise RuntimeError(
                    f"runtime.whisper_binary does not exist: {binary_path}"
                )
            if not os.access(binary_path, os.X_OK):
                raise RuntimeError(
                    f"runtime.whisper_binary is not executable: {binary_path}"
                )
            return

        if not shutil.which(binary):
            raise RuntimeError(
                f"runtime.whisper_binary not found in PATH: {binary}. "
                "Set runtime.whisper_binary to an absolute path "
                "or switch runtime.whisper_mode to remote."
            )
        return

    if config.runtime.whisper_mode == "remote":
        if not config.runtime.whisper_ssh_host:
            raise RuntimeError(
                "whisper_mode=remote requires runtime.whisper_ssh_host"
            )
        return

    raise RuntimeError(f"Unsupported runtime.whisper_mode: {config.runtime.whisper_mode}")
