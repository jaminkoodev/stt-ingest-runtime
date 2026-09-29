"""
log.py — Structured JSON logging setup.

All log records are emitted as single-line JSON to stdout.
stderr is reserved for Python's own exception traces.

Design: stdlib-only, no external dependencies.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional


class _JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    LEVEL_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        obj: dict = {
            "ts": self.formatTime(record, datefmt=None),
            "level": self.LEVEL_MAP.get(record.levelno, record.levelname),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        # Carry any extra fields attached via logger.info(..., extra={...})
        for key in ("job_id", "policy_key", "source_path", "fingerprint"):
            val = record.__dict__.get(key)
            if val is not None:
                obj[key] = val
        return json.dumps(obj, ensure_ascii=False)

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:  # noqa: N802
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created))


def setup_logging(level: str = "INFO", logger_name: str = "wir") -> logging.Logger:
    """
    Configure root logger with JSON formatter.
    Returns the named application logger.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(numeric)
    # Remove default handlers added by other imports
    root.handlers.clear()
    root.addHandler(handler)

    return logging.getLogger(logger_name)


def get_logger(name: str = "wir") -> logging.Logger:
    return logging.getLogger(name)
