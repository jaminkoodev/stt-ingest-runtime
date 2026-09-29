#!/usr/bin/env python3
"""
Entry point for stt-ingest-runtime.

Usage:
    python run.py [--config config.yaml] [--dry-run] [--once]

See run.py --help or README.md for full documentation.
"""
import sys
from whisper_ingest.cli import main

if __name__ == "__main__":
    sys.exit(main())
