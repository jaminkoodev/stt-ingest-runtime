"""
config.py — Configuration loading and dataclasses.

Single source of truth: config.yaml.
All paths are resolved to absolute Path objects at load time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import yaml


@dataclass
class PolicyConfig:
    """STT policy: defines model, language, and optional prompt for a drop folder."""
    language: str
    model: str
    prompt_file: str = ""
    whisper_args: list[str] = field(default_factory=list)
    postprocess_enabled: bool = True
    postprocess_repeat_min_consecutive: int = 3
    postprocess_repeat_min_segment_seconds: float = 8.0
    postprocess_repeat_max_text_chars: int = 60
    postprocess_micro_repeat_min_consecutive: int = 8
    postprocess_micro_repeat_max_segment_seconds: float = 1.5
    postprocess_micro_repeat_max_text_chars: int = 12
    postprocess_micro_repeat_max_unique_chars: int = 2
    postprocess_micro_repeat_min_total_seconds: float = 4.0
    postprocess_hallucination_phrases: list[str] = field(default_factory=list)
    postprocess_hallucination_min_segment_seconds: float = 20.0
    postprocess_hallucination_short_file_max_segments: int = 2
    postprocess_hallucination_short_file_max_total_seconds: float = 8.0

    def __post_init__(self) -> None:
        if not self.language:
            raise ValueError("PolicyConfig.language must not be empty")
        if not self.model:
            raise ValueError("PolicyConfig.model must not be empty")
        if not isinstance(self.whisper_args, list):
            raise ValueError("PolicyConfig.whisper_args must be a list of strings")
        for item in self.whisper_args:
            if not isinstance(item, str):
                raise ValueError("PolicyConfig.whisper_args must be a list of strings")
        if self.postprocess_repeat_min_consecutive < 2:
            raise ValueError("PolicyConfig.postprocess_repeat_min_consecutive must be >= 2")
        if self.postprocess_repeat_min_segment_seconds < 0:
            raise ValueError("PolicyConfig.postprocess_repeat_min_segment_seconds must be >= 0")
        if self.postprocess_repeat_max_text_chars <= 0:
            raise ValueError("PolicyConfig.postprocess_repeat_max_text_chars must be > 0")
        if self.postprocess_micro_repeat_min_consecutive < 2:
            raise ValueError("PolicyConfig.postprocess_micro_repeat_min_consecutive must be >= 2")
        if self.postprocess_micro_repeat_max_segment_seconds < 0:
            raise ValueError("PolicyConfig.postprocess_micro_repeat_max_segment_seconds must be >= 0")
        if self.postprocess_micro_repeat_max_text_chars <= 0:
            raise ValueError("PolicyConfig.postprocess_micro_repeat_max_text_chars must be > 0")
        if self.postprocess_micro_repeat_max_unique_chars <= 0:
            raise ValueError("PolicyConfig.postprocess_micro_repeat_max_unique_chars must be > 0")
        if self.postprocess_micro_repeat_min_total_seconds < 0:
            raise ValueError("PolicyConfig.postprocess_micro_repeat_min_total_seconds must be >= 0")
        if not isinstance(self.postprocess_hallucination_phrases, list):
            raise ValueError("PolicyConfig.postprocess_hallucination_phrases must be a list of strings")
        for item in self.postprocess_hallucination_phrases:
            if not isinstance(item, str):
                raise ValueError("PolicyConfig.postprocess_hallucination_phrases must be a list of strings")
        if self.postprocess_hallucination_min_segment_seconds < 0:
            raise ValueError("PolicyConfig.postprocess_hallucination_min_segment_seconds must be >= 0")
        if self.postprocess_hallucination_short_file_max_segments < 0:
            raise ValueError("PolicyConfig.postprocess_hallucination_short_file_max_segments must be >= 0")
        if self.postprocess_hallucination_short_file_max_total_seconds < 0:
            raise ValueError("PolicyConfig.postprocess_hallucination_short_file_max_total_seconds must be >= 0")


@dataclass
class PathsConfig:
    inbox_root: Path
    work_root: Path
    output_root: Path

    @property
    def queue_dir(self) -> Path:
        return self.work_root / "queue"

    @property
    def processing_dir(self) -> Path:
        return self.work_root / "processing"

    @property
    def done_dir(self) -> Path:
        return self.work_root / "done"

    @property
    def failed_dir(self) -> Path:
        return self.work_root / "failed"

    def ensure_dirs(self) -> None:
        """Create all required work directories."""
        for d in (self.queue_dir, self.processing_dir, self.done_dir, self.failed_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)


@dataclass
class RuntimeConfig:
    scan_interval_seconds: int = 30
    processing_timeout_seconds: int = 21600
    worker_count: int = 1          # MVP: always 1
    dry_run: bool = False
    file_stable_seconds: float = 10.0  # seconds a file must be size-stable before pickup
    max_audio_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB guardrail before bundling
    # "local": run whisper CLI on this host
    # "remote": execute whisper on a SSH host and transfer only this job's WAV/SRT
    whisper_mode: str = "local"
    whisper_binary: str = "whisper"
    whisper_ssh_host: str = ""
    whisper_ssh_user: str = ""
    whisper_ssh_port: int = 22
    whisper_ssh_key: str = ""
    whisper_connect_timeout_seconds: int = 20
    whisper_remote_path: str = ""
    whisper_remote_workdir: str = "/tmp/stt-ingest-runtime"

    def __post_init__(self) -> None:
        if self.whisper_mode not in {"local", "remote"}:
            raise ValueError("runtime.whisper_mode must be either 'local' or 'remote'")
        if self.whisper_mode == "remote" and not self.whisper_ssh_host:
            raise ValueError("runtime.whisper_mode=remote requires runtime.whisper_ssh_host")
        if self.whisper_ssh_port <= 0 or self.whisper_ssh_port > 65535:
            raise ValueError("runtime.whisper_ssh_port must be a valid TCP port")


@dataclass
class Config:
    paths: PathsConfig
    runtime: RuntimeConfig
    policies: Dict[str, PolicyConfig]
    _source_path: str = field(default="", compare=False, repr=False)


def load_config(path: str = "config.yaml", overrides: dict | None = None) -> Config:
    """
    Load and validate config from YAML file.

    overrides: dict of dotted keys → values applied after file load.
    Example: {"runtime.dry_run": True}
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        for key, val in overrides.items():
            _set_nested(raw, key.split("."), val)

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        inbox_root=_resolve_path(paths_raw.get("inbox_root", ""), p.parent),
        work_root=_resolve_path(paths_raw.get("work_root", ""), p.parent),
        output_root=_resolve_path(paths_raw.get("output_root", ""), p.parent),
    )

    rt_raw = raw.get("runtime", {})
    runtime = RuntimeConfig(
        scan_interval_seconds=int(rt_raw.get("scan_interval_seconds", 30)),
        processing_timeout_seconds=int(rt_raw.get("processing_timeout_seconds", 21600)),
        worker_count=int(rt_raw.get("worker_count", 1)),
        dry_run=bool(rt_raw.get("dry_run", False)),
        file_stable_seconds=float(rt_raw.get("file_stable_seconds", 10.0)),
        max_audio_bytes=int(rt_raw.get("max_audio_bytes", 2 * 1024 * 1024 * 1024)),
        whisper_mode=str(rt_raw.get("whisper_mode", "local")).strip().lower(),
        whisper_binary=str(rt_raw.get("whisper_binary", "whisper")),
        whisper_ssh_host=str(rt_raw.get("whisper_ssh_host", "")),
        whisper_ssh_user=str(rt_raw.get("whisper_ssh_user", "")),
        whisper_ssh_port=int(rt_raw.get("whisper_ssh_port", 22)),
        whisper_ssh_key=str(rt_raw.get("whisper_ssh_key", "")),
        whisper_connect_timeout_seconds=int(rt_raw.get("whisper_connect_timeout_seconds", 20)),
        whisper_remote_path=str(rt_raw.get("whisper_remote_path", "")),
        whisper_remote_workdir=str(rt_raw.get("whisper_remote_workdir", "/tmp/stt-ingest-runtime")),
    )

    policies_raw = raw.get("policies", {})
    if not policies_raw:
        raise ValueError("config.yaml must define at least one policy under 'policies:'")

    policies: Dict[str, PolicyConfig] = {}
    for key, pol in policies_raw.items():
        if not isinstance(pol, dict):
            raise ValueError(f"Policy '{key}' must be a mapping")
        policies[key] = PolicyConfig(
            language=pol.get("language", ""),
            model=pol.get("model", ""),
            prompt_file=_resolve_optional_path(pol.get("prompt_file", ""), p.parent),
            whisper_args=_as_string_list(pol.get("whisper_args", [])),
            postprocess_enabled=bool(pol.get("postprocess_enabled", True)),
            postprocess_repeat_min_consecutive=int(pol.get("postprocess_repeat_min_consecutive", 3)),
            postprocess_repeat_min_segment_seconds=float(pol.get("postprocess_repeat_min_segment_seconds", 8.0)),
            postprocess_repeat_max_text_chars=int(pol.get("postprocess_repeat_max_text_chars", 60)),
            postprocess_micro_repeat_min_consecutive=int(pol.get("postprocess_micro_repeat_min_consecutive", 8)),
            postprocess_micro_repeat_max_segment_seconds=float(pol.get("postprocess_micro_repeat_max_segment_seconds", 1.5)),
            postprocess_micro_repeat_max_text_chars=int(pol.get("postprocess_micro_repeat_max_text_chars", 12)),
            postprocess_micro_repeat_max_unique_chars=int(pol.get("postprocess_micro_repeat_max_unique_chars", 2)),
            postprocess_micro_repeat_min_total_seconds=float(pol.get("postprocess_micro_repeat_min_total_seconds", 4.0)),
            postprocess_hallucination_phrases=_as_string_list(pol.get("postprocess_hallucination_phrases", [])),
            postprocess_hallucination_min_segment_seconds=float(pol.get("postprocess_hallucination_min_segment_seconds", 20.0)),
            postprocess_hallucination_short_file_max_segments=int(pol.get("postprocess_hallucination_short_file_max_segments", 2)),
            postprocess_hallucination_short_file_max_total_seconds=float(pol.get("postprocess_hallucination_short_file_max_total_seconds", 8.0)),
        )

    return Config(
        paths=paths,
        runtime=runtime,
        policies=policies,
        _source_path=str(p),
    )


def _resolve_path(raw: str, base: Path) -> Path:
    if not raw:
        raise ValueError(f"Missing required path in config")
    p = Path(raw)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _resolve_optional_path(raw: str, base: Path) -> str:
    """Resolve optional file-like paths. Empty string remains empty.

    Prompt files are optional per policy but should be interpreted relative to the
    config file when configured as relative paths.
    """
    if not raw:
        return ""
    p = Path(raw)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def _as_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        raise ValueError("PolicyConfig.whisper_args must be a list")
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("PolicyConfig.whisper_args must be a list of strings")
    return raw


def _set_nested(d: dict, keys: list, value) -> None:
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
