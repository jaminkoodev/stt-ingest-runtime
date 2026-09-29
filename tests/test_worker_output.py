from pathlib import Path

import pytest

from whisper_ingest.config import Config, PathsConfig, RuntimeConfig, PolicyConfig
from whisper_ingest.models import Job, JobMeta
from whisper_ingest.worker import Worker


def _make_job(config: Config) -> Job:
    job_dir = config.paths.processing_dir / "job-1"
    job_dir.mkdir(parents=True, exist_ok=True)

    meta = JobMeta(
        job_id="job-1",
        source_path="/inbox/meeting/source.wav",
        source_filename="source.wav",
        source_size=12,
        fingerprint="sha256:abc",
        policy_key="meeting",
        policy={"language": "en", "model": "tiny", "prompt_file": ""},
        created_at="2024-01-01T00:00:00Z",
    )
    (job_dir / "output.srt").write_text("new transcript")
    return Job(job_id="job-1", job_dir=job_dir, meta=meta)


def test_deliver_output_overwrites_with_replace(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.joinpath("meeting").mkdir(parents=True)
    (output_root / "meeting" / "source.srt").write_text("old transcript")

    config = Config(
        paths=PathsConfig(
            inbox_root=tmp_path / "inbox",
            work_root=tmp_path / "work",
            output_root=output_root,
        ),
        runtime=RuntimeConfig(),
        policies={
            "meeting": PolicyConfig(language="en", model="tiny"),
        },
        _source_path=str(tmp_path / "config.yaml"),
    )

    config.paths.processing_dir.mkdir(parents=True, exist_ok=True)
    worker = Worker(config)
    job = _make_job(config)
    worker._deliver_output(job)

    final = output_root / "meeting" / "source.srt"
    assert final.read_text() == "new transcript"


def test_deliver_output_rejects_symlinked_policy_dir(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (output_root / "meeting").symlink_to(target)

    config = Config(
        paths=PathsConfig(
            inbox_root=tmp_path / "inbox",
            work_root=tmp_path / "work",
            output_root=output_root,
        ),
        runtime=RuntimeConfig(),
        policies={
            "meeting": PolicyConfig(language="en", model="tiny"),
        },
        _source_path=str(tmp_path / "config.yaml"),
    )

    worker = Worker(config)
    config.paths.processing_dir.mkdir(parents=True, exist_ok=True)
    job = _make_job(config)

    with pytest.raises(RuntimeError, match="Policy output directory is a symlink"):
        worker._deliver_output(job)
