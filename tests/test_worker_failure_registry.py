from pathlib import Path

from whisper_ingest.config import Config, PathsConfig, PolicyConfig, RuntimeConfig
from whisper_ingest.fingerprint import FingerprintRegistry
from whisper_ingest.models import Job, JobMeta
from whisper_ingest.worker import Worker


def _make_job(config: Config, fingerprint: str) -> Job:
    job_dir = config.paths.processing_dir / "job-fail-1"
    job_dir.mkdir(parents=True, exist_ok=True)
    meta = JobMeta(
        job_id="job-fail-1",
        source_path="/inbox/sample.wav",
        source_filename="sample.wav",
        source_size=1,
        fingerprint=fingerprint,
        policy_key="meeting",
        policy={"language": "en", "model": "tiny", "prompt_file": ""},
        created_at="2024-01-01T00:00:00Z",
    )
    return Job(job_id=meta.job_id, job_dir=job_dir, meta=meta)


def test_fail_job_unregisters_fingerprint_when_moved_to_failed(tmp_path: Path) -> None:
    config = Config(
        paths=PathsConfig(
            inbox_root=tmp_path / "inbox",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
        ),
        runtime=RuntimeConfig(),
        policies={"meeting": PolicyConfig(language="en", model="tiny")},
        _source_path=str(tmp_path / "config.yaml"),
    )
    config.paths.processing_dir.mkdir(parents=True, exist_ok=True)
    config.paths.failed_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = "sha256:worker-fail"
    registry = FingerprintRegistry()
    registry.register(fingerprint)

    worker = Worker(config, registry)
    job = _make_job(config, fingerprint)
    worker._fail_job(job, "boom")

    assert not registry.is_known(fingerprint)
    assert (config.paths.failed_dir / "job-fail-1").exists()
