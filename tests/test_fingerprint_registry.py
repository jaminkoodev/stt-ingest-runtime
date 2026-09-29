import json
from pathlib import Path

from whisper_ingest.fingerprint import FingerprintRegistry


def _write_meta(job_dir: Path, fingerprint: str) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "meta.json").write_text(json.dumps({"fingerprint": fingerprint}))


def test_load_from_work_root_excludes_failed_jobs(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    _write_meta(work_root / "queue" / "job-q", "sha256:q")
    _write_meta(work_root / "processing" / "job-p", "sha256:p")
    _write_meta(work_root / "done" / "job-d", "sha256:d")
    _write_meta(work_root / "failed" / "job-f", "sha256:f")

    registry = FingerprintRegistry()
    loaded = registry.load_from_work_root(work_root)

    assert loaded == 3
    assert registry.is_known("sha256:q")
    assert registry.is_known("sha256:p")
    assert registry.is_known("sha256:d")
    assert not registry.is_known("sha256:f")
