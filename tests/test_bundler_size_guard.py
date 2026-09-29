from pathlib import Path

from whisper_ingest.config import Config, PathsConfig, RuntimeConfig, PolicyConfig
from whisper_ingest.fingerprint import FingerprintRegistry
from whisper_ingest.bundler import JobBundler


def test_bundler_rejects_files_larger_than_limit(tmp_path: Path) -> None:
    config = Config(
        paths=PathsConfig(
            inbox_root=tmp_path / "inbox",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
        ),
        runtime=RuntimeConfig(max_audio_bytes=2),
        policies={
            "meeting": PolicyConfig(language="en", model="tiny"),
        },
        _source_path=str(tmp_path / "config.yaml"),
    )
    config.paths.work_root.mkdir(parents=True, exist_ok=True)
    registry = FingerprintRegistry()
    bundler = JobBundler(config, registry)

    audio_file = config.paths.inbox_root / "meeting" / "large.wav"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"123456")  # 6 > 2 bytes

    jobs = bundler.bundle_all([(audio_file, "meeting")])
    assert jobs == []
    assert not (config.paths.work_root / "queue").exists()
