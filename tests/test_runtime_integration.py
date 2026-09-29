from pathlib import Path

from whisper_ingest.bundler import JobBundler
from whisper_ingest.config import Config, PathsConfig, PolicyConfig, RuntimeConfig
from whisper_ingest.fingerprint import FingerprintRegistry
from whisper_ingest.scanner import InboxScanner
from whisper_ingest.worker import Worker


def test_dry_run_ingests_once_and_delivers_complete_output(tmp_path: Path) -> None:
    paths = PathsConfig(
        inbox_root=tmp_path / "inbox",
        work_root=tmp_path / "work",
        output_root=tmp_path / "output",
    )
    config = Config(
        paths=paths,
        runtime=RuntimeConfig(dry_run=True, file_stable_seconds=0),
        policies={"meeting": PolicyConfig(language="en", model="tiny")},
    )
    paths.ensure_dirs()
    inbox = paths.inbox_root / "meeting"
    inbox.mkdir(parents=True)
    (inbox / "sample.wav").write_bytes(b"example audio")

    scanner = InboxScanner(config)
    assert scanner.scan() == []
    ready = scanner.scan()
    assert len(ready) == 1

    registry = FingerprintRegistry()
    bundler = JobBundler(config, registry)
    assert len(bundler.bundle_all(ready)) == 1
    assert bundler.bundle_all(ready) == []

    assert Worker(config, registry).drain_queue() == 1
    output = paths.output_root / "meeting" / "sample.srt"
    assert "[DRY-RUN]" in output.read_text()
    assert not list(output.parent.glob("*.tmp"))
    assert len(list(paths.done_dir.iterdir())) == 1
