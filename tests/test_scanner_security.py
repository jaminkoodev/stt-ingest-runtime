from pathlib import Path

from whisper_ingest.config import Config, PathsConfig, RuntimeConfig, PolicyConfig
from whisper_ingest.scanner import InboxScanner


def test_scanner_rejects_symlinked_audio_files(tmp_path: Path) -> None:
    config = Config(
        paths=PathsConfig(
            inbox_root=tmp_path / "inbox",
            work_root=tmp_path / "work",
            output_root=tmp_path / "output",
        ),
        runtime=RuntimeConfig(file_stable_seconds=0.0),
        policies={
            "meeting": PolicyConfig(language="en", model="tiny"),
        },
        _source_path=str(tmp_path / "config.yaml"),
    )

    policy_dir = config.paths.inbox_root / "meeting"
    policy_dir.mkdir(parents=True)

    outside = tmp_path / "outside.txt"
    outside.write_text("external")
    linked = policy_dir / "linked.wav"
    linked.symlink_to(outside)

    local = policy_dir / "local.wav"
    local.write_bytes(b"local")

    scanner = InboxScanner(config)
    assert not scanner._is_file_in_policy_dir(linked, policy_dir)
    assert scanner._is_file_in_policy_dir(local, policy_dir)
