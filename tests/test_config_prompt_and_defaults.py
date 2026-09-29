from pathlib import Path

import yaml

from whisper_ingest.config import load_config


def test_load_config_uses_code_defaults_and_resolves_prompt_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir()
    (prompts_root / "meeting.txt").write_text("hello")

    cfg = {
        "paths": {
            "inbox_root": "inbox",
            "work_root": "work",
            "output_root": "output",
        },
        "policies": {
            "meeting": {
                "language": "en",
                "model": "tiny",
                "prompt_file": "prompts/meeting.txt",
            },
        },
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    loaded = load_config(str(config_path))

    assert loaded.runtime.processing_timeout_seconds == 21600
    assert loaded.runtime.file_stable_seconds == 10.0
    assert loaded.runtime.max_audio_bytes == 2 * 1024 * 1024 * 1024
    assert loaded.policies["meeting"].prompt_file == str((tmp_path / "prompts" / "meeting.txt").resolve())
