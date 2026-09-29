# STT Ingest Runtime

A small, crash-recoverable job runtime for speech-to-text workloads.

The project focuses on the systems problem around transcription: deciding when an input is ready, preventing accidental duplicate work, claiming jobs safely, recovering from crashes, and ensuring consumers never observe partial output.

It deliberately uses an inspectable filesystem state machine instead of a database or message broker.

## Why this project exists

A speech-to-text CLI can transcribe a file. A reliable runtime must also answer:

- Is the input still being uploaded?
- Has this file already been accepted?
- Which process owns the job now?
- What happens if the process dies midway?
- Can a consumer see a half-written subtitle?
- Can an operator understand the state without a separate database?

STT Ingest Runtime addresses those operational concerns with explicit state transitions and simple filesystem contracts.

## Architecture

```text
inbox/{policy}/
      │
      │ stable-file detection
      ▼
  fingerprint
      │
      │ deduplication
      ▼
┌─────────────┐
│ queue/{id}  │
└──────┬──────┘
       │ atomic rename = claim
       ▼
┌──────────────────┐
│ processing/{id}  │
└────────┬─────────┘
         │
     ┌───┴────┐
     │        │
     ▼        ▼
 done/{id}  failed/{id}
     │
     ▼
output/{policy}/{file}.srt
```

Core components:

| Component | Responsibility |
| --- | --- |
| Scanner | Detects files whose size has remained stable long enough to be considered ready |
| Fingerprint registry | Rebuilds known job fingerprints from filesystem state |
| Bundler | Creates a self-contained job bundle and writes metadata last as a validity sentinel |
| Worker | Claims queued jobs with an atomic rename, executes STT, and delivers output |
| Sweeper | Cleans incomplete bundles and recovers orphaned or timed-out processing jobs |

## Reliability model

### Explicit states

Every valid job is represented by a directory in exactly one of:

```text
queue/
processing/
done/
failed/
```

State transitions use filesystem renames on the local work filesystem.

### Atomic job claiming

A worker claims a job by renaming its directory from `queue/` to `processing/`. On POSIX local filesystems, rename is atomic, so a job cannot be simultaneously claimed through two successful renames.

### Validity sentinel

`meta.json` is written last when a job bundle is created. A queue directory without `meta.json` is treated as an incomplete bundle and cleaned on startup.

### Crash recovery

On startup, the sweeper examines processing state before the main loop begins. Orphaned or malformed work is made explicit rather than silently disappearing.

### Partial-output protection

Final output is delivered using a temporary file followed by rename, so a consumer should not observe an intentionally published partial SRT file.

### Deduplication

The runtime uses a lightweight content fingerprint based on file size and a prefix of file content. This is designed to avoid common accidental re-ingestion without requiring a full-file hash for large audio files.

## Configuration model

Each first-level inbox directory is a `policy_key`.

The names are arbitrary and are mapped to STT settings in configuration:

```yaml
policies:
  meeting:
    language: en
    model: small
    prompt_file: ""

  lecture:
    language: ko
    model: small
    prompt_file: ""

  podcast:
    language: ja
    model: small
    prompt_file: ""
```

This keeps routing policy outside the runtime logic.

## Quick start

Requirements:

- Python 3.10+
- PyYAML
- optional Whisper-compatible CLI for real transcription

Clone the repository and install development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Start with the portable example configuration:

```bash
cp config.example.yaml config.yaml
python run.py --config config.yaml --dry-run --once
```

This command checks configuration, recovery, and one scan cycle without requiring a model or GPU. With an empty inbox it does not create a transcription. For a full dry-run example, create a `meeting/` directory under `var/inbox`, add an audio file, and run without `--once` long enough for the stability interval to pass.

For real transcription, configure a compatible STT CLI and disable dry-run mode.

## Filesystem contract

Example:

```text
var/
├── inbox/
│   ├── meeting/
│   ├── lecture/
│   └── podcast/
├── work/
│   ├── queue/
│   ├── processing/
│   ├── done/
│   └── failed/
└── output/
    ├── meeting/
    ├── lecture/
    └── podcast/
```

The work directory must be on a filesystem where the rename semantics assumed by the runtime are valid. Do not rely on the same guarantees across arbitrary network or FUSE filesystems.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

The test suite covers core behavior including configuration defaults, fingerprint reconstruction, scanner safety, size guardrails, failure-state handling, and output delivery.

## Design choices

### Filesystem state instead of a database

The runtime is intentionally optimized for a small single-worker deployment where inspectability and recovery simplicity matter more than distributed throughput.

A database or broker would enable more sophisticated concurrency, but would also introduce another state system that must remain consistent with the audio files and generated outputs.

### Polling instead of filesystem events

The scanner uses periodic polling so the runtime does not depend on local-only file notification semantics. This is useful when the inbox may be backed by a mounted share.

### Single worker

The current design is deliberately single-worker. Multi-worker execution would require revisiting fingerprint registration and concurrency guarantees rather than assuming the current in-memory registry is sufficient.

## Limitations

- This is not a distributed queue.
- The current runtime is designed around one worker.
- The lightweight fingerprint is not a full-file content hash. Equal-size files with the same first 64 KiB can be treated as duplicates even if later bytes differ.
- Size stability does not detect an in-place upload that rewrites bytes without changing the file size.
- Atomicity assumptions apply to state transitions on the local work filesystem.
- If the process stops after publishing output but before moving the job to `done/`, recovery may mark the job failed even though a complete output file exists. Inspect the output before re-ingesting.
- STT model quality, GPU scheduling, and distributed inference are outside the core runtime's reliability guarantees.

## Project scope

The public repository contains only the portable runtime and generic examples.

Production deployment configuration, machine-specific integrations, personal routing policies, and private infrastructure are intentionally kept outside this repository.

## License

MIT.
