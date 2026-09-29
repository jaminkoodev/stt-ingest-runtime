"""
worker.py — Job processor: claim, run Whisper, deliver output.

State Transitions Performed Here
---------------------------------
  queue/{job_id}/ → processing/{job_id}/   (atomic rename = claim)
  processing/{job_id}/ → done/{job_id}/    (on success)
  processing/{job_id}/ → failed/{job_id}/  (on error)

Atomicity
---------
Job claim uses Path.rename() which maps to rename(2) on POSIX.
rename(2) is atomic: exactly one worker can rename a given source path.
In a multi-worker scenario, an OSError on rename means another worker
claimed the job first — we simply skip to the next.

Output Safety
-------------
Whisper writes output as {input_name}.srt inside the job dir.
We then do: srt_candidate → output.srt.tmp → output.srt (atomic rename)
Final delivery to output_root also uses temp+rename:
  output_root/{policy}/{stem}.srt.tmp → output_root/{policy}/{stem}.srt
This guarantees a reader of output_root never sees a partial file.

Crash Between output_root Write and done/ Move
----------------------------------------------
If we crash after writing to output_root but before renaming to done/,
the sweeper will move the job from processing/ to failed/ on next startup.
The file in output_root is already complete and correct — the only
"damage" is the job being marked failed rather than done. This is an
acceptable tradeoff: output is delivered, state tracking is imperfect.
Operators can inspect failed/ to confirm no audio was lost.
"""

from __future__ import annotations

import os
import json
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .fingerprint import FingerprintRegistry
from .log import get_logger
from .models import Job, JobMeta
from .paths import read_json, write_json_atomic

log = get_logger()


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Worker:
    """
    Single-threaded job processor.

    process_next() claims the oldest queued job and runs it to completion.
    Returns True if a job was found and processed (success or failure).
    Returns False if the queue is empty.
    """

    def __init__(self, config: Config, registry: Optional[FingerprintRegistry] = None) -> None:
        self._config = config
        self._registry = registry
        self._remote_capability_verified = False

    def drain_queue(self) -> int:
        """Process all queued jobs. Returns count of jobs processed."""
        count = 0
        while self.process_next():
            count += 1
        return count

    def process_next(self) -> bool:
        job = self._claim_next()
        if job is None:
            return False

        log.info(f"Processing job: {job.job_id}", extra={"job_id": job.job_id})
        try:
            self._run_job(job)
        except Exception as exc:
            log.error(
                f"Job failed: {job.job_id} — {exc}",
                extra={"job_id": job.job_id},
                exc_info=True,
            )
            self._fail_job(job, str(exc))

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _claim_next(self) -> Optional[Job]:
        """
        Atomically claim the oldest valid job from queue/.

        Iterates jobs in lexicographic order (job IDs begin with timestamp,
        so lexicographic ≈ FIFO). Uses rename(2) for atomic claim.
        """
        queue_dir = self._config.paths.queue_dir
        if not queue_dir.exists():
            return None

        for entry in sorted(queue_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta_file = entry / "meta.json"
            if not meta_file.exists():
                continue  # incomplete bundle — sweeper handles on restart

            target = self._config.paths.processing_dir / entry.name
            try:
                entry.rename(target)
            except OSError:
                # Another worker (or the sweeper) claimed it first
                continue

            try:
                meta_data = read_json(target / "meta.json")
                meta = JobMeta.from_dict(meta_data)
                return Job(job_id=entry.name, job_dir=target, meta=meta)
            except Exception as exc:
                log.error(f"Could not read meta.json for {entry.name}: {exc}")
                # Move back to queue so sweeper/operator can inspect
                try:
                    target.rename(entry)
                except OSError:
                    pass
                continue

        return None

    def _run_job(self, job: Job) -> None:
        # --- Write started.json ---
        started_info = {
            "started_at": _utcnow(),
            "pid": os.getpid(),
        }
        job.started_file.write_text(
            json.dumps(started_info, indent=2)
        )

        # --- Run STT ---
        if self._config.runtime.dry_run:
            self._run_mock(job)
        else:
            self._run_whisper(job)

        # --- Conservative SRT cleanup for repeated hallucination loops ---
        self._postprocess_output(job)

        # --- Deliver output atomically to output_root ---
        self._deliver_output(job)

        # --- Move job to done/ ---
        done_dir = self._config.paths.done_dir / job.job_id
        job.job_dir.rename(done_dir)

        log.info(
            f"Job done: {job.job_id}",
            extra={"job_id": job.job_id, "policy_key": job.meta.policy_key},
        )

    def _run_whisper(self, job: Job) -> None:
        """
        Execute the whisper CLI and produce output.srt in the job dir.
        """
        timeout = self._config.runtime.processing_timeout_seconds
        mode = self._config.runtime.whisper_mode

        if mode == "remote":
            self._run_whisper_remote(job, timeout)
        elif mode == "local":
            self._run_whisper_local(job, timeout)
        else:
            raise ValueError(f"Unsupported runtime.whisper_mode: {mode}")

        self._finalize_whisper_output(job)

    def _run_whisper_local(self, job: Job, timeout: int) -> None:
        policy = job.meta.policy
        prompt = self._load_prompt(policy)
        binary = self._resolve_whisper_binary(self._config.runtime.whisper_binary)
        cmd = self._build_whisper_command(
            binary,
            job.input_audio,
            job.job_dir,
            policy,
            prompt,
        )

        log.info(
            f"Running whisper (local): model={policy['model']} lang={policy['language']}",
            extra={"job_id": job.job_id},
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Local whisper binary not found while executing '{cmd[0]}'. "
                "Check runtime.whisper_binary in config.yaml."
            ) from exc

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
            raise RuntimeError(f"whisper exited {result.returncode}: {stderr_tail}")

    def _resolve_whisper_binary(self, binary: str) -> str:
        """
        Resolve and validate the local whisper binary.

        Raises a clear RuntimeError before subprocess invocation when the
        executable is missing, so we do not surface raw FileNotFoundError.
        """
        binary = (binary or "").strip()
        if not binary:
            raise RuntimeError("runtime.whisper_binary is empty in config.yaml")

        binary_path = Path(binary)
        if binary_path.is_absolute():
            if not binary_path.exists():
                raise RuntimeError(
                    f"runtime.whisper_binary does not exist: {binary_path}"
                )
            if not os.access(binary_path, os.X_OK):
                raise RuntimeError(
                    f"runtime.whisper_binary is not executable: {binary_path}"
                )
            return str(binary_path)

        resolved = shutil.which(binary)
        if not resolved:
            raise RuntimeError(
                f"Local whisper command not found in PATH: {binary}. "
                "Set runtime.whisper_binary to a full path (for example: /usr/bin/whisper) "
                "or switch runtime.whisper_mode to remote."
            )
        return resolved

    def _run_whisper_remote(self, job: Job, timeout: int) -> None:
        rt = self._config.runtime
        policy = job.meta.policy
        prompt = self._load_prompt(policy)

        remote_job_dir = Path(rt.whisper_remote_workdir) / job.job_id
        remote_input = remote_job_dir / job.input_audio.name
        remote_output = remote_job_dir / f"{job.input_audio.stem}.srt"
        target = self._ssh_target()
        ssh_opts = self._ssh_opts()

        try:
            prepare_cmd = [
                "ssh",
                *ssh_opts,
                target,
                f"mkdir -p {shlex.quote(str(remote_job_dir))}",
            ]
            self._run_checked(
                prepare_cmd,
                timeout=60,
                stage="Prepare remote job dir",
                check_stderr=True,
            )

            upload_cmd = [
                "scp",
                *self._scp_opts(),
                str(job.input_audio),
                f"{target}:{remote_input}",
            ]
            self._run_checked(
                upload_cmd,
                timeout=max(120, timeout),
                stage="Upload source audio",
                check_stderr=True,
            )

            self._ensure_remote_whisper_binary(target, ssh_opts)

            whisper_cmd = self._build_whisper_command(
                rt.whisper_binary,
                remote_input,
                remote_job_dir,
                policy,
                prompt,
            )
            remote_shell = self._build_remote_shell(
                " ".join(shlex.quote(part) for part in whisper_cmd)
            )

            remote_output_candidates = None
            last_error: str | None = None
            for attempt in range(1, 3):
                if attempt > 1:
                    log.warning(
                        f"Retrying remote whisper + output discovery (attempt {attempt})",
                        extra={"job_id": job.job_id},
                    )
                    # Re-upload audio + rerun whisper in case the first attempt
                    # failed before producing .srt outputs.
                    self._run_checked(
                        upload_cmd,
                        timeout=max(120, timeout),
                        stage="Re-upload source audio",
                        check_stderr=True,
                    )

                log.info(
                    f"Running whisper (remote): model={policy['model']} lang={policy['language']} host={rt.whisper_ssh_host}",
                    extra={"job_id": job.job_id, "attempt": attempt},
                )
                self._run_checked(
                    ["ssh", *ssh_opts, target, remote_shell],
                    timeout=timeout,
                    stage="Run remote whisper",
                    check_stderr=True,
                )

                try:
                    remote_output_candidates = self._remote_srt_candidates(
                        target, ssh_opts, remote_job_dir
                    )
                    break
                except RuntimeError as exc:
                    last_error = str(exc)
                    diag = self._inspect_remote_job_dir(target, ssh_opts, remote_job_dir)
                    log.warning(
                        f"Remote output discovery failed on attempt {attempt}",
                        extra={
                            "job_id": job.job_id,
                            "attempt": attempt,
                            "remote_job_dir": str(remote_job_dir),
                            "diagnostics": diag,
                        },
                    )
                    if attempt == 1:
                        continue
                    raise RuntimeError(
                        f"Remote whisper output discovery failed after retry: {last_error}"
                    )

            if remote_output_candidates is None:
                raise RuntimeError("Remote whisper produced no .srt files")

            remote_output = self._select_remote_output(
                remote_output_candidates, remote_output, remote_input
            )
            remote_output_name = Path(remote_output).name

            download_cmd = [
                "scp",
                *self._scp_opts(),
                f"{target}:{remote_output}",
                str(job.job_dir / remote_output_name),
            ]
            self._run_checked(
                download_cmd,
                timeout=max(120, timeout),
                stage="Download whisper output",
                check_stderr=True,
            )
        finally:
            cleanup_cmd = [
                "ssh",
                *ssh_opts,
                target,
                f"rm -rf {shlex.quote(str(remote_job_dir))}",
            ]
            try:
                self._run_checked(
                    cleanup_cmd,
                    timeout=20,
                   stage="Cleanup remote job dir",
                   check_stderr=True,
                )
            except Exception:
                pass

    def _ensure_remote_whisper_binary(self, target: str, ssh_opts: list[str]) -> None:
        if self._remote_capability_verified:
            return

        binary = self._config.runtime.whisper_binary.strip()
        if not binary:
            raise RuntimeError("Runtime config is missing runtime.whisper_binary")

        quoted_binary = shlex.quote(binary)
        if "/" in binary:
            check = f"test -x {quoted_binary}"
        else:
            check = f"command -v {quoted_binary} >/dev/null 2>&1"

        cmd = ["ssh", *ssh_opts, target, check]
        self._run_checked(
            cmd,
            timeout=60,
            stage="Check remote whisper binary",
            check_stderr=True,
        )
        self._remote_capability_verified = True

    def _remote_srt_candidates(self, target: str, ssh_opts: list[str], remote_job_dir: Path) -> list[str]:
        cmd = [
            "ssh",
            *ssh_opts,
            target,
            f"find {shlex.quote(str(remote_job_dir))} -maxdepth 1 -type f -name '*.srt' -print",
        ]
        proc = self._run_checked_with_output(
            cmd,
            timeout=120,
            stage="List remote whisper outputs",
            check_stderr=True,
        )

        files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not files:
            raise RuntimeError(f"Remote whisper produced no .srt files under {remote_job_dir}")
        return files

    def _inspect_remote_job_dir(self, target: str, ssh_opts: list[str], remote_job_dir: Path) -> str:
        cmd = [
            "ssh",
            *ssh_opts,
            target,
            (
                f'if [ -d {shlex.quote(str(remote_job_dir))} ]; then '
                f"ls -la {shlex.quote(str(remote_job_dir))}; fi"
            ),
        ]
        try:
            proc = self._run_checked_with_output(
                cmd,
                timeout=30,
                stage="Inspect remote job dir",
                check_stderr=True,
            )
            output = proc.stdout.strip()
            return output if output else "(empty)"
        except Exception as exc:  # pragma: no cover - best-effort diagnostic
            return f"(failed to inspect remote job dir: {exc})"

    def _build_remote_shell(self, shell_cmd: str) -> str:
        extra_path = self._config.runtime.whisper_remote_path.strip()
        if not extra_path:
            return shell_cmd

        sanitized_parts = [part.strip() for part in extra_path.split(":") if part.strip()]
        if not sanitized_parts:
            return shell_cmd

        path_prefix = ":".join(sanitized_parts)
        return f"PATH={shlex.quote(path_prefix)}:$PATH {shell_cmd}"

    def _select_remote_output(self, candidates: list[str], expected_output: Path, source_audio: Path) -> str:
        expected = str(expected_output)
        if expected in candidates:
            return expected

        expected_stem = source_audio.stem
        stem_matches = [
            path
            for path in candidates
            if Path(path).stem == expected_stem
        ]
        if len(stem_matches) == 1:
            return stem_matches[0]

        if len(candidates) == 1:
            return candidates[0]

        raise RuntimeError(
            "Remote output file is ambiguous. "
            f"Candidates={candidates}, expected={expected}"
        )

    def _run_checked(
        self,
        cmd: list[str],
        *,
        timeout: int,
        stage: str,
        check_stderr: bool = False,
    ) -> None:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_tail = exc.stdout[-2000:] if exc.stdout else "(no stdout)"
            stderr_tail = exc.stderr[-2000:] if exc.stderr else "(no stderr)"
            raise RuntimeError(
                f"{stage} timed out after {timeout}s: cmd={cmd} "
                f"stdout={stdout_tail} stderr={stderr_tail}"
            )

        if result.returncode != 0:
            stdout_tail = result.stdout[-2000:] if result.stdout else "(no stdout)"
            if check_stderr:
                stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"{stage} failed with exit {result.returncode}: cmd={cmd} stdout={stdout_tail} stderr={stderr_tail}"
                )
            raise RuntimeError(f"{stage} failed with exit {result.returncode}: cmd={cmd} stdout={stdout_tail}")

    def _run_checked_with_output(
        self,
        cmd: list[str],
        *,
        timeout: int,
        stage: str,
        check_stderr: bool = False,
    ):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_tail = exc.stdout[-2000:] if exc.stdout else "(no stdout)"
            stderr_tail = exc.stderr[-2000:] if exc.stderr else "(no stderr)"
            raise RuntimeError(f"{stage} timed out after {timeout}s: cmd={cmd} stdout={stdout_tail} stderr={stderr_tail}")

        stdout_tail = result.stdout[-2000:] if result.stdout else "(no stdout)"
        if result.returncode != 0:
            if check_stderr:
                stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
            else:
                stderr_tail = "(suppressed)"
            raise RuntimeError(
                f"{stage} failed with exit {result.returncode}: cmd={cmd} stdout={stdout_tail} stderr={stderr_tail}"
            )

        return result

    def _load_prompt(self, policy: dict[str, str]) -> Optional[str]:
        prompt: Optional[str] = None
        prompt_file_str = policy.get("prompt_file", "")
        if prompt_file_str:
            pf = Path(prompt_file_str)
            if pf.exists():
                prompt = pf.read_text().strip() or None
        return prompt

    def _build_whisper_command(
        self,
        binary: str,
        source: Path,
        output_dir: Path,
        policy: dict[str, object],
        prompt: Optional[str],
    ) -> list[str]:
        is_mlx = self._is_mlx_whisper_binary(binary)
        model = self._resolve_model_for_cli(str(policy["model"]), is_mlx)
        flag_output_format = self._cli_flag("--output_format", is_mlx)
        flag_output_dir = self._cli_flag("--output_dir", is_mlx)
        flag_initial_prompt = self._cli_flag("--initial_prompt", is_mlx)

        cmd = [
            binary,
            str(source),
            "--model", model,
            "--language", policy["language"],
            flag_output_format, "srt",
            flag_output_dir, str(output_dir),
        ]
        if prompt:
            cmd.extend([flag_initial_prompt, prompt])
        extra_args = policy.get("whisper_args")
        if isinstance(extra_args, list):
            cmd.extend(self._normalize_extra_args(extra_args, is_mlx))
        return cmd

    def _is_mlx_whisper_binary(self, binary: str) -> bool:
        name = Path(binary).name.lower()
        return "mlx_whisper" in name or "mlx-whisper" in name

    def _cli_flag(self, flag: str, is_mlx: bool) -> str:
        if not is_mlx:
            return flag
        return flag.replace("_", "-")

    def _normalize_extra_args(self, args: list[str], is_mlx: bool) -> list[str]:
        if not is_mlx:
            return args

        normalized: list[str] = []
        for arg in args:
            if arg.startswith("--"):
                normalized.append(arg.replace("_", "-"))
            else:
                normalized.append(arg)
        return normalized

    def _resolve_model_for_cli(self, model: str, is_mlx: bool) -> str:
        if not is_mlx:
            return model
        if "/" in model:
            return model

        mapping = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "turbo": "mlx-community/whisper-large-v3-turbo",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        }
        return mapping.get(model, model)

    def _ssh_base_opts(self) -> list[str]:
        rt = self._config.runtime
        opts = [
            "-o", f"ConnectTimeout={rt.whisper_connect_timeout_seconds}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
        ]
        if rt.whisper_ssh_key:
            opts.extend(["-i", rt.whisper_ssh_key])
        return opts

    def _ssh_opts(self) -> list[str]:
        rt = self._config.runtime
        opts = self._ssh_base_opts()
        opts.insert(0, "-p")
        opts.insert(1, str(rt.whisper_ssh_port))
        return opts

    def _scp_opts(self) -> list[str]:
        rt = self._config.runtime
        opts = self._ssh_base_opts()
        opts.insert(0, "-P")
        opts.insert(1, str(rt.whisper_ssh_port))
        return opts

    def _ssh_target(self) -> str:
        rt = self._config.runtime
        if rt.whisper_ssh_user:
            return f"{rt.whisper_ssh_user}@{rt.whisper_ssh_host}"
        return rt.whisper_ssh_host

    def _finalize_whisper_output(self, job: Job) -> None:
        # Whisper writes {input_stem}.srt next to the input file.
        # Rename to canonical output.srt so delivery path is stable.
        input_stem = job.input_audio.stem
        srt_candidate = job.job_dir / f"{input_stem}.srt"

        if not srt_candidate.exists():
            # Fallback: find any .srt whisper may have written
            candidates = list(job.job_dir.glob("*.srt"))
            if not candidates:
                raise RuntimeError("whisper produced no .srt file in job dir")
            srt_candidate = candidates[0]

        # Atomic rename to canonical name
        tmp = job.job_dir / "output.srt.tmp"
        srt_candidate.rename(tmp)
        tmp.rename(job.output_srt)

    def _postprocess_output(self, job: Job) -> None:
        """
        Remove obvious repeated hallucination loops conservatively.

        Supports two collapse modes for consecutive identical subtitles:
        - long-segment loop mode (existing): each segment duration is long enough
        - micro-segment loop mode (new): very short, low-diversity text repeated many times
        """
        policy = job.meta.policy
        if not bool(policy.get("postprocess_enabled", True)):
            return
        if not job.output_srt.exists():
            return

        min_consecutive = int(policy.get("postprocess_repeat_min_consecutive", 3))
        min_segment_seconds = float(policy.get("postprocess_repeat_min_segment_seconds", 8.0))
        max_text_chars = int(policy.get("postprocess_repeat_max_text_chars", 60))
        micro_repeat_min_consecutive = int(policy.get("postprocess_micro_repeat_min_consecutive", 8))
        micro_repeat_max_segment_seconds = float(
            policy.get("postprocess_micro_repeat_max_segment_seconds", 1.5)
        )
        micro_repeat_max_text_chars = int(policy.get("postprocess_micro_repeat_max_text_chars", 12))
        micro_repeat_max_unique_chars = int(policy.get("postprocess_micro_repeat_max_unique_chars", 2))
        micro_repeat_min_total_seconds = float(
            policy.get("postprocess_micro_repeat_min_total_seconds", 4.0)
        )
        low_info_max_segment_seconds = float(
            policy.get("postprocess_low_info_max_segment_seconds", 2.5)
        )
        low_info_min_compact_chars = int(
            policy.get("postprocess_low_info_min_compact_chars", 5)
        )
        low_info_max_unique_chars = int(
            policy.get("postprocess_low_info_max_unique_chars", 2)
        )
        hallucination_phrases_raw = policy.get("postprocess_hallucination_phrases", [])
        hallucination_min_segment_seconds = float(
            policy.get("postprocess_hallucination_min_segment_seconds", 20.0)
        )
        short_file_max_segments = int(
            policy.get("postprocess_hallucination_short_file_max_segments", 2)
        )
        short_file_max_total_seconds = float(
            policy.get("postprocess_hallucination_short_file_max_total_seconds", 8.0)
        )

        hallucination_phrases = []
        if isinstance(hallucination_phrases_raw, list):
            hallucination_phrases = [str(item) for item in hallucination_phrases_raw if str(item).strip()]
        if (
            min_consecutive < 2
            or max_text_chars <= 0
            or micro_repeat_min_consecutive < 2
            or micro_repeat_max_segment_seconds < 0
            or micro_repeat_max_text_chars <= 0
            or micro_repeat_max_unique_chars <= 0
            or micro_repeat_min_total_seconds < 0
            or low_info_max_segment_seconds < 0
            or low_info_min_compact_chars <= 0
            or low_info_max_unique_chars <= 0
        ):
            return

        raw = job.output_srt.read_text()
        segments = self._parse_srt(raw)
        if not segments:
            return

        filtered, removed_repeats = self._collapse_repeated_long_segments(
            segments,
            min_consecutive=min_consecutive,
            min_segment_seconds=min_segment_seconds,
            max_text_chars=max_text_chars,
            micro_min_consecutive=micro_repeat_min_consecutive,
            micro_max_segment_seconds=micro_repeat_max_segment_seconds,
            micro_max_text_chars=micro_repeat_max_text_chars,
            micro_max_unique_chars=micro_repeat_max_unique_chars,
            micro_min_total_seconds=micro_repeat_min_total_seconds,
        )
        filtered, removed_low_info = self._remove_low_information_segments(
            filtered,
            max_segment_seconds=low_info_max_segment_seconds,
            min_compact_chars=low_info_min_compact_chars,
            max_unique_chars=low_info_max_unique_chars,
        )
        filtered, removed_phrase = self._remove_known_hallucination_phrases(
            filtered,
            hallucination_phrases=hallucination_phrases,
            min_segment_seconds=hallucination_min_segment_seconds,
            short_file_max_segments=short_file_max_segments,
            short_file_max_total_seconds=short_file_max_total_seconds,
        )
        filtered, removed_pattern = self._remove_repetitive_pattern_segments(
            filtered,
            min_segment_seconds=min_segment_seconds,
        )

        removed = removed_repeats + removed_low_info + removed_phrase + removed_pattern
        if removed <= 0:
            return

        self._write_srt(job.output_srt, filtered)
        log.info(
            f"Postprocess removed repeated subtitle segments: {removed}",
            extra={
                "job_id": job.job_id,
                "policy_key": job.meta.policy_key,
                "removed_segments": removed,
                "removed_repeat_segments": removed_repeats,
                "removed_low_info_segments": removed_low_info,
                "removed_phrase_segments": removed_phrase,
                "removed_pattern_segments": removed_pattern,
            },
        )

    def _parse_srt(self, raw: str) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        blocks = re.split(r"\n\s*\n", raw.strip())
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 2:
                continue

            idx = 0
            if lines[0].strip().isdigit():
                idx = 1
            if idx >= len(lines):
                continue

            timing = lines[idx].strip()
            if "-->" not in timing:
                continue
            start_raw, end_raw = [part.strip() for part in timing.split("-->", 1)]
            start = self._srt_timestamp_to_seconds(start_raw)
            end = self._srt_timestamp_to_seconds(end_raw)
            if start is None or end is None or end < start:
                continue

            text_lines = [line.rstrip() for line in lines[idx + 1:] if line.strip()]
            if not text_lines:
                continue

            segments.append(
                {
                    "start_raw": start_raw,
                    "end_raw": end_raw,
                    "start": start,
                    "end": end,
                    "text_lines": text_lines,
                }
            )
        return segments

    def _srt_timestamp_to_seconds(self, ts: str) -> Optional[float]:
        match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts.strip())
        if not match:
            return None
        hh, mm, ss, ms = [int(part) for part in match.groups()]
        return float(hh * 3600 + mm * 60 + ss) + (ms / 1000.0)

    def _normalize_segment_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text).lower()
        normalized = normalized.rstrip("。．.!！?？,，、…")
        return normalized

    def _count_text_unique_chars(self, text_norm: str) -> int:
        stripped = re.sub(r"[、,，。．.!！?？…・·:：;；~〜\-_ー]", "", text_norm)
        return len(set(stripped)) if stripped else 0

    def _collapse_repeated_long_segments(
        self,
        segments: list[dict[str, Any]],
        *,
        min_consecutive: int,
        min_segment_seconds: float,
        max_text_chars: int,
        micro_min_consecutive: int,
        micro_max_segment_seconds: float,
        micro_max_text_chars: int,
        micro_max_unique_chars: int,
        micro_min_total_seconds: float,
    ) -> tuple[list[dict[str, Any]], int]:
        filtered: list[dict[str, Any]] = []
        removed = 0
        i = 0

        while i < len(segments):
            current_text = " ".join(segments[i]["text_lines"])
            current_norm = self._normalize_segment_text(current_text)
            j = i + 1
            while j < len(segments):
                next_text = " ".join(segments[j]["text_lines"])
                next_norm = self._normalize_segment_text(next_text)
                if next_norm != current_norm:
                    break
                j += 1

            run_len = j - i
            if current_norm and run_len >= min(min_consecutive, micro_min_consecutive):
                durations = [segments[k]["end"] - segments[k]["start"] for k in range(i, j)]
                total_duration = sum(durations)
                long_repeat = (
                    run_len >= min_consecutive
                    and
                    min(durations) >= min_segment_seconds
                    and len(current_norm) <= max_text_chars
                )
                micro_repeat = (
                    run_len >= micro_min_consecutive
                    and max(durations) <= micro_max_segment_seconds
                    and total_duration >= micro_min_total_seconds
                    and len(current_norm) <= micro_max_text_chars
                    and self._count_text_unique_chars(current_norm) <= micro_max_unique_chars
                )
                if long_repeat or micro_repeat:
                    filtered.append(segments[i])
                    removed += run_len - 1
                    i = j
                    continue

            filtered.extend(segments[i:j])
            i = j

        return filtered, removed

    def _remove_known_hallucination_phrases(
        self,
        segments: list[dict[str, Any]],
        *,
        hallucination_phrases: list[str],
        min_segment_seconds: float,
        short_file_max_segments: int,
        short_file_max_total_seconds: float,
    ) -> tuple[list[dict[str, Any]], int]:
        if not hallucination_phrases:
            return segments, 0

        normalized_phrases = [
            self._normalize_segment_text(phrase)
            for phrase in hallucination_phrases
            if self._normalize_segment_text(phrase)
        ]
        if not normalized_phrases:
            return segments, 0

        total_duration = sum(float(seg["end"] - seg["start"]) for seg in segments)
        short_file = (
            short_file_max_segments > 0
            and len(segments) <= short_file_max_segments
            and total_duration <= short_file_max_total_seconds
        )

        filtered: list[dict[str, Any]] = []
        removed = 0
        for seg in segments:
            text_norm = self._normalize_segment_text(" ".join(seg["text_lines"]))
            if not text_norm:
                filtered.append(seg)
                continue

            matched = any(phrase in text_norm for phrase in normalized_phrases)
            if not matched:
                filtered.append(seg)
                continue

            duration = float(seg["end"] - seg["start"])
            if duration >= min_segment_seconds or short_file:
                removed += 1
                continue

            filtered.append(seg)

        return filtered, removed

    def _remove_repetitive_pattern_segments(
        self,
        segments: list[dict[str, Any]],
        *,
        min_segment_seconds: float,
    ) -> tuple[list[dict[str, Any]], int]:
        filtered: list[dict[str, Any]] = []
        removed = 0

        for seg in segments:
            duration = float(seg["end"] - seg["start"])
            text = " ".join(seg["text_lines"])
            text_norm = self._normalize_segment_text(text)

            if duration < min_segment_seconds or len(text_norm) < 24:
                filtered.append(seg)
                continue

            if self._is_repetitive_hallucination(text_norm):
                removed += 1
                continue

            filtered.append(seg)

        return filtered, removed

    def _remove_low_information_segments(
        self,
        segments: list[dict[str, Any]],
        *,
        max_segment_seconds: float,
        min_compact_chars: int,
        max_unique_chars: int,
    ) -> tuple[list[dict[str, Any]], int]:
        filtered: list[dict[str, Any]] = []
        removed = 0

        for seg in segments:
            duration = float(seg["end"] - seg["start"])
            if duration > max_segment_seconds:
                filtered.append(seg)
                continue

            text = " ".join(seg["text_lines"])
            text_norm = self._normalize_segment_text(text)
            compact = re.sub(r"[、,，。．.!！?？…・·:：;；~〜\-_ー]", "", text_norm)
            if len(compact) < min_compact_chars:
                filtered.append(seg)
                continue

            unique_chars = len(set(compact))
            repeated_unit = bool(re.fullmatch(r"(.{1,2})\1{3,}", compact))
            if unique_chars <= max_unique_chars or repeated_unit:
                removed += 1
                continue

            filtered.append(seg)

        return filtered, removed

    def _is_repetitive_hallucination(self, text_norm: str) -> bool:
        if not text_norm:
            return False

        stripped = re.sub(r"[、,，。．.!！?？…・·:：;；~〜\-_ー]", "", text_norm)
        if len(stripped) >= 20:
            char_counts: dict[str, int] = {}
            for ch in stripped:
                char_counts[ch] = char_counts.get(ch, 0) + 1
            dominant = max(char_counts.values())
            if dominant / max(len(stripped), 1) >= 0.85:
                return True

        max_chunk_len = min(64, len(text_norm) // 2)
        for chunk_len in range(4, max_chunk_len + 1):
            chunk = text_norm[:chunk_len]
            if not chunk:
                continue

            repetitions = 0
            idx = 0
            while idx + chunk_len <= len(text_norm) and text_norm[idx : idx + chunk_len] == chunk:
                repetitions += 1
                idx += chunk_len

            coverage = idx / max(len(text_norm), 1)
            if repetitions >= 5 and coverage >= 0.72:
                return True

        return False

    def _write_srt(self, path: Path, segments: list[dict[str, Any]]) -> None:
        if not segments:
            tmp = path.with_suffix(path.suffix + ".post.tmp")
            tmp.write_text("")
            tmp.replace(path)
            return

        lines: list[str] = []
        for idx, segment in enumerate(segments, start=1):
            lines.append(str(idx))
            lines.append(f"{segment['start_raw']} --> {segment['end_raw']}")
            lines.extend(segment["text_lines"])
            lines.append("")

        content = "\n".join(lines).rstrip() + "\n"
        tmp = path.with_suffix(path.suffix + ".post.tmp")
        tmp.write_text(content)
        tmp.replace(path)

    def _run_mock(self, job: Job) -> None:
        """
        Dry-run mode: simulate transcription with a 2-second sleep.
        Writes a clearly-marked mock SRT file.
        """
        log.info(
            f"[DRY-RUN] Simulating whisper for {job.job_id}",
            extra={"job_id": job.job_id},
        )
        time.sleep(2)

        mock_content = (
            "1\n"
            "00:00:00,000 --> 00:00:05,000\n"
            f"[DRY-RUN] Mock transcription — policy={job.meta.policy_key}\n"
            "\n"
            "2\n"
            "00:00:05,000 --> 00:00:10,000\n"
            f"[DRY-RUN] Source: {job.meta.source_filename}\n"
            "\n"
        )
        job.output_srt.write_text(mock_content)

    def _deliver_output(self, job: Job) -> None:
        """
        Copy output.srt to output_root/{policy_key}/{stem}.srt atomically.

        Uses temp+rename to ensure the output file in output_root is never
        partial. If the destination already exists (e.g. same filename
        reprocessed), it is overwritten — this is intentional.
        """
        output_root = self._safe_output_dir(job.meta.policy_key)
        output_root.mkdir(parents=True, exist_ok=True)

        stem = Path(job.meta.source_filename).stem
        final_srt = output_root / f"{stem}.srt"
        tmp_srt = output_root / f".{stem}.srt.tmp"

        shutil.copy2(job.output_srt, tmp_srt)
        tmp_srt.replace(final_srt)

        log.info(
            f"Output delivered: {final_srt}",
            extra={"job_id": job.job_id, "policy_key": job.meta.policy_key},
        )

    def _safe_output_dir(self, policy_key: str) -> Path:
        """
        Return a safe output directory for a policy key.

        The output directory must stay under output_root and must not be a symlink.
        """
        base_output = self._config.paths.output_root.resolve()
        candidate = self._config.paths.output_root / policy_key
        if candidate.is_symlink():
            raise RuntimeError(f"Policy output directory is a symlink: {candidate}")
        output_root = candidate.resolve()
        if not output_root.is_relative_to(base_output):
            raise RuntimeError(f"Invalid output policy directory: {policy_key}")
        return output_root

    def _fail_job(self, job: Job, error: str) -> None:
        """Write error.json and move job to failed/."""
        moved_to_failed = False
        try:
            error_info = {
                "error": error,
                "failed_at": _utcnow(),
                "pid": os.getpid(),
            }
            write_json_atomic(job.error_file, error_info)
        except Exception:
            pass  # best-effort; don't mask the original error

        try:
            failed_dir = self._config.paths.failed_dir / job.job_id
            job.job_dir.rename(failed_dir)
            moved_to_failed = True
        except OSError as exc:
            log.error(
                f"Could not move job to failed/: {exc}",
                extra={"job_id": job.job_id},
            )
        if moved_to_failed and self._registry and job.meta.fingerprint:
            self._registry.unregister(job.meta.fingerprint)
