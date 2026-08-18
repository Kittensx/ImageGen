from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from image_gen.webui.job_runtime_events import (
    _FAILURE_BUNDLE_LINE,
    _GENERATION_SEED_LINE,
    _IMAGE_LINE,
    _LIVE_PREVIEW_SUMMARY_LINE,
    _MEMORY_STATUS_LINE,
    _MODEL_DIAGNOSTIC_LINE,
    _OUTPUT_QUALITY_DIAGNOSTIC_LINE,
    _PROMPT_PARSER_DIAGNOSTIC_LINE,
    _RUNTIME_DIAGNOSTIC_LINE,
    _STEP_PREVIEW_LINE,
    _STEP_PROGRESS_LINE,
    _normalize_live_memory_status,
)
from image_gen.webui.job_store import _utc_now

_SUBPROCESS_STREAM_LIMIT = 16 * 1024 * 1024


class IsolatedJobExecutorMixin:
    async def _run_job_isolated(self, job: GenerationJob) -> None:
        self._transition_job(job, status="preparing_model", worker_stage="loading_model")
        if job.execution_mode == "pending":
            job.execution_mode = "isolated_subprocess"
        job.started_at = _utc_now()
        job.updated_at = job.started_at
        job.status_changed_at = job.started_at
        job_root = self.context.data_root / "webui" / "jobs" / job.job_id
        job_root.mkdir(parents=True, exist_ok=True)
        job.job_root = str(job_root)
        request_path = job_root / "request.json"
        console_path = job_root / "console.log"
        job.console_log_path = str(console_path)

        request_payload = self.normalize_generation_request(job.request)
        request_model_path_before_lock = request_payload.get("model_path")
        authoritative_model_path = str(job.model_selection.get("resolved_path") or "").strip()
        if authoritative_model_path:
            request_payload["model_path"] = authoritative_model_path
        job.request = dict(request_payload)
        request_payload["save_images"] = True
        request_payload.setdefault("output_dir", str(self.context.txt2img_output_root))
        request_payload.setdefault("output_prefix", "{index:05d}-{seed}")

        preview_values = self._live_preview_request_values(job_root)
        live_preview_root = Path(preview_values["live_preview_root"])
        job.live_preview_root = str(live_preview_root)
        job.live_preview_latest_path = str(live_preview_root / "latest.json")
        self._merge_runtime_preview_values(request_payload, preview_values)
        job.request = dict(request_payload)
        job.model_diagnostics["preflight"] = {
            "authoritative_model_path": authoritative_model_path,
            "request_model_path_before_lock": request_model_path_before_lock,
            "request_model_path_after_lock": request_payload.get("model_path"),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "cwd": os.getcwd(),
            "project_root": str(self.context.project_root),
            "job_root": str(job_root),
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        }
        job.model_diagnostics["request_file"] = {
            "model_path": request_payload.get("model_path"),
            "request_path": str(request_path),
        }
        request_path.write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (job_root / "model-selection.json").write_text(
            json.dumps(
                {
                    "selection": job.model_selection,
                    "diagnostics": job.model_diagnostics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        source_root = str(self.context.project_root / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            [source_root, str(self.context.project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        command = [
            sys.executable,
            "-m",
            "modules.txt2img.cli",
            "run",
            "--project-root",
            str(self.context.project_root),
            "--config",
            str(request_path),
        ]
        if not bool(job.request.get("save_txt", False)):
            command.append("--no-txt")
        if not bool(job.request.get("save_json", True)):
            command.append("--no-json")
        if not bool(job.request.get("save_diagnostics_json", False)):
            command.append("--no-diagnostics-json")
        job.model_diagnostics["pipeline_parity"] = {
            "shares_cli_runner_with_run_bat": True,
            "run_bat_entrypoint": "python -m modules.txt2img.cli run --interactive --save",
            "webui_entrypoint": "python -m modules.txt2img.cli run --config <request.json>",
            "execution_path": [
                "src/image_gen/webui/jobs.py::GenerationJobManager._run_job",
                "modules.txt2img.cli:run",
                "modules.txt2img.txt2img_runner",
                "src/image_gen.runtime.txt2img_runner",
            ],
            "command": subprocess.list2cmdline(command),
            "request_path": str(request_path),
            "request_contains_live_preview_overlay": True,
            "live_preview_overlay_keys": sorted(preview_values.keys()),
            "webui_only_keys_do_not_change_final_decode": True,
        }
        (job_root / "command.txt").write_text(
            subprocess.list2cmdline(command),
            encoding="utf-8",
        )
        self._persist_job(job)
        self._publish_event(job, "job-started")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.context.project_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # Diagnostic JSONL lines can be much larger than asyncio's
                # default 64 KiB reader limit.
                limit=_SUBPROCESS_STREAM_LIMIT,
            )
            job.process = process
            assert process.stdout is not None
            with console_path.open("w", encoding="utf-8", newline="\n") as console:
                while True:
                    raw = await process.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    job.log_lines.append(line)
                    console.write(line + "\n")
                    console.flush()
                    seed_match = _GENERATION_SEED_LINE.match(line)
                    if seed_match:
                        try:
                            seed_payload = json.loads(seed_match.group(1))
                        except json.JSONDecodeError:
                            seed_payload = {}
                        self._transition_job(job, status="running", worker_stage="running")
                        try:
                            job.resolved_seed = int(seed_payload.get("base_seed"))
                        except (TypeError, ValueError):
                            job.resolved_seed = None
                        raw_seeds = seed_payload.get("image_seeds") or []
                        job.resolved_seeds = []
                        for value in raw_seeds:
                            try:
                                job.resolved_seeds.append(int(value))
                            except (TypeError, ValueError):
                                continue
                        job.updated_at = _utc_now()
                        self._persist_job(job)
                        self._publish_event(
                            job,
                            "job-progress",
                            resolved_seed=job.resolved_seed,
                            resolved_seeds=list(job.resolved_seeds),
                        )
                    preview_summary_match = _LIVE_PREVIEW_SUMMARY_LINE.match(line)
                    if preview_summary_match:
                        self._transition_job(job, status="finalizing", worker_stage="finalizing")
                        try:
                            preview_summary = json.loads(preview_summary_match.group(1))
                        except json.JSONDecodeError:
                            preview_summary = {}
                        if isinstance(preview_summary, dict):
                            job.live_preview_metrics.update(preview_summary)
                            job.live_preview_metrics["sse_clients_connected"] = job.sse_clients_connected
                            job.live_preview_metrics["sse_clients_peak"] = job.sse_clients_peak
                            job.live_preview_metrics["stale_preview_events_ignored"] = job.stale_preview_events_ignored
                            job.model_diagnostics["live_preview"] = self.diagnostics_payload(job)["phase09h_validation"]
                            self._persist_job(job)
                            self._publish_event(
                                job,
                                "job-progress",
                                live_preview_metrics=dict(job.live_preview_metrics),
                            )
                    memory_status_match = _MEMORY_STATUS_LINE.search(line)
                    if memory_status_match:
                        try:
                            memory_payload = json.loads(memory_status_match.group(1))
                        except json.JSONDecodeError:
                            memory_payload = {}
                        if isinstance(memory_payload, dict):
                            status_payload = _normalize_live_memory_status(
                                memory_payload.get("status") or {}
                            )
                            job.memory_status = {
                                **status_payload,
                                "event": memory_payload.get("event"),
                                "stage": memory_payload.get("stage"),
                                "active_stage": memory_payload.get("active_stage"),
                                "updated_at": _utc_now(),
                            }
                            self._persist_job(job)
                            self._publish_event(
                                job,
                                "job-progress",
                                memory_status=dict(job.memory_status),
                            )
                    image_match = _IMAGE_LINE.match(line)
                    if image_match:
                        self._transition_job(job, status="finalizing", worker_stage="saving_output")
                        image_path = image_match.group("path")
                        if image_path not in job.output_paths:
                            job.output_paths.append(image_path)
                        job.final_output_url = self._output_url_for_path(image_path)
                        if job.resolved_seed is None:
                            try:
                                job.resolved_seed = int(image_match.group("seed"))
                            except (TypeError, ValueError):
                                pass
                        job.updated_at = _utc_now()
                        self._persist_job(job)
                        self._publish_event(
                            job,
                            "job-output-produced",
                            latest_output_path=image_path,
                            latest_output_url=job.final_output_url,
                            output_count=len(job.output_paths),
                        )
                    failure_match = _FAILURE_BUNDLE_LINE.search(line)
                    if failure_match:
                        job.failure_bundle_path = failure_match.group(1).strip()
                    runtime_diag_match = _RUNTIME_DIAGNOSTIC_LINE.match(line)
                    if runtime_diag_match:
                        try:
                            payload = json.loads(runtime_diag_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            job.model_diagnostics["runtime_environment"] = payload
                    model_match = _MODEL_DIAGNOSTIC_LINE.match(line)
                    if model_match:
                        try:
                            payload = json.loads(model_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            job.model_diagnostics["runtime"] = payload
                    output_quality_match = _OUTPUT_QUALITY_DIAGNOSTIC_LINE.match(line)
                    if output_quality_match:
                        try:
                            payload = json.loads(output_quality_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            job.output_quality_diagnostics = dict(payload)
                            self._persist_job(job)
                            self._publish_event(
                                job,
                                "job-progress",
                                output_quality_diagnostics=dict(job.output_quality_diagnostics),
                            )

                    prompt_parser_match = _PROMPT_PARSER_DIAGNOSTIC_LINE.match(line)
                    if prompt_parser_match:
                        try:
                            payload = json.loads(prompt_parser_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            job.prompt_parser_diagnostics = dict(payload)
                            self._persist_job(job)
                            self._publish_event(
                                job,
                                "job-progress",
                                prompt_parser_diagnostics=dict(job.prompt_parser_diagnostics),
                            )
                    step_progress_match = _STEP_PROGRESS_LINE.search(line)
                    if step_progress_match:
                        try:
                            payload = json.loads(step_progress_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            self._apply_step_progress_payload(job, payload)

                    # tqdm-based samplers can write their carriage-return progress text
                    # immediately before the structured preview marker. Search the full
                    # console line instead of requiring the marker at column zero.
                    step_preview_match = _STEP_PREVIEW_LINE.search(line)
                    if step_preview_match:
                        try:
                            payload = json.loads(step_preview_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            self._apply_step_preview_payload(job, payload)
            job.return_code = await process.wait()
            if job.status == "cancelling":
                self._transition_job(job, status="cancelled", worker_stage="cancelled")
            elif job.return_code == 0:
                runtime_model = dict(job.model_diagnostics.get("runtime") or {})
                expected_model = str(job.model_selection.get("resolved_path") or "").strip()
                loaded_model = str(
                    runtime_model.get("loaded_path")
                    or runtime_model.get("resolved_path")
                    or runtime_model.get("requested_path")
                    or ""
                ).strip()
                model_paths_match: bool | None = None
                if expected_model and loaded_model:
                    expected_token = os.path.normcase(str(Path(expected_model).expanduser().resolve()))
                    loaded_token = os.path.normcase(str(Path(loaded_model).expanduser().resolve()))
                    model_paths_match = expected_token == loaded_token
                    if not model_paths_match:
                        self._transition_job(job, status="failed", worker_stage="failed")
                        job.error = (
                            "Model parity violation: the WebUI selected checkpoint was not the "
                            "checkpoint loaded by the canonical CLI runtime. "
                            f"Selected: {expected_model}. Loaded: {loaded_model}."
                        )
                    else:
                        self._transition_job(job, status="completed", worker_stage="completed")
                else:
                    self._transition_job(job, status="completed", worker_stage="completed")
                job.model_diagnostics["model_parity"] = {
                    "selected_path": expected_model,
                    "loaded_path": loaded_model,
                    "matches": model_paths_match,
                    "enforced": bool(expected_model),
                }
            else:
                self._transition_job(job, status="failed", worker_stage="failed")
                job.error = next(
                    (line for line in reversed(job.log_lines) if "ERROR" in line.upper()),
                    f"Generation process exited with code {job.return_code}.",
                )
            job.worker_stage = job.status
        except Exception as exc:
            self._transition_job(job, status="failed", worker_stage="failed")
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.process = None
            job.completed_at = _utc_now()
            job.updated_at = job.completed_at
            job.model_diagnostics["live_preview"] = self.diagnostics_payload(job)["phase09h_validation"]
            if job.job_root:
                (Path(job.job_root) / "model-diagnostics.json").write_text(
                    json.dumps(job.model_diagnostics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            self._persist_job(job)
            terminal_event = {
                "completed": "job-completed",
                "cancelled": "job-cancelled",
                "failed": "job-failed",
            }.get(job.status, "job-progress")
            self._publish_terminal_once(job, terminal_event)
            self.cleanup_preview_directories()
