from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from image_gen.webui.job_request_normalization import _coerce_boolean, _coerce_top_level_number
from image_gen.webui.job_store import _utc_now
from image_gen.webui.model_runtime import ModelRuntimeUnavailable
from image_gen.runtime.model_load_variant import (
    MODEL_LOAD_VARIANT_FIELDS,
    sanitize_model_load_runtime_settings,
)
from image_gen.webui.randomization import apply_parameter_ranges, iter_seed_plan, parse_seed_plan


class ResidentJobExecutorMixin:
    async def activate_model(
        self,
        model_path: str,
        *,
        selection: Mapping[str, Any] | None = None,
        runtime_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_path = str(model_path or "").strip()
        if not resolved_path:
            raise ValueError("A checkpoint model path is required for activation.")
        runtime_settings = self._model_runtime_settings()
        if runtime_overrides:
            activation_keys = (
                "_advanced_model_resolved",
                "advanced_models_enabled",
                "advanced_model_family",
                "advanced_model_components",
                "advanced_model_allow_digital_components",
                "advanced_model_t5_device",
                "text_encoder_3_device",
                *MODEL_LOAD_VARIANT_FIELDS,
            )
            for key in activation_keys:
                if key in runtime_overrides:
                    runtime_settings[key] = runtime_overrides[key]
        runtime_settings = sanitize_model_load_runtime_settings(runtime_settings)
        completion = await self.model_runtime.activate(
            resolved_path,
            runtime_settings=runtime_settings,
        )
        if not completion.get("ok"):
            raise RuntimeError(str(completion.get("error") or "Model activation failed."))
        result = dict(completion.get("result") or {})
        status = self.model_runtime.status()
        current_model_path = str(status.get("current_model_path") or "").strip()
        if not current_model_path:
            raise RuntimeError("Model activation completed without a resident checkpoint path.")
        expected = os.path.normcase(str(Path(resolved_path).expanduser().resolve()))
        actual = os.path.normcase(str(Path(current_model_path).expanduser().resolve()))
        if actual != expected:
            raise RuntimeError(
                "Model activation completed for a different checkpoint than the dropdown selection."
            )
        generation_ready = status.get("generation_ready")
        legacy_gpu_loaded = status.get("gpu_loaded")
        readiness_failed = generation_ready is False or (
            generation_ready is None
            and torch.cuda.is_available()
            and legacy_gpu_loaded is False
        )
        if readiness_failed:
            devices = dict(status.get("component_devices") or {})
            raise RuntimeError(
                "The selected model composition did not become generation-ready. "
                f"Component devices: {devices or 'unavailable'}"
            )
        if selection:
            result["selection"] = dict(selection)
        return {**completion, "status": status, "result": result}

    async def supersede_model_activation(self, model_path: str) -> bool:
        return await self.model_runtime.supersede_activation(str(model_path or ""))

    async def unload_model(self) -> dict[str, Any]:
        completion = await self.model_runtime.unload()
        if not completion.get("ok"):
            raise RuntimeError(str(completion.get("error") or "Model unload failed."))
        return dict(completion.get("result") or {})

    def model_runtime_status(self) -> dict[str, Any]:
        return self.model_runtime.status()

    async def _restore_resident_runtime_after_skip(self, job: GenerationJob) -> None:
        model_path = str(
            job.model_selection.get("resolved_path")
            or job.request.get("model_path")
            or ""
        ).strip()
        if not model_path:
            raise ModelRuntimeUnavailable(
                "The current image was skipped, but no checkpoint path was available to restore the resident runtime."
            )
        self._transition_job(job, status="preparing_model", worker_stage="restoring_after_skip")
        self._persist_job(job)
        self._publish_event(job, "job-progress", worker_stage=job.worker_stage)
        await self.activate_model(
            model_path,
            selection=job.model_selection,
            runtime_overrides=job.request,
        )
        if job.status not in {"cancelling", "cancelled"}:
            self._transition_job(job, status="running", worker_stage="skip_recovery_complete")
            self._touch_job_runtime(job, progress=True)
            self._persist_job(job)
            self._publish_event(job, "job-progress", worker_stage=job.worker_stage)

    async def _pause_between_images_if_requested(
        self,
        job: GenerationJob,
        *,
        has_more_images: bool,
    ) -> bool:
        if not job.pause_after_current_requested:
            return False
        job.pause_after_current_requested = False
        global_hold = not self._queue_resume_event.is_set()
        if not global_hold:
            self._queue_pause_requested_at = None
            self._queue_pause_owner_job_id = None
        if not has_more_images:
            self._persist_job(job)
            self._publish_event(
                job,
                "job-progress",
                queue_pause_requested=global_hold,
                queue_paused_after_job=global_hold,
            )
            return False

        paused_at = _utc_now()
        job.paused_at = paused_at
        job.queue_paused_from_status = "running"
        self._transition_job(job, status="paused", worker_stage="paused_between_images")
        self._touch_job_runtime(job, progress=True)
        self._persist_job(job)
        self._publish_event(
            job,
            "job-paused",
            paused_at=paused_at,
            queue_pause_requested=not self._queue_resume_event.is_set(),
            queue_item_paused=True,
        )
        if self._worker_task is not None and asyncio.current_task() is self._worker_task:
            job.scheduler_suspended = True
            self._persist_job(job)
            return True

        # Direct-run compatibility used by focused tests/tools: there is no
        # scheduler task to release, so wait on this job rather than requeueing it.
        job.scheduler_suspended = False
        resume_event = self._resume_event_for_job(job.job_id)
        await resume_event.wait()
        if job.status in {"cancelling", "cancelled"}:
            return False
        if job.status == "paused":
            self._transition_job(job, status="running", worker_stage="resuming_queue")
        self._touch_job_runtime(job, progress=True)
        self._persist_job(job)
        return False

    async def _run_job_resident(self, job: GenerationJob) -> None:
        """Run WebUI requests as one resident-runtime command per image.

        Splitting a user batch into image-scoped commands preserves the requested seed
        sequence while creating safe control boundaries for pause-after-current and
        skip-current-image. The checkpoint remains resident unless skip requires the
        active worker process to be restarted.
        """

        self._transition_job(job, status="preparing_model", worker_stage="preparing_model")
        job.execution_mode = "resident_model"
        request_path, console_path, preview_values = self._prepare_job_request(job)
        job_root = Path(job.job_root or request_path.parent)
        base_request = json.loads(request_path.read_text(encoding="utf-8"))
        requested_batch_count = max(
            1,
            int(_coerce_top_level_number(base_request.get("batch_count"), integer=True, default=1) or 1),
        )
        requested_batch_size = max(
            1,
            int(_coerce_top_level_number(base_request.get("batch_size"), integer=True, default=1) or 1),
        )
        requested_image_count = requested_batch_count * requested_batch_size
        unlimited = _coerce_boolean(base_request.get("unlimited", False), default=False)
        seed_plan = parse_seed_plan(
            base_request.get("seed"),
            mode=base_request.get("batch_seed_mode"),
            range_min=base_request.get("seed_range_min"),
            range_max=base_request.get("seed_range_max"),
            unique=base_request.get("seed_no_duplicates", True),
        )
        resume_image_index = max(0, int(job.resume_image_index or 0))
        resume_completed_images = max(0, int(job.resume_completed_images or 0))
        seed_iterator = iter_seed_plan(
            seed_plan,
            start_index=resume_image_index,
            exclude=job.batch_seed_history,
        )

        job.model_diagnostics["pipeline_parity"] = {
            "shares_canonical_runner_with_run_bat": True,
            "run_bat_entrypoint": "python -m modules.txt2img.cli run --interactive --save",
            "webui_entrypoint": "resident modules.txt2img.model_runtime JSONL command",
            "execution_path": [
                "src/image_gen/webui/jobs.py::GenerationJobManager._run_job_resident",
                "src/image_gen/webui/model_runtime.py::ResidentModelRuntimeClient",
                "modules.txt2img.model_runtime",
                "src/image_gen.runtime.txt2img_runner",
            ],
            "request_path": str(request_path),
            "request_contains_live_preview_overlay": True,
            "live_preview_overlay_keys": sorted(preview_values.keys()),
            "selected_model_resident_until_replaced": True,
            "webui_batch_orchestration": "one resident-runtime command per image slot",
        }
        previous_orchestration = dict(job.model_runtime_diagnostics.get("batch_orchestration") or {})
        orchestration = {
            "mode": "unlimited" if unlimited else "batch_count",
            "requested_batch_count": requested_batch_count,
            "requested_batch_size": requested_batch_size,
            "requested_image_count": None if unlimited else requested_image_count,
            "attempted_images": resume_image_index,
            "completed_images": resume_completed_images,
            "skipped_images": int(job.skipped_images),
            "completed_batches": resume_image_index // requested_batch_size,
            "current_batch": (resume_image_index // requested_batch_size) + (1 if resume_image_index else 0),
            "current_image": resume_image_index,
            "current_image_in_batch": (resume_image_index % requested_batch_size) if resume_image_index else 0,
            "command_completions": list(previous_orchestration.get("command_completions") or []),
            "resume_count": int(job.resume_count),
        }
        job.model_runtime_diagnostics["batch_orchestration"] = orchestration
        (job_root / "command.txt").write_text(
            "resident model runtime: WebUI-managed image iteration commands\n"
            + json.dumps(
                {
                    "job_id": job.job_id,
                    "base_request_path": str(request_path),
                    "mode": orchestration["mode"],
                    "requested_batch_count": requested_batch_count,
                    "requested_batch_size": requested_batch_size,
                    "requested_image_count": orchestration["requested_image_count"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._persist_job(job)
        self._publish_event(
            job,
            "job-started",
            worker_stage=job.worker_stage,
            batch_number=orchestration["current_batch"],
            batch_count=requested_batch_count,
            batch_size=requested_batch_size,
            image_number=resume_image_index,
            image_count=orchestration["requested_image_count"],
            completed_images=resume_completed_images,
            unlimited=unlimited,
            resumed=bool(resume_image_index),
        )

        attempted_images = resume_image_index
        completed_images = resume_completed_images
        last_completion: dict[str, Any] = {}
        console_mode = "a" if resume_image_index and console_path.exists() else "w"
        with console_path.open(console_mode, encoding="utf-8", newline="\n") as console:
            async def on_line(line: str) -> None:
                job.log_lines.append(line)
                console.write(line + "\n")
                console.flush()
                self._apply_runtime_line(job, line)

            while unlimited or attempted_images < requested_image_count:
                if job.status in {"cancelling", "cancelled"}:
                    break

                image_number = attempted_images + 1
                parent_batch_number = ((image_number - 1) // requested_batch_size) + 1
                image_in_batch = ((image_number - 1) % requested_batch_size) + 1
                image_seed = next(seed_iterator)
                iteration_request = json.loads(json.dumps(base_request, ensure_ascii=False))
                iteration_request, range_resolution = apply_parameter_ranges(iteration_request)
                iteration_request["batch_count"] = 1
                iteration_request["batch_size"] = 1
                iteration_request["unlimited"] = False
                iteration_request["seed"] = image_seed
                iteration_request["_webui_parent_batch_number"] = parent_batch_number
                iteration_request["_webui_parent_batch_count"] = requested_batch_count
                iteration_request["_webui_parent_batch_size"] = requested_batch_size
                iteration_request["_webui_parent_image_in_batch"] = image_in_batch
                iteration_request["_webui_parent_image_number"] = image_number
                iteration_request["_webui_parent_image_count"] = None if unlimited else requested_image_count
                iteration_request["_webui_parent_unlimited"] = unlimited

                iteration_preview_root = job_root / "live-preview" / f"batch_{image_number:05d}"
                iteration_preview_root.mkdir(parents=True, exist_ok=True)
                iteration_request["live_preview_root"] = str(iteration_preview_root)
                iteration_request_path = job_root / f"request-batch-{image_number:05d}.json"
                iteration_request_path.write_text(
                    json.dumps(iteration_request, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                job.current_step = 0
                job.total_steps = max(
                    1,
                    int(_coerce_top_level_number(iteration_request.get("steps"), integer=True, default=1) or 1),
                )
                job.progress_percent = 0.0
                job.live_preview_url = None
                job.live_preview_path = None
                job.live_preview_decode_mode = None
                job.live_preview_history = []
                job.live_cfg_step_series = {}
                job.live_preview_root = str(iteration_preview_root)
                job.live_preview_latest_path = str(iteration_preview_root / "latest.json")
                job.skip_current_requested = False
                job.skip_requested_at = None
                self._transition_job(job, status="running", worker_stage="starting_image")

                orchestration.update(
                    {
                        "current_batch": parent_batch_number,
                        "current_image": image_number,
                        "current_image_in_batch": image_in_batch,
                        "attempted_images": attempted_images,
                        "completed_images": completed_images,
                        "skipped_images": int(job.skipped_images),
                        "current_seed": image_seed,
                        "seed_plan": seed_plan.to_dict(),
                        "randomization_resolved": range_resolution,
                        "current_request_path": str(iteration_request_path),
                    }
                )
                job.model_runtime_diagnostics["batch_orchestration"] = orchestration
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    worker_stage=job.worker_stage,
                    batch_number=parent_batch_number,
                    batch_count=requested_batch_count,
                    batch_size=requested_batch_size,
                    image_number=image_number,
                    image_in_batch=image_in_batch,
                    image_count=None if unlimited else requested_image_count,
                    completed_images=completed_images,
                    skipped_images=job.skipped_images,
                    unlimited=unlimited,
                    live_preview_url=None,
                    live_preview_path=None,
                    current_step=0,
                    total_steps=job.total_steps,
                    progress_percent=0.0,
                )
                console.write(
                    "WEBUI_IMAGE_START_JSON: "
                    + json.dumps(
                        {
                            "image_number": image_number,
                            "image_count": None if unlimited else requested_image_count,
                            "batch_number": parent_batch_number,
                            "image_in_batch": image_in_batch,
                            "batch_count": requested_batch_count,
                            "batch_size": requested_batch_size,
                            "unlimited": unlimited,
                            "seed": image_seed,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                console.flush()

                output_count_before = len(job.output_paths)
                try:
                    runtime_kwargs = {
                        "job_id": job.job_id,
                        "config_path": iteration_request_path,
                        "save_txt": bool(job.request.get("save_txt", False)),
                        "save_json": bool(job.request.get("save_json", True)),
                        "save_diagnostics_json": bool(
                            job.request.get("save_diagnostics_json", False)
                        ),
                        "on_line": on_line,
                    }
                    while True:
                        try:
                            completion = await self.model_runtime.run_job(**runtime_kwargs)
                            break
                        except TypeError as exc:
                            if (
                                "save_diagnostics_json" not in str(exc)
                                or "save_diagnostics_json" not in runtime_kwargs
                            ):
                                raise
                            runtime_kwargs.pop("save_diagnostics_json", None)
                except ModelRuntimeUnavailable as exc:
                    if job.skip_current_requested and job.status not in {"cancelling", "cancelled"}:
                        attempted_images += 1
                        job.batch_seed_history.append(int(image_seed))
                        output_completed_before_cancel = len(job.output_paths) > output_count_before
                        skip_event = {
                            "timestamp": _utc_now(),
                            "image_number": image_number,
                            "batch_number": parent_batch_number,
                            "image_in_batch": image_in_batch,
                            "seed": int(image_seed),
                            "runtime_error": f"{type(exc).__name__}: {exc}",
                            "output_completed_before_cancel": output_completed_before_cancel,
                        }
                        if output_completed_before_cancel:
                            completed_images += 1
                            skip_event["outcome"] = "completed_before_skip_reached_runtime"
                        else:
                            job.skipped_images += 1
                            job.skipped_image_seeds.append(int(image_seed))
                            skip_event["outcome"] = "skipped"
                        job.skip_events.append(skip_event)
                        job.skip_current_requested = False
                        job.skip_requested_at = None
                        orchestration["command_completions"].append(
                            {
                                "image_number": image_number,
                                "batch_number": parent_batch_number,
                                "ok": False,
                                "skipped": not output_completed_before_cancel,
                                "output_completed_before_cancel": output_completed_before_cancel,
                                "error": str(exc),
                            }
                        )
                        orchestration.update(
                            {
                                "attempted_images": attempted_images,
                                "completed_images": completed_images,
                                "skipped_images": int(job.skipped_images),
                                "completed_batches": attempted_images // requested_batch_size,
                                "last_skip_event": skip_event,
                            }
                        )
                        job.model_runtime_diagnostics["batch_orchestration"] = orchestration
                        self._touch_job_runtime(job, progress=True)
                        self._persist_job(job)
                        self._publish_event(
                            job,
                            "job-image-skipped",
                            skip_event=skip_event,
                            completed_images=completed_images,
                            skipped_images=job.skipped_images,
                        )
                        console.write(
                            "WEBUI_IMAGE_SKIP_JSON: "
                            + json.dumps(skip_event, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                        console.flush()
                        await self._restore_resident_runtime_after_skip(job)
                        has_more_images = unlimited or attempted_images < requested_image_count
                        if await self._pause_between_images_if_requested(
                            job,
                            has_more_images=has_more_images,
                        ):
                            job.resume_image_index = attempted_images
                            job.resume_completed_images = completed_images
                            orchestration["suspended_at"] = _utc_now()
                            job.model_runtime_diagnostics["batch_orchestration"] = orchestration
                            self._persist_job(job)
                            return
                        continue
                    raise

                last_completion = dict(completion)
                completion_summary = {
                    "image_number": image_number,
                    "batch_number": parent_batch_number,
                    "image_in_batch": image_in_batch,
                    "ok": bool(completion.get("ok")),
                    "command_id": completion.get("command_id"),
                    "result": dict(completion.get("result") or {}),
                    "error": completion.get("error"),
                }
                orchestration["command_completions"].append(completion_summary)

                if not completion.get("ok"):
                    job.return_code = 1
                    self._transition_job(job, status="failed", worker_stage="failed")
                    job.error = str(completion.get("error") or "Model runtime generation failed.")
                    traceback_text = str(completion.get("traceback") or "").strip()
                    if traceback_text:
                        job.log_lines.extend(traceback_text.splitlines()[-40:])
                    break

                runtime_result = dict(completion.get("result") or {})
                runtime_completed = int(runtime_result.get("completed_batches") or 0)
                if runtime_completed != 1:
                    job.return_code = 1
                    self._transition_job(job, status="failed", worker_stage="failed")
                    job.error = (
                        "Resident runtime image contract violation: each WebUI image iteration must "
                        f"complete exactly one runtime batch, but reported {runtime_completed}."
                    )
                    break

                attempted_images += 1
                completed_images += 1
                job.batch_seed_history.append(int(image_seed))
                orchestration.update(
                    {
                        "attempted_images": attempted_images,
                        "completed_images": completed_images,
                        "skipped_images": int(job.skipped_images),
                        "completed_batches": attempted_images // requested_batch_size,
                        "current_batch": parent_batch_number,
                        "current_image": image_number,
                        "last_completed_at": _utc_now(),
                    }
                )
                job.model_runtime_diagnostics["batch_orchestration"] = orchestration
                self._touch_job_runtime(job, progress=True)
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    worker_stage="image_completed",
                    batch_number=parent_batch_number,
                    batch_count=requested_batch_count,
                    batch_size=requested_batch_size,
                    image_number=image_number,
                    image_in_batch=image_in_batch,
                    image_count=None if unlimited else requested_image_count,
                    completed_images=completed_images,
                    skipped_images=job.skipped_images,
                    unlimited=unlimited,
                    output_count=len(job.output_paths),
                )
                console.write(
                    "WEBUI_IMAGE_COMPLETE_JSON: "
                    + json.dumps(
                        {
                            "image_number": image_number,
                            "completed_images": completed_images,
                            "skipped_images": job.skipped_images,
                            "output_count": len(job.output_paths),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                console.flush()

                has_more_images = unlimited or attempted_images < requested_image_count
                if await self._pause_between_images_if_requested(
                    job,
                    has_more_images=has_more_images,
                ):
                    job.resume_image_index = attempted_images
                    job.resume_completed_images = completed_images
                    orchestration["suspended_at"] = _utc_now()
                    job.model_runtime_diagnostics["batch_orchestration"] = orchestration
                    self._persist_job(job)
                    return
                await asyncio.sleep(0)

        job.model_runtime_diagnostics["command_completion"] = last_completion
        orchestration["attempted_images"] = attempted_images
        orchestration["completed_images"] = completed_images
        orchestration["skipped_images"] = int(job.skipped_images)
        orchestration["completed_batches"] = attempted_images // requested_batch_size
        orchestration["finished_at"] = _utc_now()
        job.model_runtime_diagnostics["batch_orchestration"] = orchestration

        if job.status in {"cancelling", "cancelled"}:
            self._transition_job(job, status="cancelled", worker_stage="cancelled")
            job.return_code = 130
        elif job.status == "failed":
            job.return_code = 1
        else:
            job.return_code = 0
            self._transition_job(job, status="completed", worker_stage="completed")
            self._apply_model_parity(job)

        job.resume_image_index = 0
        job.resume_completed_images = 0
        self._job_resume_events.pop(job.job_id, None)
        self._finalize_resident_job(job)

    def _finalize_resident_job(self, job: GenerationJob) -> None:
        job.process = None
        job.completed_at = _utc_now()
        job.updated_at = job.completed_at
        final_status = self.model_runtime.status()
        final_memory = dict(final_status.get("memory") or {})
        if final_memory:
            job.memory_status = {
                **final_memory,
                "event": "model_runtime_final_status",
                "stage": final_status.get("stage"),
                "active_stage": final_status.get("active_stage"),
                "updated_at": job.completed_at,
            }
        job.model_diagnostics["live_preview"] = self.diagnostics_payload(job)["phase09h_validation"]
        job.model_diagnostics["resident_model"] = {
            **dict(job.model_runtime_diagnostics),
            "final_status": final_status,
            "execution_mode": job.execution_mode,
        }
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
