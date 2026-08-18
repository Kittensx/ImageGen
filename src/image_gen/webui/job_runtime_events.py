from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from image_gen.systems.memory.telemetry import normalize_cuda_memory_payload
from image_gen.webui.job_request_normalization import _coerce_top_level_number
from image_gen.webui.job_store import _utc_now
from modules.prompt_parsers import default_prompt_parser_registry

_IMAGE_LINE = re.compile(r"^\s*Image \[seed (?P<seed>[^\]]+)\]:\s*(?P<path>.+?)\s*$")
_FAILURE_BUNDLE_LINE = re.compile(r"\(failure bundle:\s*(.+?)\)\s*$", re.IGNORECASE)
_MODEL_DIAGNOSTIC_LINE = re.compile(r"^MODEL_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_PROMPT_PARSER_DIAGNOSTIC_LINE = re.compile(r"^PROMPT_PARSER_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_OUTPUT_QUALITY_DIAGNOSTIC_LINE = re.compile(r"^OUTPUT_QUALITY_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_RUNTIME_DIAGNOSTIC_LINE = re.compile(r"^RUNTIME_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_STEP_PROGRESS_LINE = re.compile(r"STEP_PROGRESS_JSON:\s*(\{.*\})\s*$")
_STEP_PREVIEW_LINE = re.compile(r"STEP_PREVIEW_JSON:\s*(\{.*\})\s*$")
_GENERATION_SEED_LINE = re.compile(r"^GENERATION_SEED_JSON:\s*(\{.*\})\s*$")
_LIVE_PREVIEW_SUMMARY_LINE = re.compile(r"^LIVE_PREVIEW_SUMMARY_JSON:\s*(\{.*\})\s*$")
_MEMORY_STATUS_LINE = re.compile(r"MEMORY_STATUS_JSON:\s*(\{.*\})\s*$")
_MODEL_RUNTIME_STATUS_LINE = re.compile(r"^MODEL_RUNTIME_STATUS_JSON:\s*(\{.*\})\s*$")
_ASYNC_OUTPUT_SAVE_STATUS_LINE = re.compile(r"^ASYNC_OUTPUT_SAVE_STATUS_JSON:\s*(\{.*\})\s*$")
_ASYNC_OUTPUT_SAVE_ERROR_LINE = re.compile(r"^ASYNC_OUTPUT_SAVE_ERROR_JSON:\s*(\{.*\})\s*$")

def _normalize_live_memory_status(value: Mapping[str, Any] | None) -> dict[str, Any]:
    status = dict(value or {})
    snapshot = dict(status.get("latest_snapshot") or {})
    snapshot["cuda"] = normalize_cuda_memory_payload(
        dict(snapshot.get("cuda") or {})
    )
    status["latest_snapshot"] = snapshot
    if status.get("job_peak_allocated_vram_bytes") is None:
        status["job_peak_allocated_vram_bytes"] = status.get(
            "peak_allocated_vram_bytes"
        )
    if status.get("job_peak_reserved_vram_bytes") is None:
        status["job_peak_reserved_vram_bytes"] = status.get(
            "peak_reserved_vram_bytes"
        )
    return status


class JobRuntimeEventsMixin:
    def _apply_step_progress_payload(self, job: GenerationJob, payload: Mapping[str, Any]) -> bool:
        if job.status in {"cancelling", "cancelled", "failed", "completed"}:
            return False

        step_number = max(
            0,
            int(_coerce_top_level_number(payload.get("step"), integer=True, default=0) or 0),
        )
        total_steps = max(
            step_number,
            int(
                _coerce_top_level_number(
                    payload.get("total_steps"), integer=True, default=step_number
                )
                or step_number
            ),
        )
        phase_index = max(
            0,
            int(_coerce_top_level_number(payload.get("phase_index"), integer=True, default=0) or 0),
        )
        previous_phase = int(job.sampling_timing.get("phase_index") or 0)
        if phase_index == previous_phase and step_number < int(job.current_step or 0):
            return False

        progress_percent = _coerce_top_level_number(
            payload.get("progress_percent"), integer=False, default=None
        )
        if progress_percent is None:
            progress_percent = (step_number / max(total_steps, 1)) * 100.0
        progress_percent = min(max(float(progress_percent), 0.0), 100.0)
        updated_at = _utc_now()

        if step_number > 0 and total_steps > 0 and step_number >= total_steps:
            self._transition_job(job, status="finalizing", worker_stage="sampling_complete")
        else:
            self._transition_job(job, status="running", worker_stage="sampling")

        job.current_step = step_number
        job.total_steps = total_steps
        job.progress_percent = progress_percent
        job.sampling_timing = {
            "schema_version": int(
                _coerce_top_level_number(payload.get("schema_version"), integer=True, default=1)
                or 1
            ),
            "phase_index": phase_index,
            "description": str(payload.get("description") or "Sampling"),
            "unit": str(payload.get("unit") or "step"),
            "step": step_number,
            "total_steps": total_steps,
            "step_duration_ms": _coerce_top_level_number(
                payload.get("step_duration_ms"), integer=False, default=None
            ),
            "average_step_ms": _coerce_top_level_number(
                payload.get("average_step_ms"), integer=False, default=None
            ),
            "rolling_average_step_ms": _coerce_top_level_number(
                payload.get("rolling_average_step_ms"), integer=False, default=None
            ),
            "sampling_elapsed_ms": _coerce_top_level_number(
                payload.get("sampling_elapsed_ms"), integer=False, default=0.0
            ),
            "estimated_remaining_ms": _coerce_top_level_number(
                payload.get("estimated_remaining_ms"), integer=False, default=None
            ),
            "timed_step_count": int(
                _coerce_top_level_number(payload.get("timed_step_count"), integer=True, default=0)
                or 0
            ),
            "updated_at": updated_at,
        }
        job.updated_at = updated_at
        job.last_runtime_line_at = updated_at
        job.last_progress_at = updated_at
        self._persist_job(job)
        self._publish_event(
            job,
            "job-progress",
            current_step=job.current_step,
            total_steps=job.total_steps,
            progress_percent=job.progress_percent,
            sampling_timing=dict(job.sampling_timing),
        )
        return True

    def _apply_step_preview_payload(self, job: GenerationJob, payload: Mapping[str, Any]) -> bool:
        step_number = max(1, int(_coerce_top_level_number(payload.get("step"), integer=True, default=1) or 1))
        is_final = bool(payload.get("is_final", False))
        incoming_filename = str(payload.get("filename") or "")
        incoming_preview_path = str(payload.get("preview_path") or "")
        incoming_preview_suspended = bool(payload.get("preview_image_suspended", False))
        incoming_has_preview_image = bool(
            not incoming_preview_suspended
            and (incoming_filename or incoming_preview_path)
        )
        if job.status in {"cancelling", "cancelled", "failed"}:
            job.stale_preview_events_ignored += 1
            return False
        # Per-step telemetry is emitted immediately, while asynchronous image
        # encoding may finish several sampler steps later. Reject stale
        # telemetry, but retain delayed image frames without rolling progress
        # backward.
        if (
            step_number < int(job.current_step or 0)
            and not is_final
            and not incoming_has_preview_image
        ):
            job.stale_preview_events_ignored += 1
            job.live_preview_metrics["stale_preview_events_ignored"] = job.stale_preview_events_ignored
            return False
        total_steps = max(step_number, int(_coerce_top_level_number(payload.get("total_steps"), integer=True, default=step_number) or step_number))
        progress_percent = float(payload.get("progress_percent") or (step_number / max(total_steps, 1)) * 100.0)
        progress_percent = min(max(progress_percent, 0.0), 100.0)
        updated_at = str(payload.get("updated_at") or _utc_now())
        record = {
            "step": step_number,
            "total_steps": total_steps,
            "progress_percent": progress_percent,
            "decode_mode": str(payload.get("decode_mode") or "fast"),
            "filename": incoming_filename,
            "is_final": is_final,
            "telemetry_only": bool(payload.get("telemetry_only", False)),
            "updated_at": updated_at,
            "sampler_name": payload.get("sampler_name"),
            "scheduler_name": payload.get("scheduler_name"),
            "preview_path": incoming_preview_path,
            "image_width": int(_coerce_top_level_number(payload.get("image_width"), integer=True, default=0) or 0),
            "image_height": int(_coerce_top_level_number(payload.get("image_height"), integer=True, default=0) or 0),
            "sigma": _coerce_top_level_number(payload.get("sigma"), integer=False, default=None),
            "timestep": _coerce_top_level_number(payload.get("model_timestep"), integer=False, default=None),
            "requested_cfg_scale": _coerce_top_level_number(payload.get("requested_cfg_scale"), integer=False, default=None),
            "effective_cfg_scale": _coerce_top_level_number(payload.get("effective_cfg_scale"), integer=False, default=None),
            "guidance_mode": str(payload.get("guidance_mode") or payload.get("cfg_guidance_mode") or "flat"),
            "cfg_rescale": _coerce_top_level_number(payload.get("cfg_rescale"), integer=False, default=0.0),
            "cfg_rescale_applied": bool(payload.get("cfg_rescale_applied", False)),
            "override_source": str(payload.get("override_source") or "base_request"),
            "transition_id": payload.get("transition_id"),
            "preview_image_suspended": incoming_preview_suspended,
            "preview_image_suspension_reason": str(payload.get("preview_image_suspension_reason") or ""),
            "preview_image_suspension_source": str(payload.get("preview_image_suspension_source") or ""),
            "preview_decoder_released": bool(payload.get("preview_decoder_released", False)),
            "cfg_telemetry_continues": bool(payload.get("cfg_telemetry_continues", False)),
        }
        root = self.live_preview_root_path(job)
        if root is not None:
            filename = record["filename"]
            if filename and not record["preview_path"]:
                record["preview_path"] = str(root / filename)
        keep_history = str(job.request.get("live_preview_keep_history") or "current_job").strip().lower()
        has_preview_image = bool(
            not record["preview_image_suspended"]
            and (record["filename"] or record["preview_path"])
        )
        record["preview_url"] = (
            (
                self._preview_latest_url(job, updated_at=updated_at)
                if keep_history == "latest_only"
                else self._preview_step_url(job, step_number, updated_at=updated_at)
            )
            if has_preview_image
            else ""
        )

        previous_step = int(job.current_step or 0)
        previous_total = int(job.total_steps or 0)
        previous_percent = float(job.progress_percent or 0.0)
        if is_final or (step_number >= total_steps and step_number >= previous_step):
            self._transition_job(job, status="finalizing", worker_stage="finalizing")
        elif job.status != "finalizing":
            self._transition_job(job, status="running", worker_stage="sampling")
        job.current_step = max(previous_step, step_number)
        job.total_steps = max(previous_total, total_steps, job.current_step)
        job.progress_percent = max(previous_percent, progress_percent)
        job.updated_at = updated_at
        job.last_runtime_line_at = updated_at
        job.last_progress_at = updated_at
        if has_preview_image:
            job.live_preview_decode_mode = record["decode_mode"]
            job.live_preview_path = record["preview_path"] or None
            job.live_preview_url = record["preview_url"]

            history = [item for item in job.live_preview_history if int(item.get("step", 0)) != step_number]
            history.append(record)
            history.sort(key=lambda item: int(item.get("step", 0)))
            if len(history) > self._live_preview_history_limit:
                history = history[-self._live_preview_history_limit:]
            job.live_preview_history = history
        elif record["preview_image_suspended"]:
            job.live_preview_metrics["image_decode_suspended"] = True
            job.live_preview_metrics["image_decode_suspension_reason"] = record[
                "preview_image_suspension_reason"
            ]
            job.live_preview_metrics["image_decode_suspension_source"] = record[
                "preview_image_suspension_source"
            ]
            job.live_preview_metrics["preview_decoder_released"] = bool(
                record["preview_decoder_released"]
            )
            job.live_preview_metrics["cfg_telemetry_continues_during_preview_suspension"] = True

        requested_cfg = record.get("requested_cfg_scale")
        effective_cfg = record.get("effective_cfg_scale")
        if bool(job.request.get("cfg_lab_enabled", False)) and (requested_cfg is not None or effective_cfg is not None):
            if requested_cfg is None:
                requested_cfg = effective_cfg
            if effective_cfg is None:
                effective_cfg = requested_cfg
            series = dict(job.live_cfg_step_series or {})
            points = [
                dict(item)
                for item in (series.get("points") or [])
                if int(item.get("step_index", -1)) != step_number - 1
            ]
            points.append({
                "step_index": step_number - 1,
                "requested_cfg_scale": float(requested_cfg),
                "effective_cfg_scale": float(effective_cfg),
                "sigma": record.get("sigma"),
                "timestep": record.get("timestep"),
                "guidance_mode": record.get("guidance_mode") or "flat",
                "cfg_rescale": float(record.get("cfg_rescale") or 0.0),
                "cfg_rescale_applied": bool(record.get("cfg_rescale_applied", False)),
                "override_source": record.get("override_source") or "base_request",
                "transition_id": record.get("transition_id"),
            })
            points.sort(key=lambda item: int(item.get("step_index", 0)))
            series.update({
                "schema_version": 1,
                "coordinate": "live_denoising_step",
                "source": "preview_stream",
                "supports_future_step_overrides": True,
                "points": points,
            })
            job.live_cfg_step_series = series

        self._persist_job(job)
        self._publish_event(
            job,
            "step-preview",
            step=step_number,
            total_steps=total_steps,
            progress_percent=progress_percent,
            live_preview_url=job.live_preview_url,
            live_preview_path=job.live_preview_path,
            live_preview_decode_mode=job.live_preview_decode_mode,
            requested_cfg_scale=record.get("requested_cfg_scale"),
            effective_cfg_scale=record.get("effective_cfg_scale"),
            guidance_mode=record.get("guidance_mode"),
            cfg_rescale=record.get("cfg_rescale"),
            live_cfg_step_series=job.live_cfg_step_series,
        )
        self._publish_event(
            job,
            "job-progress",
            current_step=job.current_step,
            total_steps=job.total_steps,
            progress_percent=job.progress_percent,
            live_preview_url=job.live_preview_url,
            live_preview_path=job.live_preview_path,
            live_preview_decode_mode=job.live_preview_decode_mode,
            requested_cfg_scale=record.get("requested_cfg_scale"),
            effective_cfg_scale=record.get("effective_cfg_scale"),
            guidance_mode=record.get("guidance_mode"),
            cfg_rescale=record.get("cfg_rescale"),
            live_cfg_step_series=job.live_cfg_step_series,
        )
        return True

    def _apply_output_save_status_payload(
        self,
        job: GenerationJob,
        payload: Mapping[str, Any],
    ) -> None:
        normalized = dict(payload or {})
        event = str(normalized.get("event") or "")
        if not event:
            return
        job.pending_save_batches = max(0, int(_coerce_top_level_number(normalized.get("pending_batches"), integer=True, default=0) or 0))
        job.completed_save_batches = max(0, int(_coerce_top_level_number(normalized.get("completed_batches"), integer=True, default=0) or 0))
        job.failed_save_batches = max(0, int(_coerce_top_level_number(normalized.get("failed_batches"), integer=True, default=0) or 0))
        job.output_save_status = normalized
        job.output_save_events.append({**normalized, "updated_at": _utc_now()})
        if len(job.output_save_events) > 24:
            job.output_save_events = job.output_save_events[-24:]
        if event in {"enqueued", "started", "completed", "failed"} and job.status not in {"completed", "cancelled", "failed"}:
            if job.status != "running" or event in {"started", "failed"}:
                self._transition_job(job, status="finalizing", worker_stage="saving_output")
        self._touch_job_runtime(job, progress=False)
        self._persist_job(job)
        self._publish_event(
            job,
            "job-progress",
            output_save_status=dict(job.output_save_status),
            pending_save_batches=job.pending_save_batches,
            completed_save_batches=job.completed_save_batches,
            failed_save_batches=job.failed_save_batches,
        )

    def _apply_runtime_line(self, job: GenerationJob, line: str) -> None:
        runtime_status_match = _MODEL_RUNTIME_STATUS_LINE.match(line)
        if runtime_status_match:
            try:
                status_payload = json.loads(runtime_status_match.group(1))
            except json.JSONDecodeError:
                status_payload = {}
            if isinstance(status_payload, dict):
                stage = str(status_payload.get("stage") or "preparing_model")
                batch_orchestration = dict(
                    job.model_runtime_diagnostics.get("batch_orchestration") or {}
                )
                job.model_runtime_diagnostics = dict(status_payload)
                if batch_orchestration:
                    job.model_runtime_diagnostics["batch_orchestration"] = batch_orchestration
                runtime_memory = dict(status_payload.get("memory") or {})
                component_devices = dict(status_payload.get("component_devices") or {})
                if runtime_memory or component_devices:
                    active_gpu_components = [
                        str(component)
                        for component, device in component_devices.items()
                        if str(device or "").lower().startswith("cuda")
                    ]
                    offloaded_components = [
                        str(component)
                        for component, device in component_devices.items()
                        if not str(device or "").lower().startswith("cuda")
                    ]
                    previous = dict(job.memory_status or {})
                    previous_snapshot = dict(previous.get("latest_snapshot") or {})
                    previous_cuda = dict(previous_snapshot.get("cuda") or {})
                    current_allocated = runtime_memory.get("allocated_bytes")
                    current_reserved = runtime_memory.get("reserved_bytes")
                    previous_peak_allocated = previous.get("peak_allocated_vram_bytes")
                    previous_peak_reserved = previous.get("peak_reserved_vram_bytes")
                    normalized_cuda = normalize_cuda_memory_payload(
                        {
                            **previous_cuda,
                            "available": bool(runtime_memory),
                            "device_name": runtime_memory.get("device_name"),
                            "allocated_vram_bytes": current_allocated,
                            "reserved_vram_bytes": current_reserved,
                            "free_vram_bytes": runtime_memory.get("free_bytes"),
                            "total_vram_bytes": runtime_memory.get("total_bytes"),
                        }
                    )
                    job.memory_status = {
                        **previous,
                        "event": "model_runtime_status",
                        "stage": stage,
                        "active_stage": stage,
                        "active_gpu_components": active_gpu_components,
                        "offloaded_components": offloaded_components,
                        "latest_snapshot": {
                            **previous_snapshot,
                            "pipeline_stage": stage,
                            "cuda": normalized_cuda,
                        },
                        "peak_allocated_vram_bytes": max(
                            int(previous_peak_allocated or 0),
                            int(current_allocated or 0),
                        ),
                        "peak_reserved_vram_bytes": max(
                            int(previous_peak_reserved or 0),
                            int(current_reserved or 0),
                        ),
                        "job_peak_allocated_vram_bytes": max(
                            int(previous.get("job_peak_allocated_vram_bytes") or 0),
                            int(previous_peak_allocated or 0),
                            int(current_allocated or 0),
                        ),
                        "job_peak_reserved_vram_bytes": max(
                            int(previous.get("job_peak_reserved_vram_bytes") or 0),
                            int(previous_peak_reserved or 0),
                            int(current_reserved or 0),
                        ),
                        "telemetry_source": "resident_model_runtime",
                        "updated_at": _utc_now(),
                    }
                if stage in {"preparing_model", "loading_tokenizer"}:
                    self._transition_job(job, status="preparing_model", worker_stage=stage)
                elif stage in {"loading_checkpoint", "moving_to_gpu", "reusing_checkpoint", "model_ready"}:
                    self._transition_job(job, status="warming_model", worker_stage=stage)
                elif stage in {"applying_retention_policy", "ready"} and job.status not in {"completed", "cancelled", "failed"}:
                    self._transition_job(job, status="finalizing", worker_stage=stage)
                elif stage == "running":
                    self._transition_job(job, status="running", worker_stage=stage)
                else:
                    self._transition_job(job, worker_stage=stage)
                self._touch_job_runtime(job, progress=stage == "running")
                if stage == "failed" and job.status not in {"cancelled", "cancelling"}:
                    job.error = str(status_payload.get("error") or status_payload.get("last_error") or "Model runtime failed.")
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    worker_stage=job.worker_stage,
                    model_runtime=dict(status_payload),
                    memory_status=dict(job.memory_status),
                )
            return

        output_save_status_match = _ASYNC_OUTPUT_SAVE_STATUS_LINE.match(line)
        if output_save_status_match:
            try:
                payload = json.loads(output_save_status_match.group(1))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                self._apply_output_save_status_payload(job, payload)
            return

        output_save_error_match = _ASYNC_OUTPUT_SAVE_ERROR_LINE.match(line)
        if output_save_error_match:
            try:
                payload = json.loads(output_save_error_match.group(1))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                merged = dict(job.output_save_status or {})
                merged.update(payload)
                merged["event"] = merged.get("event") or "failed"
                self._apply_output_save_status_payload(job, merged)
            return

        seed_match = _GENERATION_SEED_LINE.match(line)
        if seed_match:
            try:
                seed_payload = json.loads(seed_match.group(1))
            except json.JSONDecodeError:
                seed_payload = {}
            try:
                job.resolved_seed = int(seed_payload.get("base_seed"))
            except (TypeError, ValueError):
                job.resolved_seed = None
            job.resolved_seeds = []
            for value in seed_payload.get("image_seeds") or []:
                try:
                    job.resolved_seeds.append(int(value))
                except (TypeError, ValueError):
                    continue
            self._touch_job_runtime(job)
            self._persist_job(job)
            self._publish_event(
                job,
                "job-progress",
                resolved_seed=job.resolved_seed,
                resolved_seeds=list(job.resolved_seeds),
            )
            return

        preview_summary_match = _LIVE_PREVIEW_SUMMARY_LINE.match(line)
        if preview_summary_match:
            try:
                preview_summary = json.loads(preview_summary_match.group(1))
            except json.JSONDecodeError:
                preview_summary = {}
            if isinstance(preview_summary, dict):
                self._transition_job(job, status="finalizing", worker_stage="finalizing")
                job.live_preview_metrics.update(preview_summary)
                job.live_preview_metrics["sse_clients_connected"] = job.sse_clients_connected
                job.live_preview_metrics["sse_clients_peak"] = job.sse_clients_peak
                job.live_preview_metrics["stale_preview_events_ignored"] = job.stale_preview_events_ignored
                self._touch_job_runtime(job)
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    live_preview_metrics=dict(job.live_preview_metrics),
                )
            return

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
                updated_at = _utc_now()
                job.memory_status = {
                    **status_payload,
                    "event": memory_payload.get("event"),
                    "stage": memory_payload.get("stage"),
                    "active_stage": memory_payload.get("active_stage"),
                    "updated_at": updated_at,
                }
                job.updated_at = updated_at
                job.last_runtime_line_at = updated_at
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    memory_status=dict(job.memory_status),
                )
            return

        image_match = _IMAGE_LINE.match(line)
        if image_match:
            self._record_job_output(
                job,
                image_match.group("path"),
                seed_text=image_match.group("seed"),
            )
            return

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
                job.model_runtime_diagnostics["resident_reuse_benefited"] = bool(payload.get("cache_reused"))

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

        step_preview_match = _STEP_PREVIEW_LINE.search(line)
        if step_preview_match:
            try:
                payload = json.loads(step_preview_match.group(1))
            except json.JSONDecodeError:
                payload = {"parse_error": line}
            if isinstance(payload, dict):
                self._apply_step_preview_payload(job, payload)

    def _apply_model_parity(self, job: GenerationJob) -> None:
        runtime_model = dict(job.model_diagnostics.get("runtime") or {})
        expected_model = str(job.model_selection.get("resolved_path") or job.request.get("model_path") or "").strip()
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
                    "checkpoint loaded by the runtime. "
                    f"Selected: {expected_model}. Loaded: {loaded_model}."
                )
        job.model_diagnostics["model_parity"] = {
            "selected_path": expected_model,
            "loaded_path": loaded_model,
            "matches": model_paths_match,
            "enforced": bool(expected_model),
        }

    def diagnostics_payload(self, job: GenerationJob) -> dict[str, Any]:
        payload = job.to_dict()
        metrics = dict(job.live_preview_metrics)
        payload["phase11d_scheduler"] = {
            "requested_settings": dict(job.scheduler_settings_requested),
            "effective_settings": dict(job.scheduler_settings_effective),
            "compatibility_policy": dict(job.scheduler_compatibility_policy),
            "validation_warnings": list(job.scheduler_validation_warnings),
            "validation_warning_count": len(job.scheduler_validation_warnings),
            "preset_reference": dict(job.scheduler_preset_reference),
            "requested_hash": job.scheduler_requested_hash,
            "effective_hash": job.scheduler_effective_hash,
            "step_count_source": job.scheduler_step_count_source,
            "warnings_acknowledged": job.scheduler_warnings_acknowledged,
        }
        live_cfg_points = list((job.live_cfg_step_series or {}).get("points") or [])
        payload["phase11h1_live_cfg_preview"] = {
            "visual_enabled": bool(job.request.get("live_preview_cfg_visual_enabled", False)),
            "series": {
                **dict(job.live_cfg_step_series or {}),
                "points": live_cfg_points,
            },
            "latest_requested_cfg_scale": (
                live_cfg_points[-1].get("requested_cfg_scale") if live_cfg_points else None
            ),
            "latest_effective_cfg_scale": (
                live_cfg_points[-1].get("effective_cfg_scale") if live_cfg_points else None
            ),
        }
        payload["phase13_memory"] = {
            **dict(job.memory_status or {}),
            "requested_policy": job.request.get("memory_policy", "auto"),
            "vram_safety_margin_mb": job.request.get("memory_vram_safety_margin_mb", 1024),
            "allow_preview_suspension_on_oom": job.request.get(
                "memory_allow_preview_suspension_on_oom", True
            ),
            "cfg_telemetry_continues_during_preview_suspension": True,
        }
        payload["phase13c_async_output_save_pipeline"] = {
            "pending_save_batches": int(job.pending_save_batches),
            "completed_save_batches": int(job.completed_save_batches),
            "failed_save_batches": int(job.failed_save_batches),
            "latest_status": dict(job.output_save_status),
            "recent_events": list(job.output_save_events[-8:]),
        }
        payload["phase14k7_preview_memory_policy"] = {
            "requested_policy": job.request.get("preview_policy", "normal"),
            "image_decode_suspended": bool(
                metrics.get("image_decode_suspended")
                or (job.memory_status or {}).get("preview_image_decode_suspended")
            ),
            "suspension_reason": str(
                metrics.get("image_decode_suspension_reason")
                or (job.memory_status or {}).get("preview_image_decode_suspension_reason")
                or ""
            ),
            "suspension_source": str(
                metrics.get("image_decode_suspension_source")
                or (job.memory_status or {}).get("preview_image_decode_suspension_source")
                or ""
            ),
            "preview_decoder_released": bool(
                metrics.get("preview_decoder_released")
                or (job.memory_status or {}).get("preview_decoder_released")
            ),
            "cfg_telemetry_continues": True,
        }
        payload["phase13c_prompt_parser"] = {
            "requested_parser": job.request.get("prompt_parser_name", "legacy"),
            "requested_options": dict(job.request.get("prompt_parser_kwargs") or {}),
            "shortcut_profile_name": job.request.get("prompt_shortcut_profile_name", "legacy_default"),
            "shortcut_profile_snapshot": dict(job.request.get("prompt_shortcut_profile_snapshot") or {}),
            "parser_preset_name": job.request.get("prompt_parser_preset_name", ""),
            "runtime": dict(job.prompt_parser_diagnostics),
            "available": default_prompt_parser_registry().has(
                job.request.get("prompt_parser_name", "legacy"),
                require_available=True,
            ),
        }
        payload["model_residency"] = {
            "execution_mode": job.execution_mode,
            "worker_stage": job.worker_stage,
            "model_runtime_status": self.model_runtime.status(),
            "job_runtime_diagnostics": dict(job.model_runtime_diagnostics),
            "watchdog": dict(self._watchdog_report),
            "resident_reuse_benefited": bool(
                (job.model_diagnostics.get("runtime") or {}).get("cache_reused")
                or job.model_runtime_diagnostics.get("resident_reuse_benefited")
            ),
        }
        payload["phase09h_validation"] = {
            "preview_enabled": bool(job.request.get("live_preview_enabled", False)),
            "preview_mode": job.request.get("live_preview_mode"),
            "preview_interval": job.request.get("live_preview_interval"),
            "preview_width": job.request.get("live_preview_width"),
            "preview_format": job.request.get("live_preview_format"),
            "preview_frames_emitted": metrics.get("preview_frames_emitted", metrics.get("frames_processed", 0)),
            "preview_frames_failed": metrics.get("preview_frames_failed", metrics.get("worker_failures", 0)),
            "preview_decode_time_total_ms": metrics.get("preview_decode_time_total_ms", 0.0),
            "preview_encode_time_total_ms": metrics.get("preview_encode_time_total_ms", 0.0),
            "preview_last_step": metrics.get("preview_last_step", job.current_step),
            "sse_clients_connected": job.sse_clients_connected,
            "sse_clients_peak": job.sse_clients_peak,
            "stale_preview_events_ignored": job.stale_preview_events_ignored,
            "runtime_python": platform.python_version(),
            "runtime_torch": torch.__version__,
            "model_path": job.request.get("model_path"),
            "model_architecture": dict(job.model_selection.get("architecture_contract") or {}),
            "model_architecture_summary": job.model_selection.get("architecture_summary"),
            "sampler": job.request.get("sampler_name"),
            "scheduler": job.request.get("scheduler_name"),
            "seed": job.resolved_seed if job.resolved_seed is not None else job.request.get("seed"),
        }
        return payload
