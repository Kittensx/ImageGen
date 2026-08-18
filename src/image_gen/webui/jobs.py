from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping
from urllib.parse import quote

import torch

from image_gen.contracts import (
    PROMPT_ASSET_CONTRACT_VERSION,
    extract_hires_failure_stage,
    hires_failure_stage_label,
    normalize_prompt_asset_list,
)
from image_gen.systems.outpainting import (
    OUTPAINT_ANCHORS,
    OUTPAINT_CONTEXT_SEED_MODES,
    OUTPAINT_LATENT_STRATEGIES,
    OUTPAINT_PROMPT_MODES,
    OUTPAINT_SHAPE_TARGET_MODES,
    OUTPAINT_SOURCE_HANDOFF_MODES,
    resolve_outpaint_shape_target,
    extract_outpaint_failure_stage,
    outpaint_failure_label,
)
from image_gen.systems.image_conditioning import (
    DEFAULT_HIRES_STEP_POLICY,
    SUPPORTED_HIRES_STEP_POLICIES,
)
from image_gen.systems.memory.telemetry import normalize_cuda_memory_payload
from image_gen.systems.registry import RuntimeRegistrySystem
from image_gen.runtime_options import (
    RUNTIME_REPLAY_JOB_FIELDS,
    build_runtime_startup_status,
    resolve_runtime_startup_options,
    runtime_request_settings,
)
from image_gen.runtime.hires_sizing import apply_hires_dimensions
from image_gen.runtime.scheduler_settings import (
    normalize_scheduler_payload,
    scheduler_resolution_from_payload,
)
from image_gen.webui.schema_utils import coerce_value_by_schema, normalize_config_schema
from image_gen.webui.selection import WebUISelectionResolver
from image_gen.webui.model_runtime import ModelRuntimeUnavailable, ResidentModelRuntimeClient
from image_gen.webui.randomization import (
    apply_parameter_ranges,
    iter_seed_plan,
    normalize_parameter_ranges,
    parse_seed_plan,
)
from image_gen.webui.store import DEFAULT_FORCED_LIVE_PREVIEW_INTERVAL, FORCED_LIVE_PREVIEW_MODE
from modules.project_context import ProjectContext
from modules.sdxl_runtime_profile import profile_for_sdxl_filename, sdxl_profile_recommendation_warnings
from modules.checkpoint_inspector import CheckpointInspector
from modules.sd3_runtime_profile import profile_from_checkpoint_variant, sd3_profile_recommendation_warnings
from modules.prompt_parsers import PromptProcessingPreflight, default_prompt_parser_registry
from modules.prompt_shortcuts import PromptShortcutProfileDescriptor, default_prompt_shortcut_registry, validate_prompt_shortcut_profile

from image_gen.webui.job_request_normalization import (
    JobRequestNormalizationMixin,
    _coerce_boolean,
    _coerce_top_level_number,
    _drop_default_or_empty_values,
    _normalize_top_level_request,
    apply_vae_selection_policy,
    normalize_generation_request,
)
from image_gen.webui.job_store import JobStoreMixin, _ACTIVE_JOB_STATUSES, _utc_now
from image_gen.webui.job_preview import JobPreviewMixin
from image_gen.webui.job_runtime_events import (
    JobRuntimeEventsMixin,
    _ASYNC_OUTPUT_SAVE_ERROR_LINE,
    _ASYNC_OUTPUT_SAVE_STATUS_LINE,
    _FAILURE_BUNDLE_LINE,
    _GENERATION_SEED_LINE,
    _IMAGE_LINE,
    _LIVE_PREVIEW_SUMMARY_LINE,
    _MEMORY_STATUS_LINE,
    _MODEL_DIAGNOSTIC_LINE,
    _MODEL_RUNTIME_STATUS_LINE,
    _OUTPUT_QUALITY_DIAGNOSTIC_LINE,
    _PROMPT_PARSER_DIAGNOSTIC_LINE,
    _RUNTIME_DIAGNOSTIC_LINE,
    _STEP_PREVIEW_LINE,
    _STEP_PROGRESS_LINE,
    _normalize_live_memory_status,
)
from image_gen.webui.job_queue import (
    JobQueueControlMixin,
    _CANCELLABLE_JOB_STATUSES,
    _ReorderableJobQueue,
)
from image_gen.webui.job_watchdog import JobWatchdogMixin, _timestamp_from_iso
from image_gen.webui.job_resident_executor import ResidentJobExecutorMixin
from image_gen.webui.job_isolated_executor import IsolatedJobExecutorMixin, _SUBPROCESS_STREAM_LIMIT




















@dataclass
class GenerationJob:
    job_id: str
    request: dict[str, Any]
    status: str = "queued"
    worker_stage: str = "queued"
    execution_mode: str = "pending"
    model_runtime_diagnostics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str | None = None
    status_changed_at: str = field(default_factory=_utc_now)
    last_progress_at: str | None = None
    last_runtime_line_at: str | None = None
    return_code: int | None = None
    output_paths: list[str] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    error: str | None = None
    job_root: str | None = None
    console_log_path: str | None = None
    failure_bundle_path: str | None = None
    live_preview_root: str | None = None
    live_preview_latest_path: str | None = None
    live_preview_path: str | None = None
    live_preview_url: str | None = None
    live_preview_decode_mode: str | None = None
    live_preview_history: list[dict[str, Any]] = field(default_factory=list)
    live_cfg_step_series: dict[str, Any] = field(default_factory=lambda: {
        "schema_version": 1,
        "coordinate": "live_denoising_step",
        "source": "preview_stream",
        "supports_future_step_overrides": True,
        "points": [],
    })
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float | None = None
    resolved_seed: int | None = None
    resolved_seeds: list[int] = field(default_factory=list)
    final_output_url: str | None = None
    live_preview_metrics: dict[str, Any] = field(default_factory=dict)
    sampling_timing: dict[str, Any] = field(default_factory=dict)
    memory_status: dict[str, Any] = field(default_factory=dict)
    sse_clients_connected: int = 0
    sse_clients_peak: int = 0
    stale_preview_events_ignored: int = 0
    terminal_events_emitted: int = 0
    model_selection: dict[str, Any] = field(default_factory=dict)
    model_diagnostics: dict[str, Any] = field(default_factory=dict)
    prompt_parser_diagnostics: dict[str, Any] = field(default_factory=dict)
    output_quality_diagnostics: dict[str, Any] = field(default_factory=dict)
    prompt_preflight: dict[str, Any] = field(default_factory=dict)
    scheduler_settings_requested: dict[str, Any] = field(default_factory=dict)
    scheduler_settings_effective: dict[str, Any] = field(default_factory=dict)
    scheduler_validation_warnings: list[str] = field(default_factory=list)
    scheduler_compatibility_policy: dict[str, Any] = field(default_factory=dict)
    scheduler_preset_reference: dict[str, Any] = field(default_factory=dict)
    scheduler_requested_hash: str | None = None
    scheduler_effective_hash: str | None = None
    scheduler_step_count_source: str | None = None
    scheduler_warnings_acknowledged: bool = False
    output_save_status: dict[str, Any] = field(default_factory=dict)
    output_save_events: list[dict[str, Any]] = field(default_factory=list)
    pending_save_batches: int = 0
    completed_save_batches: int = 0
    failed_save_batches: int = 0
    pause_after_current_requested: bool = False
    pause_requested_at: str | None = None
    paused_at: str | None = None
    resumed_at: str | None = None
    resume_count: int = 0
    resume_image_index: int = 0
    resume_completed_images: int = 0
    batch_seed_history: list[int] = field(default_factory=list)
    scheduler_suspended: bool = False
    queue_paused_from_status: str | None = None
    skip_current_requested: bool = False
    skip_requested_at: str | None = None
    skipped_images: int = 0
    skipped_image_seeds: list[int] = field(default_factory=list)
    skip_events: list[dict[str, Any]] = field(default_factory=list)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "request": self.request,
            "status": self.status,
            "worker_stage": self.worker_stage,
            "execution_mode": self.execution_mode,
            "model_runtime_diagnostics": dict(self.model_runtime_diagnostics),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "status_changed_at": self.status_changed_at,
            "last_progress_at": self.last_progress_at,
            "last_runtime_line_at": self.last_runtime_line_at,
            "return_code": self.return_code,
            "output_paths": list(self.output_paths),
            "log_lines": self.log_lines[-80:],
            "error": self.error,
            "failure_stage_code": (
                extract_outpaint_failure_stage(self.error)
                or extract_hires_failure_stage(self.error)
            ),
            "failure_stage_label": (
                outpaint_failure_label(extract_outpaint_failure_stage(self.error))
                if extract_outpaint_failure_stage(self.error)
                else (
                    hires_failure_stage_label(extract_hires_failure_stage(self.error))
                    if extract_hires_failure_stage(self.error)
                    else ""
                )
            ),
            "failure_stage_domain": (
                "outpaint" if extract_outpaint_failure_stage(self.error)
                else ("hires" if extract_hires_failure_stage(self.error) else "")
            ),
            "job_root": self.job_root,
            "console_log_path": self.console_log_path,
            "failure_bundle_path": self.failure_bundle_path,
            "live_preview_root": self.live_preview_root,
            "live_preview_latest_path": self.live_preview_latest_path,
            "live_preview_path": self.live_preview_path,
            "live_preview_url": self.live_preview_url,
            "live_preview_decode_mode": self.live_preview_decode_mode,
            "live_preview_history": self.live_preview_history[-40:],
            "live_cfg_step_series": {
                **dict(self.live_cfg_step_series or {}),
                "points": list((self.live_cfg_step_series or {}).get("points") or []),
            },
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "resolved_seed": self.resolved_seed,
            "resolved_seeds": list(self.resolved_seeds),
            "final_output_url": self.final_output_url,
            "live_preview_metrics": dict(self.live_preview_metrics),
            "sampling_timing": dict(self.sampling_timing),
            "memory_status": dict(self.memory_status),
            "prompt_parser_diagnostics": dict(self.prompt_parser_diagnostics),
            "output_quality_diagnostics": dict(self.output_quality_diagnostics),
            "prompt_preflight": dict(self.prompt_preflight or self.request.get("prompt_preflight") or {}),
            "sse_clients_connected": int(self.sse_clients_connected),
            "sse_clients_peak": int(self.sse_clients_peak),
            "stale_preview_events_ignored": int(self.stale_preview_events_ignored),
            "terminal_events_emitted": int(self.terminal_events_emitted),
            "scheduler_name": self.request.get("scheduler_name"),
            "scheduler_preset_reference": dict(self.scheduler_preset_reference),
            "scheduler_preset_name": self.scheduler_preset_reference.get("name"),
            "scheduler_validation_warning_count": len(self.scheduler_validation_warnings),
            "scheduler_validation_warnings": list(self.scheduler_validation_warnings),
            "scheduler_compatibility_policy": dict(self.scheduler_compatibility_policy),
            "scheduler_requested_hash": self.scheduler_requested_hash,
            "scheduler_effective_hash": self.scheduler_effective_hash,
            "scheduler_step_count_source": self.scheduler_step_count_source,
            "scheduler_warnings_acknowledged": self.scheduler_warnings_acknowledged,
            "output_save_status": dict(self.output_save_status),
            "output_save_events": list(self.output_save_events[-16:]),
            "pending_save_batches": int(self.pending_save_batches),
            "completed_save_batches": int(self.completed_save_batches),
            "failed_save_batches": int(self.failed_save_batches),
            "pause_after_current_requested": bool(self.pause_after_current_requested),
            "pause_requested_at": self.pause_requested_at,
            "paused_at": self.paused_at,
            "resumed_at": self.resumed_at,
            "resume_count": int(self.resume_count),
            "resume_image_index": int(self.resume_image_index),
            "resume_completed_images": int(self.resume_completed_images),
            "batch_seed_history": list(self.batch_seed_history),
            "scheduler_suspended": bool(self.scheduler_suspended),
            "queue_paused_from_status": self.queue_paused_from_status,
            "skip_current_requested": bool(self.skip_current_requested),
            "skip_requested_at": self.skip_requested_at,
            "skipped_images": int(self.skipped_images),
            "skipped_image_seeds": list(self.skipped_image_seeds),
            "skip_events": list(self.skip_events[-32:]),
            "model_selection": dict(self.model_selection),
            "model_diagnostics": dict(self.model_diagnostics),
        }




class GenerationJobManager(
    JobRequestNormalizationMixin,
    JobStoreMixin,
    JobPreviewMixin,
    JobRuntimeEventsMixin,
    JobQueueControlMixin,
    JobWatchdogMixin,
    ResidentJobExecutorMixin,
    IsolatedJobExecutorMixin,
):
    """One-GPU reorderable scheduler backed by the canonical resident model runtime."""

    def __init__(
        self,
        context: ProjectContext,
        *,
        settings_provider: Callable[[], Mapping[str, Any]] | None = None,
        recent_output_provider: Callable[[Path], Mapping[str, Any] | None] | None = None,
        output_record_callback: Callable[[Path], Any] | None = None,
    ) -> None:
        self.context = context
        self.settings_provider = settings_provider
        self.recent_output_provider = recent_output_provider
        self.output_record_callback = output_record_callback
        self.registry = RuntimeRegistrySystem(project_context=context)
        self.selections = WebUISelectionResolver(self.registry)
        self.jobs: dict[str, GenerationJob] = {}
        # Reorderable one-GPU scheduler. Queued jobs live here; paused jobs do not
        # consume a queue slot and can be reinserted independently.
        self._queue: _ReorderableJobQueue = _ReorderableJobQueue()
        self._queue_available = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._started = False
        self._event_subscribers: dict[str, set[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._live_preview_history_limit = 64
        self._terminal_events_emitted: set[str] = set()
        self._queue_resume_event = asyncio.Event()
        self._queue_resume_event.set()
        self._job_resume_events: dict[str, asyncio.Event] = {}
        self._queue_pause_requested_at: str | None = None
        self._queue_pause_owner_job_id: str | None = None
        self.runtime_startup_options: dict[str, Any] = {}
        self._last_cleanup_report: dict[str, Any] = {}
        self._last_job_cache_report: dict[str, Any] = {}
        self.model_runtime = ResidentModelRuntimeClient(context)
        self._watchdog_task: asyncio.Task[None] | None = None
        self._watchdog_report: dict[str, Any] = {
            "enabled": True,
            "running": False,
            "interval_seconds": 5,
            "running_stall_timeout_seconds": 180,
            "transition_stall_timeout_seconds": 120,
            "checks": 0,
            "recoveries": 0,
            "last_check_at": None,
            "last_recovery_at": None,
            "last_recovery_reason": None,
            "last_recovery_job_id": None,
        }

    def _runtime_request_values(self) -> dict[str, Any]:
        if self.runtime_startup_options:
            options = self.runtime_startup_options
        else:
            options = resolve_runtime_startup_options(
                environment={},
                settings=self._application_settings(),
            )
        values = runtime_request_settings(options)
        application_settings = self._application_settings()
        raw_overrides = application_settings.get("runtime_job_overrides")
        overrides = dict(raw_overrides) if isinstance(raw_overrides, Mapping) else {}
        if not overrides:
            return values

        normalized = runtime_request_settings(
            resolve_runtime_startup_options(environment={}, settings=overrides)
        )
        for key in RUNTIME_REPLAY_JOB_FIELDS:
            if key in overrides and key in normalized:
                values[key] = normalized[key]
        return values

    def _live_preview_request_values(self, job_root: Path) -> dict[str, Any]:
        application_settings = (
            dict(self.settings_provider() or {})
            if callable(self.settings_provider)
            else {}
        )
        live_preview_root = job_root / "live-preview"
        return {
            "live_preview_enabled": application_settings.get(
                "live_preview_enabled", True
            ),
            # Step/progress/CFG telemetry is independent of decoded image frames.
            # This keeps the graph and sampler timing alive when the image-preview
            # checkbox is disabled or image decoding is suspended.
            "live_preview_telemetry_enabled": True,
            "live_preview_mode": FORCED_LIVE_PREVIEW_MODE,
            "live_preview_interval": application_settings.get(
                "live_preview_interval", DEFAULT_FORCED_LIVE_PREVIEW_INTERVAL
            ),
            "live_preview_width": application_settings.get(
                "live_preview_width", 384
            ),
            "live_preview_format": application_settings.get(
                "live_preview_format", "webp"
            ),
            "live_preview_keep_history": application_settings.get(
                "live_preview_keep_history", "current_job"
            ),
            "live_preview_batch_index": application_settings.get(
                "live_preview_batch_index", 0
            ),
            "live_preview_quality": application_settings.get(
                "live_preview_quality", 78
            ),
            "live_preview_adaptive_throttle": application_settings.get(
                "live_preview_adaptive_throttle", True
            ),
            "live_preview_adaptive_target_ratio": application_settings.get(
                "live_preview_adaptive_target_ratio", 0.75
            ),
            "live_preview_adaptive_recovery_ratio": application_settings.get(
                "live_preview_adaptive_recovery_ratio", 0.40
            ),
            "live_preview_adaptive_max_interval": application_settings.get(
                "live_preview_adaptive_max_interval", 8
            ),
            "live_preview_adaptive_window": application_settings.get(
                "live_preview_adaptive_window", 6
            ),
            "live_preview_adaptive_suspend_on_overhead": application_settings.get(
                "live_preview_adaptive_suspend_on_overhead", False
            ),
            "cfg_lab_enabled": application_settings.get("cfg_lab_enabled", False),
            "live_preview_cfg_visual_enabled": bool(
                application_settings.get("cfg_lab_enabled", False)
                and application_settings.get("live_preview_cfg_visual_enabled", False)
            ),
            "diagnostics": self._diagnostics_request_settings(application_settings),
            "external_vae_override_enabled": True,
            "vae_mode": "checkpoint_embedded_auto",
            **self._runtime_request_values(),
            "memory_pinned_cpu_memory": application_settings.get(
                "memory_pinned_cpu_memory", False
            ),
            "memory_allow_tiled_vae_fallback": application_settings.get(
                "memory_allow_tiled_vae_fallback", True
            ),
            "memory_allow_preview_suspension_on_oom": application_settings.get(
                "memory_allow_preview_suspension_on_oom", True
            ),
            "live_preview_root": str(live_preview_root),
            "live_preview_clone_tensors": False,
            "live_preview_async": True,
            "progress_json": True,
        }

    @staticmethod
    def _merge_runtime_preview_values(
        request_payload: dict[str, Any],
        preview_values: Mapping[str, Any],
    ) -> None:
        replay_fields = set(RUNTIME_REPLAY_JOB_FIELDS)
        for key, value in preview_values.items():
            if key in replay_fields and key in request_payload:
                continue
            request_payload[key] = value

    def _application_settings(self) -> dict[str, Any]:
        return dict(self.settings_provider() or {}) if callable(self.settings_provider) else {}

    @staticmethod
    def _diagnostics_request_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
        mode = str(settings.get("diagnostics_mode") or "failures_only").strip().lower()
        diagnostic_decode_enabled = bool(settings.get("diagnostic_decode_enabled", False))
        if mode == "off":
            return {
                "mode": "off",
                "failure_bundles": False,
                "export_events": False,
                "tensor_summaries": False,
                "tensor_statistics": False,
                "capture_output_quality": False,
                "diagnostic_decode_enabled": diagnostic_decode_enabled,
            }
        if mode == "every_run":
            return {
                "mode": "every_run",
                "failure_bundles": True,
                "export_events": True,
                "tensor_summaries": True,
                "tensor_statistics": False,
                "capture_output_quality": False,
                "diagnostic_decode_enabled": diagnostic_decode_enabled,
            }
        if mode == "deep_tensor":
            return {
                "mode": "deep_tensor",
                "failure_bundles": True,
                "export_events": True,
                "tensor_summaries": True,
                "tensor_statistics": True,
                "capture_output_quality": True,
                "diagnostic_decode_enabled": diagnostic_decode_enabled,
                "sampler_trace": {
                    "enabled": True,
                    "export_json": True,
                    "export_csv": False,
                    "export_txt_summary": True,
                    "capture_latents": False,
                    "capture_latent_every_n": 0,
                },
            }
        return {
            "mode": "failures_only",
            "failure_bundles": True,
            "export_events": False,
            "tensor_summaries": True,
            "tensor_statistics": False,
            "capture_output_quality": False,
            "diagnostic_decode_enabled": diagnostic_decode_enabled,
        }

    def _model_runtime_settings(self) -> dict[str, Any]:
        settings = self._application_settings()
        return {
            **self._runtime_request_values(),
            "memory_pinned_cpu_memory": settings.get("memory_pinned_cpu_memory", False),
            "memory_allow_tiled_vae_fallback": settings.get("memory_allow_tiled_vae_fallback", True),
            "memory_allow_preview_suspension_on_oom": settings.get("memory_allow_preview_suspension_on_oom", True),
        }









    def _touch_job_runtime(self, job: GenerationJob, *, progress: bool = False) -> str:
        now = _utc_now()
        job.updated_at = now
        job.last_runtime_line_at = now
        if progress:
            job.last_progress_at = now
        return now

    def _transition_job(
        self,
        job: GenerationJob,
        *,
        status: str | None = None,
        worker_stage: str | None = None,
    ) -> str:
        now = _utc_now()
        status_value = str(status or job.status)
        stage_value = str(worker_stage or job.worker_stage)
        if status_value != job.status:
            job.status = status_value
            job.status_changed_at = now
        if stage_value != job.worker_stage:
            job.worker_stage = stage_value
        job.updated_at = now
        return now

    def _recent_output_payload(self, image_path: str | Path) -> dict[str, Any] | None:
        try:
            resolved = Path(image_path).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

        payload: Mapping[str, Any] | None = None
        provider = self.recent_output_provider
        if callable(provider):
            try:
                payload = provider(resolved)
            except TypeError:
                payload = provider(Path(resolved))
            except Exception:
                payload = None
        else:
            try:
                from image_gen.webui.catalog import WebUICatalog

                payload = WebUICatalog(self.context).output_summary_from_path(resolved)
            except Exception:
                payload = None

        if isinstance(payload, Mapping):
            return dict(payload)
        return None

    def _record_job_output(
        self,
        job: GenerationJob,
        image_path: str | Path,
        *,
        seed_text: str | None = None,
    ) -> dict[str, Any] | None:
        image_value = str(image_path)
        self._transition_job(job, status="finalizing", worker_stage="saving_output")
        is_new_output = image_value not in job.output_paths
        if is_new_output:
            job.output_paths.append(image_value)
            if callable(self.output_record_callback):
                try:
                    self.output_record_callback(Path(image_value))
                except Exception:
                    # Profile/statistics accounting must never fail a generation job.
                    pass
        job.final_output_url = self._output_url_for_path(image_value)
        if job.resolved_seed is None and seed_text not in (None, ""):
            try:
                job.resolved_seed = int(str(seed_text))
            except (TypeError, ValueError):
                pass
        job.updated_at = _utc_now()
        self._persist_job(job)
        recent_output = self._recent_output_payload(image_value)
        payload = {
            "latest_output_path": image_value,
            "latest_output_url": job.final_output_url,
            "output_count": len(job.output_paths),
        }
        if recent_output is not None:
            payload["recent_output"] = recent_output
        self._publish_event(job, "job-output-produced", **payload)
        return recent_output











    async def start(self) -> None:
        self._started = True
        self.clear_job_cache(preserve_active=True, startup=True)
        self.cleanup_preview_directories()
        self._watchdog_settings()
        if self._worker_task is None or self._worker_task.done():
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_loop())
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def stop(self) -> None:
        self._started = False
        self._stopping = True
        self._queue_resume_event.set()
        self._queue_available.set()
        for resume_event in self._job_resume_events.values():
            resume_event.set()
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
        active_resident_job = next(
            (
                job
                for job in self.jobs.values()
                if job.execution_mode == "resident_model" and job.status in _ACTIVE_JOB_STATUSES
            ),
            None,
        )
        if active_resident_job is not None:
            await self.model_runtime.cancel_active(active_resident_job.job_id)
        for job in self.jobs.values():
            if job.process is not None and job.status in _ACTIVE_JOB_STATUSES:
                job.process.terminate()
        await self.model_runtime.stop()
        for subscribers in self._event_subscribers.values():
            for queue in list(subscribers):
                self._offer_event(queue, None)
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._watchdog_task is not None:
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        self._watchdog_report["running"] = False




    @staticmethod
    def _model_profile_recommendation_warnings(request: Mapping[str, Any]) -> list[str]:
        if bool(request.get("advanced_models_enabled")):
            # Advanced Models owns the learned component composition. The disabled
            # whole-checkpoint selector and donor filenames are source locations,
            # not an authoritative runtime-profile/recommendation contract.
            return []
        model_path = str(request.get("model_path") or "").strip()
        if not model_path:
            return []
        path = Path(model_path).expanduser()
        if path.is_file() and path.suffix.lower() == ".safetensors":
            try:
                report = CheckpointInspector().inspect(str(path), compute_sha256=False)
            except Exception:
                report = None
            if report is not None and str(report.architecture or "").strip().lower() == "sd3.x":
                profile = profile_from_checkpoint_variant(report.architecture_variant)
                if profile is not None:
                    return list(sd3_profile_recommendation_warnings(dict(request), profile))
        profile = profile_for_sdxl_filename(path.name)
        if profile.family not in {"lightning", "turbo"}:
            return []
        return list(sdxl_profile_recommendation_warnings(dict(request), profile))

    def preflight_scheduler(self, request: dict[str, Any]) -> dict[str, Any]:
        normalized = self.normalize_generation_request(request)
        resolution = scheduler_resolution_from_payload(normalized)
        warnings = list(resolution.get("validation_warnings") or [])
        warnings.extend(self._model_profile_recommendation_warnings(normalized))
        warnings = list(dict.fromkeys(warnings))
        return {
            "ok": True,
            "scheduler_name": normalized.get("scheduler_name"),
            "steps": normalized.get("steps"),
            "scheduler_kwargs": dict(normalized.get("scheduler_kwargs") or {}),
            "requested_settings": dict(resolution.get("requested_settings") or {}),
            "effective_settings": dict(resolution.get("effective_settings") or {}),
            "compatibility_policy": dict(resolution.get("compatibility_policy") or {}),
            "validation_warnings": warnings,
            "validation_warning_count": len(warnings),
            "preset_reference": dict(resolution.get("preset_reference") or {}),
            "step_count_source": resolution.get("step_count_source"),
            "requested_hash": resolution.get("requested_hash"),
            "effective_hash": resolution.get("effective_hash"),
            "fallback_applied": bool(resolution.get("fallback_applied", False)),
        }

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _write_scheduler_artifacts(
        self,
        *,
        job_root: Path,
        requested_generation: Mapping[str, Any],
        normalized_generation: Mapping[str, Any],
        resolution: Mapping[str, Any],
    ) -> None:
        self._write_json(job_root / "requested-generation.json", dict(requested_generation))
        self._write_json(job_root / "normalized-generation.json", dict(normalized_generation))
        self._write_json(
            job_root / "scheduler-settings-requested.json",
            dict(resolution.get("requested_settings") or {}),
        )
        self._write_json(
            job_root / "scheduler-settings-effective.json",
            dict(resolution.get("effective_settings") or {}),
        )
        self._write_json(
            job_root / "scheduler-validation-warnings.json",
            list(resolution.get("validation_warnings") or []),
        )
        self._write_json(
            job_root / "scheduler-preset-reference.json",
            dict(resolution.get("preset_reference") or {}),
        )

    async def submit(
        self,
        request: dict[str, Any],
        *,
        model_selection: Mapping[str, Any] | None = None,
    ) -> GenerationJob:
        job_id = uuid.uuid4().hex[:12]
        requested_generation = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
        normalized = self.normalize_generation_request(request)
        resolution = scheduler_resolution_from_payload(normalized)
        scheduler_warnings = list(resolution.get("validation_warnings") or [])
        profile_warnings = self._model_profile_recommendation_warnings(normalized)
        warnings = list(dict.fromkeys([*scheduler_warnings, *profile_warnings]))
        acknowledged = _coerce_boolean(
            request.get("_webui_scheduler_warnings_acknowledged", not warnings),
            default=not warnings,
        )
        # Warnings are advisory by contract. The interactive WebUI may ask the
        # user to acknowledge them before it calls submit(), but the backend
        # never upgrades a warning into a queue-time failure. Genuine invalid
        # scheduler settings must be reported as validation errors instead.
        selected = dict(model_selection or {})
        job_root = self.context.data_root / "webui" / "jobs" / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        job = GenerationJob(
            job_id=job_id,
            request=normalized,
            job_root=str(job_root),
            prompt_preflight=dict(normalized.get("prompt_preflight") or {}),
            model_selection=selected,
            scheduler_settings_requested=dict(resolution.get("requested_settings") or {}),
            scheduler_settings_effective=dict(resolution.get("effective_settings") or {}),
            scheduler_validation_warnings=warnings,
            scheduler_compatibility_policy=dict(resolution.get("compatibility_policy") or {}),
            scheduler_preset_reference=dict(resolution.get("preset_reference") or {}),
            scheduler_requested_hash=resolution.get("requested_hash"),
            scheduler_effective_hash=resolution.get("effective_hash"),
            scheduler_step_count_source=resolution.get("step_count_source"),
            scheduler_warnings_acknowledged=acknowledged,
            model_diagnostics={
                "submission": {
                    "browser_requested_path": request.get("_webui_model_requested_path")
                    or request.get("model_path"),
                    "browser_resolved_path": request.get("_webui_model_browser_resolved_path"),
                    "browser_matches_active": request.get("_webui_model_browser_matches_active"),
                    "browser_resolve_error": request.get("_webui_model_browser_resolve_error"),
                    "browser_selection_id": request.get("_webui_model_selection_id"),
                    "backend_active_path": selected.get("resolved_path"),
                    "backend_selection_id": selected.get("selection_id"),
                    "normalized_request_path": normalized.get("model_path"),
                    "server_python_executable": sys.executable,
                    "server_python_version": sys.version,
                    "server_cwd": os.getcwd(),
                    "project_root": str(self.context.project_root),
                    "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
                },
                "scheduler_settings": dict(resolution),
            },
        )
        prompt_preflight_payload = dict(normalized.get("prompt_preflight") or {})
        if prompt_preflight_payload:
            (job_root / "prompt-preflight.json").write_text(
                json.dumps(prompt_preflight_payload, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        self._write_scheduler_artifacts(
            job_root=job_root,
            requested_generation=requested_generation,
            normalized_generation=normalized,
            resolution=resolution,
        )
        self.jobs[job_id] = job
        self._persist_job(job)
        if self._started and (self._worker_task is None or self._worker_task.done()):
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_loop())
        self._enqueue_job_id(job_id)
        return job













    def get_job(self, job_id: str) -> GenerationJob | None:
        return self.jobs.get(job_id)



    def status(self) -> dict[str, Any]:
        active_job = self._active_generation_job()
        active = active_job.job_id if active_job is not None else None
        queue_order = self._queued_order()
        paused_jobs = [job.job_id for job in self.jobs.values() if job.status == "paused"]
        queue_pause_requested = (not self._queue_resume_event.is_set()) or bool(active_job and active_job.pause_after_current_requested)
        return {
            "online": self._worker_task is not None and not self._worker_task.done(),
            "active_job_id": active,
            "queued": len(queue_order),
            "paused": len(paused_jobs),
            "queue_order": queue_order,
            "paused_job_ids": paused_jobs,
            "queue_pause_requested": queue_pause_requested,
            "queue_paused": bool(
                ((not self._queue_resume_event.is_set()) and active_job is None)
                or (paused_jobs and not active_job and not queue_order)
            ),
            "queue_pause_requested_at": self._queue_pause_requested_at,
            "queue_pause_owner_job_id": self._queue_pause_owner_job_id,
            "sse_clients_connected": sum(job.sse_clients_connected for job in self.jobs.values()),
            "preview_cleanup": dict(self._last_cleanup_report),
            "job_cache_cleanup": dict(self._last_job_cache_report),
            "watchdog": dict(self._watchdog_report),
            "model_runtime": self.model_runtime.status(),
        }

    def _offer_event(self, queue: asyncio.Queue[dict[str, Any] | None], payload: dict[str, Any] | None) -> None:
        try:
            queue.put_nowait(payload)
            return
        except asyncio.QueueFull:
            pass
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def _build_event_payload(self, job: GenerationJob, event_type: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "type": event_type,
            "timestamp": _utc_now(),
            "job_id": job.job_id,
            "status": job.status,
            "job": job.to_dict(),
        }
        payload.update(extra)
        return payload

    def _publish_event(self, job: GenerationJob, event_type: str, **extra: Any) -> None:
        payload = self._build_event_payload(job, event_type, **extra)
        subscribers = self._event_subscribers.get(job.job_id, set())
        for queue in list(subscribers):
            self._offer_event(queue, payload)

    def _publish_terminal_once(self, job: GenerationJob, event_type: str) -> bool:
        if job.job_id in self._terminal_events_emitted:
            return False
        self._terminal_events_emitted.add(job.job_id)
        job.terminal_events_emitted += 1
        self._publish_event(job, event_type)
        return True

    async def subscribe(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=1)
        subscribers = self._event_subscribers.setdefault(job_id, set())
        subscribers.add(queue)
        job = self.jobs.get(job_id)
        if job is not None:
            job.sse_clients_connected = len(subscribers)
            job.sse_clients_peak = max(job.sse_clients_peak, job.sse_clients_connected)
            job.live_preview_metrics["sse_clients_connected"] = job.sse_clients_connected
            job.live_preview_metrics["sse_clients_peak"] = job.sse_clients_peak
            initial = self._build_event_payload(
                job,
                "job-progress",
                current_step=job.current_step,
                total_steps=job.total_steps,
                progress_percent=job.progress_percent,
                live_preview_url=job.live_preview_url,
                live_preview_path=job.live_preview_path,
                live_preview_decode_mode=job.live_preview_decode_mode,
            )
            self._offer_event(queue, initial)
        try:
            while True:
                payload = await queue.get()
                if payload is None:
                    break
                yield payload
        finally:
            subscribers.discard(queue)
            if job is not None:
                job.sse_clients_connected = len(subscribers)
                job.live_preview_metrics["sse_clients_connected"] = job.sse_clients_connected
            if not subscribers:
                self._event_subscribers.pop(job_id, None)

    async def _worker_loop(self) -> None:
        while not self._stopping:
            await self._queue_resume_event.wait()
            await self._queue_available.wait()
            await self._queue_resume_event.wait()
            if self._stopping:
                break

            job_id: str | None = None
            while self._queue:
                candidate = self._queue.popleft()
                job = self.jobs.get(candidate)
                if job is not None and job.status == "queued":
                    job_id = candidate
                    break
            if not self._queue:
                self._queue_available.clear()
            if job_id is None:
                await asyncio.sleep(0)
                continue

            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Defensive boundary: never strand later queue items.
                job.error = f"Unhandled queue runtime error: {type(exc).__name__}: {exc}"
                job.return_code = 1
                job.log_lines.extend(traceback.format_exc().splitlines()[-60:])
                self._transition_job(job, status="failed", worker_stage="failed")
                if job.execution_mode == "resident_model":
                    self._finalize_resident_job(job)
                else:
                    job.completed_at = _utc_now()
                    job.updated_at = job.completed_at
                    self._persist_job(job)
                    self._publish_terminal_once(job, "job-failed")














    async def _run_job(self, job: GenerationJob) -> None:
        try:
            await self._run_job_resident(job)
            return
        except ModelRuntimeUnavailable as exc:
            if job.status == "cancelling":
                self._transition_job(job, status="cancelled", worker_stage="cancelled")
                self._finalize_resident_job(job)
                return
            safe_to_retry = (
                not job.output_paths
                and int(job.current_step or 0) == 0
                and not job.skip_events
            )
            if safe_to_retry:
                job.model_runtime_diagnostics.setdefault("recovery_actions", []).append({
                    "timestamp": _utc_now(),
                    "action": "restart_and_reactivate",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                try:
                    await self.model_runtime.stop()
                    model_path = str(job.model_selection.get("resolved_path") or job.request.get("model_path") or "").strip()
                    if model_path:
                        await self.activate_model(
                            model_path,
                            selection=job.model_selection,
                            runtime_overrides=job.request,
                        )
                    await self._run_job_resident(job)
                    return
                except Exception as retry_exc:
                    job.model_runtime_diagnostics["retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
                    exc = ModelRuntimeUnavailable(str(retry_exc))
            self._transition_job(job, status="failed", worker_stage="failed")
            job.error = f"Model runtime unavailable: {exc}"
            self._finalize_resident_job(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._transition_job(job, status="failed", worker_stage="failed")
            job.error = f"Generation failed: {type(exc).__name__}: {exc}"
            job.return_code = 1
            job.log_lines.extend(traceback.format_exc().splitlines()[-60:])
            self._finalize_resident_job(job)

    def _prepare_job_request(
        self,
        job: GenerationJob,
    ) -> tuple[Path, Path, dict[str, Any]]:
        started = job.started_at or _utc_now()
        job.started_at = started
        job.updated_at = _utc_now()
        job.status_changed_at = job.status_changed_at or started
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
        request_payload["save_images"] = True
        current_settings = (
            dict(self.settings_provider() or {}) if self.settings_provider is not None else {}
        )
        runtime_startup_status = build_runtime_startup_status(
            self.runtime_startup_options,
            current_settings,
            worker_ready=(self.model_runtime.status().get("ready") or None),
            worker_status=self.model_runtime.status(),
        )
        request_payload["runtime_startup_status"] = runtime_startup_status
        request_payload.setdefault("output_dir", str(self.context.txt2img_output_root))
        request_payload.setdefault("output_prefix", "{index:05d}-{seed}")

        preview_values = self._live_preview_request_values(job_root)
        live_preview_root = Path(preview_values["live_preview_root"])
        job.live_preview_root = str(live_preview_root)
        job.live_preview_latest_path = str(live_preview_root / "latest.json")
        self._merge_runtime_preview_values(request_payload, preview_values)
        job.request = dict(request_payload)
        job.model_diagnostics["diagnostics_mode"] = str(
            (request_payload.get("diagnostics") or {}).get("mode") or "failures_only"
        )
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
        job.model_diagnostics["runtime_startup_status"] = runtime_startup_status
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
        return request_path, console_path, preview_values






