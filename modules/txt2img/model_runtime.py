from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
import traceback
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from image_gen.runtime_options import (
    RuntimeStartupOptions,
    build_runtime_startup_status,
    add_runtime_startup_arguments,
    argv_for_primary_parser,
    prebootstrap_runtime_startup,
    runtime_request_settings,
)

# Apply import-time runtime environment before Torch, xformers, MSLK, Triton,
# registry, or model modules are imported.
_PREBOOTSTRAP_RUNTIME_STARTUP_OPTIONS = prebootstrap_runtime_startup(sys.argv[1:])

import torch

from image_gen.runtime.scheduler_settings import normalize_scheduler_payload
from image_gen.runtime.residency_policy import (
    GENERATION_RESIDENCY_COLD_LOAD,
    GENERATION_RESIDENCY_HOT_REUSE,
    GENERATION_RESIDENCY_HOT_STAGED_REUSE,
    GENERATION_RESIDENCY_MANAGED_REUSE,
    GENERATION_RESIDENCY_MODEL_SWITCH,
    MODEL_RESIDENCY_MODE_HOT,
    POST_JOB_RESIDENCY_FORCED_RELEASE,
    POST_JOB_RESIDENCY_HOT_HOLD,
    POST_JOB_RESIDENCY_HOT_RESTORE,
    POST_JOB_RESIDENCY_HOT_STAGED_HOLD,
    POST_JOB_RESIDENCY_MANAGED_RETENTION,
    RESIDENCY_STATE_HOT_GPU,
    RESIDENCY_STATE_HOT_STAGED,
    build_residency_diagnostics,
    normalize_model_residency_mode,
    resolve_effective_residency_state,
)
from image_gen.runtime.model_load_variant import (
    MODEL_LOAD_VARIANT_FIELDS,
    model_load_variant_comparison,
    sanitize_model_load_runtime_settings,
    update_model_load_runtime_settings,
)
from image_gen.systems.registry import RuntimeRegistrySystem
from modules.project_context import ProjectContext
from modules.txt2img.async_output_pipeline import AsyncOutputSaveQueue, OutputSaveTicket
from modules.txt2img.cli import _build_prompt_adapter
from modules.txt2img.request_loader import load_request_payload, payload_to_generation_request
from modules.txt2img.seed_utils import iter_batch_base_seeds, offset_seed
from modules.txt2img.txt2img_runner import Txt2ImgRunner

_STATUS_PREFIX = "MODEL_RUNTIME_STATUS_JSON: "
_ASYNC_OUTPUT_SAVE_STATUS_PREFIX = "ASYNC_OUTPUT_SAVE_STATUS_JSON: "
_ASYNC_OUTPUT_SAVE_ERROR_PREFIX = "ASYNC_OUTPUT_SAVE_ERROR_JSON: "
_READY_PREFIX = "MODEL_RUNTIME_READY_JSON: "
_COMPLETE_PREFIX = "MODEL_RUNTIME_COMMAND_COMPLETE_JSON: "


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)


def _utc_timestamp() -> float:
    return time.time()


def _normalized_resolved_path(value: str | None) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    try:
        return os.path.normcase(str(Path(token).expanduser().resolve()))
    except OSError:
        return os.path.normcase(token)


def _model_file_signature(value: str | None) -> tuple[int, int] | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        stat = Path(token).expanduser().resolve().stat()
    except OSError:
        return None
    return int(stat.st_size), int(stat.st_mtime_ns)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _append_trace_stage(
    trace: dict[str, Any],
    name: str,
    started: float,
    **details: Any,
) -> float:
    elapsed = _elapsed_ms(started)
    stage = {"name": str(name), "elapsed_ms": elapsed}
    if details:
        stage.update(details)
    trace.setdefault("stages", []).append(stage)
    return elapsed


def _load_variant_comparison(
    requested_values: dict[str, Any],
    resident_status: dict[str, Any],
) -> dict[str, Any]:
    return model_load_variant_comparison(requested_values, resident_status)


def _checkpoint_identity_comparison(
    requested_model_path: str | None,
    resident_status: dict[str, Any],
) -> dict[str, Any]:
    requested_path = _normalized_resolved_path(requested_model_path)
    resident_path = _normalized_resolved_path(resident_status.get("model_path"))
    path_matches = bool(resident_path and requested_path == resident_path)
    resident_identity = dict(resident_status.get("runtime_checkpoint_identity") or {})
    resident_identity_path = _normalized_resolved_path(resident_identity.get("path"))
    if resident_identity_path and resident_path and resident_identity_path != resident_path:
        path_matches = False

    requested_signature = _model_file_signature(requested_model_path)
    resident_size = resident_identity.get("file_size_bytes")
    resident_modified = resident_identity.get("modified_ns")
    resident_signature = None
    if resident_size is not None and resident_modified is not None:
        try:
            resident_signature = (int(resident_size), int(resident_modified))
        except (TypeError, ValueError):
            resident_signature = None

    if resident_signature is not None and requested_signature is not None:
        source_signature_matches = requested_signature == resident_signature
        proof = (
            "resident_sha256_bound_to_source_file_signature"
            if str(resident_identity.get("sha256") or "").strip()
            else "source_file_signature"
        )
    elif resident_identity and resident_signature is None and requested_signature is None:
        source_signature_matches = True
        proof = "symbolic_path_only"
    elif resident_identity:
        source_signature_matches = False
        proof = "resident_source_identity_incomplete"
    else:
        # Legacy/fake runners predate CNRR-02 source identity publication. Preserve
        # their path-only behavior rather than making compatibility tests fail shut.
        source_signature_matches = True
        proof = "legacy_path_only"

    matches = bool(path_matches and source_signature_matches)
    return {
        "matches": matches,
        "path_matches": path_matches,
        "requested_path": requested_path,
        "resident_path": resident_path,
        "proof": proof,
        "requested_source_signature": (
            {"file_size_bytes": requested_signature[0], "modified_ns": requested_signature[1]}
            if requested_signature is not None
            else {}
        ),
        "resident_checkpoint_identity": resident_identity,
        "source_signature_matches": bool(source_signature_matches),
    }


def _resident_identity_comparison(
    requested_model_path: str | None,
    requested_values: dict[str, Any],
    resident_status: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = _checkpoint_identity_comparison(requested_model_path, resident_status)
    requested_composition = str(requested_values.get("advanced_model_composition_sha256") or "")
    resident_composition = str(
        resident_status.get("advanced_model_composition_sha256")
        or (
            resident_status.get("composition_sha256")
            if str(resident_status.get("model_identity") or "").startswith("advanced:")
            else ""
        )
        or ""
    )
    composition_matches = requested_composition == resident_composition
    load_variant = _load_variant_comparison(requested_values, resident_status)
    load_variant_matches = bool(load_variant.get("matches"))
    resident_present = bool(resident_status.get("resident"))
    matches = bool(
        resident_present
        and checkpoint.get("matches")
        and composition_matches
        and load_variant_matches
    )

    invalidation_reasons: list[str] = []
    if not resident_present:
        invalidation_reasons.append("no_resident_model")
    if resident_present and not checkpoint.get("path_matches"):
        invalidation_reasons.append("checkpoint_path_changed")
    if resident_present and checkpoint.get("path_matches") and not checkpoint.get("source_signature_matches"):
        invalidation_reasons.append("checkpoint_source_changed")
    if resident_present and not composition_matches:
        invalidation_reasons.append("advanced_composition_changed")
    if resident_present and not load_variant_matches:
        invalidation_reasons.append("effective_load_contract_changed")

    return {
        "matches": matches,
        "resident_present": resident_present,
        "checkpoint_identity": checkpoint,
        "path_matches": bool(checkpoint.get("path_matches")),
        "requested_path": checkpoint.get("requested_path"),
        "resident_path": checkpoint.get("resident_path"),
        "composition_matches": composition_matches,
        "requested_composition_sha256": requested_composition,
        "resident_composition_sha256": resident_composition,
        "load_variant_matches": load_variant_matches,
        "load_variant": load_variant,
        "invalidation_reasons": invalidation_reasons,
        "reuse_reason": "canonical_whole_checkpoint_fast_path" if matches else "",
    }


class ResidentTxt2ImgModelRuntime:
    def __init__(
        self,
        context: ProjectContext,
        runtime_startup_options: RuntimeStartupOptions | None = None,
        *,
        emit_status_protocol: bool = True,
    ) -> None:
        self.context = context
        self.emit_status_protocol = bool(emit_status_protocol)
        self.runtime_startup_options = (
            runtime_startup_options or _PREBOOTSTRAP_RUNTIME_STARTUP_OPTIONS
        )
        self.registry_system = RuntimeRegistrySystem(project_context=context)
        self.live_sampler_map = self.registry_system.legacy_map("sampler")
        self.live_scheduler_map = self.registry_system.legacy_map("scheduler")
        self.runner: Txt2ImgRunner | None = None
        self.stage = "idle"
        self.last_error: str | None = None
        self.last_transition_unix = _utc_timestamp()
        self.current_job_id: str | None = None
        self.selected_model_path: str | None = None
        self.timings: dict[str, Any] = {
            "initial_activation_time_ms": None,
            "next_job_preparation_time_ms": None,
            "first_step_latency_ms": None,
            "cpu_to_gpu_promotion_time_ms": None,
            "checkpoint_hydration_time_ms": None,
            "request_setup_time_ms": None,
            "generation_execution_time_ms": None,
            "post_generation_residency_time_ms": None,
            "output_save_wait_time_ms": None,
            "post_generation_finalize_time_ms": None,
            "command_wall_time_ms": None,
        }
        self.residency_mode_requested = "managed"
        self._last_effective_residency_state = "empty"
        self.last_residency_transition = _utc_timestamp()
        self.last_residency_reason = "runtime_initialized"
        self.retention_suppressed_for_hot = False
        self.hot_residency_active = False
        self.hot_since: float | None = None
        self._hot_resident_signature: tuple[str, str, str, str] | None = None
        self._hot_model_file_signature: tuple[int, int] | None = None
        self.post_job_residency_action: str | None = None
        self.last_residency_report: dict[str, Any] = {}
        self.hot_reuse_count = 0
        self.cold_or_switch_load_count = 0
        self.last_generation_residency_classification: str | None = None
        self.last_resident_change_classification: str | None = None
        self.last_load_classification: str | None = None
        self.last_activation_trace: dict[str, Any] = {}
        self.last_preparation_trace: dict[str, Any] = {}
        self._last_payload_load_trace: dict[str, Any] = {}
        self.recent_job_reports: list[dict[str, Any]] = []
        self.residency_transition_history: list[dict[str, Any]] = []
        self.runtime_settings: dict[str, Any] = runtime_request_settings(
            self.runtime_startup_options
        )
        if not torch.cuda.is_available():
            self.runtime_settings["model_runtime_execution_device"] = "cpu"
            self.runtime_settings["model_runtime_retention_device"] = "cpu"
        self._output_save_queue: AsyncOutputSaveQueue | None = None



    def _emit_async_output_save_status(self, snapshot: dict[str, Any]) -> None:
        print(
            _ASYNC_OUTPUT_SAVE_STATUS_PREFIX
            + json.dumps(snapshot, ensure_ascii=True, sort_keys=True),
            flush=True,
        )

    def _handle_async_save_enqueued(
        self,
        ticket: OutputSaveTicket,
        snapshot: Any,
    ) -> None:
        del ticket
        self._emit_async_output_save_status(dict(snapshot.to_dict()))

    def _handle_async_save_started(
        self,
        ticket: OutputSaveTicket,
        snapshot: Any,
    ) -> None:
        del ticket
        self._emit_async_output_save_status(dict(snapshot.to_dict()))

    def _handle_async_save_success(
        self,
        ticket: OutputSaveTicket,
        records: list[Any],
        snapshot: Any,
    ) -> None:
        del ticket
        snapshot_payload = dict(snapshot.to_dict())
        snapshot_payload["saved_paths"] = [str(record.image_path) for record in records]
        metadata_warnings = [
            str(warning)
            for record in records
            for warning in (getattr(record, "metadata_warnings", None) or [])
            if str(warning or "").strip()
        ]
        if metadata_warnings:
            snapshot_payload["warnings"] = metadata_warnings
        self._emit_async_output_save_status(snapshot_payload)
        for record in records:
            seed_label = "unknown" if getattr(record, "seed", None) is None else str(record.seed)
            print(f"  Image [seed {seed_label}]: {record.image_path}", flush=True)
            if getattr(record, "txt_path", None):
                print(f"  TXT:   {record.txt_path}", flush=True)
            if getattr(record, "json_path", None):
                print(f"  JSON:  {record.json_path}", flush=True)

    def _handle_async_save_error(
        self,
        ticket: OutputSaveTicket,
        error: BaseException,
        snapshot: Any,
    ) -> None:
        self._emit_async_output_save_status(dict(snapshot.to_dict()))
        print(
            _ASYNC_OUTPUT_SAVE_ERROR_PREFIX
            + json.dumps(
                {
                    "job_id": ticket.job_id,
                    "batch_number": ticket.batch_number,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            flush=True,
        )

    def _ensure_output_save_queue(self) -> AsyncOutputSaveQueue:
        if self._output_save_queue is not None:
            return self._output_save_queue
        runner = self._ensure_runner()
        self._output_save_queue = AsyncOutputSaveQueue(
            runner.output_system.save_prepared,
            on_enqueued=self._handle_async_save_enqueued,
            on_started=self._handle_async_save_started,
            on_saved=self._handle_async_save_success,
            on_error=self._handle_async_save_error,
        )
        return self._output_save_queue

    def _ensure_runner(self) -> Txt2ImgRunner:
        if self.runner is not None:
            return self.runner
        from modules.load_safetensors_model import LoadModel

        self.runner = Txt2ImgRunner(
            prompt_adapter_factory=_build_prompt_adapter,
            model_loader=LoadModel(project_context=self.context),
            project_context=self.context,
            registry_system=self.registry_system,
        )
        return self.runner

    @staticmethod
    def _resident_signature(status: dict[str, Any] | None) -> tuple[str, str, str, str] | None:
        current = dict(status or {})
        model_path = _normalized_resolved_path(current.get("model_path"))
        if not model_path:
            return None
        return (
            model_path,
            str(current.get("model_identity") or ""),
            str(current.get("composition_sha256") or ""),
            str(
                current.get("runtime_effective_load_variant_fingerprint")
                or current.get("runtime_load_variant_fingerprint")
                or ""
            ),
        )

    def _invalidate_hot_residency(self, reason: str, *, forced_release: bool = False) -> None:
        self.hot_residency_active = False
        self.hot_since = None
        self._hot_resident_signature = None
        self._hot_model_file_signature = None
        self.retention_suppressed_for_hot = False
        self.last_residency_reason = str(reason or "hot_residency_invalidated")
        self.last_residency_transition = _utc_timestamp()
        if forced_release:
            self.post_job_residency_action = POST_JOB_RESIDENCY_FORCED_RELEASE

    def _set_residency_mode_requested(self, value: Any) -> str:
        mode = normalize_model_residency_mode(value)
        previous = self.residency_mode_requested
        self.residency_mode_requested = mode
        self.runtime_settings["model_residency_mode"] = mode
        if previous == MODEL_RESIDENCY_MODE_HOT and mode != MODEL_RESIDENCY_MODE_HOT:
            self._invalidate_hot_residency("hot_policy_disabled")
        return mode

    def _apply_runtime_residency_policy(
        self,
        runner: Any,
        settings: dict[str, Any],
        *,
        reason: str,
        post_job: bool = False,
    ) -> dict[str, Any]:
        mode = normalize_model_residency_mode(settings.get("model_residency_mode", self.residency_mode_requested))
        self.residency_mode_requested = mode
        self.runtime_settings["model_residency_mode"] = mode

        if mode == MODEL_RESIDENCY_MODE_HOT and callable(getattr(runner, "apply_hot_residency", None)):
            report = dict(runner.apply_hot_residency(settings, reason=reason) or {})
            status = dict(report.get("status") or runner.resident_model_status() or {})
            signature = self._resident_signature(status)
            active = bool(report.get("applied") and status.get("resident") and signature is not None)
            previous_signature = self._hot_resident_signature
            self.hot_residency_active = active
            self.retention_suppressed_for_hot = active
            if active:
                if self.hot_since is None or previous_signature != signature:
                    self.hot_since = _utc_timestamp()
                self._hot_resident_signature = signature
                self._hot_model_file_signature = _model_file_signature(status.get("model_path"))
            else:
                self.hot_since = None
                self._hot_resident_signature = None
                self._hot_model_file_signature = None
            promotion_ms = report.get("promotion_time_ms")
            if promotion_ms is not None:
                self.timings["cpu_to_gpu_promotion_time_ms"] = promotion_ms
            action = str(report.get("residency_action") or (
                POST_JOB_RESIDENCY_HOT_STAGED_HOLD
                if str(report.get("effective_hot_state") or "") == RESIDENCY_STATE_HOT_STAGED
                else POST_JOB_RESIDENCY_HOT_HOLD
            ))
        else:
            report = dict(runner.apply_resident_retention(settings) or {})
            self.hot_residency_active = False
            self.hot_since = None
            self._hot_resident_signature = None
            self._hot_model_file_signature = None
            self.retention_suppressed_for_hot = False
            action = POST_JOB_RESIDENCY_MANAGED_RETENTION
            report.setdefault("residency_action", action)

        if post_job:
            self.post_job_residency_action = action
            report["post_job_residency_action"] = action
        self.last_residency_report = dict(report)
        policy_reason = str(report.get("degradation_reason") or report.get("staging_reason") or "").strip()
        if policy_reason:
            self.last_residency_reason = policy_reason
            self.last_residency_transition = _utc_timestamp()
        return report

    def _record_first_step_timing(self, payload: dict[str, Any] | None = None) -> None:
        event = dict(payload or {})
        value = event.get("first_step_latency_ms")
        try:
            self.timings["first_step_latency_ms"] = round(max(0.0, float(value)), 3)
        except (TypeError, ValueError):
            self.timings["first_step_latency_ms"] = None

    def _status_payload(self, *, stage: str | None = None, **extra: Any) -> dict[str, Any]:
        if stage is not None:
            self.stage = str(stage)
            self.last_transition_unix = _utc_timestamp()
        residency = self.runner.resident_model_status() if self.runner is not None else {
            "resident": False,
            "model_path": None,
            "model_identity": "",
            "composition_sha256": "",
            "composition_identity_version": "",
            "composition_contract": {},
            "component_sources": {},
            "composition_projection": {},
            "advanced_model_composition_sha256": "",
            "runtime_load_variant_fingerprint": "",
            "runtime_effective_load_variant_fingerprint": "",
            "runtime_effective_load_variant": {},
            "runtime_checkpoint_identity": {},
            "cache_entries": 0,
            "cpu_loaded": False,
            "gpu_loaded": False,
            "generation_ready": False,
            "staged_runtime": False,
            "architecture": "",
            "component_devices": {},
            "cuda_memory": {"allocated_bytes": 0, "reserved_bytes": 0},
        }
        resident_signature = self._resident_signature(residency)
        hot_active = bool(
            self.hot_residency_active
            and resident_signature is not None
            and resident_signature == self._hot_resident_signature
        )
        if self.hot_residency_active and not hot_active:
            self._invalidate_hot_residency("resident_identity_no_longer_matches_hot_signature")
        effective_state = resolve_effective_residency_state(
            requested_mode=self.residency_mode_requested,
            stage=self.stage,
            resident=bool(residency.get("resident")),
            gpu_loaded=bool(residency.get("gpu_loaded")),
            staged_runtime=bool(residency.get("staged_runtime")),
            hot_residency_active=hot_active,
            hot_gpu_ready=bool(residency.get("hot_gpu_ready", residency.get("gpu_loaded"))),
        )
        policy_reason = str(
            self.last_residency_report.get("degradation_reason")
            or self.last_residency_report.get("staging_reason")
            or ""
        ).strip()
        reason = str(
            extra.get("action")
            or (policy_reason if effective_state == "hot_staged" and policy_reason else None)
            or stage
            or self.stage
            or "status"
        )
        if effective_state != self._last_effective_residency_state:
            previous_state = self._last_effective_residency_state
            self._last_effective_residency_state = effective_state
            self.last_residency_transition = _utc_timestamp()
            self.last_residency_reason = reason
            self.residency_transition_history.append({
                "timestamp": self.last_residency_transition,
                "from": previous_state,
                "to": effective_state,
                "reason": reason,
                "model_path": residency.get("model_path"),
            })
            self.residency_transition_history = self.residency_transition_history[-40:]
        diagnostics = build_residency_diagnostics(
            requested_mode=self.residency_mode_requested,
            effective_state=effective_state,
            resident_status=residency,
            last_residency_transition=self.last_residency_transition,
            last_residency_reason=self.last_residency_reason,
            retention_suppressed_for_hot=self.retention_suppressed_for_hot,
            hot_reuse_count=self.hot_reuse_count,
            cold_or_switch_load_count=self.cold_or_switch_load_count,
            last_generation_residency_classification=self.last_generation_residency_classification,
            hot_since=self.hot_since,
        )
        payload = {
            "schema_version": 1,
            "worker_pid": os.getpid(),
            "stage": self.stage,
            # Backward-compatible field consumed by existing WebUI/tests. HMR uses
            # residency_state_effective for the normalized policy-aware contract.
            "residency_state": "resident" if residency.get("resident") else "empty",
            "selected_model_path": self.selected_model_path,
            "current_model_path": residency.get("model_path"),
            "model_identity": str(residency.get("model_identity") or ""),
            "composition_sha256": str(residency.get("composition_sha256") or ""),
            "composition_identity_version": str(residency.get("composition_identity_version") or ""),
            "composition_contract": dict(residency.get("composition_contract") or {}),
            "component_sources": {
                str(role): dict(source)
                for role, source in dict(residency.get("component_sources") or {}).items()
            },
            "composition_projection": dict(residency.get("composition_projection") or {}),
            "advanced_model_composition_sha256": str(residency.get("advanced_model_composition_sha256") or ""),
            "runtime_load_variant_fingerprint": str(residency.get("runtime_load_variant_fingerprint") or ""),
            "runtime_effective_load_variant_fingerprint": str(residency.get("runtime_effective_load_variant_fingerprint") or ""),
            "runtime_effective_load_variant": dict(residency.get("runtime_effective_load_variant") or {}),
            "runtime_checkpoint_identity": dict(residency.get("runtime_checkpoint_identity") or {}),
            "cpu_loaded": bool(residency.get("cpu_loaded")),
            "gpu_loaded": bool(residency.get("gpu_loaded")),
            "hot_gpu_ready": bool(residency.get("hot_gpu_ready", residency.get("gpu_loaded"))),
            "generation_ready": bool(residency.get("generation_ready")),
            "staged_runtime": bool(residency.get("staged_runtime")),
            "architecture": str(residency.get("architecture") or ""),
            "component_devices": dict(residency.get("component_devices") or {}),
            "composition_execution_lease": dict(residency.get("composition_execution_lease") or {}),
            "memory": dict(residency.get("cuda_memory") or {}),
            "cache_entries": int(residency.get("cache_entries") or 0),
            "current_job_id": self.current_job_id,
            "last_transition_unix": self.last_transition_unix,
            "last_error": self.last_error,
            "cuda_available": bool(torch.cuda.is_available()),
            "execution_device_policy": str(self.runtime_settings.get("model_runtime_execution_device") or "cuda_preferred"),
            "retention_device_policy": str(self.runtime_settings.get("model_runtime_retention_device") or "auto"),
            "execution_device": str(
                self.runtime_settings.get("last_execution_device")
                or (residency.get("component_devices") or {}).get("unet")
                or (residency.get("component_devices") or {}).get("transformer")
                or ("cuda" if torch.cuda.is_available() else "cpu")
            ),
            "cpu_fallback_reason": self.runtime_settings.get("cpu_fallback_reason"),
            "timings": dict(self.timings),
            "last_load_classification": self.last_load_classification,
            "post_job_residency_action": self.post_job_residency_action,
            "last_resident_change_classification": self.last_resident_change_classification,
            "last_activation_trace": dict(self.last_activation_trace),
            "last_preparation_trace": dict(self.last_preparation_trace),
            "recent_job_reports": list(self.recent_job_reports[-10:]),
            "residency_transition_history": list(self.residency_transition_history[-20:]),
            **extra,
        }
        payload.update(diagnostics)
        return payload

    def emit_status(self, stage: str | None = None, **extra: Any) -> dict[str, Any]:
        payload = self._status_payload(stage=stage, **extra)
        if self.emit_status_protocol:
            _emit(_STATUS_PREFIX, payload)
        return payload

    def _runner_event(self, payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage") or "preparing_model")
        if stage == "model_ready":
            self.timings["checkpoint_hydration_time_ms"] = payload.get(
                "checkpoint_hydration_time_ms"
            )
            self.timings["gpu_transfer_included"] = payload.get("gpu_transfer_included")
            self.runtime_settings["cpu_fallback_reason"] = payload.get("cpu_fallback_reason")
            self.runtime_settings["last_execution_device"] = payload.get("execution_device")
        self.emit_status(stage, **{key: value for key, value in payload.items() if key != "stage"})

    def activate(self, command: dict[str, Any]) -> dict[str, Any]:
        activation_started = time.perf_counter()
        trace: dict[str, Any] = {
            "schema_version": 1,
            "kind": "activation",
            "stages": [],
        }
        model_path = str(command.get("model_path") or "").strip()
        if not model_path:
            raise ValueError("A model_path is required for model activation.")
        extras = dict(command.get("runtime_settings") or {})
        self._set_residency_mode_requested(extras.get("model_residency_mode"))
        self.selected_model_path = model_path
        self.current_job_id = None
        self.last_error = None
        self.emit_status("preparing_model", action="activate")

        stage_started = time.perf_counter()
        startup_runtime = runtime_request_settings(self.runtime_startup_options)
        extras.update(startup_runtime)
        if not torch.cuda.is_available():
            extras["model_runtime_execution_device"] = "cpu"
            extras["model_runtime_retention_device"] = "cpu"
        extras = sanitize_model_load_runtime_settings(extras)
        update_model_load_runtime_settings(self.runtime_settings, extras)
        _append_trace_stage(trace, "prepare_runtime_settings", stage_started)

        stage_started = time.perf_counter()
        runner = self._ensure_runner()
        _append_trace_stage(trace, "ensure_runner", stage_started)

        stage_started = time.perf_counter()
        current = runner.resident_model_status()
        _append_trace_stage(trace, "resident_status_before", stage_started)
        current_path = str(current.get("model_path") or "")
        identity_comparison = _resident_identity_comparison(model_path, extras, current)
        extras["model_runtime_event_callback"] = self._runner_event
        reused_resident_model = bool(identity_comparison.get("matches"))
        trace["identity_comparison"] = {
            **identity_comparison,
            "reused_resident_model": reused_resident_model,
        }

        if current_path and not reused_resident_model:
            load_variant_mismatches = list(
                dict(identity_comparison.get("load_variant") or {}).get("mismatch_fields") or []
            )
            unsafe_transition_fields = {
                "sd2_dedicated_generation",
                "sd2_runtime_profile_override",
                "sdxl_runtime_profile_override",
                "sd3_runtime_profile_override",
            }
            unsafe_contract_changes = sorted(
                str(item.get("field") or "")
                for item in load_variant_mismatches
                if str(item.get("field") or "") in unsafe_transition_fields
            )
            transition_eligibility_fn = getattr(runner, "component_transition_eligibility", None)
            if unsafe_contract_changes:
                transition_eligibility = {
                    "eligible": False,
                    "reason": "runtime_profile_contract_changed",
                    "unsafe_contract_fields": unsafe_contract_changes,
                }
            else:
                transition_eligibility = (
                    transition_eligibility_fn(model_path)
                    if callable(transition_eligibility_fn)
                    else {"eligible": False, "reason": "runner_component_transition_api_unavailable"}
                )
            trace["component_transition_eligibility"] = dict(transition_eligibility)
            self._invalidate_hot_residency("automatic_model_swap")
            if bool(transition_eligibility.get("eligible")):
                extras["_component_transition_requested"] = True
                self.emit_status(
                    "switching",
                    action="component_aware_model_swap",
                    previous_model_path=current_path,
                    next_model_path=model_path,
                    component_transition_eligibility=dict(transition_eligibility),
                )
            else:
                self.emit_status(
                    "unloading",
                    action="automatic_model_swap",
                    previous_model_path=current_path,
                    next_model_path=model_path,
                    component_transition_eligibility=dict(transition_eligibility),
                )
                stage_started = time.perf_counter()
                runner.clear_model_cache()
                _append_trace_stage(trace, "clear_incompatible_resident", stage_started)
            reused_resident_model = False

        if reused_resident_model:
            self.emit_status("applying_retention_policy", action="reuse_resident_model")
            stage_started = time.perf_counter()
            retention = self._apply_runtime_residency_policy(
                runner,
                extras,
                reason="activate_reuse",
            )
            _append_trace_stage(trace, "reuse_retention_policy", stage_started)
            retention_status = dict(retention.get("status") or {})
            activate_time_ms = _elapsed_ms(activation_started)
            trace["total_ms"] = activate_time_ms
            trace["resident_fast_path"] = True
            self.last_activation_trace = copy.deepcopy(trace)
            self.timings.update(
                {
                    "activate_time_ms": activate_time_ms,
                    "checkpoint_hydration_time_ms": None,
                    "first_step_warmup_time_ms": None,
                    "first_step_warmup_performed": False,
                }
            )
            return self.emit_status(
                "ready",
                action="activate_reused_resident_model",
                activate_time_ms=self.timings["activate_time_ms"],
                model_provenance={},
                retention=retention,
                resident_fast_path=True,
                resident_fast_path_before=current,
                resident_fast_path_after=retention_status,
                activation_trace=copy.deepcopy(trace),
            )

        self.last_load_classification = (
            GENERATION_RESIDENCY_MODEL_SWITCH if bool(current.get("resident")) else GENERATION_RESIDENCY_COLD_LOAD
        )
        self.cold_or_switch_load_count += 1
        stage_started = time.perf_counter()
        result = runner.preload_model(model_path, extras)
        _append_trace_stage(
            trace,
            "preload_model",
            stage_started,
            cache_reused=bool((result.get("model_provenance") or {}).get("cache_reused")),
        )
        preload_trace = dict(result.get("preload_trace") or {})
        if preload_trace:
            trace["preload_trace"] = preload_trace
        component_transition_report = dict(result.get("component_transition_report") or {})
        if component_transition_report:
            trace["component_transition_report"] = component_transition_report
        self.emit_status("applying_retention_policy")
        stage_started = time.perf_counter()
        retention = self._apply_runtime_residency_policy(
            runner,
            extras,
            reason="activate_complete",
        )
        _append_trace_stage(trace, "post_preload_retention_policy", stage_started)
        elapsed = _elapsed_ms(activation_started)
        trace["total_ms"] = elapsed
        trace["resident_fast_path"] = False
        self.last_activation_trace = copy.deepcopy(trace)
        if self.last_load_classification == GENERATION_RESIDENCY_COLD_LOAD and self.timings.get("initial_activation_time_ms") is None:
            self.timings["initial_activation_time_ms"] = elapsed
        self.timings.update(
            {
                "activate_time_ms": result.get("preload_time_ms"),
                "checkpoint_hydration_time_ms": (
                    result.get("model_provenance") or {}
                ).get("checkpoint_hydration_time_ms"),
                "first_step_warmup_time_ms": None,
                "first_step_warmup_performed": False,
            }
        )
        return self.emit_status(
            "ready",
            action="activate_complete",
            activate_time_ms=elapsed,
            model_provenance=dict(result.get("model_provenance") or {}),
            retention=retention,
            activation_trace=copy.deepcopy(trace),
        )

    def _load_payload(self, config_path: str) -> tuple[Any, dict[str, Any]]:
        trace_started = time.perf_counter()
        trace: dict[str, Any] = {"schema_version": 1, "kind": "request_load", "stages": []}
        stage_started = time.perf_counter()
        payload = load_request_payload(
            config_path=config_path,
            base_payload=self.context.generation_defaults(),
        )
        _append_trace_stage(trace, "load_request_payload", stage_started)
        stage_started = time.perf_counter()
        payload, _resolution = normalize_scheduler_payload(payload)
        _append_trace_stage(trace, "normalize_scheduler_payload", stage_started)
        stage_started = time.perf_counter()
        request, payload_extras = payload_to_generation_request(payload)
        _append_trace_stage(trace, "payload_to_generation_request", stage_started)
        stage_started = time.perf_counter()
        sanitized = sanitize_model_load_runtime_settings(payload_extras)
        _append_trace_stage(trace, "sanitize_model_load_runtime_settings", stage_started)
        trace["total_ms"] = _elapsed_ms(trace_started)
        self._last_payload_load_trace = trace
        return request, sanitized

    def run_job(self, command: dict[str, Any]) -> dict[str, Any]:
        command_started = time.perf_counter()
        trace_enabled = bool(command.get("trace_preparation"))
        preparation_trace: dict[str, Any] = {
            "schema_version": 1,
            "kind": "resident_command",
            "stages": [],
        }
        job_id = str(command.get("job_id") or uuid.uuid4().hex[:12])
        config_path = str(command.get("config_path") or "").strip()
        if not config_path:
            raise ValueError("A config_path is required for resident generation.")
        self.current_job_id = job_id
        self.last_error = None
        stage_started = time.perf_counter()
        runner = self._ensure_runner()
        if trace_enabled:
            _append_trace_stage(preparation_trace, "ensure_runner", stage_started)
        stage_started = time.perf_counter()
        runner.reset_runtime_state()
        if trace_enabled:
            _append_trace_stage(preparation_trace, "reset_runtime_state_before_request", stage_started)
        stage_started = time.perf_counter()
        request, payload_extras = self._load_payload(config_path)
        if trace_enabled:
            _append_trace_stage(preparation_trace, "load_request_payload", stage_started)
            preparation_trace["request_load_trace"] = copy.deepcopy(self._last_payload_load_trace)
            payload_extras["model_runtime_trace_enabled"] = True
        self._set_residency_mode_requested(payload_extras.get("model_residency_mode"))
        self.selected_model_path = str(payload_extras.get("model_path") or "") or None
        stage_started = time.perf_counter()
        resident_before = runner.resident_model_status()
        if trace_enabled:
            _append_trace_stage(preparation_trace, "resident_status_before", stage_started)
        identity_comparison = _resident_identity_comparison(
            self.selected_model_path,
            payload_extras,
            resident_before,
        )
        resident_reuse_candidate = bool(identity_comparison.get("matches"))
        if trace_enabled:
            preparation_trace["identity_comparison"] = {
                **identity_comparison,
                "resident_reuse_candidate": resident_reuse_candidate,
            }
        if resident_reuse_candidate:
            self.last_resident_change_classification = (
                "resident_compatible_restaging"
                if bool(resident_before.get("staged_runtime"))
                else "request_only"
            )
        elif resident_before.get("resident"):
            self.last_resident_change_classification = "resident_identity_change"
        else:
            self.last_resident_change_classification = "resident_identity_change"

        hot_source_file_matches = bool(
            self._hot_model_file_signature == _model_file_signature(self.selected_model_path)
        )
        hot_signature_matches = bool(
            self.hot_residency_active
            and self._resident_signature(resident_before) is not None
            and self._resident_signature(resident_before) == self._hot_resident_signature
            and hot_source_file_matches
        )
        hot_source_changed = bool(
            resident_before.get("resident")
            and identity_comparison.get("path_matches")
            and identity_comparison.get("composition_matches")
            and identity_comparison.get("load_variant_matches")
            and self.residency_mode_requested == MODEL_RESIDENCY_MODE_HOT
            and self.hot_residency_active
            and not hot_source_file_matches
        )
        if hot_source_changed:
            # The checkpoint at the selected path changed after Hot activation.
            # Reuse must not survive a same-filename replacement; clear the
            # hydrated cache so the canonical loader can establish a new model
            # identity and file-stat cache key.
            self.emit_status(
                "switching",
                action="hot_source_file_changed",
                previous_file_signature=self._hot_model_file_signature,
                current_file_signature=_model_file_signature(self.selected_model_path),
            )
            self._invalidate_hot_residency("hot_source_file_changed")
            runner.clear_model_cache()
            resident_reuse_candidate = False
        if resident_reuse_candidate and self.residency_mode_requested == MODEL_RESIDENCY_MODE_HOT and hot_signature_matches:
            if bool(resident_before.get("staged_runtime")) or not bool(
                resident_before.get("hot_gpu_ready", resident_before.get("gpu_loaded"))
            ):
                generation_residency_classification = GENERATION_RESIDENCY_HOT_STAGED_REUSE
            else:
                generation_residency_classification = GENERATION_RESIDENCY_HOT_REUSE
            self.hot_reuse_count += 1
        elif resident_reuse_candidate:
            generation_residency_classification = GENERATION_RESIDENCY_MANAGED_REUSE
        elif resident_before.get("resident"):
            generation_residency_classification = GENERATION_RESIDENCY_MODEL_SWITCH
        else:
            generation_residency_classification = GENERATION_RESIDENCY_COLD_LOAD
        self.last_generation_residency_classification = generation_residency_classification
        if not resident_reuse_candidate and self.selected_model_path:
            activation_settings = dict(self.runtime_settings)
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
                if key in payload_extras:
                    activation_settings[key] = payload_extras[key]
            activation_settings = sanitize_model_load_runtime_settings(activation_settings)
            stage_started = time.perf_counter()
            activation_result = self.activate({
                "model_path": self.selected_model_path,
                "runtime_settings": activation_settings,
            })
            if trace_enabled:
                _append_trace_stage(preparation_trace, "activate_for_request", stage_started)
                preparation_trace["activation_trace"] = dict(
                    activation_result.get("activation_trace")
                    or self.last_activation_trace
                    or {}
                )
            resident_before = runner.resident_model_status()
            resident_reuse_candidate = bool(resident_before.get("resident"))
        self.emit_status(
            "preparing_model",
            action="run_job",
            resident_reuse_candidate=resident_reuse_candidate,
            generation_residency_classification=generation_residency_classification,
        )

        stage_started = time.perf_counter()
        extras = {
            "live_sampler_map": self.live_sampler_map,
            "live_scheduler_map": self.live_scheduler_map,
            "model_runtime_event_callback": self._runner_event,
            "model_runtime_first_step_callback": self._record_first_step_timing,
        }
        extras.update(payload_extras)
        startup_runtime = runtime_request_settings(self.runtime_startup_options)
        extras.update(startup_runtime)
        if not torch.cuda.is_available():
            extras["model_runtime_execution_device"] = "cpu"
            extras["model_runtime_retention_device"] = "cpu"
        self.runtime_settings.update({
            key: value
            for key, value in extras.items()
            if key.startswith("model_runtime_") or key.startswith("memory_") or key == "model_residency_mode"
        })
        if trace_enabled:
            _append_trace_stage(preparation_trace, "prepare_generation_runtime_extras", stage_started)
        stage_started = time.perf_counter()
        batch_count = int(extras.pop("batch_count", 1) or 1)
        unlimited = bool(extras.pop("unlimited", False))
        if batch_count < 1:
            raise ValueError("batch_count must be at least 1")
        if request.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        requested_seed = request.seed
        base_seed_iterator = iter_batch_base_seeds(
            requested_seed,
            batch_size=request.batch_size,
        )
        completed_batches = 0
        total_saved = 0
        job_started = time.perf_counter()
        model_cache_reused = False
        first_generation_started: float | None = None
        save_tickets: list[OutputSaveTicket] = []
        generation_execution_seconds = 0.0
        if trace_enabled:
            _append_trace_stage(preparation_trace, "prepare_batch_iteration_state", stage_started)

        while unlimited or completed_batches < batch_count:
            # Clear request-scoped runtime state before each batch iteration while keeping resident model components loaded.
            batch_prepare_started = time.perf_counter()
            runner.reset_runtime_state()
            batch_number = completed_batches + 1
            batch_request = replace(
                request,
                seed=next(base_seed_iterator),
                resolved_seeds=[],
                scheduler_kwargs=dict(request.scheduler_kwargs),
                sampler_kwargs=dict(request.sampler_kwargs),
                prompt_parser_name=str(request.prompt_parser_name or "legacy"),
                prompt_parser_kwargs=dict(request.prompt_parser_kwargs),
                prompt_semantic_pass_records=dict(request.prompt_semantic_pass_records or {}),
                prompt_semantic_recorded=dict(request.prompt_semantic_recorded or {}),
                prompt_semantic_replay_mode=str(request.prompt_semantic_replay_mode or "reconstruct"),
                region_pass_records=dict(request.region_pass_records or {}),
                region_recorded=dict(request.region_recorded or {}),
                region_replay_mode=str(request.region_replay_mode or "reconstruct"),
                prompt_shortcut_profile_name=str(request.prompt_shortcut_profile_name or "legacy_default"),
                prompt_shortcut_profile_snapshot=dict(request.prompt_shortcut_profile_snapshot),
                prompt_parser_preset_name=str(request.prompt_parser_preset_name or ""),
                base_prompt_parser_name=str(request.base_prompt_parser_name or request.prompt_parser_name or "legacy"),
                base_shortcut_profile_name=str(request.base_shortcut_profile_name or request.prompt_shortcut_profile_name or "legacy_default"),
                hires_prompt_parser_mode=str(request.hires_prompt_parser_mode or "same_as_base"),
                hires_prompt_parser_name=str(request.hires_prompt_parser_name or request.prompt_parser_name or "legacy"),
                hires_prompt_parser_kwargs=dict(request.hires_prompt_parser_kwargs),
                hires_shortcut_profile_mode=str(request.hires_shortcut_profile_mode or "same_as_base"),
                hires_shortcut_profile_name=str(request.hires_shortcut_profile_name or request.prompt_shortcut_profile_name or "legacy_default"),
                hires_shortcut_profile_snapshot=dict(request.hires_shortcut_profile_snapshot),
                hires_positive_prompt=str(request.hires_positive_prompt or request.positive_prompt),
                hires_negative_prompt=str(request.hires_negative_prompt if request.hires_negative_prompt is not None else request.negative_prompt),
                hires_size_mode=str(request.hires_size_mode or "same_as_base"),
                hires_scale=(float(request.hires_scale) if request.hires_scale is not None else None),
                hires_width=int(request.hires_width or 0),
                hires_height=int(request.hires_height or 0),
                hires_dimension_plan=dict(request.hires_dimension_plan),
                hires_axis_scale_width=float(request.hires_axis_scale_width),
                hires_axis_scale_height=float(request.hires_axis_scale_height),
                hires_uniform_scale=(
                    float(request.hires_uniform_scale)
                    if request.hires_uniform_scale is not None
                    else None
                ),
                hires_aspect_ratio_changed=bool(request.hires_aspect_ratio_changed),
                prompt_preflight=dict(request.prompt_preflight),
                prompt_shadow_compare=bool(request.prompt_shadow_compare),
                prompt_route_plan=dict(request.prompt_route_plan),
                hires_prompt_route_plan=dict(request.hires_prompt_route_plan),
                parser_kwargs=dict(request.parser_kwargs),
                diagnostics=dict(request.diagnostics),
            )
            resolved_image_seeds = [
                offset_seed(int(batch_request.seed), index)
                for index in range(int(batch_request.batch_size))
            ]
            print(
                "GENERATION_SEED_JSON: "
                + json.dumps(
                    {
                        "batch_number": batch_number,
                        "base_seed": int(batch_request.seed),
                        "image_seeds": resolved_image_seeds,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )
            batch_extras = dict(extras)
            batch_extras.update(
                {
                    "batch_number": batch_number,
                    "batch_count": batch_count,
                    "unlimited": unlimited,
                    "generation_mode": "unlimited" if unlimited else "batch_count",
                }
            )
            if first_generation_started is None:
                first_generation_started = time.perf_counter()
                preparation_ms = round(
                    (first_generation_started - command_started) * 1000.0,
                    3,
                )
                self.timings["next_job_preparation_time_ms"] = preparation_ms
                self.timings["request_setup_time_ms"] = preparation_ms
                self.timings["first_step_latency_ms"] = None
                if trace_enabled:
                    _append_trace_stage(
                        preparation_trace,
                        "first_batch_request_preparation",
                        batch_prepare_started,
                    )
                    preparation_trace["pre_generation_wall_time_ms"] = preparation_ms
            self.emit_status("running", batch_number=batch_number, batch_count=batch_count)
            run_request_kwargs = {
                "save_txt": bool(command.get("save_txt", True)),
                "save_json": bool(command.get("save_json", True)),
                "save_diagnostics_json": bool(command.get("save_diagnostics_json", True)),
                "defer_output_save": True,
            }
            while True:
                generation_call_started = time.perf_counter()
                try:
                    result = runner.run_request(
                        batch_request,
                        batch_extras,
                        **run_request_kwargs,
                    )
                except TypeError as exc:
                    generation_execution_seconds += time.perf_counter() - generation_call_started
                    message = str(exc)
                    unsupported = next(
                        (
                            key
                            for key in ("save_diagnostics_json", "defer_output_save")
                            if key in run_request_kwargs and key in message
                        ),
                        None,
                    )
                    if unsupported is None:
                        raise
                    # Preserve compatibility with test/plugin runner shims that
                    # predate this optional output-sidecar argument.
                    run_request_kwargs.pop(unsupported, None)
                    continue
                generation_execution_seconds += time.perf_counter() - generation_call_started
                break
            if trace_enabled:
                _append_trace_stage(
                    preparation_trace,
                    f"generation_call_{batch_number}",
                    generation_call_started,
                )

            live_preview_summary = dict(result.pipeline_result.metadata.get("live_preview") or {})
            print(
                "LIVE_PREVIEW_SUMMARY_JSON: "
                + json.dumps(live_preview_summary, ensure_ascii=True, sort_keys=True),
                flush=True,
            )
            output_quality_diagnostic = dict(
                result.pipeline_result.metadata.get("output_quality") or {}
            )
            print(
                "OUTPUT_QUALITY_DIAGNOSTIC_JSON: "
                + json.dumps(output_quality_diagnostic, ensure_ascii=True, sort_keys=True),
                flush=True,
            )

            prompt_parser_diagnostic = dict(
                result.pipeline_result.metadata.get("prompt_parser") or {}
            )
            print(
                "PROMPT_PARSER_DIAGNOSTIC_JSON: "
                + json.dumps(prompt_parser_diagnostic, ensure_ascii=True, sort_keys=True),
                flush=True,
            )
            model_diagnostic = dict(result.request_extras.get("model_provenance") or {})
            model_cache_reused = model_cache_reused or bool(model_diagnostic.get("cache_reused"))
            print(
                "MODEL_DIAGNOSTIC_JSON: "
                + json.dumps(model_diagnostic, ensure_ascii=True, sort_keys=True),
                flush=True,
            )

            if result.request.save_images:
                prepared_save_request = result.prepared_save_request
                if prepared_save_request is None:
                    raise RuntimeError(
                        "Generation completed, but no prepared output save request was returned."
                    )
                if int(result.expected_saved_count or 0) < 1:
                    raise RuntimeError(
                        "Generation completed, but output saving was requested and no save artifacts were prepared."
                    )
                save_ticket = self._ensure_output_save_queue().enqueue(
                    prepared_save_request,
                    job_id=job_id,
                    batch_number=batch_number,
                )
                save_tickets.append(save_ticket)
                saved_paths: list[str] = []
            else:
                saved_paths = [record.image_path for record in result.saved_records]

            if requested_seed is not None and int(requested_seed) >= 0:
                expected_base_seed = offset_seed(
                    int(requested_seed), completed_batches * int(request.batch_size)
                )
                if int(result.request.seed) != expected_base_seed:
                    raise RuntimeError(
                        "Fixed seed changed unexpectedly: "
                        f"requested batch base {expected_base_seed}, runtime used {result.request.seed}."
                    )
                expected_image_seeds = [
                    offset_seed(expected_base_seed, index)
                    for index in range(int(request.batch_size))
                ]
                if list(result.request.resolved_seeds or []) != expected_image_seeds:
                    raise RuntimeError(
                        "Fixed per-image seed sequence changed unexpectedly: "
                        f"expected {expected_image_seeds}, runtime used {result.request.resolved_seeds}."
                    )
            completed_batches += 1
            if not result.request.save_images:
                total_saved += len(saved_paths)
                for record in result.saved_records:
                    seed_label = "unknown" if record.seed is None else str(record.seed)
                    print(f"  Image [seed {seed_label}]: {record.image_path}", flush=True)
                    if record.txt_path:
                        print(f"  TXT:   {record.txt_path}", flush=True)
                    if record.json_path:
                        print(f"  JSON:  {record.json_path}", flush=True)

        generation_completed = time.perf_counter()
        self.timings["generation_execution_time_ms"] = round(
            generation_execution_seconds * 1000.0,
            3,
        )

        # Request-scoped mutable state must never leak across resident jobs. Clear
        # it before applying model residency so Hot retains model components only.
        stage_started = time.perf_counter()
        runner.reset_runtime_state()
        if trace_enabled:
            _append_trace_stage(preparation_trace, "reset_runtime_state_after_generation", stage_started)

        # The save queue owns CPU images and metadata only. Establish the selected
        # model's post-job residency before draining disk persistence so Hot GPU
        # restoration can overlap saving without invoking Managed retention.
        self.emit_status("applying_retention_policy", action="restore_selected_model_residency")
        residency_started = time.perf_counter()
        retention = self._apply_runtime_residency_policy(
            runner,
            extras,
            reason="successful_job_completion",
            post_job=True,
        )
        self.timings["post_generation_residency_time_ms"] = round(
            (time.perf_counter() - residency_started) * 1000.0,
            3,
        )
        if trace_enabled:
            _append_trace_stage(preparation_trace, "post_generation_residency", residency_started)

        save_wait_started = time.perf_counter()
        if save_tickets:
            self.emit_status(
                "saving_output",
                action="drain_async_output_save_queue",
                pending_save_batches=len(save_tickets),
            )
            for ticket in save_tickets:
                records = ticket.result()
                total_saved += len(records)
        self.timings["output_save_wait_time_ms"] = round(
            (time.perf_counter() - save_wait_started) * 1000.0,
            3,
        )
        if trace_enabled:
            _append_trace_stage(preparation_trace, "output_save_wait", save_wait_started)
        self.timings["post_generation_finalize_time_ms"] = round(
            (time.perf_counter() - generation_completed) * 1000.0,
            3,
        )
        total_ms = round((time.perf_counter() - job_started) * 1000.0, 3)
        command_wall_time_ms = round((time.perf_counter() - command_started) * 1000.0, 3)
        self.timings.update(
            {
                "last_job_total_ms": total_ms,
                "command_wall_time_ms": command_wall_time_ms,
                "resident_reuse_benefited_last_job": model_cache_reused,
                "cold_or_switch_load_last_job": not model_cache_reused,
            }
        )
        if trace_enabled:
            preparation_trace["command_wall_time_ms"] = command_wall_time_ms
            preparation_trace["execution_window_time_ms"] = total_ms
            preparation_trace["generation_execution_time_ms"] = self.timings.get("generation_execution_time_ms")
            preparation_trace["post_generation_finalize_time_ms"] = self.timings.get("post_generation_finalize_time_ms")
            self.last_preparation_trace = copy.deepcopy(preparation_trace)
        else:
            self.last_preparation_trace = {}
        post_status = runner.resident_model_status()
        job_report = {
            "schema_version": 1,
            "timestamp": _utc_timestamp(),
            "job_id": job_id,
            "model_path": self.selected_model_path,
            "architecture": post_status.get("architecture"),
            "prompt_parser_preset_name": str(getattr(request, "prompt_parser_preset_name", "") or ""),
            "residency_mode_requested": self.residency_mode_requested,
            "residency_state_effective": resolve_effective_residency_state(
                requested_mode=self.residency_mode_requested,
                stage="ready",
                resident=bool(post_status.get("resident")),
                gpu_loaded=bool(post_status.get("gpu_loaded")),
                staged_runtime=bool(post_status.get("staged_runtime")),
                hot_residency_active=self.hot_residency_active,
                hot_gpu_ready=bool(post_status.get("hot_gpu_ready", post_status.get("gpu_loaded"))),
            ),
            "generation_residency_classification": generation_residency_classification,
            "resident_change_classification": self.last_resident_change_classification,
            "post_job_residency_action": self.post_job_residency_action,
            "residency_reason": self.last_residency_reason,
            "degradation_reason": self.last_residency_report.get("degradation_reason"),
            "staging_reason": self.last_residency_report.get("staging_reason"),
            "memory_policy": str(self.runtime_settings.get("memory_policy") or "auto"),
            "retention_device_policy": str(self.runtime_settings.get("model_runtime_retention_device") or "auto"),
            "execution_device_policy": str(self.runtime_settings.get("model_runtime_execution_device") or "cuda_preferred"),
            "timings": dict(self.timings),
            "preparation_trace": copy.deepcopy(self.last_preparation_trace),
            "memory": dict(post_status.get("cuda_memory") or {}),
            "component_devices": dict(post_status.get("component_devices") or {}),
            "hot_reuse_count": self.hot_reuse_count,
            "cold_or_switch_load_count": self.cold_or_switch_load_count,
        }
        self.recent_job_reports.append(job_report)
        self.recent_job_reports = self.recent_job_reports[-10:]
        print("MODEL_RESIDENCY_REPORT_JSON: " + json.dumps(job_report, ensure_ascii=True, sort_keys=True), flush=True)
        self.current_job_id = None
        status = self.emit_status(
            "ready",
            action="job_complete",
            completed_batches=completed_batches,
            total_saved=total_saved,
            resident_reuse_benefited=model_cache_reused,
            generation_residency_classification=generation_residency_classification,
            post_job_residency_action=self.post_job_residency_action,
            job_total_ms=total_ms,
            command_wall_time_ms=command_wall_time_ms,
            retention=retention,
        )
        return {
            "job_id": job_id,
            "completed_batches": completed_batches,
            "total_saved": total_saved,
            "resident_reuse_benefited": model_cache_reused,
            "status": status,
        }

    def set_residency_mode(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.current_job_id:
            raise RuntimeError("Cannot change model residency policy while a generation is active.")
        settings = dict(command.get("runtime_settings") or {})
        mode = self._set_residency_mode_requested(settings.get("model_residency_mode"))
        settings["model_residency_mode"] = mode
        update_model_load_runtime_settings(self.runtime_settings, settings)
        if self.runner is None or not self.runner.resident_model_status().get("resident"):
            return self.emit_status(
                "idle",
                action="residency_policy_changed",
                residency_policy_applied=False,
            )
        self.emit_status("applying_retention_policy", action="residency_policy_change")
        retention = self._apply_runtime_residency_policy(
            self.runner,
            settings,
            reason="webui_residency_policy_change",
        )
        return self.emit_status(
            "ready",
            action="residency_policy_changed",
            residency_policy_applied=True,
            retention=retention,
        )

    def unload(self) -> dict[str, Any]:
        if self.current_job_id:
            raise RuntimeError("Cannot unload the checkpoint while a generation is active.")
        released = self.runner.clear_model_cache() if self.runner is not None else {
            "cached_entries_released": 0,
            "previous_model_path": None,
            "unload_time_ms": 0.0,
        }
        self.selected_model_path = None
        self.last_error = None
        self._invalidate_hot_residency("explicit_unload", forced_release=True)
        return self.emit_status(
            "idle",
            action="unload",
            post_job_residency_action=POST_JOB_RESIDENCY_FORCED_RELEASE,
            released=released,
        )

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        name = str(command.get("command") or "status").strip().lower()
        if name == "status":
            return self.emit_status()
        if name == "activate":
            return self.activate(command)
        if name == "set_residency_mode":
            return self.set_residency_mode(command)
        if name == "unload":
            return self.unload()
        if name == "run":
            return self.run_job(command)
        if name == "shutdown":
            released = self.runner.clear_model_cache() if self.runner is not None else {
                "cached_entries_released": 0,
                "previous_model_path": None,
                "unload_time_ms": 0.0,
            }
            self.selected_model_path = None
            self._invalidate_hot_residency("runtime_shutdown", forced_release=True)
            if self._output_save_queue is not None:
                self._output_save_queue.shutdown(wait=True)
                self._output_save_queue = None
            result = self.emit_status("idle", action="shutdown", released=released)
            result["shutdown"] = True
            return result
        raise ValueError(f"Unsupported model runtime command: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IMAGE_GEN resident txt2img model runtime")
    parser.add_argument("--project-root")
    parser.add_argument("--project-config")
    add_runtime_startup_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    parser_argv = argv_for_primary_parser(raw_argv)
    args = build_parser().parse_args(parser_argv)
    args._runtime_argv = raw_argv
    context = ProjectContext.load(
        project_root=args.project_root,
        config_path=args.project_config,
    )
    runtime_startup_options = _PREBOOTSTRAP_RUNTIME_STARTUP_OPTIONS
    args.runtime_startup_options = runtime_startup_options
    worker = ResidentTxt2ImgModelRuntime(context, runtime_startup_options)
    _emit(
        _READY_PREFIX,
        {
            "schema_version": 1,
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "project_root": str(context.project_root),
            "runtime_startup_options": runtime_startup_options.to_dict(),
            "runtime_startup_status": build_runtime_startup_status(
                runtime_startup_options,
                {"mslk_fmha": runtime_startup_options.mslk_fmha.to_dict()},
            ),
        },
    )
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        command_id = ""
        command: dict[str, Any] = {}
        try:
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ValueError("Model runtime commands must be JSON objects.")
            command_id = str(command.get("command_id") or uuid.uuid4().hex)
            result = worker.handle(command)
            _emit(
                _COMPLETE_PREFIX,
                {
                    "command_id": command_id,
                    "command": command.get("command"),
                    "ok": True,
                    "result": result,
                },
            )
            if str(command.get("command") or "").lower() == "shutdown":
                return 0
        except BaseException as exc:
            worker.last_error = f"{type(exc).__name__}: {exc}"
            worker.current_job_id = None
            worker.emit_status("failed", error=worker.last_error)
            _emit(
                _COMPLETE_PREFIX,
                {
                    "command_id": command_id,
                    "command": command.get("command"),
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
