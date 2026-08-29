from __future__ import annotations

import argparse
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
from image_gen.runtime.model_load_variant import (
    MODEL_LOAD_VARIANT_FIELDS,
    model_load_variant_matches_resident,
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
    print(prefix + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


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


class ResidentTxt2ImgModelRuntime:
    def __init__(
        self,
        context: ProjectContext,
        runtime_startup_options: RuntimeStartupOptions | None = None,
    ) -> None:
        self.context = context
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
        self.timings: dict[str, Any] = {}
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
            + json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
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
                ensure_ascii=False,
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

    def _status_payload(self, *, stage: str | None = None, **extra: Any) -> dict[str, Any]:
        if stage is not None:
            self.stage = str(stage)
            self.last_transition_unix = _utc_timestamp()
        residency = self.runner.resident_model_status() if self.runner is not None else {
            "resident": False,
            "model_path": None,
            "cache_entries": 0,
            "cpu_loaded": False,
            "gpu_loaded": False,
            "generation_ready": False,
            "architecture": "",
            "component_devices": {},
            "cuda_memory": {"allocated_bytes": 0, "reserved_bytes": 0},
        }
        return {
            "schema_version": 1,
            "worker_pid": os.getpid(),
            "stage": self.stage,
            "residency_state": "resident" if residency.get("resident") else "empty",
            "selected_model_path": self.selected_model_path,
            "current_model_path": residency.get("model_path"),
            "cpu_loaded": bool(residency.get("cpu_loaded")),
            "gpu_loaded": bool(residency.get("gpu_loaded")),
            "generation_ready": bool(residency.get("generation_ready")),
            "staged_runtime": bool(residency.get("staged_runtime")),
            "architecture": str(residency.get("architecture") or ""),
            "component_devices": dict(residency.get("component_devices") or {}),
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
            **extra,
        }

    def emit_status(self, stage: str | None = None, **extra: Any) -> dict[str, Any]:
        payload = self._status_payload(stage=stage, **extra)
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
        model_path = str(command.get("model_path") or "").strip()
        if not model_path:
            raise ValueError("A model_path is required for model activation.")
        self.selected_model_path = model_path
        self.current_job_id = None
        self.last_error = None
        self.emit_status("preparing_model", action="activate")
        started = time.perf_counter()
        extras = dict(command.get("runtime_settings") or {})
        startup_runtime = runtime_request_settings(self.runtime_startup_options)
        extras.update(startup_runtime)
        if not torch.cuda.is_available():
            extras["model_runtime_execution_device"] = "cpu"
            extras["model_runtime_retention_device"] = "cpu"
        extras = sanitize_model_load_runtime_settings(extras)
        update_model_load_runtime_settings(self.runtime_settings, extras)
        runner = self._ensure_runner()
        current = runner.resident_model_status()
        current_path = str(current.get("model_path") or "")
        requested_path = _normalized_resolved_path(model_path)
        current_resolved = _normalized_resolved_path(current_path)
        requested_composition = str(extras.get("advanced_model_composition_sha256") or "")
        current_composition = str(current.get("composition_sha256") or "")
        composition_matches = requested_composition == current_composition
        load_variant_matches = model_load_variant_matches_resident(extras, current)
        extras["model_runtime_event_callback"] = self._runner_event
        reused_resident_model = bool(
            current.get("resident")
            and current_path
            and current_resolved == requested_path
            and composition_matches
            and load_variant_matches
        )
        if current_path and (current_resolved != requested_path or not composition_matches or not load_variant_matches):
            self.emit_status(
                "unloading",
                action="automatic_model_swap",
                previous_model_path=current_path,
                next_model_path=model_path,
            )
            runner.clear_model_cache()
            reused_resident_model = False

        if reused_resident_model:
            self.emit_status("applying_retention_policy", action="reuse_resident_model")
            retention = runner.apply_resident_retention(extras)
            retention_status = dict(retention.get("status") or {})
            self.timings.update(
                {
                    "activate_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
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
            )

        result = runner.preload_model(model_path, extras)
        self.emit_status("applying_retention_policy")
        retention = runner.apply_resident_retention(extras)
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
        elapsed = round((time.perf_counter() - started) * 1000.0, 3)
        return self.emit_status(
            "ready",
            action="activate_complete",
            activate_time_ms=elapsed,
            model_provenance=dict(result.get("model_provenance") or {}),
            retention=retention,
        )

    def _load_payload(self, config_path: str) -> tuple[Any, dict[str, Any]]:
        payload = load_request_payload(
            config_path=config_path,
            base_payload=self.context.generation_defaults(),
        )
        payload, _resolution = normalize_scheduler_payload(payload)
        request, payload_extras = payload_to_generation_request(payload)
        return request, sanitize_model_load_runtime_settings(payload_extras)

    def run_job(self, command: dict[str, Any]) -> dict[str, Any]:
        job_id = str(command.get("job_id") or uuid.uuid4().hex[:12])
        config_path = str(command.get("config_path") or "").strip()
        if not config_path:
            raise ValueError("A config_path is required for resident generation.")
        self.current_job_id = job_id
        self.last_error = None
        runner = self._ensure_runner()
        runner.reset_runtime_state()
        request, payload_extras = self._load_payload(config_path)
        self.selected_model_path = str(payload_extras.get("model_path") or "") or None
        resident_before = runner.resident_model_status()
        requested_composition = str(payload_extras.get("advanced_model_composition_sha256") or "")
        current_composition = str(resident_before.get("composition_sha256") or "")
        resident_reuse_candidate = bool(
            resident_before.get("resident")
            and resident_before.get("model_path")
            and os.path.normcase(str(Path(str(resident_before.get("model_path"))).resolve()))
            == os.path.normcase(str(Path(str(self.selected_model_path)).resolve()))
            and requested_composition == current_composition
            and model_load_variant_matches_resident(payload_extras, resident_before)
        )
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
            self.activate({
                "model_path": self.selected_model_path,
                "runtime_settings": activation_settings,
            })
            resident_before = runner.resident_model_status()
            resident_reuse_candidate = bool(resident_before.get("resident"))
        self.emit_status(
            "preparing_model",
            action="run_job",
            resident_reuse_candidate=resident_reuse_candidate,
        )

        extras = {
            "live_sampler_map": self.live_sampler_map,
            "live_scheduler_map": self.live_scheduler_map,
            "model_runtime_event_callback": self._runner_event,
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
            if key.startswith("model_runtime_") or key.startswith("memory_")
        })
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

        while unlimited or completed_batches < batch_count:
            # Clear request-scoped runtime state before each batch iteration while keeping resident model components loaded.
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
                    ensure_ascii=False,
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
            self.emit_status("running", batch_number=batch_number, batch_count=batch_count)
            run_request_kwargs = {
                "save_txt": bool(command.get("save_txt", True)),
                "save_json": bool(command.get("save_json", True)),
                "save_diagnostics_json": bool(command.get("save_diagnostics_json", True)),
                "defer_output_save": True,
            }
            while True:
                try:
                    result = runner.run_request(
                        batch_request,
                        batch_extras,
                        **run_request_kwargs,
                    )
                    break
                except TypeError as exc:
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

            live_preview_summary = dict(result.pipeline_result.metadata.get("live_preview") or {})
            print(
                "LIVE_PREVIEW_SUMMARY_JSON: "
                + json.dumps(live_preview_summary, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            output_quality_diagnostic = dict(
                result.pipeline_result.metadata.get("output_quality") or {}
            )
            print(
                "OUTPUT_QUALITY_DIAGNOSTIC_JSON: "
                + json.dumps(output_quality_diagnostic, ensure_ascii=False, sort_keys=True),
                flush=True,
            )

            prompt_parser_diagnostic = dict(
                result.pipeline_result.metadata.get("prompt_parser") or {}
            )
            print(
                "PROMPT_PARSER_DIAGNOSTIC_JSON: "
                + json.dumps(prompt_parser_diagnostic, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            model_diagnostic = dict(result.request_extras.get("model_provenance") or {})
            model_cache_reused = model_cache_reused or bool(model_diagnostic.get("cache_reused"))
            print(
                "MODEL_DIAGNOSTIC_JSON: "
                + json.dumps(model_diagnostic, ensure_ascii=False, sort_keys=True),
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

        # The save queue owns CPU images and metadata only. Restore the selected
        # checkpoint residency now so GPU transfers overlap disk persistence and
        # the next unchanged-model generation is ready as soon as saving ends.
        self.emit_status("applying_retention_policy", action="restore_selected_model_residency")
        retention = runner.apply_resident_retention(extras)

        if save_tickets:
            self.emit_status(
                "saving_output",
                action="drain_async_output_save_queue",
                pending_save_batches=len(save_tickets),
            )
            for ticket in save_tickets:
                records = ticket.result()
                total_saved += len(records)
        total_ms = round((time.perf_counter() - job_started) * 1000.0, 3)
        self.timings.update(
            {
                "last_job_total_ms": total_ms,
                "resident_reuse_benefited_last_job": model_cache_reused,
                "cold_or_switch_load_last_job": not model_cache_reused,
            }
        )
        runner.reset_runtime_state()
        self.current_job_id = None
        status = self.emit_status(
            "ready",
            action="job_complete",
            completed_batches=completed_batches,
            total_saved=total_saved,
            resident_reuse_benefited=model_cache_reused,
            job_total_ms=total_ms,
            retention=retention,
        )
        return {
            "job_id": job_id,
            "completed_batches": completed_batches,
            "total_saved": total_saved,
            "resident_reuse_benefited": model_cache_reused,
            "status": status,
        }

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
        return self.emit_status("idle", action="unload", released=released)

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        name = str(command.get("command") or "status").strip().lower()
        if name == "status":
            return self.emit_status()
        if name == "activate":
            return self.activate(command)
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
