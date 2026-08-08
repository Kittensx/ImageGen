from __future__ import annotations

import json
import os
import gc
import inspect
import time
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import torch

from modules.component_placement import place_component

from image_gen.contracts import GenerationRequest, GenerationResult
from image_gen.runtime.composition import PipelineCompositionRoot
from image_gen.runtime.lora_runtime import LoRARuntimeManager
from image_gen.systems.diagnostics import (
    DiagnosticSession,
    DiagnosticsSystem,
    PipelineStageError,
)
from image_gen.systems.model_loading import ModelLoadingSystem
from image_gen.contracts.vae_provenance import read_vae_provenance
from image_gen.systems.memory.telemetry import normalize_cuda_memory_payload
from image_gen.systems.output import OutputSystem, PreparedOutputSaveRequest
from image_gen.systems.outpainting import (
    OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION,
    build_post_generation_shape_action,
    format_outpaint_failure,
    normalize_outpaint_source_handoff_mode,
    resolve_outpaint_shape_target,
)
from image_gen.systems.upscaling import (
    StandaloneNeuralUpscaler,
    UpscalerModelRegistry,
    discover_upscalers,
)
from image_gen.systems.registry import RuntimeRegistrySystem
from modules.pipeline.progress_reporter import ProgressReporter
from modules.project_context import ProjectContext
from modules.shared_state import SharedState
from modules.txt2img.output_saver import SavedImageRecord
from modules.txt2img.request_loader import load_request_payload, payload_to_generation_request


@dataclass
class Txt2ImgRunResult:
    request: GenerationRequest
    request_extras: dict[str, Any] = field(default_factory=dict)
    pipeline_result: GenerationResult = field(default_factory=GenerationResult)
    manifest: Any | None = None
    saved_records: list[SavedImageRecord] = field(default_factory=list)
    generation_time_sec: float | None = None
    run_id: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    prepared_save_request: PreparedOutputSaveRequest | None = None
    expected_saved_count: int = 0




def _prepare_output_directory(
    project_context: ProjectContext,
    request: GenerationRequest,
    *,
    should_save: bool,
) -> Path | None:
    """Create the runtime-owned output directory before path validation."""
    request.save_images = bool(should_save)
    if not should_save:
        return None

    configured_output = request.output_dir or project_context.txt2img_output_root
    output_path = Path(str(configured_output)).expanduser()
    if not output_path.is_absolute():
        output_path = project_context.resolve_project_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()
    request.output_dir = str(output_path)
    return output_path


def _verify_saved_records(records: list[SavedImageRecord]) -> list[SavedImageRecord]:
    """Require at least one real image file when persistence was requested."""
    if not records:
        raise RuntimeError(
            "Image saving was requested, but the output system returned no saved records."
        )
    missing = [record.image_path for record in records if not Path(record.image_path).is_file()]
    if missing:
        raise RuntimeError(
            "The output system reported image paths that do not exist: "
            + ", ".join(missing)
        )
    return records




def _compact_memory_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the live WebUI subset of a memory-manager event.

    The full memory-manager status contains component residency histories and
    other large diagnostic structures. Those belong in manifests and failure
    bundles, not in the console transport used by the WebUI. The compact event
    preserves the live panel contract without reintroducing large per-stage
    JSON writes.
    """

    source = dict(payload or {})
    status = dict(source.get("status") or {})
    snapshot = dict(status.get("latest_snapshot") or {})
    cuda = normalize_cuda_memory_payload(dict(snapshot.get("cuda") or {}))
    estimate = dict(status.get("latest_estimate") or {})
    stage_peaks = [
        dict(value or {})
        for value in dict(status.get("peak_vram_by_stage") or {}).values()
        if isinstance(value, dict)
    ]

    def _maximum_peak(key: str, fallback: Any = None) -> int | None:
        values: list[int] = []
        for item in stage_peaks:
            try:
                candidate = int(item.get(key))
            except (TypeError, ValueError):
                continue
            if candidate >= 0:
                values.append(candidate)
        try:
            candidate = int(fallback)
        except (TypeError, ValueError):
            candidate = -1
        if candidate >= 0:
            values.append(candidate)
        return max(values) if values else None

    compact_snapshot = {
        "pipeline_stage": snapshot.get("pipeline_stage"),
        "timestamp": snapshot.get("timestamp"),
        "cuda": {
            key: cuda.get(key)
            for key in (
                "available",
                "device_index",
                "device_name",
                "allocated_vram_bytes",
                "reserved_vram_bytes",
                "free_vram_bytes",
                "total_vram_bytes",
                "peak_allocated_vram_bytes",
                "peak_reserved_vram_bytes",
                "physical_measurement_available",
                "physical_total_vram_bytes",
                "physical_free_vram_bytes",
                "physical_used_vram_bytes",
                "physical_measurement_source",
                "allocator_committed_vram_bytes",
                "allocator_overcommit_bytes",
                "allocator_oversubscribed",
                "allocator_measurement_semantics",
            )
            if key in cuda
        },
    }
    compact_status = {
        key: status.get(key)
        for key in (
            "requested_policy",
            "effective_policy",
            "active_stage",
            "active_gpu_components",
            "offloaded_components",
            "component_transfer_count",
            "peak_allocated_vram_bytes",
            "peak_reserved_vram_bytes",
            "preview_policy",
            "preview_image_decode_suspended",
            "preview_image_decode_suspension_reason",
            "preview_image_decode_suspension_source",
            "preview_decoder_released",
        )
        if key in status
    }
    compact_status["job_peak_allocated_vram_bytes"] = _maximum_peak(
        "peak_allocated_vram_bytes",
        status.get("peak_allocated_vram_bytes"),
    )
    compact_status["job_peak_reserved_vram_bytes"] = _maximum_peak(
        "peak_reserved_vram_bytes",
        status.get("peak_reserved_vram_bytes"),
    )
    compact_status["latest_snapshot"] = compact_snapshot
    if estimate:
        compact_status["latest_estimate"] = {
            key: estimate.get(key)
            for key in (
                "stage",
                "estimated_expected_bytes",
                "safety_adjusted_required_bytes",
                "available_bytes",
                "headroom_bytes",
                "feasible",
                "confidence",
            )
            if key in estimate
        }
    actions = list(status.get("automatic_actions") or [])
    if actions:
        compact_status["automatic_actions"] = actions[-5:]
    return {
        "event": source.get("event"),
        "stage": source.get("stage"),
        "active_stage": source.get("active_stage"),
        "status": compact_status,
        "transport": "compact_live_status",
    }


def _build_memory_event_callback(
    *,
    console_verbose: bool,
    console_mode: str = "json",
    progress_reporter: Any | None = None,
) -> Callable[[dict[str, Any]], None]:
    """Build the memory event callback used by CLI and WebUI workers.

    WebUI workers retain the compact JSON transport they parse. Human CLI runs
    can instead fold the same requested/available/used figures into the active
    sampling line, or suppress memory output entirely. Verbose diagnostics
    always retain the full structured payload.
    """

    resolved_mode = str(console_mode or "json").strip().lower()
    if resolved_mode not in {"off", "compact", "json"}:
        resolved_mode = "json"
    if console_verbose:
        resolved_mode = "json"

    def _emit(payload: dict[str, Any]) -> None:
        compact = _compact_memory_event_payload(payload)
        if resolved_mode == "compact":
            updater = getattr(progress_reporter, "update_memory_status", None)
            if callable(updater):
                updater(payload)
            return
        if resolved_mode == "off":
            return
        output = payload if console_verbose else compact
        print(
            "MEMORY_STATUS_JSON: "
            + json.dumps(output, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )

    setattr(_emit, "_image_gen_default_memory_callback", True)
    setattr(_emit, "_image_gen_console_mode", resolved_mode)
    return _emit


class Txt2ImgRunner:
    """Use-case coordinator for loading, generation, diagnostics, and output."""

    def __init__(
        self,
        *,
        prompt_adapter: Any | None = None,
        scheduler_adapter: Any | None = None,
        sampler_adapter: Any | None = None,
        prompt_adapter_factory: Callable[..., Any] | None = None,
        scheduler_adapter_factory: Callable[..., Any] | None = None,
        sampler_adapter_factory: Callable[..., Any] | None = None,
        model_loader: Any | None = None,
        project_context: ProjectContext | None = None,
        state: Any | None = None,
        tokenizer: Any = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        latent_scale_factor: int = 8,
        vae_scaling_factor: float = 0.18215,
        registry_system: RuntimeRegistrySystem | None = None,
        output_system: OutputSystem | None = None,
        model_loading_system: ModelLoadingSystem | None = None,
        diagnostics_system: DiagnosticsSystem | None = None,
        system_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.prompt_adapter = prompt_adapter
        self.scheduler_adapter = scheduler_adapter
        self.sampler_adapter = sampler_adapter
        self.prompt_adapter_factory = prompt_adapter_factory
        self.scheduler_adapter_factory = scheduler_adapter_factory
        self.sampler_adapter_factory = sampler_adapter_factory

        inherited_context = getattr(model_loader, "context", None)
        self.project_context = project_context or inherited_context or ProjectContext.load()
        if model_loader is None:
            from modules.load_safetensors_model import LoadModel

            model_loader = LoadModel(project_context=self.project_context)
        self.model_loader = model_loader
        self.state = state or SharedState()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.latent_scale_factor = int(latent_scale_factor)
        self.vae_scaling_factor = float(vae_scaling_factor)
        self.system_overrides = dict(system_overrides or {})

        override_diagnostics = self.system_overrides.get("diagnostics")
        self.diagnostics_system = (
            diagnostics_system
            or override_diagnostics
            or DiagnosticsSystem.from_project_context(self.project_context)
        )
        self.system_overrides["diagnostics"] = self.diagnostics_system

        self.registry_system = registry_system or RuntimeRegistrySystem(
            self.state, project_context=self.project_context
        )
        self.registry_system.bind_state(self.state)
        self.output_system = output_system or OutputSystem()
        self.model_loading_system = model_loading_system or ModelLoadingSystem(self.model_loader)
        self.tokenizer = tokenizer
        self._loaded_model_cache: dict[tuple[str, str, str, int, int], Any] = {}
        self.last_loaded_model: Any | None = None
        self.lora_runtime_manager = LoRARuntimeManager(self.project_context)


    def clear_model_cache(self) -> dict[str, Any]:
        """Release cached checkpoint components and return before/after telemetry.

        CUDA's process context and third-party allocations may remain visible in system
        monitors, but IMAGE_GEN-owned module parameters are moved to CPU and allocator
        caches are explicitly released before the status response is produced.
        """
        started = time.perf_counter()
        cached_entries = len(self._loaded_model_cache)
        previous = self.resident_model_status()
        cuda_before = dict(previous.get("cuda_memory") or {})
        released_components: list[str] = []
        placement_errors: list[dict[str, str]] = []

        loaded_objects: list[Any] = list(self._loaded_model_cache.values())
        if self.last_loaded_model is not None:
            loaded_objects.append(self.last_loaded_model)
        seen: set[int] = set()
        for loaded in loaded_objects:
            if id(loaded) in seen:
                continue
            seen.add(id(loaded))
            components = getattr(loaded, "components", None)
            if components is None:
                continue
            for name in ("unet", "text_encoder", "vae"):
                module = getattr(components, name, None)
                if module is None or not callable(getattr(module, "to", None)):
                    continue
                try:
                    module.to(device=torch.device("cpu"))
                    released_components.append(name)
                except Exception as exc:
                    placement_errors.append(
                        {"component": name, "error_type": type(exc).__name__, "error": str(exc)}
                    )

        self._loaded_model_cache.clear()
        self.last_loaded_model = None
        self.lora_runtime_manager.reset()
        gc.collect()
        cuda_cleanup_errors: list[str] = []
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                cuda_cleanup_errors.append(f"synchronize: {type(exc).__name__}: {exc}")
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                cuda_cleanup_errors.append(f"empty_cache: {type(exc).__name__}: {exc}")
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except Exception as exc:
                    cuda_cleanup_errors.append(f"ipc_collect: {type(exc).__name__}: {exc}")

        after = self.resident_model_status()
        return {
            "cached_entries_released": cached_entries,
            "previous_model_path": previous.get("model_path"),
            "components_moved_to_cpu": sorted(set(released_components)),
            "component_release_errors": placement_errors,
            "cuda_cleanup_errors": cuda_cleanup_errors,
            "cuda_memory_before": cuda_before,
            "cuda_memory_after": dict(after.get("cuda_memory") or {}),
            "note": (
                "System GPU monitors may still show the CUDA context or non-IMAGE_GEN allocations; "
                "allocated_bytes is the authoritative IMAGE_GEN/PyTorch tensor allocation value."
            ),
            "unload_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def reset_runtime_state(self) -> None:
        """Reset request-scoped mutable state while retaining loaded components."""
        self.state = SharedState()
        self.registry_system.bind_state(self.state)

    @staticmethod
    def _emit_model_runtime_event(extras: dict[str, Any], stage: str, **payload: Any) -> None:
        callback = extras.get("model_runtime_event_callback")
        if not callable(callback):
            return
        try:
            callback({"stage": str(stage), **payload})
        except Exception:
            return

    def resident_model_status(self) -> dict[str, Any]:
        loaded = self.last_loaded_model
        components = getattr(loaded, "components", None)
        report = getattr(getattr(loaded, "load_plan", None), "report", None)
        model_path = str(getattr(report, "model_path", "") or "")
        component_devices: dict[str, str] = {}
        if components is not None:
            for name in ("text_encoder", "unet", "vae"):
                module = getattr(components, name, None)
                device = "unknown"
                if module is not None:
                    try:
                        parameter = next(module.parameters())
                        device = str(parameter.device)
                    except (StopIteration, AttributeError, TypeError):
                        device = str(getattr(module, "device", "unknown"))
                component_devices[name] = device
        known_component_devices = [
            value for value in component_devices.values() if value and value != "unknown"
        ]
        gpu_loaded = bool(known_component_devices) and all(
            value.startswith("cuda") for value in known_component_devices
        )
        cpu_loaded = any(value.startswith("cpu") for value in known_component_devices)
        cuda_memory = {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "free_bytes": None,
            "total_bytes": None,
            "device_name": None,
        }
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                cuda_memory = {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                    "device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
                }
            except Exception:
                pass
        return {
            "resident": loaded is not None and bool(self._loaded_model_cache),
            "model_path": model_path or None,
            "cache_entries": len(self._loaded_model_cache),
            "cpu_loaded": cpu_loaded,
            "gpu_loaded": gpu_loaded,
            "component_devices": component_devices,
            "cuda_memory": cuda_memory,
        }

    def apply_resident_retention(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        """Keep the selected checkpoint components resident between jobs."""
        loaded = self.last_loaded_model
        components = getattr(loaded, "components", None)
        if components is None:
            return {"applied": False, "reason": "no cached model"}
        values = dict(settings or {})
        retain = {
            "unet": bool(values.get("memory_retain_checkpoint_between_jobs", True)),
            "vae": bool(values.get("memory_retain_vae_between_jobs", True)),
            "text_encoder": bool(values.get("model_runtime_retain_text_encoder_between_jobs", True)),
        }
        retention_policy = str(values.get("model_runtime_retention_device") or "cuda_preferred").strip().lower()
        execution_device, fallback_reason = self._resolve_execution_device(values)
        if retention_policy == "cpu":
            target_device = torch.device("cpu")
        elif retention_policy == "cuda":
            if not torch.cuda.is_available():
                target_device = torch.device("cpu")
                fallback_reason = "GPU retention was requested, but CUDA is unavailable; retained on CPU."
            else:
                target_device = torch.device("cuda")
        else:
            target_device = execution_device
        moves: list[dict[str, Any]] = []
        for name, keep_on_target in retain.items():
            module = getattr(components, name, None)
            if module is None or not hasattr(module, "to"):
                continue
            target = target_device if keep_on_target else torch.device("cpu")
            before = "unknown"
            try:
                before = str(next(module.parameters()).device)
            except (StopIteration, AttributeError, TypeError):
                pass
            if before == str(target) or (before.startswith("cuda") and target.type == "cuda"):
                continue
            module.to(target)
            moves.append({"component": name, "from": before, "to": str(target)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "applied": True,
            "retain": retain,
            "retention_device": str(target_device),
            "execution_device": str(execution_device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_fallback_reason": fallback_reason,
            "moves": moves,
            "status": self.resident_model_status(),
        }

    def preload_model(
        self,
        model_path: str,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load and retain checkpoint components without running a generation."""
        preload_extras = dict(extras or {})
        preload_extras["model_path"] = str(model_path)
        defaults = dict(self.project_context.generation_defaults() or {})
        defaults.update({
            "positive_prompt": defaults.get("positive_prompt") or "warmup",
            "negative_prompt": defaults.get("negative_prompt") or "",
            "model_path": str(model_path),
            "save_images": False,
        })
        request, payload_extras = payload_to_generation_request(defaults)
        payload_extras.update(preload_extras)
        effective_config_fn = getattr(self.project_context, "effective_config", None)
        effective_config = effective_config_fn() if callable(effective_config_fn) else {
            "project_root": str(getattr(self.project_context, "project_root", "."))
        }
        session = self.diagnostics_system.start(
            request,
            effective_config=effective_config,
            request_extras=payload_extras,
        )
        started = time.perf_counter()
        try:
            request.device = str(self.device)
            self._configure_runtime_state(payload_extras, session)
            request, payload_extras = self.registry_system.apply_resolution(request, payload_extras)
            self._build_pipeline(request, payload_extras, session)
            self.diagnostics_system.complete(session)
            status = self.resident_model_status()
            status.update({
                "preload_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "model_provenance": dict(payload_extras.get("model_provenance") or {}),
            })
            return status
        except Exception as exc:
            raise self.diagnostics_system.fail_unassigned(
                session, exc, system="model_loading", operation="preload_model"
            ) from exc
        finally:
            self.reset_runtime_state()

    def _build_local_tokenizer(self):
        local_dir = self.project_context.tokenizer_root
        if not local_dir.exists():
            raise FileNotFoundError(f"Missing local tokenizer directory: {local_dir}")
        from transformers import CLIPTokenizer

        return CLIPTokenizer.from_pretrained(str(local_dir), local_files_only=True)

    def _resolve_adapter(
        self,
        direct: Any,
        factory: Callable[..., Any] | None,
        *,
        request: GenerationRequest,
        extras: dict[str, Any],
        resolved_entry_key: str | None = None,
        resolved_descriptor_key: str | None = None,
    ) -> Any:
        if direct is not None:
            return direct
        if resolved_descriptor_key:
            adapter = self.registry_system.instantiate_adapter(extras.get(resolved_descriptor_key))
            if adapter is not None:
                return adapter
        if resolved_entry_key:
            adapter = self.registry_system.instantiate_adapter(extras.get(resolved_entry_key))
            if adapter is not None:
                return adapter
        if factory is None:
            raise ValueError("Missing adapter and factory for required pipeline component.")
        return factory(request=request, extras=extras, state=self.state)

    def _resolve_adapters(
        self,
        request: GenerationRequest,
        extras: dict[str, Any],
    ) -> tuple[Any, Any, Any]:
        return (
            self._resolve_adapter(
                self.prompt_adapter,
                self.prompt_adapter_factory,
                request=request,
                extras=extras,
            ),
            self._resolve_adapter(
                self.scheduler_adapter,
                self.scheduler_adapter_factory,
                request=request,
                extras=extras,
                resolved_entry_key="resolved_scheduler_entry",
                resolved_descriptor_key="resolved_scheduler_descriptor",
            ),
            self._resolve_adapter(
                self.sampler_adapter,
                self.sampler_adapter_factory,
                request=request,
                extras=extras,
                resolved_entry_key="resolved_sampler_entry",
                resolved_descriptor_key="resolved_sampler_descriptor",
            ),
        )

    def _configure_runtime_state(
        self,
        extras: dict[str, Any],
        session: DiagnosticSession,
    ) -> None:
        progress_reporter = ProgressReporter(
            enabled=session.config.progress,
            desc="Sampling",
            unit="step",
            machine_readable=bool(extras.get("progress_json", False)),
        )
        extras["progress_reporter"] = progress_reporter
        if hasattr(self.state, "extra"):
            # Live-preview writers and sinks are job-scoped. A resident model worker
            # survives across jobs, so retaining a closed writer here makes a UI
            # setting change appear to require a full process restart.
            previous_writer = self.state.extra.pop("live_preview_frame_writer", None)
            if previous_writer is not None and callable(getattr(previous_writer, "close", None)):
                try:
                    previous_writer.close()
                except Exception:
                    pass
            for preview_key in (
                "live_preview_sink",
                "live_preview_callback",
                "live_preview_warning_callback",
                "live_preview_event_callback",
                "live_preview_memory_event_callback",
                "live_preview_warnings",
            ):
                self.state.extra.pop(preview_key, None)

            self.state.extra["progress_reporter"] = progress_reporter
            console_verbose = bool(extras.get("_console_verbose", False))
            console_memory_mode = str(
                extras.get("_console_memory_mode")
                or ("json" if extras.get("progress_json", False) else "compact")
            ).strip().lower()
            existing_memory_callback = self.state.extra.get("memory_event_callback")
            replace_default_callback = bool(
                getattr(
                    existing_memory_callback,
                    "_image_gen_default_memory_callback",
                    False,
                )
            )
            if not callable(existing_memory_callback) or replace_default_callback:
                self.state.extra["memory_event_callback"] = _build_memory_event_callback(
                    console_verbose=console_verbose,
                    console_mode=console_memory_mode,
                    progress_reporter=progress_reporter,
                )
            for key in (
                "live_preview_callback",
                "live_preview_warning_callback",
                "live_preview_sink_factory",
                "live_preview_enabled",
                "live_preview_telemetry_enabled",
                "live_preview_mode",
                "live_preview_interval",
                "live_preview_width",
                "live_preview_format",
                "live_preview_keep_history",
                "live_preview_batch_index",
                "live_preview_quality",
                "live_preview_root",
                "live_preview_max_failures",
                "live_preview_clone_tensors",
                "live_preview_event_callback",
                "live_preview_async",
                "live_preview_adaptive_throttle",
                "live_preview_adaptive_target_ratio",
                "live_preview_adaptive_recovery_ratio",
                "live_preview_adaptive_max_interval",
                "live_preview_adaptive_window",
                "live_preview_adaptive_suspend_on_overhead",
                "live_preview_adaptive_suspend_ratio",
                "live_preview_adaptive_suspend_min_work_ms",
                "live_preview_adaptive_suspend_min_samples",
                "progress_json",
                "memory_policy",
                "memory_vram_safety_margin_mb",
                "memory_retain_checkpoint_between_jobs",
                "memory_retain_vae_between_jobs",
                "memory_pinned_cpu_memory",
                "memory_allow_tiled_vae_fallback",
                "memory_allow_preview_suspension_on_oom",
                "attention_slicing",
                "vae_tiling",
                "vae_slicing",
                "vae_device",
                "preview_policy",
                "hires_memory_profile",
                "pre_hires_cleanup",
                "oom_retry_profile",
                "oom_retry_limit",
                "runtime_profile",
                "runtime_startup_options",
                "runtime_replay_conformance_source",
                "model_runtime_retain_text_encoder_between_jobs",
                "cuda_allocator_environment",
                "cuda_allocator_diagnostics",
            ):
                if key in extras:
                    self.state.extra[key] = extras[key]
        if hasattr(self.state, "d"):
            self.state.d.device = self.device
            self.state.d.device_type = self.device.type
            if self.device.index is not None:
                self.state.d.device_index = self.device.index

    def _resolve_execution_device(self, extras: dict[str, Any]) -> tuple[torch.device, str | None]:
        policy = str(extras.get("model_runtime_execution_device") or "cuda_preferred").strip().lower()
        if policy == "cpu":
            return torch.device("cpu"), "CPU execution was explicitly selected."
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        if policy == "cuda_required":
            raise RuntimeError("CUDA execution is required, but torch.cuda.is_available() is false in this worker.")
        return torch.device("cpu"), "CUDA is unavailable in the model runtime; CPU fallback was activated."

    def _place_loaded_components(
        self,
        loaded: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[dict[str, Any]]:
        components = getattr(loaded, "components", None)
        if components is None:
            return []
        reports: list[dict[str, Any]] = []
        for name in ("unet", "text_encoder", "vae"):
            module = getattr(components, name, None)
            if module is None:
                continue
            reports.append(
                place_component(
                    module,
                    device=device,
                    dtype=dtype,
                    owner="Txt2ImgRunner.execution_promotion",
                    component_name=name,
                ).to_dict()
            )
        return reports

    def _build_pipeline(
        self,
        request: GenerationRequest,
        extras: dict[str, Any],
        session: DiagnosticSession,
    ):
        prompt_adapter, scheduler_adapter, sampler_adapter = self._resolve_adapters(request, extras)
        hires_scheduler_descriptor = extras.get("resolved_hires_scheduler_descriptor")
        hires_sampler_descriptor = extras.get("resolved_hires_sampler_descriptor")
        hires_scheduler_name = str(
            getattr(hires_scheduler_descriptor, "name", "")
            or getattr(request, "hires_scheduler_name", "")
            or request.scheduler_name
            or ""
        )
        hires_sampler_name = str(
            getattr(hires_sampler_descriptor, "name", "")
            or getattr(request, "hires_sampler_name", "")
            or request.sampler_name
            or ""
        )
        hires_scheduler_adapter = (
            scheduler_adapter
            if hires_scheduler_name == str(request.scheduler_name or "")
            else self.registry_system.instantiate_adapter(hires_scheduler_descriptor)
        )
        hires_sampler_adapter = (
            sampler_adapter
            if hires_sampler_name == str(request.sampler_name or "")
            else self.registry_system.instantiate_adapter(hires_sampler_descriptor)
        )
        if hires_scheduler_adapter is None or hires_sampler_adapter is None:
            raise ValueError(
                "Unable to instantiate the requested hires sampler/scheduler adapters."
            )
        if hasattr(self.state, "extra") and isinstance(self.state.extra, dict):
            self.state.extra["hires_scheduler_adapter"] = hires_scheduler_adapter
            self.state.extra["hires_sampler_adapter"] = hires_sampler_adapter
            self.state.extra["hires_plugin_compatibility"] = dict(
                extras.get("hires_plugin_compatibility") or {}
            )
            self.state.extra["hires_pair_qualification"] = dict(
                extras.get("hires_pair_qualification") or {}
            )
            self.state.extra["hires_sampler_inherited"] = bool(
                extras.get("hires_sampler_inherited", True)
            )
            self.state.extra["hires_scheduler_inherited"] = bool(
                extras.get("hires_scheduler_inherited", True)
            )
        memory_policy = str(extras.get("memory_policy") or "auto").strip().lower().replace(" ", "_")
        if memory_policy == "cpu_fallback":
            runtime_device, fallback_reason = torch.device("cpu"), "Memory policy explicitly selected CPU fallback."
        else:
            runtime_device, fallback_reason = self._resolve_execution_device(extras)
        runtime_dtype = torch.float32 if runtime_device.type == "cpu" else self.dtype
        extras["execution_device"] = str(runtime_device)
        extras["cuda_available"] = bool(torch.cuda.is_available())
        extras["cpu_fallback_reason"] = fallback_reason
        if hasattr(self.state, "extra") and isinstance(self.state.extra, dict):
            self.state.extra["execution_device"] = str(runtime_device)
            self.state.extra["cuda_available"] = bool(torch.cuda.is_available())
            self.state.extra["cpu_fallback_reason"] = fallback_reason
        request.device = str(runtime_device)
        request.dtype = runtime_dtype
        if hasattr(self.state, "d"):
            self.state.d.device = runtime_device
            self.state.d.device_type = runtime_device.type
            self.state.d.device_index = runtime_device.index
        model_path = extras.get("model_path") or self.model_loading_system.default_model_path
        if not model_path:
            raise ValueError("No model_path provided in request extras and no default MODEL_PATH found.")

        if self.tokenizer is None:
            self._emit_model_runtime_event(extras, "loading_tokenizer")
            self.tokenizer = self.diagnostics_system.run_stage(
                session,
                "model_loading",
                "load_tokenizer",
                self._build_local_tokenizer,
            )
        model_file = Path(str(model_path)).expanduser().resolve()
        model_provenance: dict[str, Any] = {
            "requested_path": str(model_path),
            "resolved_path": str(model_file),
            "loaded_path": "",
            "file_name": model_file.name,
            "file_size_bytes": None,
            "modified_ns": None,
            "sha256": "",
            "architecture": "",
            "prediction_type": "",
            "conditioning_dimension": None,
            "architecture_summary": "",
            "architecture_source": "",
            "architecture_contract": {},
            "checkpoint_kind": "",
            "cache_reused": False,
            "loader": "ModelLoadingSystem.load",
            "device": str(runtime_device),
            "dtype": str(runtime_dtype),
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_fallback_reason": fallback_reason,
            "execution_device_policy": str(extras.get("model_runtime_execution_device") or "cuda_preferred"),
        }
        if model_file.is_file():
            stat = model_file.stat()
            load_path = str(model_file)
            model_provenance["file_size_bytes"] = int(stat.st_size)
            model_provenance["modified_ns"] = int(stat.st_mtime_ns)
            cache_key = (
                load_path,
                str(runtime_dtype),
                str(runtime_device),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        else:
            # Injected/fake loaders used by focused tests may use a symbolic path.
            load_path = str(model_path)
            cache_key = (str(model_file), str(runtime_dtype), str(runtime_device), -1, -1)
        extras["model_provenance"] = dict(model_provenance)
        session.request_extras["model_provenance"] = dict(model_provenance)
        self.diagnostics_system.emit(
            session,
            "INFO",
            "model_loading",
            "select_checkpoint",
            "checkpoint selected for loading",
            **model_provenance,
        )
        loaded = self._loaded_model_cache.get(cache_key)
        load_started = time.perf_counter()
        if loaded is None:
            self._emit_model_runtime_event(
                extras,
                "loading_checkpoint",
                model_path=str(model_file),
                target_device=str(runtime_device),
                target_dtype=str(runtime_dtype),
            )
            load_method = self.model_loading_system.load
            try:
                load_parameters = inspect.signature(load_method).parameters
            except (TypeError, ValueError):
                load_parameters = {}
            load_kwargs = {
                "tokenizer": self.tokenizer,
                "dtype": runtime_dtype,
            }
            if (
                "device" in load_parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in load_parameters.values()
                )
            ):
                load_kwargs["device"] = runtime_device
            if runtime_device.type == "cuda":
                self._emit_model_runtime_event(
                    extras,
                    "moving_to_gpu",
                    model_path=str(model_file),
                    target_device=str(runtime_device),
                    transfer_strategy="checkpoint_loader",
                    note="Checkpoint hydration and GPU transfer are performed by the canonical loader.",
                )
            loaded = self.diagnostics_system.run_stage(
                session,
                "model_loading",
                "load_checkpoint",
                lambda: load_method(load_path, **load_kwargs),
            )
            self._loaded_model_cache.clear()
            self._loaded_model_cache[cache_key] = loaded
        else:
            model_provenance["cache_reused"] = True
            self._emit_model_runtime_event(
                extras,
                "reusing_checkpoint",
                model_path=str(model_file),
                target_device=str(runtime_device),
            )
            self.diagnostics_system.emit(
                session,
                "info",
                "model_loading",
                "reuse_checkpoint",
                "reused loaded checkpoint components",
                model_path=str(model_file),
            )
            self._emit_model_runtime_event(
                extras,
                "moving_to_gpu" if runtime_device.type == "cuda" else "preparing_model",
                model_path=str(model_file),
                target_device=str(runtime_device),
                transfer_strategy="cached_component_promotion",
            )
            extras["execution_placement_reports"] = self._place_loaded_components(
                loaded,
                device=runtime_device,
                dtype=runtime_dtype,
            )
        report = getattr(getattr(loaded, "load_plan", None), "report", None)
        report_path = getattr(report, "model_path", None)
        model_provenance.update(
            {
                "loaded_path": str(Path(str(report_path or load_path)).expanduser().resolve()),
                "file_name": str(getattr(report, "file_name", model_file.name) or model_file.name),
                "file_size_bytes": int(
                    getattr(report, "file_size_bytes", model_provenance["file_size_bytes"] or 0)
                    or 0
                ),
                "sha256": str(getattr(report, "sha256", "") or ""),
                "architecture": str(getattr(report, "architecture", "") or ""),
                "prediction_type": str(getattr(report, "prediction_type", "") or ""),
                "conditioning_dimension": getattr(report, "model_dimension", None),
                "architecture_summary": str(getattr(report, "architecture_summary", "") or ""),
                "architecture_source": str(
                    getattr(report, "architecture_source", getattr(report, "prediction_type_source", "")) or ""
                ),
                "architecture_contract": dict(
                    getattr(getattr(report, "architecture_contract", None), "to_dict", lambda: {})() or {}
                ),
                "checkpoint_kind": str(getattr(report, "checkpoint_kind", "") or ""),
            }
        )
        component_devices: dict[str, str] = {}
        component_dtypes: dict[str, str] = {}
        for component_name in ("unet", "text_encoder", "vae"):
            module = getattr(getattr(loaded, "components", None), component_name, None)
            if module is None:
                continue
            try:
                parameter = next(module.parameters())
                component_devices[component_name] = str(parameter.device)
                component_dtypes[component_name] = str(parameter.dtype)
            except (StopIteration, AttributeError, TypeError):
                component_devices[component_name] = str(getattr(module, "device", "unknown"))
                component_dtypes[component_name] = str(getattr(module, "dtype", "unknown"))
        canonical_vae = read_vae_provenance(loaded.components.vae)
        vae_provenance = {
            **canonical_vae,
            "mode": str(
                extras.get("vae_mode")
                or (
                    "manual_external_selection"
                    if extras.get("vae_path")
                    else "checkpoint_embedded_auto"
                )
            ),
            "selection_enabled": bool(extras.get("external_vae_override_enabled", True)),
            "requested_path": str(
                extras.get("vae_override_requested_path")
                or extras.get("vae_path")
                or ""
            ),
            "effective_source": str(canonical_vae.get("source_kind") or "runtime_component"),
            "effective_path": str(canonical_vae.get("source_path") or ""),
            "loaded_from_checkpoint": bool(
                canonical_vae.get("embedded_in_checkpoint", False)
            ),
            "component_device": component_devices.get("vae", "unknown"),
            "component_dtype": component_dtypes.get("vae", "unknown"),
        }
        model_provenance["component_devices"] = component_devices
        model_provenance["component_dtypes"] = component_dtypes
        attention_backend = dict(
            getattr(
                getattr(loaded, "built_components", None),
                "attention_backend_report",
                {},
            )
            or {}
        )
        if attention_backend:
            model_provenance["attention_backend"] = dict(attention_backend)
            extras["attention_backend"] = dict(attention_backend)
            session.request_extras["attention_backend"] = dict(attention_backend)
        model_provenance["vae_mode"] = vae_provenance["mode"]
        model_provenance["vae_provenance"] = dict(vae_provenance)
        extras["vae_provenance"] = dict(vae_provenance)
        session.request_extras["vae_provenance"] = dict(vae_provenance)
        model_provenance["execution_placement_reports"] = list(extras.get("execution_placement_reports") or [])
        model_provenance["checkpoint_hydration_time_ms"] = round(
            (time.perf_counter() - load_started) * 1000.0, 3
        )
        model_provenance["gpu_transfer_included_in_hydration"] = bool(runtime_device.type == "cuda")
        extras["model_provenance"] = dict(model_provenance)
        session.request_extras["model_provenance"] = dict(model_provenance)
        self._emit_model_runtime_event(
            extras,
            "model_ready",
            model_path=str(model_provenance.get("loaded_path") or model_file),
            cache_reused=bool(model_provenance.get("cache_reused")),
            checkpoint_hydration_time_ms=model_provenance["checkpoint_hydration_time_ms"],
            gpu_transfer_included=bool(runtime_device.type == "cuda"),
            execution_device=str(runtime_device),
            cuda_available=bool(torch.cuda.is_available()),
            cpu_fallback_reason=fallback_reason,
        )
        self.diagnostics_system.emit(
            session,
            "INFO",
            "model_loading",
            "checkpoint_loaded",
            "checkpoint components loaded",
            **model_provenance,
        )
        self.last_loaded_model = loaded
        checkpoint_family_hint = str(
            model_provenance.get("architecture")
            or model_provenance.get("architecture_summary")
            or model_provenance.get("checkpoint_kind")
            or ""
        )
        checkpoint_family = checkpoint_family_hint.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        try:
            resolved_lora_stack = self.lora_runtime_manager.prepare_request(
                request,
                extras,
                checkpoint_family=checkpoint_family,
            )
            self.lora_runtime_manager.apply(
                components=loaded.components,
                stack=resolved_lora_stack,
                extras=extras,
            )
        except Exception as exc:
            raise ValueError(f"LoRA request validation failed: {exc}") from exc
        session.request_extras["resolved_lora_stack"] = list(extras.get("resolved_lora_stack") or [])
        self.diagnostics_system.update_components(session, loaded.components)
        model_load_memory = dict(getattr(loaded, "memory_telemetry", {}) or {})
        if model_load_memory:
            extras["model_load_memory"] = model_load_memory
            session.request_extras["model_load_memory"] = model_load_memory
            self.diagnostics_system.emit(
                session,
                "INFO",
                "memory",
                "checkpoint_load_memory",
                "checkpoint component load memory telemetry captured",
                **model_load_memory,
            )
        pipeline = PipelineCompositionRoot(
            components=loaded.components,
            prompt_adapter=prompt_adapter,
            scheduler_adapter=scheduler_adapter,
            sampler_adapter=sampler_adapter,
            latent_scale_factor=self.latent_scale_factor,
            vae_scaling_factor=self.vae_scaling_factor,
            device=runtime_device,
            dtype=runtime_dtype,
            system_overrides=self.system_overrides,
        ).build(state=self.state)
        pixel_requested = bool(getattr(request, "hires_enabled", False))
        if pixel_requested:
            discovery = self.diagnostics_system.run_stage(
                session,
                "upscaler_discovery",
                "discover_for_pixel_hires",
                lambda: discover_upscalers(self.project_context, mode="unidentified"),
            )
            registry = UpscalerModelRegistry.from_discovery(
                discovery,
                memory_manager=pipeline.memory_manager,
            )
            pipeline.configure_neural_upscaling(
                registry=registry,
                runtime=StandaloneNeuralUpscaler(registry),
            )
            setattr(request, "_hires_upscaler_registry", registry)
            discovery_record = discovery.to_dict() if hasattr(discovery, "to_dict") else {}
            extras["hires_upscaler_discovery"] = discovery_record
            session.request_extras["hires_upscaler_discovery"] = discovery_record
        return pipeline

    def _generate_with_optional_shape_expansion(
        self,
        *,
        pipeline: Any,
        request: GenerationRequest,
        session: DiagnosticSession,
    ) -> GenerationResult:
        if not bool(getattr(request, "outpaint_shape_expansion_enabled", False)):
            return pipeline.generate(request, diagnostic_session=session)

        base_width = int(getattr(request, "outpaint_shape_base_width", 0) or request.width)
        base_height = int(getattr(request, "outpaint_shape_base_height", 0) or request.height)
        request.outpaint_shape_base_width = base_width
        request.outpaint_shape_base_height = base_height
        try:
            target = resolve_outpaint_shape_target(
                base_width=base_width,
                base_height=base_height,
                target_mode=str(getattr(request, "outpaint_shape_target_mode", "square") or "square"),
                target_width=int(getattr(request, "outpaint_shape_target_width", 0) or 0),
                target_height=int(getattr(request, "outpaint_shape_target_height", 0) or 0),
            )
        except Exception as exc:
            raise RuntimeError(format_outpaint_failure("outpaint_source_handoff", str(exc))) from exc

        base_request = replace(
            request,
            width=base_width,
            height=base_height,
            outpaint_prototype_enabled=False,
        )
        base_request.outpaint_shape_base_width = base_width
        base_request.outpaint_shape_base_height = base_height
        base_result = pipeline.generate(base_request, diagnostic_session=session)
        if not torch.is_tensor(base_result.images):
            raise RuntimeError(format_outpaint_failure(
                "outpaint_live_source_capture",
                "Fresh txt2img base generation did not return an image tensor for P-3 shape expansion.",
            ))
        if not torch.is_tensor(base_result.latents):
            raise RuntimeError(format_outpaint_failure(
                "outpaint_live_source_capture",
                "Fresh txt2img base generation did not return a sampled latent for P-3 shape expansion.",
            ))

        expansion_request = replace(
            base_request,
            width=int(target["target_width"]),
            height=int(target["target_height"]),
            outpaint_target_width=int(target["target_width"]),
            outpaint_target_height=int(target["target_height"]),
            outpaint_prototype_enabled=True,
            outpaint_source_image="",
            outpaint_anchor=str(getattr(request, "outpaint_shape_anchor", "center") or "center"),
            outpaint_source_x=-1,
            outpaint_source_y=-1,
            outpaint_context_seed_mode=str(
                getattr(request, "outpaint_shape_context_seed_mode", "edge_pad_v1") or "edge_pad_v1"
            ),
            outpaint_denoising_strength=float(
                getattr(request, "outpaint_shape_denoising_strength", 0.40) or 0.40
            ),
            # P-2 demonstrated that a provisional context seed is useful only
            # when the encoded provisional canvas participates in the new area.
            outpaint_latent_strategy="canvas_regional_noise_v1",
            outpaint_prompt_mode=str(
                getattr(request, "outpaint_shape_prompt_mode", "overlay_only_v1") or "overlay_only_v1"
            ),
            outpaint_overlay_positive_prompt=str(
                getattr(request, "outpaint_shape_overlay_positive_prompt", "") or ""
            ),
            outpaint_overlay_negative_prompt=str(
                getattr(request, "outpaint_shape_overlay_negative_prompt", "") or ""
            ),
        )
        expansion_request.outpaint_shape_expansion_enabled = True
        expansion_request.outpaint_shape_target_width = int(target["target_width"])
        expansion_request.outpaint_shape_target_height = int(target["target_height"])
        expansion_request.outpaint_shape_base_width = base_width
        expansion_request.outpaint_shape_base_height = base_height
        setattr(expansion_request, "_outpaint_runtime_source_tensor", base_result.images)
        setattr(expansion_request, "_outpaint_runtime_source_latent", base_result.latents)
        setattr(
            expansion_request,
            "_outpaint_runtime_source_handoff_requested",
            str(getattr(request, "outpaint_shape_source_handoff", "auto") or "auto"),
        )

        expanded_result = pipeline.generate(expansion_request, diagnostic_session=session)
        if bool(getattr(request, "outpaint_shape_save_base", False)):
            expanded_result.auxiliary_images["outpaint_pre_expansion_base"] = (
                base_result.images.detach().clone()
            )

        outpaint_record = dict(expanded_result.metadata.get("outpaint_prototype") or {})
        source_handoff = dict(outpaint_record.get("source_handoff") or {})
        source_handoff_contract = dict(outpaint_record.get("source_handoff_contract") or {})
        runtime_record = {
            "contract_version": OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION,
            "enabled": True,
            "source_kind": "fresh_txt2img_generation",
            "source_origin": str(source_handoff_contract.get("source_origin") or "fresh_generation"),
            "disk_round_trip": False,
            "base_generation_width": base_width,
            "base_generation_height": base_height,
            "base_latent_shape": list(base_result.latents.shape),
            "base_latent_dtype": str(base_result.latents.dtype),
            "base_latent_device_at_handoff": str(base_result.latents.device),
            "target_mode": str(target["target_mode"]),
            "target_width": int(target["target_width"]),
            "target_height": int(target["target_height"]),
            "anchor": str(expansion_request.outpaint_anchor),
            "context_seed_mode": str(expansion_request.outpaint_context_seed_mode),
            "source_handoff_requested": str(
                getattr(request, "outpaint_shape_source_handoff", "auto") or "auto"
            ),
            "source_handoff_requested_stable": normalize_outpaint_source_handoff_mode(
                getattr(request, "outpaint_shape_source_handoff", "auto") or "auto",
                default="auto",
            ),
            "source_handoff_actual": str(source_handoff.get("actual") or ""),
            "source_handoff_actual_stable": str(source_handoff_contract.get("actual_source_handoff") or ""),
            "source_handoff_fallback_reason": str(source_handoff.get("fallback_reason") or source_handoff_contract.get("source_handoff_fallback_reason") or ""),
            "latent_grid_alignment": dict(source_handoff_contract.get("latent_grid_alignment") or source_handoff.get("alignment") or {}),
            "preservation_reference_source": str(
                source_handoff_contract.get("preservation_reference_source") or source_handoff.get("preservation_reference_source") or ""
            ),
            "source_was_vae_reencoded_for_protected_latent": bool(
                source_handoff_contract.get("source_was_vae_reencoded_for_protected_latent", source_handoff.get("source_was_vae_reencoded_for_protected_latent", True))
            ),
            "live_source_latent_reused": bool(source_handoff_contract.get("live_source_latent_reused", source_handoff.get("live_source_latent_reused", False))),
            "outpaint_prompt_mode": str(expansion_request.outpaint_prompt_mode),
            "outpaint_overlay_positive_prompt": str(expansion_request.outpaint_overlay_positive_prompt),
            "outpaint_overlay_negative_prompt": str(expansion_request.outpaint_overlay_negative_prompt),
            "outpaint_denoising_strength": float(expansion_request.outpaint_denoising_strength),
            "provisional_base_saved": bool(getattr(request, "outpaint_shape_save_base", False)),
            "expanded_result_is_primary": True,
            "post_generation_shape_action": build_post_generation_shape_action(
                base_width=base_width,
                base_height=base_height,
                target_width=int(target["target_width"]),
                target_height=int(target["target_height"]),
                anchor=str(expansion_request.outpaint_anchor),
                context_seed_mode=str(expansion_request.outpaint_context_seed_mode),
                source_handoff_policy=str(getattr(request, "outpaint_shape_source_handoff", "auto") or "auto"),
                overlay_positive_prompt=str(expansion_request.outpaint_overlay_positive_prompt),
                overlay_negative_prompt=str(expansion_request.outpaint_overlay_negative_prompt),
                denoise_strength=float(expansion_request.outpaint_denoising_strength),
                save_pre_expansion_base=bool(getattr(request, "outpaint_shape_save_base", False)),
            ),
            "geometry_fingerprint": dict(outpaint_record.get("geometry_fingerprint") or {}),
            "inference_fingerprint": dict(outpaint_record.get("inference_fingerprint") or {}),
            "audit": dict(outpaint_record.get("audit") or {}),
        }
        runtime_record["runtime_handoff_tensors_released_after_expansion"] = True
        expansion_request.outpaint_shape_runtime_record = dict(runtime_record)
        for transient_name in (
            "_outpaint_runtime_source_tensor",
            "_outpaint_runtime_source_latent",
            "_outpaint_runtime_source_handoff_requested",
        ):
            if hasattr(expansion_request, transient_name):
                delattr(expansion_request, transient_name)
        # The prototype flag is an internal implementation detail of the second
        # in-job pass. Persist only the P-3 shape-expansion contract so replay
        # regenerates the base first instead of trying to load an uploaded source.
        expansion_request.outpaint_prototype_enabled = False
        expansion_request.outpaint_source_image = ""
        expanded_result.request = expansion_request
        expanded_result.metadata["outpaint_shape_expansion"] = dict(runtime_record)
        expanded_result.metadata["base_generation"] = {
            "width": base_width,
            "height": base_height,
            "latent_shape": list(base_result.latents.shape),
            "output_dimensions": dict(base_result.metadata.get("output_dimensions") or {}),
        }
        return expanded_result


    def run_request(
        self,
        request: GenerationRequest,
        extras: dict[str, Any] | None = None,
        *,
        save_images: bool | None = None,
        save_txt: bool = True,
        save_json: bool = True,
        save_diagnostics_json: bool = True,
        defer_output_save: bool = False,
    ) -> Txt2ImgRunResult:
        should_save = request.save_images if save_images is None else bool(save_images)
        _prepare_output_directory(
            self.project_context,
            request,
            should_save=should_save,
        )

        extras = dict(extras or {})
        effective_config_fn = getattr(self.project_context, "effective_config", None)
        effective_config = (
            effective_config_fn()
            if callable(effective_config_fn)
            else {
                "project_root": str(
                    getattr(self.project_context, "project_root", ".")
                )
            }
        )
        session = self.diagnostics_system.start(
            request,
            effective_config=effective_config,
            request_extras=extras,
        )
        try:
            model_path = extras.get("model_path") or self.model_loading_system.default_model_path
            self.diagnostics_system.run_stage(
                session,
                "configuration",
                "validate_generation_paths",
                lambda: self.project_context.require_generation_ready(
                    model_path=model_path,
                    output_dir=request.output_dir,
                    require_output=should_save,
                ),
            )
            request.device = str(self.device)
            self._configure_runtime_state(extras, session)
            request, extras = self.diagnostics_system.run_stage(
                session,
                "registry",
                "resolve_plugins",
                lambda: self.registry_system.apply_resolution(request, extras),
            )
            session.request_extras.update(
                {
                    key: value
                    for key, value in extras.items()
                    if key not in {
                        "progress_reporter",
                        "live_preview_callback",
                        "live_preview_warning_callback",
                        "live_preview_sink_factory",
                        "live_preview_sink",
                        "live_preview_frame_writer",
                        "live_preview_event_callback",
                        "live_preview_memory_event_callback",
                        "memory_event_callback",
                        "model_runtime_event_callback",
                    }
                }
            )
            pipeline = self.diagnostics_system.run_stage(
                session,
                "runtime",
                "compose_pipeline",
                lambda: self._build_pipeline(request, extras, session),
            )

            performance_matrix_enabled = os.environ.get(
                "IMAGE_GEN_PERF_MATRIX", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            phase_started_unix = time.time()
            if performance_matrix_enabled:
                print(
                    "PERFORMANCE_PHASE_JSON: "
                    + json.dumps(
                        {
                            "event": "generation_start",
                            "timestamp_unix": phase_started_unix,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            started = time.perf_counter()
            generation_time_sec = 0.0
            try:
                pipeline_result = self._generate_with_optional_shape_expansion(
                    pipeline=pipeline, request=request, session=session
                )
            finally:
                generation_time_sec = time.perf_counter() - started
                if performance_matrix_enabled:
                    print(
                        "PERFORMANCE_PHASE_JSON: "
                        + json.dumps(
                            {
                                "event": "generation_end",
                                "timestamp_unix": time.time(),
                                "elapsed_sec": generation_time_sec,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            request = pipeline_result.request or request
            pipeline_result.metadata["model_provenance"] = dict(
                extras.get("model_provenance") or {}
            )
            pipeline_result.metadata["vae_provenance"] = dict(
                extras.get("vae_provenance") or {}
            )

            manifest = self.diagnostics_system.run_stage(
                session,
                "output",
                "build_manifest",
                lambda: self.output_system.build_manifest(
                    request=request,
                    extras=extras,
                    pipeline_result=pipeline_result,
                    generation_time_sec=generation_time_sec,
                    device_name=str(request.device or self.device),
                ),
            )

            saved_records: list[SavedImageRecord] = []
            prepared_save_request: PreparedOutputSaveRequest | None = None
            expected_saved_count = 0
            if should_save:
                output_dir = request.output_dir or extras.get("output_dir")
                if not output_dir:
                    raise ValueError("An output directory is required when saving images.")
                pipeline.memory_manager.capture("before_output_save")
                if hasattr(manifest, "extra"):
                    manifest.extra["memory_management"] = pipeline.memory_manager.summary()
                if defer_output_save:
                    prepared_save_request = self.diagnostics_system.run_stage(
                        session,
                        "output",
                        "prepare_save_request",
                        lambda: self.output_system.prepare_save_request(
                            pipeline_result=pipeline_result,
                            request=request,
                            manifest=manifest,
                            output_dir=str(output_dir),
                            save_txt=save_txt,
                            save_json=save_json,
                            save_diagnostics_json=save_diagnostics_json,
                        ),
                    )
                    expected_saved_count = int(prepared_save_request.expected_count or 0)
                    pipeline_result.metadata["memory_management"] = pipeline.memory_manager.summary()
                else:
                    def save_and_verify_outputs() -> list[SavedImageRecord]:
                        records = self.output_system.save(
                            pipeline_result=pipeline_result,
                            request=request,
                            manifest=manifest,
                            output_dir=str(output_dir),
                            save_txt=save_txt,
                            save_json=save_json,
                            save_diagnostics_json=save_diagnostics_json,
                        )
                        return _verify_saved_records(records)

                    saved_records = self.diagnostics_system.run_stage(
                        session,
                        "output",
                        "save",
                        lambda: pipeline.memory_manager.observe_stage(
                            "output_save",
                            save_and_verify_outputs,
                        ),
                    )
                    expected_saved_count = len(saved_records)
                    pipeline_result.metadata["memory_management"] = pipeline.memory_manager.summary()

            diagnostics_summary = self.diagnostics_system.complete(
                session, result=pipeline_result
            )
            pipeline_result.metadata["diagnostics"] = diagnostics_summary
            return Txt2ImgRunResult(
                request=request,
                request_extras=extras,
                pipeline_result=pipeline_result,
                manifest=manifest,
                saved_records=saved_records,
                generation_time_sec=generation_time_sec,
                run_id=session.run_id,
                diagnostics=diagnostics_summary,
                prepared_save_request=prepared_save_request,
                expected_saved_count=expected_saved_count,
            )
        except PipelineStageError:
            raise
        except Exception as exc:
            raise self.diagnostics_system.fail_unassigned(
                session, exc, system="runtime", operation="run_request"
            ) from exc

    def run_from_sources(
        self,
        *,
        config_path: str | None = None,
        manifest_path: str | None = None,
        infotext_path: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
        base_payload: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
        save_txt: bool = True,
        save_json: bool = True,
        save_diagnostics_json: bool = True,
    ) -> Txt2ImgRunResult:
        payload = load_request_payload(
            config_path=config_path,
            manifest_path=manifest_path,
            infotext_path=infotext_path,
            cli_overrides=cli_overrides,
            base_payload=base_payload,
        )
        request, payload_extras = payload_to_generation_request(payload)
        merged_extras = dict(extras or {})
        merged_extras.update(payload_extras)
        return self.run_request(
            request,
            merged_extras,
            save_txt=save_txt,
            save_json=save_json,
            save_diagnostics_json=save_diagnostics_json,
        )
