from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Callable

import torch

from image_gen.contracts import GenerationRequest
from image_gen.contracts.vae_provenance import read_vae_provenance
from image_gen.runtime.composition import PipelineCompositionRoot
from image_gen.runtime.component_residency import public_transition_report
from image_gen.runtime.model_preflight import _advanced_model_family
from image_gen.runtime.model_load_variant import (
    model_load_variant_fingerprint,
    model_load_variant_payload,
    model_load_variant_payload_fingerprint,
    resolved_model_load_variant_payload,
    model_load_variant_comparison,
)
from image_gen.systems.diagnostics import DiagnosticSession
from image_gen.systems.memory.telemetry import normalize_cuda_memory_payload
from image_gen.systems.upscaling import StandaloneNeuralUpscaler, UpscalerModelRegistry, discover_upscalers
from modules.checkpoint_inspector import CheckpointInspector
from modules.pipeline.progress_reporter import ProgressReporter
from modules.sd2_runtime_assets import SD2RuntimeAssetResolver
from modules.sdxl_runtime_assets import SDXLRuntimeAssetResolver


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


class PipelineFactoryMixin:
    @staticmethod
    def _emit_model_runtime_event(extras: dict[str, Any], stage: str, **payload: Any) -> None:
        callback = extras.get("model_runtime_event_callback")
        if not callable(callback):
            return
        try:
            callback({"stage": str(stage), **payload})
        except Exception:
            return

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
            first_step_callback=extras.get("model_runtime_first_step_callback"),
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
                "text_encoder_3_device",
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

    def _build_pipeline(
        self,
        request: GenerationRequest,
        extras: dict[str, Any],
        session: DiagnosticSession,
    ):
        trace_enabled = bool(extras.get("model_runtime_trace_enabled"))
        trace_started = time.perf_counter()
        pipeline_trace: dict[str, Any] = {
            "schema_version": 1,
            "kind": "pipeline_build",
            "stages": [],
        }

        def record_trace(name: str, started: float, **details: Any) -> None:
            if not trace_enabled:
                return
            item = {
                "name": str(name),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            if details:
                item.update(details)
            pipeline_trace["stages"].append(item)

        # Programmatic callers may compose directly. Normal run_request performs
        # this before registry resolution so profile-selected adapters cannot be stale.
        stage_started = time.perf_counter()
        self._apply_sdxl_runtime_preflight(request, extras)
        record_trace("sdxl_runtime_preflight", stage_started)
        stage_started = time.perf_counter()
        self._apply_sd3_runtime_preflight(request, extras)
        record_trace(
            "sd3_runtime_preflight",
            stage_started,
            resolved_profile=str(extras.get("sd3_runtime_profile_override") or ""),
        )
        stage_started = time.perf_counter()
        prompt_adapter, scheduler_adapter, sampler_adapter = self._resolve_adapters(request, extras)
        record_trace("resolve_adapters", stage_started)
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
        stage_started = time.perf_counter()
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
        record_trace(
            "resolve_execution_device",
            stage_started,
            execution_device=str(runtime_device),
            memory_policy=memory_policy,
        )
        model_path = extras.get("model_path") or self.model_loading_system.default_model_path
        if not model_path:
            raise ValueError("No model_path provided in request extras and no default MODEL_PATH found.")

        stage_started = time.perf_counter()
        model_file = Path(str(model_path)).expanduser().resolve()
        checkpoint_family = ""
        advanced_family = _advanced_model_family(extras)
        if advanced_family:
            checkpoint_family = advanced_family
            extras["checkpoint_preflight_architecture"] = {
                "family": advanced_family,
                "source": "advanced_model_registry_resolution",
                "checkpoint_reinspection_skipped": True,
            }
        elif model_file.is_file() and model_file.suffix.lower() == ".safetensors":
            try:
                checkpoint_contract = CheckpointInspector().inspect_architecture_contract(str(model_file))
                checkpoint_family = str(checkpoint_contract.family or "")
                extras["checkpoint_preflight_architecture"] = checkpoint_contract.to_dict()
            except Exception as exc:
                extras["checkpoint_preflight_architecture"] = {
                    "family": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        record_trace(
            "checkpoint_architecture_inspection",
            stage_started,
            checkpoint_family=checkpoint_family,
        )
        stage_started = time.perf_counter()
        try:
            resolved_lora_stack = self.lora_runtime_manager.prepare_request(
                request,
                extras,
                checkpoint_family=checkpoint_family,
            )
        except Exception as exc:
            details = dict(extras.get("adapter_preflight") or {})
            details.setdefault("checkpoint_family", checkpoint_family)
            details["error"] = str(exc)
            extras["adapter_preflight"] = details
            session.request_extras["adapter_preflight"] = dict(details)
            raise ValueError(f"LoRA request validation failed: {exc}") from exc
        session.request_extras["adapter_preflight"] = dict(extras.get("adapter_preflight") or {})
        session.request_extras["adapter_runtime_plans"] = list(extras.get("adapter_runtime_plans") or [])
        record_trace("lora_prepare_request", stage_started)

        stage_started = time.perf_counter()
        expected_tokenizer_identity = ""
        preflight_family = checkpoint_family
        if "SD 2" in preflight_family or preflight_family.lower().startswith("sd2"):
            profile = self._resolve_sd2_profile_hint(model_file)
            if profile is not None:
                assets = SD2RuntimeAssetResolver(self.project_context).resolve(profile)
                expected_tokenizer_identity = f"{profile.profile_id}:{assets.tokenizer_dir.resolve()}"
        elif preflight_family.strip().lower() == "sdxl":
            assets = SDXLRuntimeAssetResolver(self.project_context).resolve()
            expected_tokenizer_identity = f"sdxl:{assets.tokenizer_dir.resolve()}"
        elif preflight_family.strip().lower() == "sd3.x":
            # SD3's ComponentBuilder owns its paired local tokenizers. Avoid
            # loading the unrelated legacy SD1 tokenizer before checkpoint hydration.
            expected_tokenizer_identity = "sd3:component_builder"
        else:
            expected_tokenizer_identity = f"sd1:{self.project_context.tokenizer_root.resolve()}"

        if preflight_family.strip().lower() == "sd3.x":
            self.tokenizer = None
            self._tokenizer_identity = expected_tokenizer_identity
        elif self.tokenizer is None or (
            self._tokenizer_identity not in {"injected", expected_tokenizer_identity}
        ):
            self._emit_model_runtime_event(extras, "loading_tokenizer")
            tokenizer_result = self.diagnostics_system.run_stage(
                session,
                "model_loading",
                "load_tokenizer",
                lambda: self._build_local_tokenizer(
                    checkpoint_path=model_file,
                    checkpoint_family=checkpoint_family,
                ),
            )
            self.tokenizer, self._tokenizer_identity = tokenizer_result
        record_trace(
            "tokenizer_resolution",
            stage_started,
            tokenizer_identity=str(self._tokenizer_identity or ""),
        )
        stage_started = time.perf_counter()
        model_provenance: dict[str, Any] = {
            "requested_path": str(model_path),
            "resolved_path": str(model_file),
            "loaded_path": "",
            "file_name": model_file.name,
            "model_name": model_file.stem,
            "model_name_source": "filename",
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
        vae_override_path = str(extras.get("vae_path") or "").strip()
        vae_override_identity: tuple[Any, ...] = ("", -1, -1)
        if vae_override_path:
            resolved_vae = Path(vae_override_path).expanduser()
            if resolved_vae.is_file():
                vae_stat = resolved_vae.stat()
                vae_override_identity = (str(resolved_vae.resolve()), int(vae_stat.st_size), int(vae_stat.st_mtime_ns))
            else:
                vae_override_identity = (vae_override_path, -1, -1)
        # Keep cache identity and resident identity on the same canonical
        # composition contract. Device placement remains a separate cache
        # dimension because it can change hydration/placement without changing
        # which components logically make up the model.
        load_variant_key = (
            model_load_variant_fingerprint(extras),
            str(extras.get("text_encoder_3_device") or ""),
            *vae_override_identity,
        )
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
                *load_variant_key,
            )
        else:
            # Injected/fake loaders used by focused tests may use a symbolic path.
            load_path = str(model_path)
            cache_key = (str(model_file), str(runtime_dtype), str(runtime_device), -1, -1, *load_variant_key)
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
        cnrr07_prepared_reuse = False
        if loaded is None:
            prepared_context = dict(getattr(self, "_cnrr07_prepared_transition_context", {}) or {})
            prepared_loaded = getattr(self, "last_loaded_model", None)
            prepared_report = getattr(getattr(prepared_loaded, "load_plan", None), "report", None)
            prepared_path = str(getattr(prepared_report, "model_path", "") or prepared_context.get("model_path") or "")
            requested_path_key = str(model_file.resolve(strict=False)).casefold()
            prepared_path_key = str(Path(prepared_path).expanduser().resolve(strict=False)).casefold() if prepared_path else ""
            source_signature = dict(prepared_context.get("source_signature") or {})
            source_signature_matches = True
            if source_signature and model_file.is_file():
                stat_now = model_file.stat()
                source_signature_matches = bool(
                    int(source_signature.get("file_size_bytes") or -1) == int(stat_now.st_size)
                    and int(source_signature.get("modified_ns") or -1) == int(stat_now.st_mtime_ns)
                )
            prepared_components = getattr(prepared_loaded, "components", None)
            prepared_variant = model_load_variant_comparison(
                extras,
                {
                    "runtime_load_variant": dict(getattr(prepared_components, "runtime_load_variant", {}) or {}),
                    "runtime_load_variant_fingerprint": str(getattr(prepared_components, "runtime_load_variant_fingerprint", "") or ""),
                    "runtime_effective_load_variant": dict(getattr(prepared_components, "runtime_effective_load_variant", {}) or {}),
                    "runtime_effective_load_variant_fingerprint": str(getattr(prepared_components, "runtime_effective_load_variant_fingerprint", "") or ""),
                },
            ) if prepared_loaded is not None else {"matches": False}
            prepared_variant_matches = bool(prepared_variant.get("matches"))
            if (
                prepared_loaded is not None
                and prepared_path_key
                and prepared_path_key == requested_path_key
                and prepared_variant_matches
                and source_signature_matches
            ):
                loaded = prepared_loaded
                cnrr07_prepared_reuse = True
                self._loaded_model_cache.clear()
                self._loaded_model_cache[cache_key] = loaded
                model_provenance["cache_reused"] = True
                model_provenance["cnrr07_prepared_transition_reused"] = True
        record_trace(
            "checkpoint_cache_lookup",
            stage_started,
            cache_hit=bool(loaded is not None),
            cnrr07_prepared_transition_reused=cnrr07_prepared_reuse,
            load_variant_fingerprint=str(load_variant_key[0]),
        )
        load_started = time.perf_counter()
        stage_started = load_started
        if loaded is None:
            self._emit_model_runtime_event(
                extras,
                "loading_checkpoint",
                model_path=str(model_file),
                target_device=str(runtime_device),
                target_dtype=str(runtime_dtype),
            )
            if bool(extras.get("_component_transition_requested")) and self.last_loaded_model is not None:
                lease_bundle = getattr(self, "execution_lease_reuse_bundle", None)
                extras["_resident_component_reuse_bundle"] = (
                    lease_bundle() if callable(lease_bundle) else self.resident_component_reuse_bundle()
                )
            else:
                extras.pop("_resident_component_reuse_bundle", None)
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
            if (
                "request_extras" in load_parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in load_parameters.values()
                )
            ):
                load_kwargs["request_extras"] = extras
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
            # A conventional load supersedes any prior CNRR-07 prepared-commit
            # reuse marker.  Keeping this context beyond its exact target request
            # would make a later unrelated cache miss eligible for stale reuse.
            self._cnrr07_prepared_transition_context = {}
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
                settings=extras,
            )
        record_trace(
            "checkpoint_load_or_cached_placement",
            stage_started,
            cache_reused=bool(model_provenance.get("cache_reused")),
        )
        components = getattr(loaded, "components", None)
        report = getattr(getattr(loaded, "load_plan", None), "report", None)
        if components is not None:
            raw_load_variant = model_load_variant_payload(extras)
            setattr(components, "runtime_load_variant", raw_load_variant)
            setattr(components, "runtime_load_variant_fingerprint", model_load_variant_fingerprint(extras))

            load_plan = getattr(loaded, "load_plan", None)
            profile_ids: dict[str, str] = {}
            profile_contract_fields = (
                ("sd2_runtime_profile_override", getattr(load_plan, "sd2_contract", None)),
                ("sdxl_runtime_profile_override", getattr(load_plan, "sdxl_contract", None)),
                ("sd3_runtime_profile_override", getattr(load_plan, "sd3_contract", None)),
            )
            for field, contract in profile_contract_fields:
                profile = getattr(contract, "profile", None)
                profile_id = str(getattr(profile, "profile_id", "") or "").strip()
                if profile_id:
                    profile_ids[field] = profile_id

            runtime_profile = dict(getattr(components, "model_runtime_profile", {}) or {})
            sd3_text_encoder_sources = dict(runtime_profile.get("text_encoder_sources") or {})
            effective_load_variant = resolved_model_load_variant_payload(
                extras,
                profile_ids=profile_ids,
                sd3_text_encoder_sources=sd3_text_encoder_sources,
            )
            effective_load_variant_fingerprint = model_load_variant_payload_fingerprint(
                effective_load_variant
            )
            setattr(components, "runtime_effective_load_variant", effective_load_variant)
            setattr(
                components,
                "runtime_effective_load_variant_fingerprint",
                effective_load_variant_fingerprint,
            )

            checkpoint_identity = {
                "path": str(Path(str(getattr(report, "model_path", "") or model_file)).expanduser().resolve(strict=False)),
                "sha256": str(getattr(report, "sha256", "") or "").strip().lower(),
                "file_size_bytes": model_provenance.get("file_size_bytes"),
                "modified_ns": model_provenance.get("modified_ns"),
                "proof": (
                    "resident_sha256_bound_to_source_file_signature"
                    if str(getattr(report, "sha256", "") or "").strip()
                    else "source_file_signature"
                ),
            }
            setattr(components, "runtime_checkpoint_identity", checkpoint_identity)
        report_path = getattr(report, "model_path", None)
        model_provenance.update(
            {
                "loaded_path": str(Path(str(report_path or load_path)).expanduser().resolve()),
                "file_name": str(getattr(report, "file_name", model_file.name) or model_file.name),
                "model_name": str(getattr(report, "model_name", model_file.stem) or model_file.stem),
                "model_name_source": str(getattr(report, "model_name_source", "filename") or "filename"),
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
        for component_name in ("unet", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
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
        model_provenance["model_runtime_profile"] = dict(
            getattr(loaded.components, "model_runtime_profile", {}) or {}
        )
        model_provenance["runtime_checkpoint_identity"] = dict(
            getattr(loaded.components, "runtime_checkpoint_identity", {}) or {}
        )
        model_provenance["runtime_effective_load_variant"] = dict(
            getattr(loaded.components, "runtime_effective_load_variant", {}) or {}
        )
        model_provenance["runtime_effective_load_variant_fingerprint"] = str(
            getattr(loaded.components, "runtime_effective_load_variant_fingerprint", "") or ""
        )
        model_provenance["composition_sha256"] = str(
            getattr(loaded.components, "composition_sha256", "") or ""
        )
        model_provenance["composition_identity_version"] = str(
            getattr(loaded.components, "composition_identity_version", "") or ""
        )
        model_provenance["composition_contract"] = dict(
            getattr(loaded.components, "composition_contract", {}) or {}
        )
        model_provenance["component_sources"] = {
            str(role): dict(source)
            for role, source in dict(getattr(loaded.components, "component_sources", {}) or {}).items()
        }
        model_provenance["composition_projection"] = dict(
            getattr(loaded.components, "composition_projection", {}) or {}
        )
        model_provenance["advanced_model_composition_sha256"] = str(
            getattr(loaded.components, "advanced_model_composition_sha256", "") or ""
        )
        model_provenance["component_transition_report"] = public_transition_report(
            getattr(loaded.components, "component_transition_report", {}) or {}
        )
        model_provenance["latent_vae_contract"] = {
            "latent_channels": int(getattr(loaded.components, "latent_channels", 4) or 4),
            "latent_scale_factor": int(getattr(loaded.components, "latent_scale_factor", 8) or 8),
            "scaling_factor": float(getattr(loaded.components, "vae_scaling_factor", 0.18215)),
            "shift_factor": float(getattr(loaded.components, "vae_shift_factor", 0.0) or 0.0),
            "force_upcast": bool(getattr(loaded.components, "vae_force_upcast", False)),
            "use_quant_conv": getattr(loaded.components, "vae_use_quant_conv", None),
            "use_post_quant_conv": getattr(loaded.components, "vae_use_post_quant_conv", None),
        }
        model_provenance["vae_scaling_factor"] = float(
            getattr(loaded.components, "vae_scaling_factor", 0.18215)
        )
        model_provenance["vae_shift_factor"] = float(
            getattr(loaded.components, "vae_shift_factor", 0.0) or 0.0
        )
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
        lease_commit = getattr(self, "on_composition_committed_to_lease", None)
        if callable(lease_commit):
            try:
                extras["composition_execution_lease_commit"] = lease_commit(
                    str(model_provenance.get("loaded_path") or model_file)
                )
            except Exception as exc:
                extras["composition_execution_lease_commit"] = {
                    "updated": False,
                    "reason": "lease_commit_diagnostics_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        stage_started = time.perf_counter()
        try:
            self.lora_runtime_manager.apply(
                components=loaded.components,
                stack=resolved_lora_stack,
                extras=extras,
            )
        except Exception as exc:
            raise ValueError(f"LoRA runtime application failed: {exc}") from exc
        record_trace("lora_runtime_apply", stage_started)
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
        stage_started = time.perf_counter()
        pipeline = PipelineCompositionRoot(
            components=loaded.components,
            prompt_adapter=prompt_adapter,
            scheduler_adapter=scheduler_adapter,
            sampler_adapter=sampler_adapter,
            latent_scale_factor=self.latent_scale_factor,
            vae_scaling_factor=(
                self.vae_scaling_factor
                if self.vae_scaling_factor is not None
                else float(getattr(loaded.components, "vae_scaling_factor", 0.18215))
            ),
            device=runtime_device,
            dtype=runtime_dtype,
            system_overrides=self.system_overrides,
        ).build(state=self.state)
        record_trace("pipeline_composition_root_build", stage_started)
        pixel_requested = bool(getattr(request, "hires_enabled", False))
        if pixel_requested:
            stage_started = time.perf_counter()
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
            record_trace("hires_upscaler_discovery_and_configure", stage_started)
        if trace_enabled:
            pipeline_trace["total_ms"] = round((time.perf_counter() - trace_started) * 1000.0, 3)
            extras["pipeline_build_trace"] = pipeline_trace
            session.request_extras["pipeline_build_trace"] = dict(pipeline_trace)
        return pipeline
