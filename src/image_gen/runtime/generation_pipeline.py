from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
import inspect
import time
from typing import Any

import torch

from modules.attention_backend import attention_backend_report, temporary_attention_slicing
from modules.attention_runtime import get_execution_evidence

from image_gen.contracts import (
    ConditioningOutput,
    GenerationRequest,
    GenerationResult,
    PipelineComponents,
    PromptAdapterProtocol,
    SamplerAdapterProtocol,
    SchedulerAdapterProtocol,
    SchedulerOutput,
)
from image_gen.contracts.vae_provenance import read_vae_provenance
from image_gen.runtime.composition import GenerationSystems, PipelineCompositionRoot
from image_gen.runtime.performance_metrics import GenerationPerformanceRecorder
from image_gen.runtime.hires_fix import (
    add_hires_noise,
    build_hires_request,
    PixelNeuralHiresSourceResult,
    hires_schedule_baseline_metadata,
    resolve_hires_execution_plan,
    validate_recorded_hires_vae_identity,
)
from image_gen.systems.image_conditioning import (
    build_image_conditioned_schedule,
    build_vae_execution_fingerprint,
    build_schedule_fingerprint_record,
    build_schedule_replay_record,
    compare_schedule_conformance,
    noise_stream_metadata,
    rehydrate_schedule_replay_record,
    require_qualified_hires_pair,
    vae_encode_for_sampling,
    vae_round_trip_from_encoded_for_diagnostics,
)
from image_gen.runtime_options import (
    build_runtime_execution_record,
    compare_runtime_execution_records,
    runtime_execution_fingerprint,
)
from image_gen.systems.decoding import DecodingSystem
from image_gen.systems.sampling import SamplingSystem
from image_gen.systems.scheduling import SchedulingSystem
from image_gen.systems.upscaling import StandaloneNeuralUpscaler, UpscaleRequest
from image_gen.systems.diagnostics import DiagnosticSession, DiagnosticsSystem, PipelineStageError
from image_gen.systems.diagnostics.output_quality import (
    classify_normalized_images,
    summarize_tensor,
    write_output_quality_bundle,
)
from image_gen.systems.memory import (
    AdaptiveComponentMemoryManager,
    StageRecoveryContract,
    PixelHiresAdmissionError,
    PixelHiresCancelled,
    cancellation_requested,
    compare_preflight_to_actual,
    estimate_pixel_hires_preflight,
    perform_pre_hires_cleanup,
    raise_if_pixel_hires_cancelled,
    resolve_hires_memory_behavior,
    resolve_preview_stage_policy,
    stage_tensor_to_host,
)
from modules.pipeline.conditioning_utils import resolve_step_conditioning as resolve_step_conditioning_util
from modules.pipeline.live_preview import build_live_preview_sink
from modules.pipeline.live_preview_decode import create_live_preview_writer




def _machine_preview_transport_enabled(values: dict[str, Any] | None) -> bool:
    """Return whether preview telemetry should be printed to the console.

    Human CLI runs keep preview generation available without writing the
    machine-readable STEP_PREVIEW_JSON transport into the progress display.
    WebUI workers opt in through ``progress_json``; verbose diagnostics may
    also request the transport explicitly.
    """

    extra = dict(values or {})
    return bool(
        extra.get("progress_json", False)
        or extra.get("_console_verbose", False)
        or extra.get("_console_preview_json", False)
    )



def _execution_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        "successful_call_delta": max(
            0,
            int(after.get("successful_call_count", 0) or 0)
            - int(before.get("successful_call_count", 0) or 0),
        ),
        "failed_call_delta": max(
            0,
            int(after.get("failed_call_count", 0) or 0)
            - int(before.get("failed_call_count", 0) or 0),
        ),
    }


class GenerationPipeline:
    """Ordered, file-system-free composition of the generation systems."""

    def _sampling_recovery_contract(
        self,
        *,
        stage: str,
        source_latents: torch.Tensor,
        request: GenerationRequest,
        operation_builder: Any,
    ) -> StageRecoveryContract | None:
        """Create a pristine CPU latent boundary for one bounded sampler retry."""

        if not self.memory_manager.oom.enabled:
            return None

        holder: dict[str, Any] = {
            "latents": source_latents.detach().to(device="cpu", copy=True),
        }
        source_device = source_latents.device
        source_dtype = source_latents.dtype
        source_shape = list(source_latents.shape)

        sampler_kwargs = getattr(request, "sampler_kwargs", {}) or {}
        trace_recorder = sampler_kwargs.get("trace_recorder")
        trace_checkpoint = None
        checkpoint_method = getattr(trace_recorder, "recovery_checkpoint", None)
        if callable(checkpoint_method):
            trace_checkpoint = checkpoint_method()

        def operation_factory(attempt_index: int):
            boundary_latents = holder.get("latents")
            if boundary_latents is None:
                raise RuntimeError(
                    f"Sampling recovery boundary {stage!r} was released before attempt {attempt_index}."
                )
            attempt_latents = boundary_latents.to(
                device=source_device,
                dtype=source_dtype,
                copy=True,
            )
            return lambda: operation_builder(attempt_latents)

        def prepare_retry(profile: str, retry_index: int) -> dict[str, Any]:
            details: dict[str, Any] = {
                "profile": str(profile),
                "retry_index": int(retry_index),
                "source_shape": source_shape,
                "source_device": str(source_device),
                "source_dtype": str(source_dtype),
                "rng_recreated_from_resolved_seeds": True,
            }
            restore_method = getattr(
                trace_recorder, "restore_recovery_checkpoint", None
            )
            if callable(restore_method) and trace_checkpoint is not None:
                details["trace_restore"] = dict(
                    restore_method(trace_checkpoint) or {}
                )

            state_sampler = getattr(self.state, "samp", None)
            if state_sampler is not None:
                for attribute in ("samples", "sampler_fn"):
                    if hasattr(state_sampler, attribute):
                        setattr(state_sampler, attribute, None)
                details["shared_sampler_state_cleared"] = True

            state_extra = getattr(self.state, "extra", None)
            writer = (
                state_extra.get("live_preview_frame_writer")
                if isinstance(state_extra, dict)
                else None
            )
            drain = getattr(writer, "drain", None)
            if callable(drain):
                drain()
                details["preview_queue_drained"] = True
            release_history = getattr(writer, "release_nonfinal_history", None)
            if callable(release_history):
                details["preview_history_removed"] = int(
                    release_history() or 0
                )
            return details

        def release_boundary() -> None:
            holder["latents"] = None

        return StageRecoveryContract(
            boundary_id=f"{stage}:pristine_initial_latents",
            restart_mode="same_stage_from_pristine_cpu_latents",
            operation_factory=operation_factory,
            prepare_retry=prepare_retry,
            release_boundary=release_boundary,
            metadata={
                "sampler_state_restart": "new_sampler_invocation",
                "partial_sampler_state_reused": False,
                "source_shape": source_shape,
                "source_dtype": str(source_dtype),
            },
        )

    def _configure_live_preview_sink(
        self,
        request: GenerationRequest,
        schedule: SchedulerOutput,
        diagnostics: DiagnosticsSystem | None,
        session: Any | None,
    ) -> Any | None:
        if self.state is None:
            return None

        extra = getattr(self.state, "extra", None)
        if not isinstance(extra, dict):
            return None

        def _warning_callback(payload: dict[str, Any]) -> None:
            if diagnostics is not None and session is not None:
                diagnostics.emit(
                    session,
                    "WARNING",
                    "live_preview",
                    str(payload.get("operation", "callback")),
                    str(payload.get("message", "Live preview sink failure was isolated and generation continued.")),
                    step_index=payload.get("step_index"),
                    error_type=payload.get("error_type"),
                    error=payload.get("error"),
                    failure_count=payload.get("failure_count"),
                    disabled=payload.get("disabled"),
                    metadata=payload.get("metadata"),
                )

        writer = extra.get("live_preview_frame_writer")
        if writer is None and not callable(extra.get("live_preview_callback")):
            if _machine_preview_transport_enabled(extra):
                def _event_callback(payload: dict[str, Any]) -> None:
                    import json as _json
                    import sys as _sys

                    message = _json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    _sys.stdout.write(f"STEP_PREVIEW_JSON: {message}\n")
                    _sys.stdout.flush()

                extra.setdefault("live_preview_event_callback", _event_callback)

            extra.setdefault("live_preview_warning_callback", _warning_callback)
            extra.setdefault("live_preview_memory_event_callback", self.memory_manager.capture)
            extra.setdefault("live_preview_async", True)
            writer = create_live_preview_writer(
                extra,
                decoder=self.systems.decoding.decode,
            )
            if writer is not None:
                extra["live_preview_frame_writer"] = writer
                extra["live_preview_callback"] = writer
                extra.setdefault("live_preview_clone_tensors", False)

        preview_policy = str(self.memory_manager.settings.preview_policy or "normal")
        extra["live_preview_policy"] = preview_policy
        if writer is not None and preview_policy == "disabled":
            self.memory_manager.suspend_preview_image_decode(
                "Preview policy disabled image decoding for this job.",
                source="policy_disabled",
            )

        sink = build_live_preview_sink(
            state=self.state,
            request=request,
            schedule=schedule,
            warning_callback=_warning_callback,
        )
        writer_settings = getattr(writer, "settings", None)
        if writer_settings is None:
            writer_settings = getattr(getattr(writer, "writer", None), "settings", None)
        if (
            writer_settings is not None
            and getattr(writer_settings, "preview_policy", "normal") != "disabled"
            and writer_settings.performance_warning
        ):
            diagnostics.emit(
                session,
                "WARNING",
                "live_preview",
                "accurate_mode_performance",
                writer_settings.performance_warning,
                preview_mode=writer_settings.mode,
                preview_interval=writer_settings.interval,
                preview_width=writer_settings.width,
            )
        return sink

    MODEL_PREDICTION_TYPE = "epsilon"

    def __init__(
        self,
        *,
        components: PipelineComponents,
        systems: GenerationSystems,
        state: Any | None = None,
        device: torch.device,
        dtype: torch.dtype,
        latent_scale_factor: int = 8,
        vae_scaling_factor: float = 0.18215,
        memory_manager: AdaptiveComponentMemoryManager | None = None,
    ) -> None:
        self.components = components
        self.systems = systems
        self.state = state
        self.device = device
        self.dtype = dtype
        self.latent_scale_factor = int(latent_scale_factor)
        self.vae_scaling_factor = float(vae_scaling_factor)
        self.memory_manager = memory_manager or AdaptiveComponentMemoryManager.from_state(
            target_device=self.device,
            state=self.state,
        )
        if not self.memory_manager.registry.components:
            self.memory_manager.register_core_components(self.components)

        # Temporary compatibility attributes while historical callers migrate.
        self.prompt_adapter = getattr(systems.conditioning, "adapter", None)
        self.scheduler_adapter = getattr(systems.scheduling, "adapter", None)
        self.sampler_adapter = getattr(systems.sampling, "adapter", None)
        state_extra = getattr(self.state, "extra", None)
        self.neural_upscaler_registry = (
            state_extra.get("hires_upscaler_registry")
            if isinstance(state_extra, dict)
            else None
        )
        self.neural_upscaler = (
            state_extra.get("standalone_neural_upscaler")
            if isinstance(state_extra, dict)
            else None
        )

    def configure_neural_upscaling(
        self,
        *,
        registry: Any,
        runtime: StandaloneNeuralUpscaler | None = None,
    ) -> None:
        self.neural_upscaler_registry = registry
        self.neural_upscaler = runtime or StandaloneNeuralUpscaler(registry)
        if self.state is not None and isinstance(getattr(self.state, "extra", None), dict):
            self.state.extra["hires_upscaler_registry"] = registry
            self.state.extra["standalone_neural_upscaler"] = self.neural_upscaler

    def prepare_latents(
        self,
        request: GenerationRequest,
        schedule: SchedulerOutput | None = None,
    ) -> torch.Tensor:
        if schedule is None:
            schedule = self.systems.scheduling.build(request, self.state)
        return self.systems.latent_preparation.prepare(request, schedule)

    def predict_raw_noise(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.systems.denoising.predict_raw_noise(*args, **kwargs)

    def predict_guided_noise(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.systems.denoising.predict_guided_noise(*args, **kwargs)

    def predict_denoised(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.systems.denoising.predict_denoised(*args, **kwargs)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.systems.decoding.decode(latents)

    def resolve_step_conditioning(
        self,
        conditioning: ConditioningOutput,
        step_index: int,
        latents: torch.Tensor | None = None,
        state: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return resolve_step_conditioning_util(
            conditioning=conditioning,
            step_index=step_index,
            latents=latents,
            state=state or self.state,
        )

    @staticmethod
    def _validate_result(request: GenerationRequest, result: GenerationResult) -> GenerationResult:
        if result.latents is None or not torch.is_tensor(result.latents):
            raise TypeError("GenerationResult.latents must be a tensor.")
        if not torch.isfinite(result.latents).all():
            raise ValueError("GenerationResult.latents contains non-finite values.")
        if not request.return_latents:
            if result.images is None or not torch.is_tensor(result.images):
                raise TypeError("GenerationResult.images must be a tensor unless return_latents is enabled.")
            if bool(getattr(request, "hires_enabled", False)):
                hires_plan = dict(getattr(request, "hires_dimension_plan", {}) or {})
                expected_height = int(hires_plan.get("effective_height") or request.height)
                expected_width = int(hires_plan.get("effective_width") or request.width)
            else:
                expected_height = int(request.height)
                expected_width = int(request.width)
            expected = (request.batch_size, expected_height, expected_width)
            actual = (result.images.shape[0], result.images.shape[-2], result.images.shape[-1])
            if actual != expected:
                raise ValueError(
                    f"Decoded image dimensions must be {expected}, got {actual}."
                )
            if not torch.isfinite(result.images).all():
                raise ValueError("GenerationResult.images contains non-finite values.")
        return result

    def _invoke_neural_upscaler(
        self,
        request: UpscaleRequest,
        *,
        cancellation_check: Any,
    ) -> Any:
        """Call the owned runtime without breaking compatible test/system overrides."""

        method = self.neural_upscaler.upscale
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "cancellation_check" in signature.parameters:
            return method(request, cancellation_check=cancellation_check)
        return method(request)

    @staticmethod
    def _upscaler_snapshot_is_resident(snapshot: dict[str, Any]) -> bool:
        if bool(snapshot.get("runtime_loaded", False)):
            return True
        if bool(snapshot.get("loaded_upscaler_id")):
            return True
        for item in list(snapshot.get("neural") or []):
            if not isinstance(item, dict):
                continue
            if bool(item.get("runtime_loaded", False)):
                return True
            if int(item.get("active_leases", 0) or 0) > 0:
                return True
        active_leases = snapshot.get("active_leases")
        if isinstance(active_leases, dict):
            return any(int(value or 0) > 0 for value in active_leases.values())
        return False

    def _cleanup_pixel_hires_scoped_runtime(self, *, reason: str) -> dict[str, Any]:
        report: dict[str, Any] = {
            "schema_version": "phase14n6-cancellation-cleanup-v1",
            "reason": str(reason),
            "upscaler_unload_requested": False,
            "upscaler_resident_after_cleanup": False,
            "inactive_component_actions": [],
        }
        registry = self.neural_upscaler_registry
        if registry is not None:
            report["upscaler_unload_requested"] = True
            with suppress(Exception):
                registry.unload()
            with suppress(Exception):
                snapshot = dict(registry.snapshot() or {})
                report["registry_snapshot"] = snapshot
                report["upscaler_resident_after_cleanup"] = (
                    self._upscaler_snapshot_is_resident(snapshot)
                )
        with suppress(Exception):
            report["inactive_component_actions"] = self.memory_manager.offload_inactive_components(
                {"upscaler"},
                stage="pixel_hires_cleanup",
                reason=str(reason),
            )
        with suppress(Exception):
            self.memory_manager.release_cuda_cache(
                stage="pixel_hires_cleanup",
                reason=str(reason),
            )
        return report

    @torch.inference_mode()
    def generate(
        self,
        request: GenerationRequest,
        *,
        diagnostic_session: DiagnosticSession | None = None,
    ) -> GenerationResult:
        diagnostics = self.systems.diagnostics
        owns_session = diagnostic_session is None
        session = diagnostic_session or diagnostics.start(
            request,
            components=self.components,
        )
        diagnostics.update_components(session, self.components)
        performance = GenerationPerformanceRecorder.from_request(
            request, device=self.device
        )
        provider_execution_before_generation = get_execution_evidence()
        provider_execution_after_base = dict(provider_execution_before_generation)
        provider_execution_after_hires = dict(provider_execution_before_generation)
        auxiliary_images: dict[str, Any] = {}
        configure_vae_memory = getattr(
            self.systems.decoding, "configure_memory_controls", None
        )
        if callable(configure_vae_memory):
            base_vae_memory_controls = dict(
                configure_vae_memory(
                    tiling=bool(self.memory_manager.settings.vae_tiling),
                    slicing=bool(self.memory_manager.settings.vae_slicing),
                    device=str(self.memory_manager.settings.vae_device),
                )
                or {}
            )
        else:
            base_vae_memory_controls = {
                "implementation": "unavailable",
                "requested": {
                    "tiling": bool(self.memory_manager.settings.vae_tiling),
                    "slicing": bool(self.memory_manager.settings.vae_slicing),
                    "device": str(self.memory_manager.settings.vae_device),
                },
            }
        configure_output_quality_diagnostics = getattr(
            self.systems.decoding, "configure_output_quality_diagnostics", None
        )
        if callable(configure_output_quality_diagnostics):
            diagnostics_request = (
                dict(getattr(request, "diagnostics", None) or {})
                if isinstance(getattr(request, "diagnostics", None), dict)
                else {}
            )
            configure_output_quality_diagnostics(
                bool(diagnostics_request.get("capture_output_quality", False))
            )
        self.memory_manager.configure_oom_recovery_hooks(
            attention_context_factory=lambda mode: temporary_attention_slicing(
                self.components.unet,
                mode,
                strict=False,
            ),
            vae_memory_configurator=(
                lambda **values: configure_vae_memory(
                    tiling=values.get("tiling"),
                    slicing=values.get("slicing"),
                    device=str(self.memory_manager.settings.vae_device),
                )
                if callable(configure_vae_memory)
                else None
            ),
        )

        self.systems.denoising.configure_request(
            cfg_rescale=float(getattr(request, "cfg_rescale", 0.0) or 0.0)
        )
        pixel_hires_job = False
        pixel_hires_preflight = None
        pixel_hires_stage_timings: dict[str, float] = {}
        pixel_hires_cancelled_stage = ""

        try:
            dimension_plan = diagnostics.run_stage(
                session,
                "latent_preparation",
                "plan_dimensions",
                lambda: self.systems.latent_preparation.plan_dimensions(request),
            )
            base_dimension_plan = dimension_plan
            final_output_width = int(request.width)
            final_output_height = int(request.height)
            if self.neural_upscaler_registry is not None:
                setattr(request, "_hires_upscaler_registry", self.neural_upscaler_registry)
            hires_execution_plan = resolve_hires_execution_plan(request)
            request.hires_dimension_plan = hires_execution_plan.dimensions.to_dict()
            request.hires_steps = int(hires_execution_plan.steps)
            request.hires_denoising_strength = float(
                hires_execution_plan.denoising_strength
            )
            request.hires_step_policy = str(hires_execution_plan.step_policy)
            request.hires_strategy = str(hires_execution_plan.upscale_plan.strategy)
            request.hires_upscaler = str(hires_execution_plan.upscale_plan.legacy_value or hires_execution_plan.upscaler)
            request.hires_upscaler_id = str(hires_execution_plan.upscale_plan.upscaler_id or "")
            hires_metadata: dict[str, Any] = hires_execution_plan.to_dict()
            pixel_hires_job = bool(
                hires_execution_plan.enabled
                and hires_execution_plan.upscale_plan.strategy == "pixel_neural"
            )
            if pixel_hires_job and bool(getattr(request, "hires_memory_preflight", True)):
                descriptor = hires_execution_plan.upscale_plan.descriptor
                if descriptor is None and self.neural_upscaler_registry is not None:
                    descriptor = self.neural_upscaler_registry.resolve_neural(
                        hires_execution_plan.upscale_plan.upscaler_id
                    )
                if descriptor is None:
                    raise PixelHiresAdmissionError(
                        "Pixel-neural hires preflight could not resolve the selected neural descriptor."
                    )
                pixel_hires_preflight = estimate_pixel_hires_preflight(
                    request=request,
                    base_width=int(request.width),
                    base_height=int(request.height),
                    target_width=int(hires_execution_plan.upscale_plan.target_width),
                    target_height=int(hires_execution_plan.upscale_plan.target_height),
                    native_scale=int(descriptor.native_scale),
                    model_file_size_bytes=int(descriptor.file_size_bytes),
                    memory_manager=self.memory_manager,
                    output_dir=getattr(request, "output_dir", None),
                )
                hires_metadata["memory_preflight"] = pixel_hires_preflight.to_dict()
                if not pixel_hires_preflight.admitted:
                    raise PixelHiresAdmissionError(
                        "; ".join(pixel_hires_preflight.rejection_reasons)
                    )
            state_extra = getattr(self.state, "extra", {}) if self.state is not None else {}
            preview_mode = (
                state_extra.get("live_preview_mode", "disabled")
                if isinstance(state_extra, dict) and state_extra.get("live_preview_enabled", False)
                else "disabled"
            )
            base_preview_policy = resolve_preview_stage_policy(
                requested_policy=self.memory_manager.settings.preview_policy,
                stage="base_sampling",
                preview_mode=str(preview_mode),
                already_suspended=self.memory_manager.preview_image_decode_suspended,
                existing_suspension_reason=self.memory_manager.preview_image_decode_suspension_reason,
                existing_suspension_source=self.memory_manager.preview_image_decode_suspension_source,
            )
            preview_policy_report: dict[str, Any] = {
                "requested": str(self.memory_manager.settings.preview_policy),
                "base_sampling": base_preview_policy.to_dict(),
                "hires_second_pass": None,
            }
            self.memory_manager.set_request_context(
                request=request,
                dimension_plan=dimension_plan,
                preview_mode=base_preview_policy.effective_preview_mode,
            )
            diagnostics.run_stage(
                session,
                "latent_preparation",
                "resolve_seeds",
                lambda: self.systems.latent_preparation.resolve_seeds(request),
            )
            conditioning = diagnostics.run_stage(
                session,
                "conditioning",
                "encode",
                lambda: self.memory_manager.run_stage(
                    stage="conditioning",
                    required={"text_encoder"},
                    preferred={"unet"},
                    operation=lambda: self.systems.conditioning.encode(
                        self.components, request, self.state
                    ),
                    request=request,
                ),
            )
            diagnostics.record_tensor(
                session,
                "conditioning.cond",
                conditioning.cond,
                system="conditioning",
                operation="encode",
            )
            diagnostics.record_tensor(
                session,
                "conditioning.uncond",
                conditioning.uncond,
                system="conditioning",
                operation="encode",
            )

            schedule = diagnostics.run_stage(
                session,
                "scheduling",
                "build",
                lambda: self.memory_manager.observe_stage(
                    "scheduler_construction",
                    lambda: self.systems.scheduling.build(request, self.state),
                ),
            )
            diagnostics.update_schedule(session, schedule)
            self._configure_live_preview_sink(request, schedule, diagnostics, session)

            latents = diagnostics.run_stage(
                session,
                "latent_preparation",
                "prepare",
                lambda: self.memory_manager.observe_stage(
                    "initial_latent_allocation",
                    lambda: self.systems.latent_preparation.prepare(request, schedule),
                ),
            )
            diagnostics.record_tensor(
                session,
                "latent_preparation.latents",
                latents,
                system="latent_preparation",
                operation="prepare",
            )

            raw_model_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_raw_noise",
                self.systems.denoising.predict_raw_noise,
            )
            guided_model_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_guided_noise",
                self.systems.denoising.predict_guided_noise,
            )
            regional_conditional_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_regional_conditional_noise",
                self.systems.denoising.predict_regional_conditional_noise,
            )
            regional_guided_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_regional_guided_noise",
                self.systems.denoising.predict_regional_guided_noise,
            )
            regional_denoised_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_regional_denoised",
                self.systems.denoising.predict_regional_denoised,
            )
            setattr(
                raw_model_fn,
                "predict_regional_conditional_noise",
                regional_conditional_fn,
            )
            setattr(
                guided_model_fn,
                "predict_regional_guided_noise",
                regional_guided_fn,
            )
            setattr(
                guided_model_fn,
                "predict_regional_denoised",
                regional_denoised_fn,
            )
            guided_denoiser_adapter = self.systems.denoising.build_guided_epsilon_denoiser(
                guided_model_fn,
            )
            denoised_model_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_denoised",
                guided_denoiser_adapter,
            )
            setattr(guided_model_fn, "predict_denoised", denoised_model_fn)
            setattr(
                guided_model_fn,
                "denoising_contract",
                self.systems.denoising.guided_epsilon_contract_metadata,
            )

            def _base_sampling_operation(attempt_latents: torch.Tensor) -> Any:
                return performance.run(
                    "base_sampling",
                    lambda: self.systems.sampling.sample(
                        raw_model_fn=raw_model_fn,
                        guided_model_fn=guided_model_fn,
                        latents=attempt_latents,
                        schedule=schedule,
                        conditioning=conditioning,
                        request=request,
                        state=self.state,
                    ),
                    operation_name="sampling.sample",
                )

            base_sampling_recovery = self._sampling_recovery_contract(
                stage="sampling",
                source_latents=latents,
                request=request,
                operation_builder=_base_sampling_operation,
            )
            sample_output = diagnostics.run_stage(
                session,
                "sampling",
                "sample",
                lambda: self.memory_manager.run_stage(
                    stage="sampling",
                    required=(
                        {"unet", "vae"}
                        if base_preview_policy.requires_vae
                        else {"unet"}
                    ),
                    optional=(
                        {"vae"}
                        if base_preview_policy.optional_vae
                        else set()
                    ),
                    preview_requires_vae=base_preview_policy.suspend_on_pressure,
                    operation=lambda: _base_sampling_operation(latents),
                    request=request,
                    recovery_contract=base_sampling_recovery,
                ),
            )
            diagnostics.update_sampler(session, sample_output)
            provider_execution_after_base = get_execution_evidence()
            provider_execution_after_hires = dict(provider_execution_after_base)

            if hires_execution_plan.enabled:
                base_memory_profile = self.memory_manager.effective_profile_for_stage(
                    "sampling",
                    fallback=self.memory_manager.settings.policy,
                )
                hires_memory_behavior = resolve_hires_memory_behavior(
                    requested_profile=self.memory_manager.settings.hires_memory_profile,
                    base_memory_policy=base_memory_profile,
                    pre_hires_cleanup=self.memory_manager.settings.pre_hires_cleanup,
                    preview_policy=self.memory_manager.settings.preview_policy,
                    base_safety_margin_mb=self.memory_manager.settings.safety_margin_mb,
                )
                hires_metadata["memory_behavior"] = hires_memory_behavior.to_dict()
                hires_metadata["base_memory_profile"] = base_memory_profile
                if callable(configure_vae_memory):
                    hires_vae_memory_controls = dict(
                        configure_vae_memory(
                            tiling=bool(
                                self.memory_manager.settings.vae_tiling
                                or hires_memory_behavior.vae_tiling_requested
                            ),
                            slicing=bool(
                                self.memory_manager.settings.vae_slicing
                                or hires_memory_behavior.vae_slicing_requested
                            ),
                            device=str(self.memory_manager.settings.vae_device),
                        )
                        or {}
                    )
                else:
                    hires_vae_memory_controls = dict(base_vae_memory_controls)
                hires_metadata["vae_memory_controls"] = hires_vae_memory_controls
                lowres_artifact_requested = bool(
                    getattr(request, "hires_save_lowres", False)
                    and not bool(getattr(request, "return_latents", False))
                )
                hires_metadata["lowres_artifact"] = {
                    "requested": lowres_artifact_requested,
                    "captured": False,
                    "source": "exact_base_pass_latents",
                    "width": int(request.width),
                    "height": int(request.height),
                }
                if lowres_artifact_requested:
                    lowres_decoded = diagnostics.run_stage(
                        session,
                        "hires",
                        "decode_base_artifact",
                        lambda: self.memory_manager.run_stage(
                            stage="hires_base_artifact_decode",
                            required={"vae"},
                            operation=lambda: performance.run(
                                "hires_base_artifact_decode",
                                lambda: self.systems.decoding.decode(sample_output.latents),
                                operation_name="decoding.decode_base_artifact",
                            ),
                            request=request,
                        ),
                    )
                    if base_dimension_plan.crop_required:
                        lowres_decoded = DecodingSystem.center_crop(
                            lowres_decoded,
                            width=int(request.width),
                            height=int(request.height),
                        )
                    auxiliary_images["hires_base_lowres"] = lowres_decoded.detach().to("cpu")
                    consume_decode_report = getattr(
                        self.systems.decoding, "consume_last_decode_diagnostics", None
                    )
                    if callable(consume_decode_report):
                        consume_decode_report()
                    hires_metadata["lowres_artifact"]["captured"] = True
                    hires_metadata["lowres_artifact"]["tensor_shape"] = list(
                        auxiliary_images["hires_base_lowres"].shape
                    )

                hires_request = build_hires_request(request, hires_execution_plan)
                hires_scheduler_adapter = (
                    state_extra.get("hires_scheduler_adapter")
                    if isinstance(state_extra, dict)
                    else None
                )
                hires_sampler_adapter = (
                    state_extra.get("hires_sampler_adapter")
                    if isinstance(state_extra, dict)
                    else None
                )
                hires_scheduling_system = (
                    SchedulingSystem(hires_scheduler_adapter)
                    if hires_scheduler_adapter is not None
                    else self.systems.scheduling
                )
                hires_sampling_system = (
                    SamplingSystem(hires_sampler_adapter)
                    if hires_sampler_adapter is not None
                    else self.systems.sampling
                )
                hires_compatibility = (
                    dict(state_extra.get("hires_plugin_compatibility") or {})
                    if isinstance(state_extra, dict)
                    else {}
                )
                hires_qualification = (
                    dict(state_extra.get("hires_pair_qualification") or {})
                    if isinstance(state_extra, dict)
                    else {}
                )
                if not hires_qualification:
                    try:
                        hires_qualification = require_qualified_hires_pair(
                            hires_request.sampler_name,
                            hires_request.scheduler_name,
                            compatibility=hires_compatibility,
                        ).to_serializable_dict()
                    except ValueError as exc:
                        # Direct/injected pipelines do not pass through the runtime
                        # plugin registry. Preserve their compatibility while clearly
                        # reporting that the pair was not release-qualified. CLI and
                        # WebUI paths are rejected earlier by RuntimeRegistrySystem.
                        hires_qualification = {
                            "sampler_name": str(hires_request.sampler_name or ""),
                            "scheduler_name": str(hires_request.scheduler_name or ""),
                            "qualification_id": "direct-adapter-unqualified",
                            "qualified": False,
                            "required_matrix_pair": False,
                            "compatibility_mode": "direct_adapter",
                            "notes": [str(exc)],
                        }
                hires_metadata["runtime_pair"] = {
                    "sampler_name": str(hires_request.sampler_name or ""),
                    "scheduler_name": str(hires_request.scheduler_name or ""),
                    "sampler_inherited": bool(
                        isinstance(state_extra, dict)
                        and state_extra.get("hires_sampler_inherited", not bool(request.hires_sampler_name))
                    ),
                    "scheduler_inherited": bool(
                        isinstance(state_extra, dict)
                        and state_extra.get("hires_scheduler_inherited", not bool(request.hires_scheduler_name))
                    ),
                    "cfg_scale": float(hires_request.cfg_scale),
                    "cfg_rescale": float(hires_request.cfg_rescale),
                    "cfg_scale_inherited": request.hires_cfg_scale is None,
                    "cfg_rescale_inherited": request.hires_cfg_rescale is None,
                    "compatibility": hires_compatibility,
                    "qualification": hires_qualification,
                }
                self.systems.denoising.configure_request(
                    cfg_rescale=float(hires_request.cfg_rescale)
                )
                hires_dimension_plan = diagnostics.run_stage(
                    session,
                    "hires",
                    "plan_dimensions",
                    lambda: self.systems.latent_preparation.plan_dimensions(hires_request),
                )
                final_output_width = int(hires_request.width)
                final_output_height = int(hires_request.height)
                hires_preview_policy = resolve_preview_stage_policy(
                    requested_policy=self.memory_manager.settings.preview_policy,
                    stage="hires_second_pass",
                    preview_mode=str(preview_mode),
                    force_disabled=hires_memory_behavior.disable_preview_during_hires,
                    force_disabled_reason=(
                        f"Hires memory profile {hires_memory_behavior.effective_profile} disabled image preview decoding."
                        if hires_memory_behavior.disable_preview_during_hires
                        else ""
                    ),
                    already_suspended=self.memory_manager.preview_image_decode_suspended,
                    existing_suspension_reason=self.memory_manager.preview_image_decode_suspension_reason,
                    existing_suspension_source=self.memory_manager.preview_image_decode_suspension_source,
                )
                preview_policy_report["hires_second_pass"] = hires_preview_policy.to_dict()
                hires_metadata["preview_policy"] = hires_preview_policy.to_dict()
                self.memory_manager.set_request_context(
                    request=hires_request,
                    dimension_plan=hires_dimension_plan,
                    preview_mode=hires_preview_policy.effective_preview_mode,
                )
                pixel_source_result: PixelNeuralHiresSourceResult | None = None
                hires_latents: torch.Tensor | None = None
                if hires_execution_plan.upscale_plan.strategy == "pixel_neural":
                    if self.neural_upscaler is None or self.neural_upscaler_registry is None:
                        selected_id = str(hires_execution_plan.upscale_plan.upscaler_id or "")
                        raise RuntimeError(
                            f"Pixel-neural hires requested {selected_id!r}, but the owned standalone upscaler runtime is not configured."
                        )
                    descriptor = hires_execution_plan.upscale_plan.descriptor
                    if descriptor is None:
                        raise RuntimeError("Pixel-neural hires resolved without an exact descriptor.")

                    raise_if_pixel_hires_cancelled(
                        "base_decode", request=request, state=self.state
                    )
                    base_hires_latents = sample_output.latents
                    stage_started = time.perf_counter()
                    decoded_base = diagnostics.run_stage(
                        session,
                        "hires",
                        "pixel_decode_base",
                        lambda: self.memory_manager.run_stage(
                            stage="hires_pixel_decode_base",
                            required={"vae"},
                            operation=lambda: performance.run(
                                "hires_pixel_decode_base",
                                lambda: self.systems.decoding.decode(base_hires_latents),
                                operation_name="decoding.decode_pixel_hires_source",
                            ),
                            request=request,
                        ),
                    )
                    pixel_hires_stage_timings["base_decode_ms"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    decoded_base = DecodingSystem.center_crop(
                        decoded_base,
                        width=int(request.width),
                        height=int(request.height),
                    )
                    diagnostics_options = dict(getattr(request, "diagnostics", None) or {})
                    host_staging_policy = str(
                        getattr(request, "hires_host_staging_policy", "pageable") or "pageable"
                    )
                    host_staging_cap = max(0, int(
                        getattr(request, "hires_host_staging_cap_mb", 1024) or 0
                    )) * 1024 * 1024
                    host_staging_benchmark = bool(
                        diagnostics_options.get("benchmark_hires_host_staging", False)
                    )
                    pixel_source_cpu, base_staging_report = stage_tensor_to_host(
                        decoded_base.detach().to(dtype=torch.float32),
                        label="decoded_base",
                        policy=host_staging_policy,
                        cap_bytes=host_staging_cap,
                        benchmark=host_staging_benchmark,
                    )
                    hires_metadata.setdefault("host_staging", {})["decoded_base"] = (
                        base_staging_report.to_dict()
                    )
                    decoded_base = None
                    base_hires_latents = None
                    released_base_references = (
                        "base_conditioning",
                        "base_schedule",
                        "initial_latents",
                        "base_sampler_output",
                        "base_hires_latents_after_decode",
                    )
                    conditioning = None
                    schedule = None
                    latents = None
                    sample_output = None
                    hires_cleanup_report = diagnostics.run_stage(
                        session,
                        "hires",
                        "pre_hires_cleanup",
                        lambda: perform_pre_hires_cleanup(
                            self.memory_manager,
                            behavior=hires_memory_behavior,
                            preserved_tensors=(("pixel_source_cpu", pixel_source_cpu),),
                            released_reference_names=released_base_references,
                        ),
                    )
                    hires_metadata["pre_hires_cleanup"] = hires_cleanup_report.to_dict()

                    raise_if_pixel_hires_cancelled(
                        "neural_upscale", request=request, state=self.state
                    )
                    upscale_request = UpscaleRequest(
                        source_images=pixel_source_cpu,
                        upscaler_id=str(descriptor.upscaler_id),
                        target_width=int(hires_execution_plan.upscale_plan.target_width),
                        target_height=int(hires_execution_plan.upscale_plan.target_height),
                        tile_size=int(hires_execution_plan.upscale_plan.tile_size),
                        tile_overlap=int(hires_execution_plan.upscale_plan.tile_overlap),
                        tile_batch_size=int(hires_execution_plan.upscale_plan.tile_batch_size),
                        exact_resize_filter=str(hires_execution_plan.upscale_plan.exact_resize_filter),
                        dtype_policy="auto",
                        device_policy="auto",
                        allow_tiling=True,
                        allow_oom_retry=True,
                        host_transfer_non_blocking=bool(pixel_source_cpu.is_pinned()),
                    )
                    stage_started = time.perf_counter()
                    upscale_result = diagnostics.run_stage(
                        session,
                        "hires",
                        "standalone_neural_upscale",
                        lambda: performance.run(
                            "hires_neural_upscale",
                            lambda: self._invoke_neural_upscaler(
                                upscale_request,
                                cancellation_check=lambda: cancellation_requested(
                                    request=request, state=self.state
                                ),
                            ),
                            operation_name="standalone_neural_upscaler.upscale",
                        ),
                    )
                    pixel_hires_stage_timings["neural_upscale_ms"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    upscale_metadata = dict(upscale_result.metadata or {})
                    executed_id = str(upscale_metadata.get("upscaler_id") or "")
                    executed_hash = str(upscale_metadata.get("upscaler_sha256") or "").casefold()
                    if executed_id != descriptor.upscaler_id or executed_hash != descriptor.sha256.casefold():
                        raise RuntimeError(
                            "The executed neural upscaler identity/hash did not match the resolved hires plan."
                        )
                    exact_target_images, target_staging_report = stage_tensor_to_host(
                        upscale_result.images.detach().to(dtype=torch.float32),
                        label="exact_neural_target",
                        policy=host_staging_policy,
                        cap_bytes=host_staging_cap,
                        benchmark=host_staging_benchmark,
                    )
                    hires_metadata.setdefault("host_staging", {})["exact_neural_target"] = (
                        target_staging_report.to_dict()
                    )
                    registry_snapshot_method = getattr(
                        self.neural_upscaler_registry, "snapshot", None
                    )
                    registry_snapshot = (
                        dict(registry_snapshot_method() or {})
                        if callable(registry_snapshot_method)
                        else {
                            "runtime_loaded": False,
                            "verification": "snapshot_unavailable_on_compatible_override",
                        }
                    )
                    hires_metadata["upscaler_residency_after_stage"] = registry_snapshot
                    if self._upscaler_snapshot_is_resident(registry_snapshot):
                        raise RuntimeError(
                            "The neural upscaler remained resident after its scoped execution stage."
                        )
                    upscale_result = None
                    pixel_source_cpu = None
                    if bool(getattr(request, "hires_save_upscaled_pre_denoise", False)):
                        auxiliary_images["hires_upscaled_pre_denoise"] = exact_target_images

                    execution_vae = getattr(
                        self.systems.decoding, "vae", self.components.vae
                    )
                    module_vae_provenance = read_vae_provenance(execution_vae)
                    component_vae_provenance = dict(
                        self.components.vae_provenance or {}
                    )
                    module_hash = str(
                        module_vae_provenance.get("sha256") or ""
                    ).casefold()
                    component_hash = str(
                        component_vae_provenance.get("sha256") or ""
                    ).casefold()
                    if module_hash and component_hash and module_hash != component_hash:
                        raise RuntimeError(
                            "The loader-owned VAE provenance does not match the VAE execution component."
                        )
                    vae_provenance = (
                        module_vae_provenance if module_hash else component_vae_provenance
                    )
                    vae_sha256 = str(vae_provenance.get("sha256") or "").casefold()
                    if len(vae_sha256) != 64 or any(
                        character not in "0123456789abcdef" for character in vae_sha256
                    ):
                        raise RuntimeError(
                            "Pixel-neural hires requires loader-owned VAE provenance with a complete SHA-256."
                        )
                    validate_recorded_hires_vae_identity(request, vae_provenance)

                    raise_if_pixel_hires_cancelled(
                        "vae_encode", request=request, state=self.state
                    )
                    stage_started = time.perf_counter()
                    encoded = diagnostics.run_stage(
                        session,
                        "hires",
                        "vae_encode_for_sampling",
                        lambda: self.memory_manager.run_stage(
                            stage="hires_vae_encode",
                            required={"vae"},
                            operation=lambda: performance.run(
                                "hires_vae_encode",
                                lambda: vae_encode_for_sampling(
                                    image=exact_target_images,
                                    vae=self.systems.decoding,
                                    scaling_factor=self.vae_scaling_factor,
                                    deterministic=True,
                                    target_width=int(hires_execution_plan.upscale_plan.target_width),
                                    target_height=int(hires_execution_plan.upscale_plan.target_height),
                                    allow_center_crop=False,
                                    latent_downsample_factor=self.latent_scale_factor,
                                    vae_identity=vae_provenance,
                                    upscale_metadata=upscale_metadata,
                                ),
                                operation_name="image_conditioning.vae_encode_for_sampling",
                            ),
                            request=hires_request,
                        ),
                    )
                    pixel_hires_stage_timings["vae_encode_ms"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    raise_if_pixel_hires_cancelled(
                        "vae_encode_complete", request=request, state=self.state
                    )
                    fingerprint_requested = bool(
                        getattr(request, "hires_diagnostic_vae_execution_fingerprint", False)
                        or (
                            isinstance(getattr(request, "diagnostics", None), dict)
                            and request.diagnostics.get(
                                "capture_hires_vae_execution_fingerprint", False
                            )
                        )
                    )
                    diagnostic_artifacts: dict[str, Any] = {
                        "vae_execution_fingerprint_requested": fingerprint_requested,
                        "vae_execution_fingerprint": (
                            build_vae_execution_fingerprint(encoded.metadata)
                            if fingerprint_requested
                            else None
                        ),
                    }
                    round_trip = None
                    if bool(getattr(request, "hires_save_vae_roundtrip", False)):
                        raise_if_pixel_hires_cancelled(
                            "vae_round_trip", request=request, state=self.state
                        )
                        stage_started = time.perf_counter()
                        round_trip = diagnostics.run_stage(
                            session,
                            "hires",
                            "vae_round_trip_diagnostic",
                            lambda: self.memory_manager.run_stage(
                                stage="hires_vae_round_trip",
                                required={"vae"},
                                operation=lambda: vae_round_trip_from_encoded_for_diagnostics(
                                    image=exact_target_images,
                                    encoded=encoded,
                                    vae=self.systems.decoding,
                                    scaling_factor=self.vae_scaling_factor,
                                ),
                                request=hires_request,
                            ),
                        )
                        pixel_hires_stage_timings["vae_round_trip_ms"] = (
                            time.perf_counter() - stage_started
                        ) * 1000.0
                        auxiliary_images["hires_vae_roundtrip"] = round_trip.image.detach().to("cpu")

                    pixel_source_result = PixelNeuralHiresSourceResult(
                        exact_target_images=exact_target_images,
                        upscale_metadata=upscale_metadata,
                        vae_encode_result=encoded,
                        vae_round_trip=round_trip,
                        diagnostic_artifacts=diagnostic_artifacts,
                    )
                    hires_metadata["pixel_source_preparation"] = pixel_source_result.to_dict()
                    hires_metadata["intermediate_artifacts"] = {
                        "upscaled_pre_denoise_requested": bool(
                            getattr(request, "hires_save_upscaled_pre_denoise", False)
                        ),
                        "vae_roundtrip_requested": bool(
                            getattr(request, "hires_save_vae_roundtrip", False)
                        ),
                        "content_hash_policy": "saved_file_hashes_are_owned_by_output_manifests; no normal-run tensor hash",
                    }
                    hires_latents = encoded.latents.to(
                        device=self.device, dtype=self.dtype
                    )
                    pixel_source_result = None
                    round_trip = None
                    exact_target_images = None

                hires_conditioning = diagnostics.run_stage(
                    session,
                    "hires",
                    "encode_conditioning",
                    lambda: self.memory_manager.run_stage(
                        stage="conditioning",
                        required={"text_encoder"},
                        preferred={"unet"},
                        operation=lambda: performance.run(
                            "hires_conditioning",
                            lambda: self.systems.conditioning.encode(
                                self.components, hires_request, self.state
                            ),
                            operation_name="conditioning.encode",
                        ),
                        request=hires_request,
                    ),
                )
                request.prompt_cfg_pass_schedules = dict(
                    getattr(request, "prompt_cfg_pass_schedules", {}) or {}
                )
                hires_prompt_cfg = dict(
                    getattr(hires_request, "prompt_cfg_schedule", {}) or {}
                )
                if hires_prompt_cfg:
                    request.prompt_cfg_pass_schedules["hires"] = hires_prompt_cfg
                    hires_metadata["prompt_cfg_schedule"] = hires_prompt_cfg
                else:
                    request.prompt_cfg_pass_schedules.pop("hires", None)

                request.prompt_expansion_pass_records = dict(
                    getattr(request, "prompt_expansion_pass_records", {}) or {}
                )
                hires_prompt_expansion = dict(
                    getattr(hires_request, "prompt_expansion_record", {}) or {}
                )
                if hires_prompt_expansion:
                    request.prompt_expansion_pass_records["hires"] = hires_prompt_expansion
                    hires_metadata["prompt_expansion"] = hires_prompt_expansion
                else:
                    request.prompt_expansion_pass_records.pop("hires", None)

                request.prompt_semantic_pass_records = dict(
                    getattr(request, "prompt_semantic_pass_records", {}) or {}
                )
                hires_prompt_semantics = dict(
                    (getattr(hires_request, "prompt_semantic_pass_records", {}) or {}).get("hires")
                    or {}
                )
                if hires_prompt_semantics:
                    request.prompt_semantic_pass_records["hires"] = hires_prompt_semantics
                    hires_metadata["prompt_semantics"] = hires_prompt_semantics
                else:
                    request.prompt_semantic_pass_records.pop("hires", None)

                request.region_pass_records = dict(
                    getattr(request, "region_pass_records", {}) or {}
                )
                hires_regions = dict(
                    (getattr(hires_request, "region_pass_records", {}) or {}).get("hires")
                    or {}
                )
                if hires_regions:
                    request.region_pass_records["hires"] = hires_regions
                    hires_metadata["regional_prompting"] = hires_regions
                else:
                    request.region_pass_records.pop("hires", None)

                replay_mode = str(
                    getattr(request, "hires_schedule_replay_mode", "reconstruct")
                    or "reconstruct"
                ).strip().lower()
                if replay_mode not in {"reconstruct", "recorded_exact"}:
                    raise ValueError(
                        "hires_schedule_replay_mode must be reconstruct or recorded_exact."
                    )

                schedule_construction_started = time.perf_counter()
                if replay_mode == "recorded_exact":
                    recorded_replay = dict(
                        getattr(request, "hires_recorded_schedule_replay", {}) or {}
                    )
                    recorded_fingerprint = dict(
                        getattr(request, "hires_recorded_schedule_fingerprint", {}) or {}
                    )
                    if not recorded_replay or not recorded_fingerprint:
                        raise ValueError(
                            "Exact recorded schedule replay was requested, but the manifest did not "
                            "contain both schedule_replay and schedule_fingerprint records."
                        )
                    expected_scheduler = str(recorded_replay.get("scheduler_identifier") or "")
                    expected_sampler = str(recorded_replay.get("sampler_name") or "")
                    if expected_scheduler and expected_scheduler != str(hires_request.scheduler_name or ""):
                        raise ValueError(
                            "Recorded schedule scheduler does not match the requested hires scheduler: "
                            f"{expected_scheduler!r} != {hires_request.scheduler_name!r}."
                        )
                    if expected_sampler and expected_sampler != str(hires_request.sampler_name or ""):
                        raise ValueError(
                            "Recorded schedule sampler does not match the requested hires sampler: "
                            f"{expected_sampler!r} != {hires_request.sampler_name!r}."
                        )
                    if str(recorded_replay.get("step_policy") or "") != str(hires_execution_plan.step_policy):
                        raise ValueError(
                            "Recorded schedule step policy does not match the replay request."
                        )
                    if int(recorded_replay.get("requested_refinement_steps", -1)) != int(
                        hires_execution_plan.steps
                    ):
                        raise ValueError(
                            "Recorded schedule refinement steps do not match the replay request."
                        )
                    if abs(
                        float(recorded_replay.get("denoising_strength", -1.0))
                        - float(hires_execution_plan.denoising_strength)
                    ) > 1.0e-9:
                        raise ValueError(
                            "Recorded schedule denoising strength does not match the replay request."
                        )
                    rehydration = diagnostics.run_stage(
                        session,
                        "hires",
                        "rehydrate_recorded_schedule",
                        lambda: rehydrate_schedule_replay_record(
                            recorded_replay,
                            expected_fingerprint=recorded_fingerprint,
                            device=self.device,
                            strict_fingerprint=True,
                        ),
                    )
                    image_conditioned_schedule = rehydration.schedule
                    full_hires_schedule = image_conditioned_schedule.full_schedule
                    hires_metadata["schedule_source"] = "recorded_exact"
                    hires_metadata["schedule_rehydration"] = rehydration.to_serializable_dict()
                else:
                    full_hires_schedule = diagnostics.run_stage(
                        session,
                        "hires",
                        "build_schedule",
                        lambda: self.memory_manager.observe_stage(
                            "scheduler_construction",
                            lambda: hires_scheduling_system.build(hires_request, self.state),
                        ),
                    )
                    image_conditioned_schedule = diagnostics.run_stage(
                        session,
                        "hires",
                        "build_image_conditioned_schedule",
                        lambda: build_image_conditioned_schedule(
                            full_hires_schedule,
                            requested_refinement_steps=int(hires_execution_plan.steps),
                            denoising_strength=hires_execution_plan.denoising_strength,
                            step_policy=hires_execution_plan.step_policy,
                            scheduler_identifier=str(hires_request.scheduler_name or ""),
                            scheduler_configuration=dict(
                                (full_hires_schedule.metadata or {}).get("validated_settings") or {}
                            ),
                            requires_terminal_zero=None,
                            sampler_requires_timestep=True,
                        ),
                    )
                    hires_metadata["schedule_source"] = "reconstructed"

                if pixel_hires_job:
                    pixel_hires_stage_timings["schedule_construction_ms"] = (
                        time.perf_counter() - schedule_construction_started
                    ) * 1000.0
                hires_schedule = image_conditioned_schedule.active_schedule
                hires_schedule.metadata["hires_sampler_name"] = str(hires_request.sampler_name or "")
                hires_schedule.metadata["hires_scheduler_name"] = str(hires_request.scheduler_name or "")
                hires_schedule.metadata["hires_noise_policy"] = str(
                    hires_execution_plan.noise_policy
                )
                hires_metadata["schedule_contract"] = image_conditioned_schedule.to_serializable_dict()
                hires_metadata["schedule_baseline"] = hires_schedule_baseline_metadata(
                    hires_schedule
                )
                if replay_mode == "recorded_exact":
                    hires_metadata["schedule_replay"] = dict(
                        getattr(request, "hires_recorded_schedule_replay", {}) or {}
                    )
                else:
                    hires_metadata["schedule_replay"] = build_schedule_replay_record(
                        image_conditioned_schedule,
                        scheduler_identifier=str(hires_request.scheduler_name or ""),
                        scheduler_configuration=dict(
                            (full_hires_schedule.metadata or {}).get("validated_settings") or {}
                        ),
                        sampler_name=str(hires_request.sampler_name or ""),
                        requires_terminal_zero=None,
                    )
                hires_metadata["schedule_fingerprint"] = build_schedule_fingerprint_record(
                    image_conditioned_schedule,
                    replay_record=hires_metadata["schedule_replay"],
                )
                conformance_source_replay = dict(
                    getattr(request, "hires_schedule_conformance_source_replay", {}) or {}
                )
                conformance_source_fingerprint = dict(
                    getattr(request, "hires_schedule_conformance_source_fingerprint", {}) or {}
                )
                if conformance_source_replay:
                    hires_metadata["schedule_conformance"] = diagnostics.run_stage(
                        session,
                        "hires",
                        "compare_schedule_conformance",
                        lambda: compare_schedule_conformance(
                            image_conditioned_schedule,
                            conformance_source_replay,
                            recorded_fingerprint=conformance_source_fingerprint,
                        ),
                    )
                else:
                    hires_metadata["schedule_conformance"] = {
                        "format": "image-gen-schedule-conformance-v1",
                        "status": "not_requested",
                        "matches": None,
                        "difference_count": 0,
                        "differences": [],
                        "comparison_mode": "reconstructed_without_sampling",
                    }
                hires_schedule.metadata["hires_schedule_baseline"] = hires_metadata["schedule_baseline"]
                hires_schedule.metadata["hires_schedule_replay"] = hires_metadata["schedule_replay"]
                hires_schedule.metadata["hires_schedule_fingerprint"] = hires_metadata["schedule_fingerprint"]
                hires_schedule.metadata["hires_schedule_conformance"] = hires_metadata["schedule_conformance"]
                schedule_guard_sha256 = str(
                    hires_metadata["schedule_fingerprint"].get("sha256") or ""
                )
                # Scheduler construction may use more transitions than the user
                # requested under fixed-step semantics. Samplers and progress
                # reporting receive the creative refinement step count.
                hires_request = replace(
                    hires_request,
                    steps=int(hires_execution_plan.steps),
                )

                if hires_execution_plan.upscale_plan.strategy != "pixel_neural":
                    raise RuntimeError("Only pixel-neural .pth hires is active.")
                if hires_latents is None:
                    raise RuntimeError("Pixel-neural hires did not produce sampling latents.")

                hires_metadata["noise_stream"] = noise_stream_metadata(
                    list(request.resolved_seeds),
                    hires_execution_plan.noise_policy,
                )
                hires_latents = diagnostics.run_stage(
                    session,
                    "hires",
                    "add_noise",
                    lambda: performance.run(
                        "hires_noise",
                        lambda: add_hires_noise(
                            hires_latents,
                            schedule=hires_schedule,
                            seeds=list(request.resolved_seeds),
                        ),
                        operation_name="add_hires_noise",
                    ),
                )

                def _hires_sampling_operation(
                    attempt_latents: torch.Tensor,
                ) -> Any:
                    return performance.run(
                        "hires_second_pass",
                        lambda: hires_sampling_system.sample(
                            raw_model_fn=raw_model_fn,
                            guided_model_fn=guided_model_fn,
                            latents=attempt_latents,
                            schedule=hires_schedule,
                            conditioning=hires_conditioning,
                            request=hires_request,
                            state=self.state,
                        ),
                        operation_name="sampling.sample",
                    )

                hires_sampling_recovery = self._sampling_recovery_contract(
                    stage="hires_second_pass",
                    source_latents=hires_latents,
                    request=hires_request,
                    operation_builder=_hires_sampling_operation,
                )

                def _run_hires_second_pass() -> Any:
                    global_slicing = str(
                        self.memory_manager.settings.attention_slicing or "off"
                    ).lower()
                    if (
                        hires_memory_behavior.attention_slicing_requested
                        and global_slicing == "off"
                    ):
                        slicing_context = temporary_attention_slicing(
                            self.components.unet,
                            "max",
                            strict=False,
                        )
                    else:
                        slicing_context = temporary_attention_slicing(
                            self.components.unet,
                            "off",
                            strict=False,
                        )
                    with slicing_context as attention_slicing_report:
                        if global_slicing != "off":
                            attention_slicing_report.update(
                                {
                                    "requested": global_slicing,
                                    "applied": True,
                                    "verified": True,
                                    "temporary": False,
                                    "inherited_global_setting": True,
                                }
                            )
                        result = self.memory_manager.run_stage(
                            stage="hires_second_pass",
                            required=(
                                {"unet", "vae"}
                                if hires_preview_policy.requires_vae
                                else {"unet"}
                            ),
                            optional=(
                                {"vae"}
                                if hires_preview_policy.optional_vae
                                else set()
                            ),
                            preview_requires_vae=hires_preview_policy.suspend_on_pressure,
                            requested_profile_override=hires_memory_behavior.planner_profile,
                            safety_margin_bytes_override=(
                                int(hires_memory_behavior.safety_margin_mb) * 1024 * 1024
                            ),
                            operation=lambda: _hires_sampling_operation(
                                hires_latents
                            ),
                            request=hires_request,
                            recovery_contract=hires_sampling_recovery,
                        )
                    hires_metadata["attention_slicing"] = dict(
                        attention_slicing_report
                    )
                    return result

                if pixel_hires_job:
                    raise_if_pixel_hires_cancelled(
                        "hires_second_pass", request=request, state=self.state
                    )
                stage_started = time.perf_counter()
                sample_output = diagnostics.run_stage(
                    session,
                    "hires",
                    "sample_second_pass",
                    _run_hires_second_pass,
                )
                if pixel_hires_job:
                    pixel_hires_stage_timings["hires_second_pass_ms"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    raise_if_pixel_hires_cancelled(
                        "hires_second_pass_complete", request=request, state=self.state
                    )
                schedule_guard_after = build_schedule_fingerprint_record(
                    image_conditioned_schedule,
                    replay_record=hires_metadata["schedule_replay"],
                )
                schedule_guard_after_sha256 = str(
                    schedule_guard_after.get("sha256") or ""
                )
                if schedule_guard_after_sha256 != schedule_guard_sha256:
                    raise RuntimeError(
                        "The hires sampler altered the validated active schedule. "
                        "Samplers must consume the selected schedule without rebuilding "
                        "or mutating its sigma/timestep tensors."
                    )
                hires_metadata["sampler_schedule_guard"] = {
                    "before_sha256": schedule_guard_sha256,
                    "after_sha256": schedule_guard_after_sha256,
                    "unchanged": True,
                    "effective_transition_count": int(hires_schedule.effective_steps),
                    "progress_transition_count": int(hires_schedule.effective_steps),
                }
                diagnostics.update_schedule(session, hires_schedule)
                diagnostics.update_sampler(session, sample_output)
                hires_region_runtime = dict(
                    (getattr(hires_request, "diagnostics", {}) or {})
                    .get("regional_runtime_passes", {})
                    .get("hires")
                    or sample_output.extra.get("regional_runtime")
                    or {}
                )
                if hires_region_runtime:
                    request.diagnostics.setdefault("regional_runtime_passes", {})["hires"] = hires_region_runtime
                    request.diagnostics["regional_runtime"] = hires_region_runtime
                    hires_metadata["regional_runtime"] = hires_region_runtime
                provider_execution_after_hires = get_execution_evidence()
                conditioning = hires_conditioning
                schedule = hires_schedule
                dimension_plan = hires_dimension_plan
                hires_metadata.update(
                    {
                        "base_dimension_plan": base_dimension_plan.to_serializable_dict(),
                        "second_pass_dimension_plan": hires_dimension_plan.to_serializable_dict(),
                        "full_schedule_steps": int(full_hires_schedule.effective_steps),
                        "effective_second_pass_steps": int(hires_schedule.effective_steps),
                        "sampler_name": str(hires_request.sampler_name or ""),
                        "scheduler_name": str(hires_request.scheduler_name or ""),
                        "cfg_scale": float(hires_request.cfg_scale),
                        "cfg_rescale": float(hires_request.cfg_rescale),
                        "step_policy": str(hires_execution_plan.step_policy),
                        "noise_policy": str(hires_execution_plan.noise_policy),
                        "selected_starting_sigma": float(hires_schedule.initial_sigma),
                        "selected_starting_timestep": (
                            None
                            if hires_schedule.timesteps is None
                            else float(hires_schedule.timesteps[0].detach().cpu().item())
                        ),
                        "schedule_start_index": int(
                            hires_schedule.metadata.get("hires_schedule_start_index", 0)
                        ),
                        "latent_shape": list(sample_output.latents.shape),
                        "execution_scope": "runtime_shared_cli_webui",
                    }
                )
                source_contract = dict(hires_metadata.get("pixel_source_preparation") or {})
                upscale_contract = dict(source_contract.get("upscale_metadata") or {})
                vae_contract = dict(source_contract.get("vae_encode") or {})
                vae_identity = dict(vae_contract.get("vae") or {})
                upscale_plan_contract = dict(hires_metadata.get("upscale_plan") or {})
                descriptor_contract = dict(upscale_plan_contract.get("descriptor") or {})
                source_shape = list(upscale_contract.get("source_shape") or [])
                hires_metadata["phase14n7_diagnostics"] = {
                    "schema_version": "phase14n7-hires-diagnostics-v1",
                    "algorithm_version": str(hires_metadata.get("algorithm_version") or ""),
                    "strategy": str(hires_execution_plan.upscale_plan.strategy),
                    "upscaler": {
                        "id": str(
                            upscale_contract.get("upscaler_id")
                            or upscale_plan_contract.get("upscaler_id")
                            or hires_execution_plan.upscaler
                        ),
                        "display_name": str(
                            upscale_contract.get("upscaler_display_name")
                            or descriptor_contract.get("display_name")
                            or hires_execution_plan.upscaler
                        ),
                        "architecture": str(
                            upscale_contract.get("upscaler_architecture")
                            or descriptor_contract.get("architecture")
                            or ("latent_interpolation" if hires_execution_plan.upscale_plan.strategy == "latent" else "")
                        ),
                        "native_scale": int(
                            upscale_contract.get("upscaler_native_scale")
                            or descriptor_contract.get("native_scale")
                            or 0
                        ),
                        "sha256": str(
                            upscale_contract.get("upscaler_sha256")
                            or descriptor_contract.get("sha256")
                            or ""
                        ),
                        "load_status": str(
                            upscale_contract.get("upscaler_load_status")
                            or descriptor_contract.get("load_status")
                            or "built_in"
                        ),
                        "device": str(upscale_contract.get("runtime_device") or ""),
                        "dtype": str(upscale_contract.get("runtime_dtype") or ""),
                    },
                    "tiling": {
                        "tile_size": int(upscale_contract.get("tile_size") or upscale_plan_contract.get("tile_size") or 0),
                        "tile_overlap": int(upscale_contract.get("tile_overlap") or upscale_plan_contract.get("tile_overlap") or 0),
                        "tile_batch_size": int(upscale_contract.get("tile_batch_size") or upscale_plan_contract.get("tile_batch_size") or 1),
                        "tile_count": int(upscale_contract.get("tile_count") or 0),
                    },
                    "dimensions": {
                        "base_width": int(source_shape[-1]) if len(source_shape) >= 2 else int(base_dimension_plan.requested_width),
                        "base_height": int(source_shape[-2]) if len(source_shape) >= 2 else int(base_dimension_plan.requested_height),
                        "target_width": int(hires_execution_plan.upscale_plan.target_width),
                        "target_height": int(hires_execution_plan.upscale_plan.target_height),
                        "exact_resize_filter": str(hires_execution_plan.upscale_plan.exact_resize_filter),
                    },
                    "refinement": {
                        "requested_steps": int(hires_execution_plan.steps),
                        "internal_steps": int(hires_execution_plan.internal_steps),
                        "effective_steps": int(hires_schedule.effective_steps),
                        "denoising_strength": float(hires_execution_plan.denoising_strength),
                        "step_policy": str(hires_execution_plan.step_policy),
                        "noise_policy": str(hires_execution_plan.noise_policy),
                        "sampler_name": str(hires_request.sampler_name or ""),
                        "scheduler_name": str(hires_request.scheduler_name or ""),
                        "cfg_scale": float(hires_request.cfg_scale),
                        "cfg_rescale": float(hires_request.cfg_rescale),
                    },
                    "vae": {
                        "identity": str(vae_identity.get("identity") or vae_identity.get("source") or ""),
                        "source_kind": str(vae_identity.get("source_kind") or ""),
                        "sha256": str(vae_identity.get("sha256") or ""),
                        "encode_dtype": str(vae_contract.get("posterior", {}).get("dtype") or vae_contract.get("sampling_latent", {}).get("dtype") or ""),
                        "scaling_factor": vae_contract.get("sampling_latent", {}).get("scaling_factor", vae_contract.get("scaling_factor")),
                    },
                    "intermediate_artifacts": {
                        "save_base": bool(getattr(request, "hires_save_lowres", False)),
                        "save_upscaled_pre_denoise": bool(getattr(request, "hires_save_upscaled_pre_denoise", False)),
                        "save_vae_roundtrip": bool(getattr(request, "hires_save_vae_roundtrip", False)),
                        "hashes": {},
                    },
                    "replay_policy": {
                        "neural_requires_exact_sha256": True,
                        "missing_model_fallback_allowed": False,
                        "legacy_missing_step_policy": "proportional_tail_v1",
                    },
                }

            diagnostic_settings = (
                dict(request.diagnostics)
                if isinstance(getattr(request, "diagnostics", None), dict)
                else {}
            )
            if "diagnostic_decode_enabled" in diagnostic_settings:
                diagnostic_decode_enabled = bool(
                    diagnostic_settings.get("diagnostic_decode_enabled", False)
                )
                diagnostic_decode_source = "explicit_request"
            else:
                # Preserve compatibility for direct callers that explicitly request
                # predicted-x0 previews but predate the separate decode toggle.
                sampler_trace_settings = dict(
                    diagnostic_settings.get("sampler_trace") or {}
                )
                diagnostic_decode_enabled = bool(
                    sampler_trace_settings.get("predicted_x0_previews", False)
                )
                diagnostic_decode_source = (
                    "legacy_predicted_x0_request"
                    if diagnostic_decode_enabled
                    else "default_off"
                )
            if diagnostic_decode_enabled:
                trace_exports = self.memory_manager.run_stage(
                    stage="diagnostic_decode",
                    required={"vae"},
                    operation=lambda: performance.run(
                        "diagnostic_decode",
                        lambda: diagnostics.finish(
                            session,
                            sample_output,
                            decoder=self.systems.decoding.decode,
                            diagnostic_decode_enabled=True,
                        ),
                        operation_name="diagnostics.finish",
                    ),
                    request=request,
                )
            else:
                trace_exports = performance.run(
                    "diagnostic_finalize",
                    lambda: diagnostics.finish(
                        session,
                        sample_output,
                        decoder=None,
                        diagnostic_decode_enabled=False,
                    ),
                    operation_name="diagnostics.finish_without_decode",
                )

            images = None
            output_quality: dict[str, Any] = {
                "contract_version": "image-gen-output-quality-v1",
                "suspect": False,
                "classification": "not_decoded",
            }
            if not request.return_latents:
                final_decode_started = time.perf_counter() if pixel_hires_job else None
                decoded_images = diagnostics.run_stage(
                    session,
                    "decoding",
                    "decode",
                    lambda: self.memory_manager.run_stage(
                        stage="final_decode",
                        required={"vae"},
                        operation=lambda: performance.run(
                            "final_decode",
                            lambda: self.systems.decoding.decode(sample_output.latents),
                            operation_name="decoding.decode",
                        ),
                        request=request,
                    ),
                )
                if final_decode_started is not None:
                    pixel_hires_stage_timings["final_decode_ms"] = float(
                        (time.perf_counter() - final_decode_started) * 1000.0
                    )
                consume_decode_report = getattr(
                    self.systems.decoding, "consume_last_decode_diagnostics", None
                )
                if callable(consume_decode_report):
                    output_quality = dict(consume_decode_report() or {})
                else:
                    classification = classify_normalized_images(decoded_images)
                    output_quality = {
                        "contract_version": "image-gen-output-quality-v1",
                        "suspect": bool(classification.get("suspect")),
                        "classification": str(classification.get("classification") or "unknown"),
                        "reasons": list(classification.get("reasons") or []),
                        "final_latents": summarize_tensor(sample_output.latents),
                        "scaled_latents_entering_vae": {"available": False},
                        "raw_vae_output": {"available": False},
                        "normalized_images": dict(classification.get("summary") or {}),
                        "classification_details": classification,
                        "diagnostic_limitation": (
                            "The active decoding override does not expose raw VAE diagnostics."
                        ),
                    }
                diagnostics.record_tensor(
                    session,
                    "decoding.images_before_crop",
                    decoded_images,
                    system="decoding",
                    operation="decode",
                )
                if dimension_plan.crop_required:
                    images = diagnostics.run_stage(
                        session,
                        "decoding",
                        "center_crop",
                        lambda: self.memory_manager.observe_stage(
                            "final_output_conversion",
                            lambda: DecodingSystem.center_crop(
                                decoded_images,
                                width=final_output_width,
                                height=final_output_height,
                            ),
                        ),
                    )
                else:
                    images = self.memory_manager.observe_stage(
                        "final_output_conversion",
                        lambda: decoded_images,
                    )
                diagnostics.record_tensor(
                    session,
                    "decoding.images",
                    images,
                    system="decoding",
                    operation="decode",
                )
                capture_every_run = bool(
                    isinstance(getattr(request, "diagnostics", None), dict)
                    and request.diagnostics.get("capture_output_quality", False)
                )
                if bool(output_quality.get("suspect")) or capture_every_run:
                    output_quality["schedule"] = {
                        "scheduler_name": str(getattr(request, "scheduler_name", "") or ""),
                        "sampler_name": str(getattr(request, "sampler_name", "") or ""),
                        "requested_steps": int(getattr(request, "steps", 0) or 0),
                        "effective_steps": int(getattr(schedule, "effective_steps", 0) or 0),
                        "sigmas": summarize_tensor(getattr(schedule, "sigmas", None)),
                        "timesteps": summarize_tensor(getattr(schedule, "timesteps", None)),
                        "timestep_mapping": dict((getattr(schedule, "extra", {}) or {}).get("timestep_mapping") or {}),
                        "validated_settings": dict((getattr(schedule, "extra", {}) or {}).get("validated_settings") or {}),
                    }
                    output_quality["request"] = {
                        "seed": getattr(request, "seed", None),
                        "width": final_output_width,
                        "height": final_output_height,
                        "base_width": int(getattr(request, "width", 0) or 0),
                        "base_height": int(getattr(request, "height", 0) or 0),
                        "cfg_scale": float(getattr(request, "cfg_scale", 0.0) or 0.0),
                        "prompt_parser_name": str(getattr(request, "prompt_parser_name", "") or ""),
                        "hires_size_mode": str(getattr(request, "hires_size_mode", "") or ""),
                        "hires_enabled": bool(getattr(request, "hires_enabled", False)),
                    }
                    output_quality["memory_management"] = self.memory_manager.summary()
                    output_quality["capture_reason"] = "suspect_output" if output_quality.get("suspect") else "diagnostics_every_run"
                    artifact_path = write_output_quality_bundle(
                        root=session.config.artifacts_root,
                        run_id=session.run_id,
                        report=output_quality,
                        images=images,
                    )
                    output_quality["artifact_path"] = str(artifact_path)
                    session.request_extras["output_quality"] = dict(output_quality)
                    session.trace_exports["output_quality_bundle"] = str(artifact_path)
                    diagnostics.emit(
                        session,
                        "WARNING" if bool(output_quality.get("suspect")) else "INFO",
                        "decoding",
                        "output_quality",
                        "Decoded output diagnostics were saved.",
                        classification=output_quality.get("classification"),
                        artifact_path=str(artifact_path),
                        reasons=list(output_quality.get("reasons") or []),
                    )
                preview_writer = getattr(self.state, "extra", {}).get(
                    "live_preview_frame_writer"
                ) if self.state is not None else None
                if preview_writer is not None:
                    try:
                        preview_writer.write_final(
                            images,
                            total_steps=int(schedule.effective_steps),
                        )
                    except Exception as exc:
                        diagnostics.emit(
                            session,
                            "WARNING",
                            "live_preview",
                            "final_frame",
                            "Final live preview frame failed without blocking generation.",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )

            effective_denoising_contract = (
                self.systems.denoising.guided_epsilon_contract_metadata()
                if sample_output.extra.get("integration_mode_used") == "dpmpp_2m"
                else self.systems.denoising.contract_metadata()
            )
            for key in (
                "model_prediction_type",
                "prediction_conversion",
                "model_input_preconditioning",
                "cfg_rescale",
                "model_dtype",
                "solver_dtype",
            ):
                if key in sample_output.extra:
                    target = "prediction_type" if key == "model_prediction_type" else key
                    effective_denoising_contract[target] = sample_output.extra[key]

            preview_writer = getattr(self.state, "extra", {}).get(
                "live_preview_frame_writer"
            ) if self.state is not None else None
            if preview_writer is not None and callable(getattr(preview_writer, "close", None)):
                try:
                    preview_writer.close()
                except Exception:
                    pass
            live_preview_summary = (
                preview_writer.summary()
                if preview_writer is not None and callable(getattr(preview_writer, "summary", None))
                else {"enabled": False}
            )

            self.memory_manager.capture("generation_complete")
            attention_report = attention_backend_report(self.components.unet)
            attention_execution_by_pass = {
                "before_generation": provider_execution_before_generation,
                "after_base_pass": provider_execution_after_base,
                "after_hires_pass": provider_execution_after_hires,
                "base_pass": _execution_delta(
                    provider_execution_before_generation, provider_execution_after_base
                ),
                "hires_pass": _execution_delta(
                    provider_execution_after_base, provider_execution_after_hires
                ),
            }
            hires_metadata["attention_execution"] = dict(
                attention_execution_by_pass["hires_pass"]
            )
            memory_summary = self.memory_manager.summary()
            if pixel_hires_preflight is not None:
                hires_metadata["memory_forecast_vs_actual"] = compare_preflight_to_actual(
                    pixel_hires_preflight, memory_summary
                )
            if pixel_hires_job:
                hires_metadata["phase14n6_stage_timings_ms"] = dict(
                    pixel_hires_stage_timings
                )
                hires_metadata["phase14n6_transfer_count"] = len(
                    list(memory_summary.get("transfers") or [])
                )
            vae_memory_report = (
                self.systems.decoding.memory_control_report()
                if callable(
                    getattr(
                        self.systems.decoding,
                        "memory_control_report",
                        None,
                    )
                )
                else dict(base_vae_memory_controls)
            )
            runtime_execution = build_runtime_execution_record(
                startup_options=(
                    state_extra.get("runtime_startup_options")
                    if isinstance(state_extra, dict)
                    else None
                ),
                attention_report=attention_report,
                memory_summary=memory_summary,
                preview_policy=preview_policy_report,
                vae_report=vae_memory_report,
                hires_metadata=hires_metadata,
                runtime_job_settings=(
                    state_extra if isinstance(state_extra, dict) else None
                ),
                execution_device=str(self.device),
                external_fallback_reasons=(
                    [str(state_extra.get("cpu_fallback_reason"))]
                    if isinstance(state_extra, dict)
                    and state_extra.get("cpu_fallback_reason")
                    else None
                ),
            )
            conformance_source = (
                state_extra.get("runtime_replay_conformance_source")
                if isinstance(state_extra, dict)
                else None
            )
            runtime_execution["conformance_fingerprint"] = runtime_execution_fingerprint(
                runtime_execution
            )
            if isinstance(conformance_source, dict):
                conformance_report = compare_runtime_execution_records(
                    conformance_source,
                    runtime_execution,
                )
                runtime_execution.setdefault("replay", {})["conformance"] = conformance_report

            performance_summary = performance.finish(
                attention_backend=attention_report,
                memory_management=memory_summary,
                hires_enabled=hires_execution_plan.enabled,
            )
            if pixel_hires_job:
                performance_index = dict(performance_summary.get("stage_index") or {})
                final_decode_metric = dict(performance_index.get("final_decode") or {})
                if final_decode_metric.get("duration_ms") is not None:
                    pixel_hires_stage_timings["final_decode_ms"] = float(
                        final_decode_metric["duration_ms"]
                    )
                hires_metadata["phase14n6_stage_timings_ms"] = dict(
                    pixel_hires_stage_timings
                )
            result = GenerationResult(
                request=request,
                images=images,
                latents=sample_output.latents,
                conditioning=conditioning,
                schedule=schedule,
                sampler=sample_output,
                trace_exports=trace_exports,
                auxiliary_images=auxiliary_images,
                metadata={
                    "model_prediction_type": self.systems.denoising.prediction_type,
                    "component_placement": self.components.placement_metadata(),
                    "attention_backend": attention_report,
                    "attention_performance": performance_summary,
                    "attention_execution_by_pass": attention_execution_by_pass,
                    "denoising_contract": effective_denoising_contract,
                    "initial_noise_sigma": schedule.initial_sigma,
                    "output_owner": "txt2img.output_saver",
                    "output_system": "image_gen.systems.output.OutputSystem",
                    "runtime": "image_gen.runtime.GenerationPipeline",
                    "live_preview": live_preview_summary,
                    "preview_policy": preview_policy_report,
                    "memory_management": memory_summary,
                    "runtime_execution": runtime_execution,
                    "oom_recovery": dict(memory_summary.get("oom_recovery") or {}),
                    "attention_and_vae_memory_controls": {
                        "attention_slicing": str(
                            self.memory_manager.settings.attention_slicing
                        ),
                        "vae": vae_memory_report,
                    },
                    "prompt_parser": dict((conditioning.extra or {}).get("prompt_parser") or {}),
                    "prompt_shortcut_profile": dict((conditioning.extra or {}).get("prompt_shortcut_profile") or {}),
                    "prompt_translation": dict((conditioning.extra or {}).get("prompt_translation") or {}),
                    "prompt_contract": dict((conditioning.extra or {}).get("prompt_contract") or {}),
                    "regional_runtime": dict(
                        (getattr(request, "diagnostics", {}) or {}).get("regional_runtime")
                        or sample_output.extra.get("regional_runtime")
                        or {}
                    ),
                    "regional_runtime_passes": dict(
                        (getattr(request, "diagnostics", {}) or {}).get("regional_runtime_passes")
                        or {}
                    ),
                    "dimension_plan": dimension_plan.to_serializable_dict(),
                    "base_dimension_plan": base_dimension_plan.to_serializable_dict(),
                    "hires_fix": hires_metadata,
                    "diagnostic_decode": {
                        "enabled": diagnostic_decode_enabled,
                        "source": diagnostic_decode_source,
                        "final_output_decode_unaffected": True,
                    },
                    "output_dimensions": {
                        "width": final_output_width,
                        "height": final_output_height,
                    },
                    "output_quality": dict(output_quality),
                },
            )
            result = diagnostics.run_stage(
                session,
                "runtime",
                "result_validation",
                lambda: self._validate_result(request, result),
            )
            result.metadata["diagnostics"] = diagnostics.summary(session)
            if owns_session:
                result.metadata["diagnostics"] = diagnostics.complete(
                    session, result=result
                )
            return result
        except PixelHiresCancelled as exc:
            pixel_hires_cancelled_stage = exc.stage
            cleanup = self._cleanup_pixel_hires_scoped_runtime(
                reason=f"cancelled:{exc.stage}"
            )
            session.request_extras["pixel_hires_cancellation_cleanup"] = cleanup
            raise diagnostics.fail_unassigned(
                session, exc, system="runtime", operation="pixel_hires_cancelled"
            ) from exc
        except PipelineStageError:
            if self.memory_manager.failure_bundle:
                session.request_extras["memory_failure_bundle"] = self.memory_manager.failure_bundle
            raise
        except Exception as exc:
            if self.memory_manager.failure_bundle:
                session.request_extras["memory_failure_bundle"] = self.memory_manager.failure_bundle
            if hasattr(exc, "to_dict") and callable(exc.to_dict):
                try:
                    session.request_extras["prompt_parser_failure"] = exc.to_dict()
                except Exception:
                    pass
            raise diagnostics.fail_unassigned(
                session, exc, system="runtime", operation="generate"
            ) from exc
        finally:
            if pixel_hires_job:
                cleanup_reason = (
                    f"cancelled:{pixel_hires_cancelled_stage}"
                    if pixel_hires_cancelled_stage
                    else "pixel_hires_job_complete_or_failed"
                )
                cleanup = self._cleanup_pixel_hires_scoped_runtime(
                    reason=cleanup_reason
                )
                with suppress(Exception):
                    self.memory_manager.record_external_stage_telemetry(
                        "pixel_hires_cleanup",
                        {"event": "completed", **cleanup},
                    )


class CustomSDPipeline(GenerationPipeline):
    """Historical constructor preserved while delegating to system composition."""

    def __init__(
        self,
        components: PipelineComponents,
        prompt_adapter: PromptAdapterProtocol,
        scheduler_adapter: SchedulerAdapterProtocol,
        sampler_adapter: SamplerAdapterProtocol,
        state: Any | None = None,
        latent_scale_factor: int = 8,
        vae_scaling_factor: float = 0.18215,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        system_overrides: dict[str, Any] | None = None,
    ) -> None:
        root = PipelineCompositionRoot(
            components=components,
            prompt_adapter=prompt_adapter,
            scheduler_adapter=scheduler_adapter,
            sampler_adapter=sampler_adapter,
            latent_scale_factor=latent_scale_factor,
            vae_scaling_factor=vae_scaling_factor,
            device=device,
            dtype=dtype,
            system_overrides=system_overrides,
        )
        systems = root.create_systems()
        super().__init__(
            components=components,
            systems=systems,
            state=state,
            device=root.device,
            dtype=root.dtype,
            latent_scale_factor=latent_scale_factor,
            vae_scaling_factor=vae_scaling_factor,
        )
