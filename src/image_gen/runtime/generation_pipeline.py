from __future__ import annotations

from contextlib import suppress
import inspect
from typing import Any

import torch

from modules.attention_backend import temporary_attention_slicing
from modules.attention_runtime import get_execution_evidence
from modules.pipeline.conditioning_utils import (
    resolve_step_conditioning as resolve_step_conditioning_util,
)
from modules.pipeline.live_preview import build_live_preview_sink
from modules.pipeline.live_preview_decode import create_live_preview_writer

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
from image_gen.runtime.composition import GenerationSystems, PipelineCompositionRoot
from image_gen.runtime.generation.conditioning_stage import ConditioningStageMixin
from image_gen.runtime.generation.context import GenerationContext
from image_gen.runtime.generation.decode_stage import DecodeStageMixin
from image_gen.runtime.generation.denoise_stage import BaseDenoiseStageMixin
from image_gen.runtime.generation.finalize_stage import FinalizeStageMixin, _execution_delta
from image_gen.runtime.generation.hires_stage import (
    HiresStageMixin,
    _target_correction_records_match,
)
from image_gen.runtime.generation.latent_stage import LatentStageMixin
from image_gen.runtime.generation.request_stage import RequestPreparationStageMixin
from image_gen.runtime.performance_metrics import GenerationPerformanceRecorder
from image_gen.systems.diagnostics import (
    DiagnosticSession,
    DiagnosticsSystem,
    PipelineStageError,
)
from image_gen.systems.memory import (
    AdaptiveComponentMemoryManager,
    PixelHiresCancelled,
    StageRecoveryContract,
)
from image_gen.systems.outpainting import (
    StrictLatentPreservationHook,
    extract_outpaint_failure_stage,
    format_outpaint_failure,
)
from image_gen.systems.upscaling import StandaloneNeuralUpscaler, UpscaleRequest


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





class GenerationPipeline(
    RequestPreparationStageMixin,
    ConditioningStageMixin,
    LatentStageMixin,
    BaseDenoiseStageMixin,
    HiresStageMixin,
    DecodeStageMixin,
    FinalizeStageMixin,
):
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
        vae_shift_factor: float = 0.0,
        memory_manager: AdaptiveComponentMemoryManager | None = None,
    ) -> None:
        self.components = components
        self.systems = systems
        self.state = state
        self.device = device
        self.dtype = dtype
        self.latent_scale_factor = int(latent_scale_factor)
        self.vae_scaling_factor = float(vae_scaling_factor)
        self.vae_shift_factor = float(vae_shift_factor)
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

    def _conditioning_required_components(self) -> set[str]:
        required = {"text_encoder"}
        if getattr(self.components, "text_encoder_2", None) is not None:
            required.add("text_encoder_2")
        if getattr(self.components, "text_encoder_3", None) is not None:
            required.add("text_encoder_3")
        return required

    def _conditioning_preferred_components(self) -> set[str]:
        # For SDXL CPU-first operation, never opportunistically pull the UNet
        # onto an 8 GB card while both text encoders are active. Legacy single-
        # encoder models retain the historical balanced preference.
        if getattr(self.components, "text_encoder_2", None) is not None:
            return set()
        return {"unet"}

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
                expected_height = int(
                    hires_plan.get("final_height")
                    or hires_plan.get("requested_height")
                    or hires_plan.get("effective_height")
                    or request.height
                )
                expected_width = int(
                    hires_plan.get("final_width")
                    or hires_plan.get("requested_width")
                    or hires_plan.get("effective_width")
                    or request.width
                )
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
        outpaint_enabled = bool(getattr(request, "outpaint_prototype_enabled", False))
        outpaint_source: torch.Tensor | None = None
        outpaint_canvas: torch.Tensor | None = None
        outpaint_masks: dict[str, torch.Tensor] = {}
        outpaint_latent_mask: torch.Tensor | None = None
        outpaint_plan = None
        outpaint_hook: StrictLatentPreservationHook | None = None
        outpaint_prompt_contract: dict[str, Any] = {}
        outpaint_metadata: dict[str, Any] = {
            "enabled": outpaint_enabled,
            "prototype": True,
            "pth_upscaler_used": False,
        }

        def _outpaint_stage(stage: str, operation: Any) -> Any:
            try:
                return operation()
            except Exception as exc:
                existing = extract_outpaint_failure_stage(exc)
                if existing:
                    raise
                raise RuntimeError(format_outpaint_failure(stage, str(exc))) from exc
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


        ctx = GenerationContext(
            request=request,
            diagnostics=diagnostics,
            session=session,
            owns_session=owns_session,
            performance=performance,
            provider_execution_before_generation=provider_execution_before_generation,
            provider_execution_after_base=provider_execution_after_base,
            provider_execution_after_hires=provider_execution_after_hires,
            auxiliary_images=auxiliary_images,
            outpaint_enabled=outpaint_enabled,
            outpaint_source=outpaint_source,
            outpaint_canvas=outpaint_canvas,
            outpaint_masks=outpaint_masks,
            outpaint_latent_mask=outpaint_latent_mask,
            outpaint_plan=outpaint_plan,
            outpaint_hook=outpaint_hook,
            outpaint_prompt_contract=outpaint_prompt_contract,
            outpaint_metadata=outpaint_metadata,
            outpaint_stage=_outpaint_stage,
            configure_vae_memory=configure_vae_memory,
            base_vae_memory_controls=base_vae_memory_controls,
            pixel_hires_job=pixel_hires_job,
            pixel_hires_preflight=pixel_hires_preflight,
            pixel_hires_stage_timings=pixel_hires_stage_timings,
            pixel_hires_cancelled_stage=pixel_hires_cancelled_stage,
        )

        try:
            self._run_request_preparation_stage(ctx)
            self._run_conditioning_stage(ctx)
            self._run_latent_stage(ctx)
            self._run_base_denoise_stage(ctx)
            self._run_hires_stage(ctx)
            self._run_decode_stage(ctx)
            return self._run_finalize_stage(ctx)
        except PixelHiresCancelled as exc:
            ctx.pixel_hires_cancelled_stage = exc.stage
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
            if ctx.pixel_hires_job:
                cleanup_reason = (
                    f"cancelled:{ctx.pixel_hires_cancelled_stage}"
                    if ctx.pixel_hires_cancelled_stage
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
            latent_scale_factor=root.latent_scale_factor,
            vae_scaling_factor=root.vae_scaling_factor,
            vae_shift_factor=root.vae_shift_factor,
        )
