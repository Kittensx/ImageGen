from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch

from modules.attention_backend import temporary_attention_slicing
from modules.attention_runtime import get_execution_evidence

from image_gen.contracts import (
    build_hires_correction_audit,
    build_hires_correction_fingerprint,
    extract_hires_failure_stage,
    format_hires_failure,
)
from image_gen.contracts.vae_provenance import read_vae_provenance
from image_gen.runtime.hires_fix import (
    BUILTIN_PIXEL_RESIZE_ID,
    BUILTIN_PIXEL_RESIZE_SHA256,
    PixelNeuralHiresSourceResult,
    add_hires_noise,
    build_hires_request,
    hires_schedule_baseline_metadata,
    validate_recorded_hires_vae_identity,
)
from image_gen.systems.decoding import DecodingSystem
from image_gen.systems.image_conditioning import (
    build_image_conditioned_schedule,
    build_schedule_fingerprint_record,
    build_schedule_replay_record,
    build_vae_execution_fingerprint,
    compare_schedule_conformance,
    image_conditioned_forward_process_metadata,
    noise_stream_metadata,
    rehydrate_schedule_replay_record,
    require_qualified_hires_pair,
    vae_encode_for_sampling,
    vae_round_trip_from_encoded_for_diagnostics,
)
from image_gen.systems.memory import (
    cancellation_requested,
    perform_pre_hires_cleanup,
    raise_if_pixel_hires_cancelled,
    resolve_hires_memory_behavior,
    resolve_preview_stage_policy,
    stage_tensor_to_host,
)
from image_gen.systems.outpainting import extract_outpaint_failure_stage
from image_gen.systems.sampling import SamplingSystem
from image_gen.systems.scheduling import SchedulingSystem
from image_gen.systems.upscaling import UpscaleRequest, plan_target_correction, resize_exact

from .context import GenerationContext

def _target_correction_records_match(recorded: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare deterministic correction records across the v1 schema's null-field addition."""

    expected = dict(recorded or {})
    observed = dict(actual or {})
    if "resize_scale" not in expected and observed.get("resize_scale") is None:
        expected["resize_scale"] = None
    return expected == observed


class HiresStageMixin:
    def _run_hires_stage(self, ctx: GenerationContext) -> None:
        diagnostics = ctx.diagnostics
        session = ctx.session
        request = ctx.request
        performance = ctx.performance
        hires_execution_plan = ctx.hires_execution_plan
        configure_vae_memory = ctx.configure_vae_memory
        base_vae_memory_controls = ctx.base_vae_memory_controls
        pixel_hires_job = ctx.pixel_hires_job
        pixel_hires_stage_timings = ctx.pixel_hires_stage_timings
        preview_mode = ctx.preview_mode
        preview_policy_report = ctx.preview_policy_report
        state_extra = ctx.state_extra
        auxiliary_images = ctx.auxiliary_images
        base_dimension_plan = ctx.base_dimension_plan
        dimension_plan = ctx.dimension_plan
        final_output_width = ctx.final_output_width
        final_output_height = ctx.final_output_height
        latents = ctx.latents
        schedule = ctx.schedule
        conditioning = ctx.conditioning
        sample_output = ctx.sample_output
        raw_model_fn = ctx.raw_model_fn
        guided_model_fn = ctx.guided_model_fn
        hires_metadata = ctx.hires_metadata
        provider_execution_after_hires = ctx.provider_execution_after_hires

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
            hires_scheduler_domain = (
                "flow_match"
                if bool(getattr(self.systems.denoising, "is_flow_match", False))
                else "sigma_additive"
            )
            hires_dimension_plan = diagnostics.run_stage(
                session,
                "hires",
                "plan_dimensions",
                lambda: self.systems.latent_preparation.plan_dimensions(hires_request),
            )
            # The second pass runs on the alignment-safe internal canvas,
            # but the user-requested hires target remains the final output
            # contract. Do not let the padded internal width/height escape.
            final_output_width = int(hires_execution_plan.dimensions.final_width)
            final_output_height = int(hires_execution_plan.dimensions.final_height)
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
            if hires_execution_plan.upscale_plan.strategy in {"pixel_neural", "pixel_resize"}:
                if hires_execution_plan.upscale_plan.strategy == "pixel_neural":
                    if self.neural_upscaler is None or self.neural_upscaler_registry is None:
                        selected_id = str(hires_execution_plan.upscale_plan.upscaler_id or "")
                        raise RuntimeError(
                            f"Pixel-neural hires requested {selected_id!r}, but the owned standalone upscaler runtime is not configured."
                        )
                    descriptor = hires_execution_plan.upscale_plan.descriptor
                    if descriptor is None:
                        raise RuntimeError("Pixel-neural hires resolved without an exact descriptor.")
                else:
                    descriptor = SimpleNamespace(
                        upscaler_id=BUILTIN_PIXEL_RESIZE_ID,
                        sha256=BUILTIN_PIXEL_RESIZE_SHA256,
                        native_scale=0,
                        tile_supported=False,
                    )

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
                    final_size_correction_filter=str(hires_execution_plan.upscale_plan.final_size_correction_filter),
                    aspect_policy=str(hires_execution_plan.upscale_plan.aspect_policy),
                    padding_mode=str(hires_execution_plan.upscale_plan.padding_mode),
                    blurred_edge_method=str(hires_execution_plan.upscale_plan.blurred_edge_method),
                    blurred_edge_compare_diagnostics=bool(
                        hires_execution_plan.upscale_plan.blurred_edge_compare_diagnostics
                    ),
                    dtype_policy="auto",
                    device_policy="auto",
                    allow_tiling=bool(hires_execution_plan.upscale_plan.allow_tiling),
                    allow_oom_retry=True,
                    host_transfer_non_blocking=bool(pixel_source_cpu.is_pinned()),
                )
                stage_started = time.perf_counter()
                try:
                    if hires_execution_plan.upscale_plan.strategy == "pixel_resize":
                        target_width = int(hires_execution_plan.upscale_plan.target_width)
                        target_height = int(hires_execution_plan.upscale_plan.target_height)
                        correction = plan_target_correction(
                            source_width=int(pixel_source_cpu.shape[-1]),
                            source_height=int(pixel_source_cpu.shape[-2]),
                            target_width=target_width,
                            target_height=target_height,
                            aspect_policy="stretch",
                            final_size_correction_filter="bicubic",
                            padding_mode=str(hires_execution_plan.upscale_plan.padding_mode),
                        ).to_dict()
                        resized = resize_exact(
                            pixel_source_cpu,
                            target_width=target_width,
                            target_height=target_height,
                            resize_filter="bicubic",
                        )
                        upscale_result = SimpleNamespace(
                            images=resized,
                            metadata={
                                "upscaler_id": BUILTIN_PIXEL_RESIZE_ID,
                                "upscaler_sha256": BUILTIN_PIXEL_RESIZE_SHA256,
                                "upscaler_native_scale": 0,
                                "actual_native_width": int(pixel_source_cpu.shape[-1]),
                                "actual_native_height": int(pixel_source_cpu.shape[-2]),
                                "native_output_shape": list(pixel_source_cpu.shape),
                                "target_width": target_width,
                                "target_height": target_height,
                                "aspect_policy": "stretch",
                                "padding_mode": str(hires_execution_plan.upscale_plan.padding_mode),
                                "final_size_correction_filter": "bicubic",
                                "target_correction": correction,
                                "builtin_resize": True,
                            },
                        )
                    else:
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
                except Exception as exc:
                    existing_stage = extract_hires_failure_stage(exc)
                    label = "Built-in pixel resize" if hires_execution_plan.upscale_plan.strategy == "pixel_resize" else "Pixel-neural hires"
                    message = str(exc) if existing_stage else format_hires_failure(
                        "pth_native_inference",
                        f"{label} failed during source preparation: {exc}",
                        model=descriptor.upscaler_id,
                        native_scale=f"x{int(descriptor.native_scale)}",
                        requested_target=f"{int(hires_execution_plan.dimensions.final_width)}x{int(hires_execution_plan.dimensions.final_height)}",
                        correction_canvas=f"{int(hires_execution_plan.upscale_plan.target_width)}x{int(hires_execution_plan.upscale_plan.target_height)}",
                        aspect_policy=hires_execution_plan.upscale_plan.aspect_policy,
                    )
                    raise RuntimeError(message) from exc
                pixel_hires_stage_timings["neural_upscale_ms"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                upscale_metadata = dict(upscale_result.metadata or {})
                executed_id = str(upscale_metadata.get("upscaler_id") or "")
                executed_hash = str(upscale_metadata.get("upscaler_sha256") or "").casefold()
                if executed_id != descriptor.upscaler_id or executed_hash != descriptor.sha256.casefold():
                    raise RuntimeError(format_hires_failure(
                        "pth_native_inference",
                        "The executed neural upscaler identity/hash did not match the resolved hires plan.",
                        model=descriptor.upscaler_id,
                    ))
                actual_native_width = int(upscale_metadata.get("actual_native_width") or 0)
                actual_native_height = int(upscale_metadata.get("actual_native_height") or 0)
                native_shape = list(upscale_metadata.get("native_output_shape") or [])
                if len(native_shape) >= 2:
                    shape_width = int(native_shape[-1])
                    shape_height = int(native_shape[-2])
                    if actual_native_width < 1:
                        actual_native_width = shape_width
                    if actual_native_height < 1:
                        actual_native_height = shape_height
                    if shape_width < 1 or shape_height < 1 or shape_width != actual_native_width or shape_height != actual_native_height:
                        raise RuntimeError(format_hires_failure(
                            "native_dimension_verification",
                            "Native .pth output dimensions are internally inconsistent.",
                            model=descriptor.upscaler_id,
                            predicted=f"{int(hires_execution_plan.upscale_plan.predicted_native_width)}x{int(hires_execution_plan.upscale_plan.predicted_native_height)}",
                            actual=f"{actual_native_width}x{actual_native_height}",
                        ))
                elif actual_native_width < 0 or actual_native_height < 0:
                    raise RuntimeError(format_hires_failure(
                        "native_dimension_verification",
                        "Native .pth output dimensions are invalid.",
                        model=descriptor.upscaler_id,
                        actual=f"{actual_native_width}x{actual_native_height}",
                    ))
                recorded_target_correction = dict(
                    getattr(request, "hires_recorded_target_correction", None) or {}
                )
                actual_target_correction = dict(upscale_metadata.get("target_correction") or {})
                if recorded_target_correction and not _target_correction_records_match(
                    recorded_target_correction, actual_target_correction
                ):
                    raise RuntimeError(format_hires_failure(
                        "target_aspect_correction",
                        "Recorded hires target-correction geometry does not match the current deterministic correction plan.",
                        model=descriptor.upscaler_id,
                        aspect_policy=hires_execution_plan.upscale_plan.aspect_policy,
                    ))
                upscale_metadata["correction_audit"] = build_hires_correction_audit(upscale_metadata)
                correction_fingerprint_enabled = bool(
                    getattr(request, "hires_correction_fingerprint_enabled", False)
                )
                if correction_fingerprint_enabled:
                    correction_fingerprint = build_hires_correction_fingerprint(
                        upscaler_id=executed_id,
                        upscaler_sha256=executed_hash,
                        native_scale=int(upscale_metadata.get("upscaler_native_scale") or descriptor.native_scale or 0),
                        actual_native_width=actual_native_width,
                        actual_native_height=actual_native_height,
                        target_width=int(hires_execution_plan.upscale_plan.target_width),
                        target_height=int(hires_execution_plan.upscale_plan.target_height),
                        aspect_policy=str(hires_execution_plan.upscale_plan.aspect_policy),
                        padding_mode=str(hires_execution_plan.upscale_plan.padding_mode),
                        resolved_filter=str(actual_target_correction.get("final_size_correction_filter_resolved") or "none"),
                        target_correction=actual_target_correction,
                        dimension_plan_version=str(hires_execution_plan.dimensions.contract_version),
                    )
                    recorded_fingerprint = dict(
                        getattr(request, "hires_recorded_correction_fingerprint", None) or {}
                    )
                    if (
                        recorded_fingerprint
                        and str(recorded_fingerprint.get("sha256") or "").casefold()
                        != str(correction_fingerprint.get("sha256") or "").casefold()
                    ):
                        raise RuntimeError(format_hires_failure(
                            "target_aspect_correction",
                            "Recorded hires correction fingerprint does not match the current deterministic correction contract.",
                            recorded_sha256=recorded_fingerprint.get("sha256"),
                            actual_sha256=correction_fingerprint.get("sha256"),
                        ))
                    upscale_metadata["correction_fingerprint"] = correction_fingerprint
                    request.hires_recorded_correction_fingerprint = dict(correction_fingerprint)
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
                try:
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
                                    shift_factor=self.vae_shift_factor,
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
                except Exception as exc:
                    raise RuntimeError(format_hires_failure(
                        "vae_encode",
                        f"Hires VAE encode failed: {exc}",
                        target=f"{int(hires_execution_plan.upscale_plan.target_width)}x{int(hires_execution_plan.upscale_plan.target_height)}",
                        model=descriptor.upscaler_id,
                    )) from exc
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
                                shift_factor=self.vae_shift_factor,
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
                    required=self._conditioning_required_components(),
                    preferred=self._conditioning_preferred_components(),
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

            if hires_execution_plan.upscale_plan.strategy not in {"pixel_neural", "pixel_resize"}:
                raise RuntimeError("Unsupported hires source preparation strategy.")
            if hires_latents is None:
                raise RuntimeError("Hires source preparation did not produce sampling latents.")

            hires_metadata["noise_stream"] = noise_stream_metadata(
                list(request.resolved_seeds),
                hires_execution_plan.noise_policy,
            )
            hires_metadata["noise_forward_process"] = image_conditioned_forward_process_metadata(
                hires_schedule,
                scheduler_domain=hires_scheduler_domain,
            )
            hires_schedule.metadata["hires_noise_forward_process"] = dict(
                hires_metadata["noise_forward_process"]
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
                        scheduler_domain=hires_scheduler_domain,
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
            try:
                sample_output = diagnostics.run_stage(
                    session,
                    "hires",
                    "sample_second_pass",
                    _run_hires_second_pass,
                )
            except Exception as exc:
                raise RuntimeError(format_hires_failure(
                    "second_pass_diffusion",
                    f"Hires second-pass diffusion failed: {exc}",
                    sampler=hires_request.sampler_name,
                    scheduler=hires_request.scheduler_name,
                    target=f"{int(hires_execution_plan.upscale_plan.target_width)}x{int(hires_execution_plan.upscale_plan.target_height)}",
                )) from exc
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
                    "noise_forward_process": dict(
                        hires_metadata.get("noise_forward_process") or {}
                    ),
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
                    "requested_final_width": int(hires_execution_plan.dimensions.final_width),
                    "requested_final_height": int(hires_execution_plan.dimensions.final_height),
                    "predicted_native_width": int(upscale_contract.get("predicted_native_width") or hires_execution_plan.upscale_plan.predicted_native_width),
                    "predicted_native_height": int(upscale_contract.get("predicted_native_height") or hires_execution_plan.upscale_plan.predicted_native_height),
                    "actual_native_width": int(upscale_contract.get("actual_native_width") or 0),
                    "actual_native_height": int(upscale_contract.get("actual_native_height") or 0),
                    "native_dimension_match": bool(upscale_contract.get("native_dimension_match", False)),
                    "aspect_policy": str(hires_execution_plan.upscale_plan.aspect_policy),
                    "padding_mode": str(hires_execution_plan.upscale_plan.padding_mode),
                    "blurred_edge_method": str(hires_execution_plan.upscale_plan.blurred_edge_method),
                    "blurred_edge_compare_diagnostics": bool(
                        hires_execution_plan.upscale_plan.blurred_edge_compare_diagnostics
                    ),
                    "final_size_correction_filter": str(hires_execution_plan.upscale_plan.final_size_correction_filter),
                    "target_correction": dict(upscale_contract.get("target_correction") or {}),
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


        ctx.dimension_plan = dimension_plan
        ctx.final_output_width = final_output_width
        ctx.final_output_height = final_output_height
        ctx.latents = latents
        ctx.schedule = schedule
        ctx.conditioning = conditioning
        ctx.sample_output = sample_output
        ctx.provider_execution_after_hires = provider_execution_after_hires
