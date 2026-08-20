from __future__ import annotations

from typing import Any

from image_gen.runtime.hires_fix import resolve_hires_execution_plan
from image_gen.systems.memory import (
    PixelHiresAdmissionError,
    estimate_pixel_hires_preflight,
    resolve_preview_stage_policy,
)

from .context import GenerationContext


class RequestPreparationStageMixin:
    def _run_request_preparation_stage(self, ctx: GenerationContext):
        diagnostics = ctx.diagnostics
        session = ctx.session
        request = ctx.request
        outpaint_enabled = ctx.outpaint_enabled
        pixel_hires_preflight = ctx.pixel_hires_preflight

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
        hires_execution_plan = resolve_hires_execution_plan(
            request,
            dimension_multiple=int(
                getattr(
                    self.systems.latent_preparation,
                    "pixel_alignment_multiple",
                    self.latent_scale_factor,
                )
                or self.latent_scale_factor
            ),
            base_dimension_multiple=int(self.latent_scale_factor),
        )
        if outpaint_enabled:
            if hires_execution_plan.enabled or bool(getattr(request, "hires_enabled", False)):
                raise ValueError(
                    "Existing-image expansion cannot run with hires or .pth upscaling enabled."
                )
            if int(getattr(request, "batch_size", 1) or 1) != 1:
                raise ValueError("Existing-image expansion currently requires batch_size=1.")
            outpaint_alignment = int(
                getattr(
                    self.systems.latent_preparation,
                    "pixel_alignment_multiple",
                    self.latent_scale_factor,
                )
                or self.latent_scale_factor
            )
            if int(request.width) % outpaint_alignment or int(request.height) % outpaint_alignment:
                raise ValueError(
                    "Existing-image expansion requires target width and height divisible by "
                    f"the active model's {outpaint_alignment}-pixel alignment requirement so source geometry is not altered."
                )
        request.hires_dimension_plan_version = str(hires_execution_plan.dimensions.contract_version)
        request.hires_dimension_plan = hires_execution_plan.dimensions.to_dict()
        request.hires_steps = int(hires_execution_plan.steps)
        request.hires_denoising_strength = float(
            hires_execution_plan.denoising_strength
        )
        request.hires_step_policy = str(hires_execution_plan.step_policy)
        request.hires_strategy = str(hires_execution_plan.upscale_plan.strategy)
        request.hires_upscaler = str(hires_execution_plan.upscale_plan.legacy_value or hires_execution_plan.upscaler)
        request.hires_upscaler_id = str(hires_execution_plan.upscale_plan.upscaler_id or "")
        request.hires_expected_native_scale = int(hires_execution_plan.upscale_plan.native_scale or 0)
        request.hires_final_size_correction_filter = str(hires_execution_plan.upscale_plan.final_size_correction_filter)
        request.hires_aspect_policy = str(hires_execution_plan.upscale_plan.aspect_policy)
        request.hires_padding_mode = str(hires_execution_plan.upscale_plan.padding_mode)
        request.hires_blurred_edge_method = str(hires_execution_plan.upscale_plan.blurred_edge_method)
        request.hires_blurred_edge_compare_diagnostics = bool(
            hires_execution_plan.upscale_plan.blurred_edge_compare_diagnostics
        )
        hires_metadata: dict[str, Any] = hires_execution_plan.to_dict()
        pixel_hires_job = bool(
            hires_execution_plan.enabled
            and hires_execution_plan.upscale_plan.strategy == "pixel_neural"
        )
        # Persist this immediately so outer failure cleanup retains pre-refactor behavior
        # if admission/preflight raises after the job has been classified.
        ctx.pixel_hires_job = pixel_hires_job
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

        ctx.dimension_plan = dimension_plan
        ctx.base_dimension_plan = base_dimension_plan
        ctx.final_output_width = final_output_width
        ctx.final_output_height = final_output_height
        ctx.hires_execution_plan = hires_execution_plan
        ctx.hires_metadata = hires_metadata
        ctx.pixel_hires_job = pixel_hires_job
        ctx.pixel_hires_preflight = pixel_hires_preflight
        ctx.state_extra = state_extra
        ctx.preview_mode = preview_mode
        ctx.base_preview_policy = base_preview_policy
        ctx.preview_policy_report = preview_policy_report
