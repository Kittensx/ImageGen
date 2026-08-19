from __future__ import annotations

import time
from typing import Any

from image_gen.systems.decoding import DecodingSystem
from image_gen.systems.diagnostics.output_quality import (
    classify_normalized_images,
    summarize_tensor,
    write_output_quality_bundle,
)
from image_gen.systems.outpainting import (
    composite_exact_protected_core,
    extract_outpaint_failure_stage,
    format_outpaint_failure,
    write_outpaint_diagnostic_artifacts,
)
from image_gen.contracts import format_hires_failure

from .context import GenerationContext


class DecodeStageMixin:
    def _run_decode_stage(self, ctx: GenerationContext):
        diagnostics = ctx.diagnostics
        session = ctx.session
        request = ctx.request
        performance = ctx.performance
        dimension_plan = ctx.dimension_plan
        final_output_width = ctx.final_output_width
        final_output_height = ctx.final_output_height
        hires_execution_plan = ctx.hires_execution_plan
        hires_metadata = ctx.hires_metadata
        outpaint_enabled = ctx.outpaint_enabled
        _outpaint_stage = ctx.outpaint_stage
        outpaint_canvas = ctx.outpaint_canvas
        outpaint_hook = ctx.outpaint_hook
        outpaint_masks = ctx.outpaint_masks
        outpaint_metadata = ctx.outpaint_metadata
        outpaint_plan = ctx.outpaint_plan
        outpaint_source = ctx.outpaint_source
        pixel_hires_job = ctx.pixel_hires_job
        pixel_hires_stage_timings = ctx.pixel_hires_stage_timings
        sample_output = ctx.sample_output
        schedule = ctx.schedule

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
            try:
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
            except Exception as exc:
                if outpaint_enabled and not extract_outpaint_failure_stage(exc):
                    raise RuntimeError(format_outpaint_failure(
                        "outpaint_decode", f"Outpaint decode failed: {exc}"
                    )) from exc
                raise
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
            decoded_height = int(decoded_images.shape[-2])
            decoded_width = int(decoded_images.shape[-1])
            hires_alignment_correction = bool(
                hires_execution_plan.enabled
                and (
                    decoded_width != final_output_width
                    or decoded_height != final_output_height
                )
                and (
                    hires_execution_plan.dimensions.alignment_correction_required
                    or decoded_width == int(hires_execution_plan.dimensions.internal_width)
                    or decoded_height == int(hires_execution_plan.dimensions.internal_height)
                )
            )
            if hires_alignment_correction:
                # Internal alignment is padding for model compatibility, not a
                # user-requested resize. Remove only the padded border so the
                # requested output keeps its exact pixel scale and neural detail.
                if decoded_width < final_output_width or decoded_height < final_output_height:
                    raise RuntimeError(format_hires_failure(
                        "final_alignment_crop",
                        "Aligned hires canvas is smaller than the requested final output.",
                        source=f"{decoded_width}x{decoded_height}",
                        target=f"{final_output_width}x{final_output_height}",
                    ))
                images = diagnostics.run_stage(
                    session,
                    "decoding",
                    "hires_final_alignment_crop",
                    lambda: self.memory_manager.observe_stage(
                        "final_output_conversion",
                        lambda: DecodingSystem.center_crop(
                            decoded_images,
                            width=final_output_width,
                            height=final_output_height,
                        ),
                    ),
                )
                hires_metadata["final_alignment_correction"] = {
                    "applied": True,
                    "source_width": decoded_width,
                    "source_height": decoded_height,
                    "target_width": final_output_width,
                    "target_height": final_output_height,
                    "operation": "center_crop",
                    "reason": "trim_architecture_alignment_padding_to_requested_hires_target",
                }
            elif dimension_plan.crop_required:
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
            if outpaint_enabled:
                assert outpaint_source is not None and outpaint_plan is not None
                decoded_before_source_composite = images
                images = _outpaint_stage(
                    "outpaint_source_composite",
                    lambda: composite_exact_protected_core(
                        decoded_before_source_composite, outpaint_source, outpaint_plan
                    ),
                )
                if outpaint_hook is not None:
                    outpaint_metadata["preservation"] = outpaint_hook.metadata()
                outpaint_metadata["final_dimensions"] = {
                    "width": int(images.shape[-1]),
                    "height": int(images.shape[-2]),
                }
                if bool(getattr(request, "outpaint_diagnostic_artifacts", False)):
                    assert outpaint_canvas is not None and outpaint_masks
                    artifact_path = write_outpaint_diagnostic_artifacts(
                        root=session.config.artifacts_root,
                        run_id=session.run_id,
                        source=outpaint_source,
                        canvas=outpaint_canvas,
                        generation_weight=outpaint_masks["generation_weight"],
                        decoded_before_composite=decoded_before_source_composite,
                        final_composite=images,
                        metadata=outpaint_metadata,
                    )
                    outpaint_metadata["diagnostic_artifact_path"] = str(artifact_path)
                    session.trace_exports["outpaint_prototype_artifacts"] = str(artifact_path)
                request.outpaint_prototype_record = dict(outpaint_metadata)
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

        ctx.diagnostic_decode_enabled = diagnostic_decode_enabled
        ctx.diagnostic_decode_source = diagnostic_decode_source
        ctx.images = images
        ctx.output_quality = output_quality
        ctx.trace_exports = trace_exports
