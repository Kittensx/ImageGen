from __future__ import annotations

from dataclasses import replace

import torch

from image_gen.contracts.vae_provenance import read_vae_provenance
from image_gen.systems.image_conditioning import (
    build_image_conditioned_schedule,
    vae_encode_for_sampling,
)
from image_gen.systems.outpainting import (
    OUTPAINT_PRESERVATION_STRATEGY,
    StrictLatentPreservationHook,
    build_outpaint_audit,
    build_outpaint_geometry_fingerprint,
    build_outpaint_inference_fingerprint,
    build_outpaint_mask_transform_contract,
    build_outpaint_noise,
    build_outpaint_source_handoff_record,
    embed_live_source_latent,
    format_outpaint_failure,
    initialize_outpaint_latents,
    inspect_live_latent_alignment,
    normalize_outpaint_source_handoff_mode,
    resize_mask_to_latent,
    resolve_outpaint_source_handoff,
)

from .context import GenerationContext


class LatentStageMixin:
    def _run_latent_stage(self, ctx: GenerationContext):
        diagnostics = ctx.diagnostics
        session = ctx.session
        request = ctx.request
        outpaint_enabled = ctx.outpaint_enabled
        outpaint_metadata = ctx.outpaint_metadata
        _outpaint_stage = ctx.outpaint_stage
        outpaint_canvas = ctx.outpaint_canvas
        outpaint_masks = ctx.outpaint_masks
        outpaint_plan = ctx.outpaint_plan
        source_metadata = ctx.source_metadata
        outpaint_latent_mask = ctx.outpaint_latent_mask
        outpaint_hook = ctx.outpaint_hook

        if outpaint_enabled:
            strength = float(getattr(request, "outpaint_denoising_strength", 0.70))
            # Build a sufficiently long full schedule, then slice it with the
            # already-qualified image-conditioned schedule contract.
            safe_strength = min(max(strength, 0.01), 0.999)
            internal_steps = max(int(request.steps), int(int(request.steps) / safe_strength))
            schedule_request = replace(request, steps=internal_steps)
            full_schedule = _outpaint_stage(
                "outpaint_noise_initialization",
                lambda: diagnostics.run_stage(
                    session,
                    "scheduling",
                    "build_outpaint_full_schedule",
                    lambda: self.memory_manager.observe_stage(
                        "scheduler_construction",
                        lambda: self.systems.scheduling.build(schedule_request, self.state),
                    ),
                ),
            )
            image_conditioned_schedule = _outpaint_stage(
                "outpaint_noise_initialization",
                lambda: build_image_conditioned_schedule(
                    full_schedule,
                    requested_refinement_steps=int(request.steps),
                    denoising_strength=strength,
                    step_policy="a1111_fixed_steps_v1",
                ),
            )
            # Diagnostics, live preview, latent initialization, and samplers all
            # consume the active SchedulerOutput.  Keep the wrapper separately
            # for image-conditioned provenance/replay metadata.
            schedule = image_conditioned_schedule.active_schedule
            diagnostics.update_schedule(session, schedule)
            self._configure_live_preview_sink(request, schedule, diagnostics, session)
            assert outpaint_canvas is not None
            vae_identity = read_vae_provenance(self.components.vae)
            encoded = _outpaint_stage(
                "outpaint_vae_encode",
                lambda: self.memory_manager.run_stage(
                    stage="outpaint_vae_encode",
                    required={"vae"},
                    operation=lambda: vae_encode_for_sampling(
                        image=outpaint_canvas,
                        vae=self.systems.decoding,
                        scaling_factor=float(self.systems.decoding.vae_scaling_factor),
                        deterministic=True,
                        target_width=int(request.width),
                        target_height=int(request.height),
                        allow_center_crop=False,
                        vae_identity=vae_identity,
                    ),
                    request=request,
                ),
            )
            assert outpaint_masks
            runtime_live_source_latent = getattr(request, "_outpaint_runtime_source_latent", None)
            requested_handoff = str(
                getattr(
                    request,
                    "_outpaint_runtime_source_handoff_requested",
                    getattr(request, "outpaint_source_handoff_mode", "pixel_vae_reencode"),
                )
                or "pixel_vae_reencode"
            )
            requested_handoff_stable = normalize_outpaint_source_handoff_mode(requested_handoff, default="auto")
            if torch.is_tensor(runtime_live_source_latent):
                alignment = _outpaint_stage(
                    "outpaint_source_handoff",
                    lambda: inspect_live_latent_alignment(
                        plan=outpaint_plan,
                        source_latent=runtime_live_source_latent,
                        expanded_latent=encoded.latents,
                        latent_scale_factor=self.latent_scale_factor,
                    ),
                )
            else:
                alignment = {
                    "contract_version": "phase14n13p3-live-shape-expansion-v1",
                    "latent_scale_factor": int(self.latent_scale_factor),
                    "aligned": False,
                    "reasons": ["live base latent unavailable"],
                    "source_pixel_placement": {
                        "x": int(outpaint_plan.source_x),
                        "y": int(outpaint_plan.source_y),
                    },
                    "source_latent_placement": {"x": None, "y": None},
                    "source_latent_shape": None,
                    "expanded_latent_shape": list(encoded.latents.shape),
                }
            handoff = _outpaint_stage(
                "outpaint_source_handoff",
                lambda: resolve_outpaint_source_handoff(
                    requested_mode=requested_handoff,
                    alignment=alignment,
                    live_latent_available=torch.is_tensor(runtime_live_source_latent),
                ),
            )
            actual_handoff_stable = normalize_outpaint_source_handoff_mode(handoff.get("actual"))
            source_origin = (
                "fresh_generation" if torch.is_tensor(runtime_live_source_latent) else "external_image"
            )
            source_handoff_contract = build_outpaint_source_handoff_record(
                requested_mode=requested_handoff_stable,
                actual_mode=actual_handoff_stable,
                source_origin=source_origin,
                alignment=alignment,
                fallback_reason=str(handoff.get("fallback_reason") or ""),
                source_asset=source_metadata,
                preservation_reference_source=(
                    "fresh_txt2img_sampled_latent"
                    if actual_handoff_stable == "live_txt2img_latent_v1"
                    else "vae_reencoded_expanded_canvas"
                ),
                source_was_vae_reencoded_for_protected_latent=(actual_handoff_stable != "live_txt2img_latent_v1"),
                live_source_latent_reused=(actual_handoff_stable == "live_txt2img_latent_v1"),
            )
            reference_latents = encoded.latents
            if handoff["actual"] == "live_latent":
                reference_latents = _outpaint_stage(
                    "outpaint_latent_canvas_build",
                    lambda: embed_live_source_latent(
                        encoded.latents,
                        runtime_live_source_latent,
                        plan=outpaint_plan,
                        latent_scale_factor=self.latent_scale_factor,
                    ),
                )
            outpaint_latent_mask = resize_mask_to_latent(
                outpaint_masks["generation_weight"], reference_latents
            )
            noise, noise_metadata = _outpaint_stage(
                "outpaint_noise_initialization",
                lambda: build_outpaint_noise(reference_latents, seeds=list(request.resolved_seeds)),
            )
            latents = _outpaint_stage(
                "outpaint_noise_initialization",
                lambda: initialize_outpaint_latents(
                    reference_latents,
                    generation_weight=outpaint_latent_mask,
                    noise=noise,
                    initial_sigma=float(schedule.initial_sigma),
                    strategy=str(getattr(request, "outpaint_latent_strategy", "noise_only_new_regions_v1")),
                ),
            )
            outpaint_hook = StrictLatentPreservationHook(reference_latents, outpaint_latent_mask)
            state_extra_for_hook = getattr(self.state, "extra", None) if self.state is not None else None
            if not isinstance(state_extra_for_hook, dict):
                raise RuntimeError(format_outpaint_failure(
                    "outpaint_sampling",
                    "Shared runtime state does not expose an extra mapping for latent preservation.",
                ))
            state_extra_for_hook["sampling_latent_step_hook"] = outpaint_hook
            # The VAE encode workspace is no longer needed once the source
            # canvas latent has been created.  The normal sampling planner only
            # reasons about components named by the sampling stage; in fast
            # preview mode that does not include the VAE, so it can otherwise
            # remain resident and carry its allocator workspace into UNet
            # sampling.  On 8 GiB-class GPUs that lost headroom is enough to
            # turn an otherwise viable 512x512 outpaint pass into a late-step
            # OOM.  Make the image-conditioned boundary explicit and measurable.
            outpaint_pre_sampling_actions = self.memory_manager.offload_inactive_components(
                ("vae", "text_encoder"),
                stage="pre_outpaint_sampling_cleanup",
                reason="outpaint source VAE encode complete; reserve VRAM for masked sampling",
            )
            self.memory_manager.release_cuda_cache(
                stage="pre_outpaint_sampling_cleanup",
                reason="outpaint pre-sampling allocator cleanup",
            )
            latent_mask_contract = build_outpaint_mask_transform_contract(
                pixel_mask_shape=list(outpaint_masks["generation_weight"].shape),
                latent_mask_shape=list(outpaint_latent_mask.shape),
                resize_method="bilinear_align_corners_false",
            )
            schedule_boundary = {
                "requested_steps": int(request.steps),
                "internal_schedule_steps": int(image_conditioned_schedule.internal_schedule_steps),
                "effective_steps": int(schedule.effective_steps),
                "start_sigma": float(image_conditioned_schedule.start_sigma),
                "start_timestep": image_conditioned_schedule.start_timestep,
                "start_index": int(image_conditioned_schedule.start_index),
            }
            geometry_fingerprint = build_outpaint_geometry_fingerprint(
                canvas=outpaint_metadata.get("canvas", {}),
                feather_px=int(getattr(request, "outpaint_feather_px", 24)),
                preservation_mode=str(outpaint_metadata.get("preservation_mode") or OUTPAINT_PRESERVATION_STRATEGY),
                mask_strategy=str(outpaint_metadata.get("mask_strategy") or "preserve_generate_feather_v1"),
                context_seed_mode=str(getattr(request, "outpaint_context_seed_mode", "edge_pad_v1") or "edge_pad_v1"),
                requested_source_handoff=str(source_handoff_contract.get("requested_source_handoff") or ""),
                actual_source_handoff=str(source_handoff_contract.get("actual_source_handoff") or ""),
                alignment=dict(source_handoff_contract.get("latent_grid_alignment") or {}),
            )
            inference_fingerprint = build_outpaint_inference_fingerprint(
                geometry_fingerprint=geometry_fingerprint,
                model_identity=getattr(self.components, "model_identity", "") or getattr(request, "model_identity", ""),
                vae_identity=vae_identity,
                sampler_name=str(getattr(request, "sampler_name", "") or ""),
                scheduler_name=str(getattr(request, "scheduler_name", "") or ""),
                steps=int(request.steps),
                cfg_scale=float(getattr(request, "cfg_scale", 0.0) or 0.0),
                denoise_strength=float(getattr(request, "outpaint_denoising_strength", 0.70)),
                schedule_boundary=schedule_boundary,
                noise_strategy_version=str(noise_metadata.get("strategy") or getattr(request, "outpaint_latent_strategy", "")),
                prompt_merge_mode=str((outpaint_metadata.get("conditioning") or {}).get("mode") or getattr(request, "outpaint_prompt_mode", "source_prompt_v1")),
                overlay_positive_prompt=str(getattr(request, "outpaint_overlay_positive_prompt", "") or ""),
                overlay_negative_prompt=str(getattr(request, "outpaint_overlay_negative_prompt", "") or ""),
            )
            outpaint_metadata.update({
                "pre_sampling_memory_cleanup": {
                    "contract_version": "phase14n13p-pre-sampling-memory-cleanup-v1",
                    "stage": "pre_outpaint_sampling_cleanup",
                    "actions": [dict(item) for item in outpaint_pre_sampling_actions],
                    "vae_offload_requested": True,
                    "text_encoder_offload_requested": True,
                    "allocator_cache_release_requested": True,
                },
                "vae_encode": dict(encoded.metadata),
                "latent_mask": {
                    "resize_method": "bilinear_align_corners_false",
                    "shape": list(outpaint_latent_mask.shape),
                    "range": [float(outpaint_latent_mask.min().item()), float(outpaint_latent_mask.max().item())],
                    "contract": latent_mask_contract,
                },
                "noise": noise_metadata,
                "source_handoff": {
                    **handoff,
                    "alignment": dict(alignment),
                    "preservation_reference_source": str(source_handoff_contract.get("preservation_reference_source") or ""),
                    "source_was_vae_reencoded_for_protected_latent": bool(source_handoff_contract.get("source_was_vae_reencoded_for_protected_latent", True)),
                    "live_source_latent_reused": bool(source_handoff_contract.get("live_source_latent_reused", False)),
                    "provisional_canvas_vae_encoded": True,
                },
                "source_handoff_contract": source_handoff_contract,
                "geometry_fingerprint": geometry_fingerprint,
                "inference_fingerprint": inference_fingerprint,
                "schedule": {
                    "contract": image_conditioned_schedule.to_serializable_dict(),
                    **schedule_boundary,
                },
            })
            outpaint_metadata["audit"] = build_outpaint_audit(outpaint_metadata)
            request.outpaint_prototype_record = dict(outpaint_metadata)
        else:
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

        ctx.outpaint_latent_mask = outpaint_latent_mask
        ctx.outpaint_hook = outpaint_hook
        ctx.latents = latents
        ctx.schedule = schedule
