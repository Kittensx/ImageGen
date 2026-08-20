from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from image_gen.systems.outpainting import (
    OUTPAINT_PRESERVATION_STRATEGY,
    build_canvas_and_masks,
    build_outpaint_canvas_contract,
    build_outpaint_model_capability_contract,
    build_outpaint_region_contract,
    compose_outpaint_prompt_overlay,
    format_outpaint_failure,
    load_source_image,
    plan_outpaint_canvas,
)

from .context import GenerationContext


class ConditioningStageMixin:
    def _run_conditioning_stage(self, ctx: GenerationContext):
        diagnostics = ctx.diagnostics
        session = ctx.session
        request = ctx.request
        outpaint_enabled = ctx.outpaint_enabled
        outpaint_metadata = ctx.outpaint_metadata
        _outpaint_stage = ctx.outpaint_stage
        outpaint_source = ctx.outpaint_source
        outpaint_canvas = ctx.outpaint_canvas
        outpaint_masks = ctx.outpaint_masks
        outpaint_plan = ctx.outpaint_plan
        outpaint_prompt_contract = ctx.outpaint_prompt_contract
        source_metadata = ctx.source_metadata

        if outpaint_enabled:
            runtime_source_tensor = getattr(request, "_outpaint_runtime_source_tensor", None)
            if torch.is_tensor(runtime_source_tensor):
                outpaint_source = _outpaint_stage(
                    "outpaint_live_source_capture",
                    lambda: runtime_source_tensor.detach().clone(),
                )
                source_metadata = {
                    "path": "",
                    "source_kind": "live_txt2img_tensor",
                    "normalized_dimensions": [
                        int(outpaint_source.shape[-1]),
                        int(outpaint_source.shape[-2]),
                    ],
                    "channel_order": "RGB",
                    "range": [0.0, 1.0],
                    "disk_round_trip": False,
                }
            else:
                outpaint_source, source_metadata = _outpaint_stage(
                    "outpaint_source_decode",
                    lambda: load_source_image(getattr(request, "outpaint_source_image", "")),
                )
            source_height = int(outpaint_source.shape[-2])
            source_width = int(outpaint_source.shape[-1])
            explicit_x = int(getattr(request, "outpaint_source_x", -1))
            explicit_y = int(getattr(request, "outpaint_source_y", -1))
            outpaint_plan = _outpaint_stage(
                "outpaint_canvas_planning",
                lambda: plan_outpaint_canvas(
                    source_width=source_width,
                    source_height=source_height,
                    target_width=int(request.width),
                    target_height=int(request.height),
                    anchor=str(getattr(request, "outpaint_anchor", "center") or "center"),
                    feather_px=int(getattr(request, "outpaint_feather_px", 24)),
                    source_x=None if explicit_x < 0 else explicit_x,
                    source_y=None if explicit_y < 0 else explicit_y,
                ),
            )
            outpaint_canvas, outpaint_masks, mask_metadata = _outpaint_stage(
                "outpaint_mask_build",
                lambda: build_canvas_and_masks(
                    outpaint_source,
                    outpaint_plan,
                    context_seed_mode=str(
                        getattr(request, "outpaint_context_seed_mode", "neutral_gray_v1") or "neutral_gray_v1"
                    ),
                ),
            )
            outpaint_canvas_contract = build_outpaint_canvas_contract(
                outpaint_plan,
                requested_target_width=int(request.width),
                requested_target_height=int(request.height),
                internal_target_width=int(request.width),
                internal_target_height=int(request.height),
                latent_scale_factor=int(self.latent_scale_factor),
                pixel_alignment_multiple=int(
                    getattr(
                        self.systems.latent_preparation,
                        "pixel_alignment_multiple",
                        self.latent_scale_factor,
                    )
                    or self.latent_scale_factor
                ),
            )
            outpaint_region_contract = build_outpaint_region_contract(
                outpaint_plan,
                feather_px=int(getattr(request, "outpaint_feather_px", 24)),
            )
            outpaint_metadata.update({
                "source": source_metadata,
                "canvas_plan": outpaint_plan.to_dict(),
                "canvas": outpaint_canvas_contract,
                "regions": outpaint_region_contract,
                "mask": mask_metadata,
                "context_seed_mode": str(getattr(request, "outpaint_context_seed_mode", "neutral_gray_v1") or "neutral_gray_v1"),
                "denoising_strength": float(getattr(request, "outpaint_denoising_strength", 0.70)),
                "latent_strategy": str(getattr(request, "outpaint_latent_strategy", "noise_only_new_regions_v1")),
                "preservation_strategy": OUTPAINT_PRESERVATION_STRATEGY,
                "preservation_mode": OUTPAINT_PRESERVATION_STRATEGY,
                "mask_strategy": "preserve_generate_feather_v1",
                "sampler_name": str(getattr(request, "sampler_name", "") or ""),
                "scheduler_name": str(getattr(request, "scheduler_name", "") or ""),
                "steps": int(getattr(request, "steps", 0) or 0),
                "cfg_scale": float(getattr(request, "cfg_scale", 0.0) or 0.0),
                "seed": int(request.resolved_seeds[0]) if request.resolved_seeds else getattr(request, "seed", None),
                "positive_prompt": str(getattr(request, "positive_prompt", "") or ""),
                "negative_prompt": str(getattr(request, "negative_prompt", "") or ""),
                "model_capabilities": build_outpaint_model_capability_contract(),
            })
            request.outpaint_prototype_record = dict(outpaint_metadata)
        conditioning_request = request
        if outpaint_enabled:
            outpaint_prompt_contract = _outpaint_stage(
                "outpaint_conditioning",
                lambda: compose_outpaint_prompt_overlay(
                    mode=str(getattr(request, "outpaint_prompt_mode", "source_prompt_v1") or "source_prompt_v1"),
                    source_positive_prompt=str(getattr(request, "positive_prompt", "") or ""),
                    source_negative_prompt=str(getattr(request, "negative_prompt", "") or ""),
                    overlay_positive_prompt=str(getattr(request, "outpaint_overlay_positive_prompt", "") or ""),
                    overlay_negative_prompt=str(getattr(request, "outpaint_overlay_negative_prompt", "") or ""),
                ),
            )
            if outpaint_prompt_contract["mode"] != "source_prompt_v1":
                conditioning_request = replace(
                    request,
                    positive_prompt=str(outpaint_prompt_contract["effective_positive_prompt"]),
                    negative_prompt=str(outpaint_prompt_contract["effective_negative_prompt"]),
                    prompt_cfg_recorded_schedules={},
                    prompt_cfg_replay_mode="reconstruct",
                    prompt_expansion_record={},
                    prompt_expansion_pass_records={},
                    prompt_expansion_recorded={},
                    prompt_expansion_replay_mode="reconstruct",
                    prompt_semantic_pass_records={},
                    prompt_semantic_recorded={},
                    prompt_semantic_replay_mode="reconstruct",
                    region_pass_records={},
                    region_recorded={},
                    region_replay_mode="reconstruct",
                    prompt_route_plan={},
                    diagnostics=dict(getattr(request, "diagnostics", {}) or {}),
                )
                outpaint_prompt_contract["replay_policy"] = "overlay_prompt_reconstruct_v1"
            outpaint_metadata["conditioning"] = dict(outpaint_prompt_contract)
            request.outpaint_prototype_record = dict(outpaint_metadata)

        def _encode_conditioning() -> Any:
            return diagnostics.run_stage(
                session,
                "conditioning",
                "encode",
                lambda: self.memory_manager.run_stage(
                    stage="conditioning",
                    required=self._conditioning_required_components(),
                    preferred=self._conditioning_preferred_components(),
                    operation=lambda: self.systems.conditioning.encode(
                        self.components, conditioning_request, self.state
                    ),
                    request=conditioning_request,
                ),
            )

        conditioning = (
            _outpaint_stage("outpaint_conditioning", _encode_conditioning)
            if outpaint_enabled
            else _encode_conditioning()
        )
        if outpaint_enabled:
            pass_records = dict(getattr(conditioning_request, "region_pass_records", {}) or {})
            base_region_record = dict(pass_records.get("base") or {})
            regional_branch_count = int(base_region_record.get("region_count", 0) or 0)
            overlay_mode = str(outpaint_prompt_contract.get("mode") or "source_prompt_v1")
            if overlay_mode != "source_prompt_v1" and regional_branch_count:
                raise RuntimeError(
                    format_outpaint_failure(
                        "outpaint_conditioning",
                        "Extension prompting requires one global conditioning branch; REGION directives are not allowed in the effective extension prompt.",
                    )
                )
            outpaint_metadata["conditioning"].update({
                "prompt_parser_name": str(getattr(conditioning_request, "prompt_parser_name", "") or ""),
                "prompt_shortcut_profile_name": str(getattr(conditioning_request, "prompt_shortcut_profile_name", "") or ""),
                "regional_branch_count": regional_branch_count,
                "single_global_branch_verified": regional_branch_count == 0,
            })
            request.outpaint_prototype_record = dict(outpaint_metadata)
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

        ctx.outpaint_source = outpaint_source
        ctx.outpaint_canvas = outpaint_canvas
        ctx.outpaint_masks = outpaint_masks
        ctx.outpaint_plan = outpaint_plan
        ctx.outpaint_prompt_contract = outpaint_prompt_contract
        ctx.source_metadata = source_metadata
        ctx.conditioning = conditioning
