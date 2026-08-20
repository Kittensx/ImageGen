from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .prototype import OutpaintPrototypePlan

OUTPAINT_BACKEND_CONTRACT_VERSION = "phase14n13a-outpaint-backend-v1"
OUTPAINT_CANVAS_CONTRACT_VERSION = "phase14n13a-outpaint-canvas-v1"
OUTPAINT_REGION_CONTRACT_VERSION = "phase14n13a-outpaint-region-v1"
OUTPAINT_SOURCE_HANDOFF_CONTRACT_VERSION = "phase14n13a-outpaint-source-handoff-v1"
OUTPAINT_MODEL_CAPABILITY_CONTRACT_VERSION = "phase14n13a-outpaint-model-capability-v1"
OUTPAINT_GEOMETRY_FINGERPRINT_VERSION = "phase14n13a-outpaint-geometry-fingerprint-v1"
OUTPAINT_INFERENCE_FINGERPRINT_VERSION = "phase14n13a-outpaint-inference-fingerprint-v1"
OUTPAINT_MASK_TRANSFORM_CONTRACT_VERSION = "phase14n13a-outpaint-mask-transform-v1"
OUTPAINT_AUDIT_CONTRACT_VERSION = "phase14n13a-outpaint-audit-v1"
OUTPAINT_POST_GENERATION_ACTION_CONTRACT_VERSION = "phase14n13a-post-generation-shape-action-v1"
OUTPAINT_PRESERVATION_MODE_STRICT = "strict_preserve"
OUTPAINT_MASK_STRATEGY_DEFAULT = "preserve_generate_feather_v1"
OUTPAINT_SOURCE_HANDOFF_IMAGE_REENCODE = "image_reencode_v1"
OUTPAINT_SOURCE_HANDOFF_LIVE_TXT2IMG_LATENT = "live_txt2img_latent_v1"
OUTPAINT_SOURCE_ORIGIN_EXTERNAL_IMAGE = "external_image"
OUTPAINT_SOURCE_ORIGIN_FRESH_GENERATION = "fresh_generation"
OUTPAINT_SOURCE_ORIGIN_REPLAY_REGENERATED = "replay_regenerated"
OUTPAINT_CONTEXT_SEED_PRODUCTION_DEFAULT = "edge_pad_v1"
OUTPAINT_CONTEXT_SEED_ADVANCED_ALTERNATIVE = "reflect_pad_v1"
OUTPAINT_CONTEXT_SEED_LEGACY_DEBUG = "neutral_gray_v1"


_OUTPAINT_SOURCE_HANDOFF_ALIASES = {
    "": OUTPAINT_SOURCE_HANDOFF_IMAGE_REENCODE,
    "auto": "auto",
    "image_reencode_v1": OUTPAINT_SOURCE_HANDOFF_IMAGE_REENCODE,
    "pixel_vae_reencode": OUTPAINT_SOURCE_HANDOFF_IMAGE_REENCODE,
    "pixel_reencode": OUTPAINT_SOURCE_HANDOFF_IMAGE_REENCODE,
    "live_txt2img_latent_v1": OUTPAINT_SOURCE_HANDOFF_LIVE_TXT2IMG_LATENT,
    "live_latent": OUTPAINT_SOURCE_HANDOFF_LIVE_TXT2IMG_LATENT,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_outpaint_source_handoff_mode(value: Any, *, default: str = OUTPAINT_SOURCE_HANDOFF_IMAGE_REENCODE) -> str:
    key = str(value or "").strip().casefold()
    return _OUTPAINT_SOURCE_HANDOFF_ALIASES.get(key, str(value or default))


def build_outpaint_canvas_contract(
    plan: OutpaintPrototypePlan,
    *,
    requested_target_width: int | None = None,
    requested_target_height: int | None = None,
    internal_target_width: int | None = None,
    internal_target_height: int | None = None,
    latent_scale_factor: int = 8,
    pixel_alignment_multiple: int | None = None,
) -> dict[str, Any]:
    requested_w = int(requested_target_width or plan.target_width)
    requested_h = int(requested_target_height or plan.target_height)
    internal_w = int(internal_target_width or plan.target_width)
    internal_h = int(internal_target_height or plan.target_height)
    latent_multiple = max(1, int(latent_scale_factor or 1))
    target_multiple = max(1, int(pixel_alignment_multiple or latent_multiple))
    source_bounds = {
        "x0": int(plan.source_x),
        "y0": int(plan.source_y),
        "x1": int(plan.source_right),
        "y1": int(plan.source_bottom),
    }
    return {
        "contract_version": OUTPAINT_CANVAS_CONTRACT_VERSION,
        "source_width": int(plan.source_width),
        "source_height": int(plan.source_height),
        "target_width": int(plan.target_width),
        "target_height": int(plan.target_height),
        "requested_final_dimensions": {"width": requested_w, "height": requested_h},
        "internal_aligned_dimensions": {"width": internal_w, "height": internal_h},
        "source_x": int(plan.source_x),
        "source_y": int(plan.source_y),
        "source_bounds": source_bounds,
        "left_expansion": int(plan.left_expansion),
        "right_expansion": int(plan.right_expansion),
        "top_expansion": int(plan.top_expansion),
        "bottom_expansion": int(plan.bottom_expansion),
        "anchor_policy": str(plan.anchor),
        "alignment_requirements": {
            "target_width_multiple_of": target_multiple,
            "target_height_multiple_of": target_multiple,
            "pixel_alignment_multiple": target_multiple,
            "vae_latent_scale_factor": latent_multiple,
            "exact_latent_embedding_requires": {
                "source_x_multiple_of": latent_multiple,
                "source_y_multiple_of": latent_multiple,
            },
        },
    }


def build_outpaint_region_contract(
    plan: OutpaintPrototypePlan,
    *,
    feather_px: int,
    mask_strategy: str = OUTPAINT_MASK_STRATEGY_DEFAULT,
) -> dict[str, Any]:
    bounds = {
        "x0": int(plan.source_x),
        "y0": int(plan.source_y),
        "x1": int(plan.source_right),
        "y1": int(plan.source_bottom),
    }
    return {
        "contract_version": OUTPAINT_REGION_CONTRACT_VERSION,
        "mask_strategy": str(mask_strategy or OUTPAINT_MASK_STRATEGY_DEFAULT),
        "regions": {
            "preserve": {
                "required": True,
                "bounds": bounds,
            },
            "generate": {
                "required": True,
                "expansion": {
                    "left": int(plan.left_expansion),
                    "right": int(plan.right_expansion),
                    "top": int(plan.top_expansion),
                    "bottom": int(plan.bottom_expansion),
                },
            },
            "feather": {
                "required": True,
                "width_px": max(0, int(feather_px)),
                "bounds": bounds,
            },
        },
    }


def build_outpaint_mask_transform_contract(
    *,
    pixel_mask_shape: Any,
    latent_mask_shape: Any,
    resize_method: str,
    normalization: str = "0_to_1",
    soft_mask_behavior: str = "continuous",
    threshold_behavior: str = "none",
) -> dict[str, Any]:
    return {
        "contract_version": OUTPAINT_MASK_TRANSFORM_CONTRACT_VERSION,
        "pixel_mask_shape": list(pixel_mask_shape) if pixel_mask_shape is not None else None,
        "latent_mask_shape": list(latent_mask_shape) if latent_mask_shape is not None else None,
        "resize_method": str(resize_method or "unknown"),
        "normalization": str(normalization or "0_to_1"),
        "soft_mask_behavior": str(soft_mask_behavior or "continuous"),
        "threshold_behavior": str(threshold_behavior or "none"),
    }


def build_outpaint_source_handoff_record(
    *,
    requested_mode: Any,
    actual_mode: Any,
    source_origin: str,
    alignment: Mapping[str, Any] | None,
    fallback_reason: str = "",
    source_asset: Mapping[str, Any] | None = None,
    preservation_reference_source: str = "",
    source_was_vae_reencoded_for_protected_latent: bool | None = None,
    live_source_latent_reused: bool | None = None,
) -> dict[str, Any]:
    requested = normalize_outpaint_source_handoff_mode(requested_mode)
    actual = normalize_outpaint_source_handoff_mode(actual_mode)
    alignment_record = dict(alignment or {})
    source_pixel = dict(alignment_record.get("source_pixel_placement") or {})
    source_latent = dict(alignment_record.get("source_latent_placement") or {})
    exact = bool(alignment_record.get("aligned", False))
    scale = int(alignment_record.get("latent_scale_factor") or 8)
    return {
        "contract_version": OUTPAINT_SOURCE_HANDOFF_CONTRACT_VERSION,
        "requested_source_handoff": requested,
        "actual_source_handoff": actual,
        "source_handoff_fallback_reason": str(fallback_reason or ""),
        "source_origin": str(source_origin or OUTPAINT_SOURCE_ORIGIN_EXTERNAL_IMAGE),
        "source_asset": dict(source_asset or {}),
        "latent_grid_alignment": {
            "contract_version": str(alignment_record.get("contract_version") or ""),
            "vae_spatial_compression_factor": scale,
            "source_x_pixels": source_pixel.get("x"),
            "source_y_pixels": source_pixel.get("y"),
            "source_x_latent_cells": source_latent.get("x"),
            "source_y_latent_cells": source_latent.get("y"),
            "latent_embedding_exact": exact,
            "reasons": list(alignment_record.get("reasons") or []),
            "source_latent_shape": alignment_record.get("source_latent_shape"),
            "expanded_latent_shape": alignment_record.get("expanded_latent_shape"),
        },
        "preservation_reference_source": str(preservation_reference_source or ""),
        "source_was_vae_reencoded_for_protected_latent": bool(source_was_vae_reencoded_for_protected_latent),
        "live_source_latent_reused": bool(live_source_latent_reused),
    }


def build_outpaint_model_capability_contract(
    *,
    supports_masked_outpaint: bool = True,
    supports_live_latent_source_handoff: bool = True,
    supports_direct_aligned_latent_embedding: bool = True,
    supports_pixel_vae_reencode_source_handoff: bool = True,
    supports_edge_pad: bool = True,
    supports_reflect_pad: bool = True,
    supports_dedicated_inpainting_conditioning: bool = False,
    supports_soft_masks: bool = True,
    supports_latent_preservation_during_sampling: bool = True,
) -> dict[str, Any]:
    return {
        "contract_version": OUTPAINT_MODEL_CAPABILITY_CONTRACT_VERSION,
        "ordinary_masked_outpaint_path": bool(supports_masked_outpaint),
        "live_latent_source_handoff": bool(supports_live_latent_source_handoff),
        "direct_aligned_latent_embedding": bool(supports_direct_aligned_latent_embedding),
        "pixel_vae_reencode_source_handoff": bool(supports_pixel_vae_reencode_source_handoff),
        "edge_pad": bool(supports_edge_pad),
        "reflect_pad": bool(supports_reflect_pad),
        "dedicated_inpainting_conditioning": bool(supports_dedicated_inpainting_conditioning),
        "soft_masks": bool(supports_soft_masks),
        "latent_preservation_during_sampling": bool(supports_latent_preservation_during_sampling),
    }


def build_outpaint_geometry_fingerprint(
    *,
    canvas: Mapping[str, Any],
    feather_px: int,
    preservation_mode: str,
    mask_strategy: str,
    context_seed_mode: str,
    requested_source_handoff: str,
    actual_source_handoff: str,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    contract = {
        "canvas_contract_version": str(canvas.get("contract_version") or ""),
        "source_width": int(canvas.get("source_width") or 0),
        "source_height": int(canvas.get("source_height") or 0),
        "target_width": int(canvas.get("target_width") or 0),
        "target_height": int(canvas.get("target_height") or 0),
        "requested_final_dimensions": dict(canvas.get("requested_final_dimensions") or {}),
        "internal_aligned_dimensions": dict(canvas.get("internal_aligned_dimensions") or {}),
        "source_x": int(canvas.get("source_x") or 0),
        "source_y": int(canvas.get("source_y") or 0),
        "source_bounds": dict(canvas.get("source_bounds") or {}),
        "left_expansion": int(canvas.get("left_expansion") or 0),
        "right_expansion": int(canvas.get("right_expansion") or 0),
        "top_expansion": int(canvas.get("top_expansion") or 0),
        "bottom_expansion": int(canvas.get("bottom_expansion") or 0),
        "anchor_policy": str(canvas.get("anchor_policy") or "center"),
        "feather_px": max(0, int(feather_px)),
        "preservation_mode": str(preservation_mode or OUTPAINT_PRESERVATION_MODE_STRICT),
        "mask_strategy": str(mask_strategy or OUTPAINT_MASK_STRATEGY_DEFAULT),
        "context_seed_mode": str(context_seed_mode or OUTPAINT_CONTEXT_SEED_PRODUCTION_DEFAULT),
        "requested_source_handoff": str(requested_source_handoff or ""),
        "actual_source_handoff": str(actual_source_handoff or ""),
        "latent_grid_alignment": dict(alignment or {}),
    }
    return {
        "schema_version": OUTPAINT_GEOMETRY_FINGERPRINT_VERSION,
        "sha256": _json_hash(contract),
        "contract": contract,
    }


def build_outpaint_inference_fingerprint(
    *,
    geometry_fingerprint: Mapping[str, Any],
    model_identity: Any,
    vae_identity: Mapping[str, Any] | None,
    sampler_name: str,
    scheduler_name: str,
    steps: int,
    cfg_scale: float,
    denoise_strength: float,
    schedule_boundary: Mapping[str, Any] | None,
    noise_strategy_version: str,
    prompt_merge_mode: str,
    overlay_positive_prompt: str,
    overlay_negative_prompt: str,
) -> dict[str, Any]:
    contract = {
        "geometry_fingerprint_sha256": str(geometry_fingerprint.get("sha256") or ""),
        "model_identity": str(model_identity or ""),
        "vae_identity": {
            "source_kind": str((vae_identity or {}).get("source_kind") or ""),
            "sha256": str((vae_identity or {}).get("sha256") or ""),
        },
        "sampler_name": str(sampler_name or ""),
        "scheduler_name": str(scheduler_name or ""),
        "steps": int(steps or 0),
        "cfg_scale": float(cfg_scale or 0.0),
        "denoise_strength": float(denoise_strength or 0.0),
        "schedule_boundary": dict(schedule_boundary or {}),
        "noise_strategy_version": str(noise_strategy_version or ""),
        "prompt_merge_mode": str(prompt_merge_mode or ""),
        "overlay_positive_prompt_sha256": hashlib.sha256(str(overlay_positive_prompt or "").encode("utf-8")).hexdigest(),
        "overlay_negative_prompt_sha256": hashlib.sha256(str(overlay_negative_prompt or "").encode("utf-8")).hexdigest(),
    }
    return {
        "schema_version": OUTPAINT_INFERENCE_FINGERPRINT_VERSION,
        "sha256": _json_hash(contract),
        "contract": contract,
    }


def build_post_generation_shape_action(
    *,
    base_width: int,
    base_height: int,
    target_width: int,
    target_height: int,
    anchor: str,
    context_seed_mode: str,
    source_handoff_policy: str,
    overlay_positive_prompt: str,
    overlay_negative_prompt: str,
    denoise_strength: float,
    save_pre_expansion_base: bool,
) -> dict[str, Any]:
    return {
        "contract_version": OUTPAINT_POST_GENERATION_ACTION_CONTRACT_VERSION,
        "post_generation_shape_action": "outpaint_expand",
        "base_width": int(base_width),
        "base_height": int(base_height),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "source_placement": str(anchor or "center"),
        "context_seed_mode": str(context_seed_mode or OUTPAINT_CONTEXT_SEED_PRODUCTION_DEFAULT),
        "source_handoff_policy": normalize_outpaint_source_handoff_mode(source_handoff_policy, default="auto"),
        "outpaint_overlay_positive_prompt": str(overlay_positive_prompt or ""),
        "outpaint_overlay_negative_prompt": str(overlay_negative_prompt or ""),
        "outpaint_denoise_strength": float(denoise_strength or 0.0),
        "save_pre_expansion_base": bool(save_pre_expansion_base),
    }


def build_outpaint_audit(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(metadata or {})
    source = dict(payload.get("source") or {})
    canvas = dict(payload.get("canvas") or payload.get("canvas_plan") or {})
    handoff = dict(payload.get("source_handoff_contract") or payload.get("source_handoff") or {})
    latent_alignment = dict(handoff.get("latent_grid_alignment") or handoff.get("alignment") or {})
    fingerprint = dict(payload.get("geometry_fingerprint") or {})
    requested = str(handoff.get("requested_source_handoff") or payload.get("requested_source_handoff") or "")
    actual = str(handoff.get("actual_source_handoff") or handoff.get("actual") or "")
    source_label_map = {
        OUTPAINT_SOURCE_HANDOFF_IMAGE_REENCODE: "Existing image / VAE reencode",
        OUTPAINT_SOURCE_HANDOFF_LIVE_TXT2IMG_LATENT: "Live txt2img latent",
    }
    source_label = source_label_map.get(actual, source_label_map.get(requested, "Outpaint source"))
    placement = dict(canvas.get("source_bounds") or source.get("source_bounds") or {})
    x0 = int(placement.get("x0") or 0)
    y0 = int(placement.get("y0") or 0)
    x1 = int(placement.get("x1") or 0)
    y1 = int(placement.get("y1") or 0)
    lines = [
        f"Source: {source_label}",
        f"Source size: {int(source.get('width') or canvas.get('source_width') or 0)}x{int(source.get('height') or canvas.get('source_height') or 0)}",
        f"Canvas: {int(canvas.get('target_width') or 0)}x{int(canvas.get('target_height') or 0)}",
        f"Placement: x={x0}..{x1}, y={y0}..{y1}",
        (
            "Expansion: "
            f"L{int(canvas.get('left_expansion') or 0)} "
            f"R{int(canvas.get('right_expansion') or 0)} "
            f"T{int(canvas.get('top_expansion') or 0)} "
            f"B{int(canvas.get('bottom_expansion') or 0)}"
        ),
        f"Context seed: {str(payload.get('context_seed_mode') or '')}",
        f"Preserve: {str(payload.get('preservation_strategy') or payload.get('preservation_mode') or '')}",
        f"Feather: {int((payload.get('regions') or {}).get('regions', {}).get('feather', {}).get('width_px') or source.get('feather_px') or 0)} px",
        f"Denoise: {float(payload.get('denoising_strength') or 0.0):.2f}",
    ]
    if actual:
        lines.append(f"Source handoff: requested={requested or 'n/a'} actual={actual}")
    if latent_alignment:
        lines.append(
            "Latent alignment: "
            + ("exact" if bool(latent_alignment.get("latent_embedding_exact", latent_alignment.get("aligned", False))) else "fallback")
        )
    if handoff:
        lines.append(
            "Source re-encoded for protected core: "
            + ("Yes" if bool(handoff.get("source_was_vae_reencoded_for_protected_latent", False)) else "No")
        )
    if fingerprint:
        lines.append(f"Geometry fingerprint: {str(fingerprint.get('sha256') or '')}")
    return {
        "contract_version": OUTPAINT_AUDIT_CONTRACT_VERSION,
        "summary_lines": lines,
        "requested_source_handoff": requested,
        "actual_source_handoff": actual,
        "geometry_fingerprint": str(fingerprint.get("sha256") or ""),
    }


__all__ = [
    "OUTPAINT_AUDIT_CONTRACT_VERSION",
    "OUTPAINT_BACKEND_CONTRACT_VERSION",
    "OUTPAINT_CANVAS_CONTRACT_VERSION",
    "OUTPAINT_CONTEXT_SEED_ADVANCED_ALTERNATIVE",
    "OUTPAINT_CONTEXT_SEED_LEGACY_DEBUG",
    "OUTPAINT_CONTEXT_SEED_PRODUCTION_DEFAULT",
    "OUTPAINT_GEOMETRY_FINGERPRINT_VERSION",
    "OUTPAINT_INFERENCE_FINGERPRINT_VERSION",
    "OUTPAINT_MASK_STRATEGY_DEFAULT",
    "OUTPAINT_MASK_TRANSFORM_CONTRACT_VERSION",
    "OUTPAINT_MODEL_CAPABILITY_CONTRACT_VERSION",
    "OUTPAINT_POST_GENERATION_ACTION_CONTRACT_VERSION",
    "OUTPAINT_PRESERVATION_MODE_STRICT",
    "OUTPAINT_REGION_CONTRACT_VERSION",
    "OUTPAINT_SOURCE_HANDOFF_CONTRACT_VERSION",
    "OUTPAINT_SOURCE_HANDOFF_IMAGE_REENCODE",
    "OUTPAINT_SOURCE_HANDOFF_LIVE_TXT2IMG_LATENT",
    "OUTPAINT_SOURCE_ORIGIN_EXTERNAL_IMAGE",
    "OUTPAINT_SOURCE_ORIGIN_FRESH_GENERATION",
    "OUTPAINT_SOURCE_ORIGIN_REPLAY_REGENERATED",
    "build_outpaint_audit",
    "build_outpaint_canvas_contract",
    "build_outpaint_geometry_fingerprint",
    "build_outpaint_inference_fingerprint",
    "build_outpaint_mask_transform_contract",
    "build_outpaint_model_capability_contract",
    "build_outpaint_region_contract",
    "build_outpaint_source_handoff_record",
    "build_post_generation_shape_action",
    "normalize_outpaint_source_handoff_mode",
]
