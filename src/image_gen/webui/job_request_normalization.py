from __future__ import annotations

import math
from typing import Any, Mapping

from image_gen.contracts import PROMPT_ASSET_CONTRACT_VERSION, normalize_prompt_asset_list
from image_gen.systems.outpainting import (
    OUTPAINT_ANCHORS,
    OUTPAINT_CONTEXT_SEED_MODES,
    OUTPAINT_LATENT_STRATEGIES,
    OUTPAINT_PROMPT_MODES,
    OUTPAINT_SHAPE_TARGET_MODES,
    OUTPAINT_SOURCE_HANDOFF_MODES,
    resolve_outpaint_shape_target,
)
from image_gen.systems.image_conditioning import DEFAULT_HIRES_STEP_POLICY, SUPPORTED_HIRES_STEP_POLICIES
from image_gen.runtime.hires_sizing import apply_hires_dimensions
from image_gen.runtime.scheduler_settings import normalize_scheduler_payload
from image_gen.webui.randomization import normalize_parameter_ranges, parse_seed_plan
from image_gen.webui.schema_utils import coerce_value_by_schema, normalize_config_schema
from modules.prompt_parsers import PromptProcessingPreflight, default_prompt_parser_registry
from modules.prompt_shortcuts import (
    PromptShortcutProfileDescriptor,
    default_prompt_shortcut_registry,
    validate_prompt_shortcut_profile,
)

_MISSING = object()
_SHARED_CFG_LAB_SAMPLER_FIELDS = frozenset({
    "cfg_guidance_mode",
    "cfg_curve_type",
    "cfg_curve_strength",
    "cfg_high_sigma_boost",
    "cfg_low_sigma_taper",
    "cfg_auto_low_cfg_threshold",
    "cfg_early_floor_enabled",
    "cfg_early_floor_value",
    "cfg_early_floor_until_fraction",
})

def _coerce_top_level_number(value: Any, *, integer: bool, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value) if integer else float(value)
    if isinstance(value, int) and integer:
        return value
    if isinstance(value, (int, float)) and not integer:
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        number = float(text)
    except ValueError:
        return default
    return int(number) if integer else float(number)

def _coerce_unit_interval_or_none(value: Any) -> float | None:
    number = _coerce_top_level_number(value, integer=False, default=None)
    if number is None:
        return None
    numeric = float(number)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        return None
    return numeric


def _coerce_boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default

def _normalize_top_level_request(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(payload or {})
    spatial_requirements = dict(normalized.get("_generation_spatial_requirements") or {})
    try:
        latent_scale_factor = max(1, int(spatial_requirements.get("latent_scale_factor") or 8))
    except (TypeError, ValueError):
        latent_scale_factor = 8
    try:
        pixel_alignment_multiple = max(
            1,
            int(spatial_requirements.get("pixel_alignment_multiple") or latent_scale_factor),
        )
    except (TypeError, ValueError):
        pixel_alignment_multiple = latent_scale_factor
    spatial_requirements["latent_scale_factor"] = latent_scale_factor
    spatial_requirements["pixel_alignment_multiple"] = pixel_alignment_multiple
    normalized["_generation_spatial_requirements"] = spatial_requirements
    normalized["width"] = _coerce_top_level_number(normalized.get("width"), integer=True, default=640)
    normalized["height"] = _coerce_top_level_number(normalized.get("height"), integer=True, default=960)
    normalized["steps"] = _coerce_top_level_number(normalized.get("steps"), integer=True, default=20)
    normalized["batch_size"] = _coerce_top_level_number(normalized.get("batch_size"), integer=True, default=1)
    normalized["batch_count"] = _coerce_top_level_number(normalized.get("batch_count"), integer=True, default=1)
    normalized["cfg_scale"] = _coerce_top_level_number(normalized.get("cfg_scale"), integer=False, default=7.0)
    normalized["cfg_rescale"] = _coerce_top_level_number(normalized.get("cfg_rescale"), integer=False, default=0.0)
    if "sd3_t5_enabled" in normalized:
        normalized["sd3_t5_enabled"] = _coerce_boolean(normalized.get("sd3_t5_enabled"), default=False)
    if "sd3_t5_source" in normalized:
        t5_source = str(normalized.get("sd3_t5_source") or "auto").strip().lower().replace("-", "_")
        if t5_source.startswith("component:"):
            digest = t5_source.split(":", 1)[1].strip().lower()
            t5_source = (
                f"component:{digest}"
                if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
                else "auto"
            )
        elif t5_source not in {"auto", "embedded", "external", "shared", "standalone"}:
            t5_source = "auto"
        normalized["sd3_t5_source"] = {"shared": "external", "standalone": "external"}.get(t5_source, t5_source)
    if "text_encoder_3_device" in normalized:
        t5_device = str(normalized.get("text_encoder_3_device") or "auto").strip().lower()
        normalized["text_encoder_3_device"] = t5_device if t5_device in {"auto", "cpu", "cuda", "off"} else "auto"
    normalized["prompt_cfg_pass_schedules"] = dict(
        normalized.get("prompt_cfg_pass_schedules") or {}
    )
    normalized["prompt_cfg_recorded_schedules"] = dict(
        normalized.get("prompt_cfg_recorded_schedules") or {}
    )
    normalized["prompt_cfg_replay_mode"] = str(
        normalized.get("prompt_cfg_replay_mode") or "reconstruct"
    ).strip().lower()
    if normalized["prompt_cfg_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError("prompt_cfg_replay_mode must be reconstruct or recorded_exact.")
    normalized["prompt_expansion_record"] = dict(
        normalized.get("prompt_expansion_record") or {}
    )
    normalized["prompt_expansion_pass_records"] = dict(
        normalized.get("prompt_expansion_pass_records") or {}
    )
    normalized["prompt_expansion_recorded"] = dict(
        normalized.get("prompt_expansion_recorded") or {}
    )
    normalized["prompt_expansion_replay_mode"] = str(
        normalized.get("prompt_expansion_replay_mode") or "reconstruct"
    ).strip().lower()
    if normalized["prompt_expansion_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError("prompt_expansion_replay_mode must be reconstruct or recorded_exact.")
    normalized["prompt_semantic_pass_records"] = dict(
        normalized.get("prompt_semantic_pass_records") or {}
    )
    normalized["prompt_semantic_recorded"] = dict(
        normalized.get("prompt_semantic_recorded") or {}
    )
    normalized["prompt_semantic_replay_mode"] = str(
        normalized.get("prompt_semantic_replay_mode") or "reconstruct"
    ).strip().lower()
    if normalized["prompt_semantic_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError("prompt_semantic_replay_mode must be reconstruct or recorded_exact.")
    normalized["region_pass_records"] = dict(
        normalized.get("region_pass_records") or {}
    )
    normalized["region_recorded"] = dict(
        normalized.get("region_recorded") or {}
    )
    normalized["region_replay_mode"] = str(
        normalized.get("region_replay_mode") or "reconstruct"
    ).strip().lower()
    if normalized["region_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError("region_replay_mode must be reconstruct or recorded_exact.")
    normalized["hires_enabled"] = _coerce_boolean(normalized.get("hires_enabled", False), default=False)
    hires_size_mode = str(normalized.get("hires_size_mode") or "scale_from_base").strip().lower()
    if hires_size_mode not in {"same_as_base", "scale_from_base", "explicit_dimensions"}:
        hires_size_mode = "scale_from_base"
    # ``same_as_base`` is retained for old metadata and disabled hires requests,
    # but it cannot produce a second-pass enlargement. Older WebUI sessions may
    # still contain it from before hires generation was wired into the UI.
    if normalized["hires_enabled"] and hires_size_mode == "same_as_base":
        hires_size_mode = "scale_from_base"
    normalized["hires_size_mode"] = hires_size_mode
    normalized["hires_scale"] = _coerce_top_level_number(normalized.get("hires_scale"), integer=False, default=1.5)
    normalized["hires_width"] = _coerce_top_level_number(normalized.get("hires_width"), integer=True, default=0)
    normalized["hires_height"] = _coerce_top_level_number(normalized.get("hires_height"), integer=True, default=0)
    apply_hires_dimensions(normalized)
    normalized["hires_steps"] = _coerce_top_level_number(normalized.get("hires_steps"), integer=True, default=20)
    normalized["hires_denoising_strength"] = _coerce_top_level_number(
        normalized.get("hires_denoising_strength"), integer=False, default=0.4
    )
    normalized["hires_step_policy"] = str(
        normalized.get("hires_step_policy") or DEFAULT_HIRES_STEP_POLICY
    ).strip().lower()
    if normalized["hires_step_policy"] not in SUPPORTED_HIRES_STEP_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_HIRES_STEP_POLICIES))
        raise ValueError(f"hires_step_policy must be one of: {supported}.")
    normalized["hires_sampler_name"] = str(
        normalized.get("hires_sampler_name") or ""
    ).strip()
    normalized["hires_scheduler_name"] = str(
        normalized.get("hires_scheduler_name") or ""
    ).strip()
    normalized["hires_cfg_scale"] = _coerce_top_level_number(
        normalized.get("hires_cfg_scale"), integer=False, default=None
    )
    # GFP-01 safety freeze: an invalid hires override must not survive request
    # normalization. ``None`` deliberately restores inheritance from the valid
    # base CFG rescale in the hires runtime instead of preserving bad input.
    normalized["hires_cfg_rescale"] = _coerce_unit_interval_or_none(
        normalized.get("hires_cfg_rescale")
    )
    normalized["hires_recorded_schedule_replay"] = dict(
        normalized.get("hires_recorded_schedule_replay") or {}
    )
    normalized["hires_recorded_schedule_fingerprint"] = dict(
        normalized.get("hires_recorded_schedule_fingerprint") or {}
    )
    normalized["hires_schedule_replay_mode"] = str(
        normalized.get("hires_schedule_replay_mode") or "reconstruct"
    ).strip().lower()
    if normalized["hires_schedule_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError(
            "hires_schedule_replay_mode must be reconstruct or recorded_exact."
        )
    normalized["hires_configuration_mode"] = str(
        normalized.get("hires_configuration_mode") or "custom"
    ).strip().casefold()
    if normalized["hires_configuration_mode"] not in {"auto", "profile", "custom"}:
        normalized["hires_configuration_mode"] = "custom"
    normalized["hires_auto_resolution_record"] = dict(
        normalized.get("hires_auto_resolution_record") or {}
    )
    normalized["hires_lifecycle_state"] = dict(
        normalized.get("hires_lifecycle_state") or {}
    )
    normalized["hires_strategy"] = str(
        normalized.get("hires_strategy") or "pixel_neural"
    ).strip().casefold()
    normalized["hires_upscaler"] = str(normalized.get("hires_upscaler") or "").strip()
    normalized["hires_upscaler_id"] = str(
        normalized.get("hires_upscaler_id") or normalized["hires_upscaler"]
    ).strip()
    if normalized["hires_strategy"] not in {"pixel_neural", "pixel_resize"}:
        if not normalized["hires_enabled"]:
            normalized["hires_strategy"] = "pixel_neural"
            normalized["hires_upscaler"] = ""
            normalized["hires_upscaler_id"] = ""
        else:
            raise ValueError("hires_strategy must be pixel_neural or pixel_resize.")
    if normalized["hires_strategy"] == "pixel_resize":
        normalized["hires_upscaler"] = normalized["hires_upscaler_id"] or "builtin.pixel_resize.bicubic"
        normalized["hires_upscaler_id"] = normalized["hires_upscaler"]
    normalized["hires_tile_size"] = _coerce_top_level_number(
        normalized.get("hires_tile_size"), integer=True, default=0
    )
    normalized["hires_tile_overlap"] = _coerce_top_level_number(
        normalized.get("hires_tile_overlap"), integer=True, default=16
    )
    normalized["hires_tile_batch_size"] = _coerce_top_level_number(
        normalized.get("hires_tile_batch_size"), integer=True, default=1
    )
    if normalized["hires_tile_size"] < 0:
        raise ValueError("hires_tile_size cannot be negative.")
    if normalized["hires_tile_overlap"] < 0:
        raise ValueError("hires_tile_overlap cannot be negative.")
    if normalized["hires_tile_batch_size"] < 1:
        raise ValueError("hires_tile_batch_size must be at least 1.")
    legacy_resize_filter = str(normalized.get("hires_exact_resize_filter") or "").strip().casefold()
    normalized["hires_exact_resize_filter"] = legacy_resize_filter or "bicubic"
    if normalized["hires_exact_resize_filter"] not in {"nearest", "bilinear", "bicubic", "area"}:
        raise ValueError("hires_exact_resize_filter must be nearest, bilinear, bicubic, or area.")
    normalized["hires_final_size_correction_filter"] = str(
        normalized.get("hires_final_size_correction_filter")
        or legacy_resize_filter
        or "auto"
    ).strip().casefold()
    if normalized["hires_final_size_correction_filter"] not in {"auto", "nearest", "bilinear", "bicubic", "area"}:
        raise ValueError(
            "hires_final_size_correction_filter must be auto, nearest, bilinear, bicubic, or area."
        )
    normalized["hires_aspect_policy"] = str(
        normalized.get("hires_aspect_policy") or "stretch"
    ).strip().casefold()
    if normalized["hires_aspect_policy"] not in {"stretch", "crop_to_fill", "pad_to_fit"}:
        raise ValueError("hires_aspect_policy must be stretch, crop_to_fill, or pad_to_fit.")
    normalized["hires_padding_mode"] = str(
        normalized.get("hires_padding_mode") or "reflect"
    ).strip().casefold()
    if normalized["hires_padding_mode"] not in {"reflect", "replicate", "blurred_edge", "black"}:
        raise ValueError("hires_padding_mode must be reflect, replicate, blurred_edge, or black.")
    normalized["hires_blurred_edge_method"] = str(
        normalized.get("hires_blurred_edge_method") or "box"
    ).strip().casefold()
    if normalized["hires_blurred_edge_method"] not in {"box", "gaussian_1d"}:
        raise ValueError("hires_blurred_edge_method must be box or gaussian_1d.")
    normalized["hires_blurred_edge_compare_diagnostics"] = _coerce_boolean(
        normalized.get("hires_blurred_edge_compare_diagnostics", False), default=False
    )
    normalized["hires_recorded_target_correction"] = dict(
        normalized.get("hires_recorded_target_correction") or {}
    )
    normalized["hires_correction_fingerprint_enabled"] = _coerce_boolean(
        normalized.get("hires_correction_fingerprint_enabled", False), default=False
    )
    normalized["hires_recorded_correction_fingerprint"] = dict(
        normalized.get("hires_recorded_correction_fingerprint") or {}
    )
    normalized["hires_expected_native_scale"] = _coerce_top_level_number(
        normalized.get("hires_expected_native_scale"), integer=True, default=0
    )
    normalized["hires_save_upscaled_pre_denoise"] = _coerce_boolean(
        normalized.get("hires_save_upscaled_pre_denoise", False), default=False
    )
    normalized["hires_save_vae_roundtrip"] = _coerce_boolean(
        normalized.get("hires_save_vae_roundtrip", False), default=False
    )
    normalized["hires_save_lowres"] = _coerce_boolean(
        normalized.get("hires_save_lowres", False), default=False
    )
    normalized["outpaint_enabled"] = _coerce_boolean(
        normalized.get("outpaint_enabled", normalized.get("outpaint_prototype_enabled", False)), default=False
    )
    normalized["outpaint_prototype_enabled"] = bool(normalized["outpaint_enabled"] or _coerce_boolean(
        normalized.get("outpaint_prototype_enabled", False), default=False
    ))
    normalized["outpaint_target_width"] = _coerce_top_level_number(
        normalized.get("outpaint_target_width", normalized.get("width")), integer=True, default=normalized.get("width", 640)
    )
    normalized["outpaint_target_height"] = _coerce_top_level_number(
        normalized.get("outpaint_target_height", normalized.get("height")), integer=True, default=normalized.get("height", 960)
    )
    normalized["outpaint_preservation_mode"] = str(
        normalized.get("outpaint_preservation_mode") or "strict_preserve"
    ).strip().lower()
    if normalized["outpaint_preservation_mode"] not in {"strict_preserve"}:
        raise ValueError("Unsupported outpaint_preservation_mode.")
    normalized["outpaint_mask_strategy"] = str(
        normalized.get("outpaint_mask_strategy") or "preserve_generate_feather_v1"
    ).strip().lower()
    if normalized["outpaint_mask_strategy"] not in {"preserve_generate_feather_v1"}:
        raise ValueError("Unsupported outpaint_mask_strategy.")
    normalized["outpaint_source_handoff_mode"] = str(
        normalized.get("outpaint_source_handoff_mode") or "image_reencode_v1"
    ).strip().lower()
    if normalized["outpaint_source_handoff_mode"] not in {"image_reencode_v1", "live_txt2img_latent_v1", "auto"}:
        raise ValueError("Unsupported outpaint_source_handoff_mode.")
    normalized["outpaint_source_image"] = str(normalized.get("outpaint_source_image") or "").strip()
    normalized["outpaint_anchor"] = str(normalized.get("outpaint_anchor") or "center").strip().lower()
    if normalized["outpaint_anchor"] not in OUTPAINT_ANCHORS:
        raise ValueError("outpaint_anchor must be center, left, right, top, or bottom.")
    normalized["outpaint_source_x"] = _coerce_top_level_number(
        normalized.get("outpaint_source_x"), integer=True, default=-1
    )
    normalized["outpaint_source_y"] = _coerce_top_level_number(
        normalized.get("outpaint_source_y"), integer=True, default=-1
    )
    normalized["outpaint_feather_px"] = _coerce_top_level_number(
        normalized.get("outpaint_feather_px"), integer=True, default=24
    )
    if not 0 <= normalized["outpaint_feather_px"] <= 64:
        raise ValueError("outpaint_feather_px must be between 0 and 64.")
    normalized["outpaint_context_seed_mode"] = str(
        normalized.get("outpaint_context_seed_mode") or "edge_pad_v1"
    ).strip().lower()
    if normalized["outpaint_context_seed_mode"] not in OUTPAINT_CONTEXT_SEED_MODES:
        raise ValueError("Unsupported outpaint_context_seed_mode.")
    normalized["outpaint_denoising_strength"] = _coerce_top_level_number(
        normalized.get("outpaint_denoising_strength"), integer=False, default=0.70
    )
    if not 0.01 <= normalized["outpaint_denoising_strength"] <= 0.999:
        raise ValueError("outpaint_denoising_strength must be between 0.01 and 0.999.")
    normalized["outpaint_latent_strategy"] = str(
        normalized.get("outpaint_latent_strategy") or "noise_only_new_regions_v1"
    ).strip().lower()
    if normalized["outpaint_latent_strategy"] not in OUTPAINT_LATENT_STRATEGIES:
        raise ValueError("Unsupported outpaint_latent_strategy.")
    normalized["outpaint_prompt_mode"] = str(
        normalized.get("outpaint_prompt_mode") or "source_prompt_v1"
    ).strip().lower()
    if normalized["outpaint_prompt_mode"] not in OUTPAINT_PROMPT_MODES:
        raise ValueError("Unsupported outpaint_prompt_mode.")
    normalized["outpaint_overlay_positive_prompt"] = str(
        normalized.get("outpaint_overlay_positive_prompt") or ""
    ).strip()
    normalized["outpaint_overlay_negative_prompt"] = str(
        normalized.get("outpaint_overlay_negative_prompt") or ""
    ).strip()
    if (
        normalized["outpaint_prototype_enabled"]
        and normalized["outpaint_prompt_mode"] == "overlay_only_v1"
        and not normalized["outpaint_overlay_positive_prompt"]
    ):
        raise ValueError(
            "Extension prompt only requires an extension prompt."
        )
    normalized["outpaint_diagnostic_artifacts"] = _coerce_boolean(
        normalized.get("outpaint_diagnostic_artifacts", False), default=False
    )
    normalized["outpaint_prototype_record"] = dict(
        normalized.get("outpaint_prototype_record") or {}
    )

    normalized["outpaint_shape_expansion_enabled"] = _coerce_boolean(
        normalized.get("outpaint_shape_expansion_enabled", False), default=False
    )
    normalized["outpaint_shape_target_mode"] = str(
        normalized.get("outpaint_shape_target_mode") or "square"
    ).strip().lower()
    if normalized["outpaint_shape_target_mode"] not in OUTPAINT_SHAPE_TARGET_MODES:
        raise ValueError("Unsupported outpaint_shape_target_mode.")
    normalized["outpaint_shape_target_width"] = _coerce_top_level_number(
        normalized.get("outpaint_shape_target_width"), integer=True, default=0
    )
    normalized["outpaint_shape_target_height"] = _coerce_top_level_number(
        normalized.get("outpaint_shape_target_height"), integer=True, default=0
    )
    normalized["outpaint_shape_base_width"] = _coerce_top_level_number(
        normalized.get("outpaint_shape_base_width"), integer=True, default=0
    )
    normalized["outpaint_shape_base_height"] = _coerce_top_level_number(
        normalized.get("outpaint_shape_base_height"), integer=True, default=0
    )
    normalized["outpaint_shape_anchor"] = str(
        normalized.get("outpaint_shape_anchor") or "center"
    ).strip().lower()
    if normalized["outpaint_shape_anchor"] not in OUTPAINT_ANCHORS:
        raise ValueError("outpaint_shape_anchor must be center, left, right, top, or bottom.")
    normalized["outpaint_shape_context_seed_mode"] = str(
        normalized.get("outpaint_shape_context_seed_mode") or "edge_pad_v1"
    ).strip().lower()
    if normalized["outpaint_shape_context_seed_mode"] not in {"edge_pad_v1", "reflect_pad_v1"}:
        raise ValueError("Post-generation expansion edge initialization must use edge pixels or mirrored edge pixels.")
    normalized["outpaint_shape_source_handoff"] = str(
        normalized.get("outpaint_shape_source_handoff") or "auto"
    ).strip().lower()
    if normalized["outpaint_shape_source_handoff"] not in OUTPAINT_SOURCE_HANDOFF_MODES:
        raise ValueError("Unsupported outpaint_shape_source_handoff.")
    normalized["outpaint_shape_prompt_mode"] = str(
        normalized.get("outpaint_shape_prompt_mode") or "overlay_only_v1"
    ).strip().lower()
    if normalized["outpaint_shape_prompt_mode"] not in OUTPAINT_PROMPT_MODES:
        raise ValueError("Unsupported outpaint_shape_prompt_mode.")
    normalized["outpaint_shape_overlay_positive_prompt"] = str(
        normalized.get("outpaint_shape_overlay_positive_prompt") or ""
    ).strip()
    normalized["outpaint_shape_overlay_negative_prompt"] = str(
        normalized.get("outpaint_shape_overlay_negative_prompt") or ""
    ).strip()
    normalized["outpaint_shape_denoising_strength"] = _coerce_top_level_number(
        normalized.get("outpaint_shape_denoising_strength"), integer=False, default=0.40
    )
    if not 0.01 <= normalized["outpaint_shape_denoising_strength"] <= 0.999:
        raise ValueError("outpaint_shape_denoising_strength must be between 0.01 and 0.999.")
    normalized["outpaint_shape_save_base"] = _coerce_boolean(
        normalized.get("outpaint_shape_save_base", False), default=False
    )
    normalized["outpaint_shape_runtime_record"] = dict(
        normalized.get("outpaint_shape_runtime_record") or {}
    )
    if normalized["outpaint_shape_expansion_enabled"]:
        if normalized["outpaint_prototype_enabled"]:
            raise ValueError("Expand After Generation cannot be combined with Expand Existing Image.")
        if normalized["hires_enabled"]:
            raise ValueError("Expand After Generation cannot run with hires or .pth upscaling enabled.")
        if int(normalized.get("batch_size") or 1) != 1:
            raise ValueError("Expand After Generation currently requires batch_size=1.")
        if (
            normalized["outpaint_shape_prompt_mode"] == "overlay_only_v1"
            and not normalized["outpaint_shape_overlay_positive_prompt"]
        ):
            raise ValueError("Extension prompt only requires an extension prompt.")
        shape_target = resolve_outpaint_shape_target(
            base_width=int(normalized.get("width") or 0),
            base_height=int(normalized.get("height") or 0),
            target_mode=normalized["outpaint_shape_target_mode"],
            target_width=int(normalized["outpaint_shape_target_width"] or 0),
            target_height=int(normalized["outpaint_shape_target_height"] or 0),
            dimension_multiple=pixel_alignment_multiple,
        )
        normalized["outpaint_shape_base_width"] = int(shape_target["base_width"])
        normalized["outpaint_shape_base_height"] = int(shape_target["base_height"])
        normalized["outpaint_shape_target_width"] = int(shape_target["target_width"])
        normalized["outpaint_shape_target_height"] = int(shape_target["target_height"])
    if normalized["outpaint_prototype_enabled"]:
        if normalized["hires_enabled"]:
            raise ValueError("Existing-image expansion cannot run with hires enabled.")
        if int(normalized.get("batch_size") or 1) != 1:
            raise ValueError("Existing-image expansion requires batch_size=1.")
        if not normalized["outpaint_source_image"]:
            raise ValueError("Choose a source image before enabling existing-image expansion.")
        if (
            int(normalized.get("width") or 0) % pixel_alignment_multiple
            or int(normalized.get("height") or 0) % pixel_alignment_multiple
        ):
            raise ValueError(
                "Existing-image expansion target width and height must be divisible by "
                f"the active model's {pixel_alignment_multiple}-pixel alignment requirement."
            )
    seed_plan = parse_seed_plan(
        normalized.get("seed"),
        mode=normalized.get("batch_seed_mode"),
        range_min=normalized.get("seed_range_min"),
        range_max=normalized.get("seed_range_max"),
        unique=normalized.get("seed_no_duplicates", True),
    )
    normalized["seed"] = seed_plan.seed
    normalized["batch_seed_mode"] = seed_plan.mode
    normalized["seed_range_min"] = int(seed_plan.minimum)
    normalized["seed_range_max"] = int(seed_plan.maximum)
    normalized["seed_no_duplicates"] = bool(seed_plan.unique)
    normalized["_seed_plan"] = seed_plan.to_dict()
    normalized["_random_ranges"] = normalize_parameter_ranges(normalized.get("_random_ranges"))
    normalized["save_images"] = _coerce_boolean(normalized.get("save_images", True), default=True)
    normalized["save_txt"] = _coerce_boolean(normalized.get("save_txt", False), default=False)
    normalized["save_json"] = _coerce_boolean(normalized.get("save_json", True), default=True)
    image_format = str(normalized.get("output_image_format") or "png").strip().casefold().lstrip(".")
    if image_format not in {"png", "webp"}:
        image_format = "png"
    normalized["output_image_format"] = image_format
    metadata_mode = str(normalized.get("embedded_metadata_mode") or "full_replay").strip().casefold().replace("-", "_")
    if metadata_mode not in {"full_replay", "compatibility"}:
        metadata_mode = "full_replay"
    normalized["embedded_metadata_mode"] = metadata_mode
    normalized["save_diagnostics_json"] = _coerce_boolean(
        normalized.get("save_diagnostics_json", False), default=False
    )
    normalized["prompt_asset_contract_version"] = str(
        normalized.get("prompt_asset_contract_version")
        or PROMPT_ASSET_CONTRACT_VERSION
    )
    normalized["loras"] = [
        asset.to_serializable_dict()
        for asset in normalize_prompt_asset_list(
            normalized.get("loras") or [],
            asset_type="lora",
        )
    ]
    normalized["textual_inversions"] = [
        asset.to_serializable_dict()
        for asset in normalize_prompt_asset_list(
            normalized.get("textual_inversions") or [],
            asset_type="textual_inversion",
        )
    ]
    return normalized

def _drop_default_or_empty_values(values: dict[str, Any], schema_properties: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for name, value in values.items():
        if value is None or value == "":
            continue
        schema = dict(schema_properties.get(name) or {})
        if schema.get("x_linked"):
            continue
        default = schema.get("default", _MISSING)
        if schema.get("x_omit_if_default") and default is not _MISSING and value == default:
            continue
        cleaned[name] = value
    return cleaned

def normalize_generation_request(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Client-independent top-level coercion and duplicate-control cleanup."""

    normalized = _normalize_top_level_request(payload)
    scheduler_kwargs = dict(normalized.get("scheduler_kwargs") or {})
    scheduler_kwargs.pop("steps", None)
    scheduler_kwargs.pop("device", None)
    normalized["scheduler_kwargs"] = scheduler_kwargs
    normalized["sampler_kwargs"] = dict(normalized.get("sampler_kwargs") or {})
    normalized["prompt_parser_kwargs"] = dict(normalized.get("prompt_parser_kwargs") or {})
    normalized["parser_kwargs"] = dict(normalized.get("parser_kwargs") or {})
    requested_parser = (
        normalized.get("prompt_parser_name")
        or normalized["parser_kwargs"].get("prompt_parser")
        or normalized["parser_kwargs"].get("prompt_parser_name")
        or "legacy"
    )
    normalized["prompt_parser_name"] = default_prompt_parser_registry().resolve_id(requested_parser)
    snapshot = normalized.get("prompt_shortcut_profile_snapshot")
    if isinstance(snapshot, Mapping) and snapshot:
        shortcut_profile = PromptShortcutProfileDescriptor.from_dict(dict(snapshot), builtin=bool(snapshot.get("builtin", False)))
        validation = validate_prompt_shortcut_profile(shortcut_profile)
        if not validation.valid:
            raise ValueError("Embedded prompt shortcut profile is invalid: " + " | ".join(issue.message for issue in validation.errors))
    else:
        parser_id = normalized["prompt_parser_name"]
        fallback_profile = "legacy_default" if parser_id == "legacy" else ("parser21_native" if parser_id == "parser21" else ("superhybrid_native" if parser_id == "superhybrid" else "canonical"))
        shortcut_profile = default_prompt_shortcut_registry().get(normalized.get("prompt_shortcut_profile_name") or fallback_profile)
    parser_id = normalized["prompt_parser_name"]
    compatible = parser_id in shortcut_profile.compatible_parsers or (
        parser_id == "combined" and any(item in shortcut_profile.compatible_parsers for item in ("legacy", "parser21", "superhybrid"))
    )
    if not compatible:
        raise ValueError(f"Prompt shortcut profile {shortcut_profile.profile_id!r} is not compatible with parser {parser_id!r}.")
    normalized["prompt_shortcut_profile_name"] = shortcut_profile.profile_id
    normalized["prompt_shortcut_profile_snapshot"] = shortcut_profile.snapshot()
    normalized["prompt_parser_preset_name"] = str(normalized.get("prompt_parser_preset_name") or "")
    report = PromptProcessingPreflight().validate(normalized)
    if not report.get("valid"):
        messages = " | ".join(str(item.get("message") or "Prompt validation failed.") for item in report.get("blocking_errors") or [])
        raise ValueError(f"Prompt preflight failed: {messages}")
    normalized.update(report.get("normalized_fields") or {})
    normalized["prompt_preflight"] = report
    return normalized

def apply_vae_selection_policy(payload: dict[str, Any] | None, settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Preserve the operator's VAE selection without Phase 13F troubleshooting overrides."""

    normalized = dict(payload or {})
    requested_path = normalized.get("vae_path")
    normalized["external_vae_override_enabled"] = True
    normalized["vae_override_requested_path"] = str(requested_path or "")
    normalized["vae_mode"] = (
        "manual_external_selection" if requested_path else "checkpoint_embedded_auto"
    )
    return normalized


class JobRequestNormalizationMixin:
    def _descriptor_schema(self, kind: str, requested_name: Any) -> dict[str, Any]:
        descriptor = self.registry.resolve_descriptor(requested_name, kind=kind)
        if descriptor is None:
            return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        return normalize_config_schema(descriptor.config_schema, kind=kind)

    def _normalize_plugin_kwargs(self, kind: str, selected_name: Any, raw_values: Any) -> dict[str, Any]:
        incoming = dict(raw_values or {})
        schema = self._descriptor_schema(kind, selected_name)
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        # A plugin that declares additionalProperties explicitly owns those
        # extra values. Preserve them for replay instead of dropping them in
        # the WebUI layer. Plugins with a closed schema still reject/drop
        # unknown settings.
        allow_unknown = bool(schema.get("additionalProperties", False))
        cleaned: dict[str, Any] = {}
        for name, value in incoming.items():
            if kind == "scheduler" and name in {"steps", "device"}:
                continue
            if name in properties:
                coerced = coerce_value_by_schema(value, properties.get(name))
                cleaned[name] = coerced
            elif kind == "sampler" and name in _SHARED_CFG_LAB_SAMPLER_FIELDS:
                # CFG Lab is a generation-wide guidance contract, not a sampler-
                # private advanced option. Preserve these shared fields even for
                # samplers with closed plugin schemas (for example Simple Euler).
                cleaned[name] = value
            elif allow_unknown and value not in (None, ""):
                cleaned[str(name)] = value
        return _drop_default_or_empty_values(cleaned, properties)

    def normalize_generation_request(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        incoming = dict(payload or {})
        incoming_scheduler = dict(incoming.get("scheduler_kwargs") or {})
        normalized = normalize_generation_request(incoming)
        selection = self.selections.normalize(
            normalized,
            fallback_payload=self.context.generation_defaults(),
            migrate_legacy_auto_fallback=True,
        )
        normalized = selection.payload
        normalized["scheduler_kwargs"] = self._normalize_plugin_kwargs(
            "scheduler",
            normalized.get("scheduler_name"),
            normalized.get("scheduler_kwargs"),
        )
        # Keep an explicitly submitted linked step value long enough for the
        # canonical resolver to diagnose/synchronize it against generation.steps.
        if "steps" in incoming_scheduler:
            normalized["scheduler_kwargs"]["steps"] = incoming_scheduler["steps"]
        normalized["sampler_kwargs"] = self._normalize_plugin_kwargs(
            "sampler",
            normalized.get("sampler_name"),
            normalized.get("sampler_kwargs"),
        )
        normalized, _compatibility = self.registry.apply_compatibility_to_payload(normalized)
        normalized, _resolution = normalize_scheduler_payload(normalized)
        normalized = apply_vae_selection_policy(normalized, self._application_settings())
        return self.selections.strip_webui_metadata(normalized)
