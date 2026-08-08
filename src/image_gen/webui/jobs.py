from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping
from urllib.parse import quote

import torch

from image_gen.contracts import (
    PROMPT_ASSET_CONTRACT_VERSION,
    extract_hires_failure_stage,
    hires_failure_stage_label,
    normalize_prompt_asset_list,
)
from image_gen.systems.outpainting import (
    OUTPAINT_ANCHORS,
    OUTPAINT_CONTEXT_SEED_MODES,
    OUTPAINT_LATENT_STRATEGIES,
    OUTPAINT_PROMPT_MODES,
    OUTPAINT_SHAPE_TARGET_MODES,
    OUTPAINT_SOURCE_HANDOFF_MODES,
    extract_outpaint_failure_stage,
    outpaint_failure_label,
)
from image_gen.systems.image_conditioning import (
    DEFAULT_HIRES_STEP_POLICY,
    SUPPORTED_HIRES_STEP_POLICIES,
)
from image_gen.systems.memory.telemetry import normalize_cuda_memory_payload
from image_gen.systems.registry import RuntimeRegistrySystem
from image_gen.runtime_options import (
    RUNTIME_REPLAY_JOB_FIELDS,
    build_runtime_startup_status,
    resolve_runtime_startup_options,
    runtime_request_settings,
)
from image_gen.runtime.hires_sizing import apply_hires_dimensions
from image_gen.runtime.scheduler_settings import (
    normalize_scheduler_payload,
    scheduler_resolution_from_payload,
)
from image_gen.webui.schema_utils import coerce_value_by_schema, normalize_config_schema
from image_gen.webui.selection import WebUISelectionResolver
from image_gen.webui.model_runtime import ModelRuntimeUnavailable, ResidentModelRuntimeClient
from image_gen.webui.store import DEFAULT_FORCED_LIVE_PREVIEW_INTERVAL, FORCED_LIVE_PREVIEW_MODE
from modules.project_context import ProjectContext
from modules.txt2img.seed_utils import iter_batch_base_seeds
from modules.prompt_parsers import PromptProcessingPreflight, default_prompt_parser_registry
from modules.prompt_shortcuts import PromptShortcutProfileDescriptor, default_prompt_shortcut_registry, validate_prompt_shortcut_profile

_IMAGE_LINE = re.compile(r"^\s*Image \[seed (?P<seed>[^\]]+)\]:\s*(?P<path>.+?)\s*$")
_FAILURE_BUNDLE_LINE = re.compile(r"\(failure bundle:\s*(.+?)\)\s*$", re.IGNORECASE)
_MODEL_DIAGNOSTIC_LINE = re.compile(r"^MODEL_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_PROMPT_PARSER_DIAGNOSTIC_LINE = re.compile(r"^PROMPT_PARSER_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_OUTPUT_QUALITY_DIAGNOSTIC_LINE = re.compile(r"^OUTPUT_QUALITY_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_RUNTIME_DIAGNOSTIC_LINE = re.compile(r"^RUNTIME_DIAGNOSTIC_JSON:\s*(\{.*\})\s*$")
_STEP_PROGRESS_LINE = re.compile(r"STEP_PROGRESS_JSON:\s*(\{.*\})\s*$")
_STEP_PREVIEW_LINE = re.compile(r"STEP_PREVIEW_JSON:\s*(\{.*\})\s*$")
_GENERATION_SEED_LINE = re.compile(r"^GENERATION_SEED_JSON:\s*(\{.*\})\s*$")
_LIVE_PREVIEW_SUMMARY_LINE = re.compile(r"^LIVE_PREVIEW_SUMMARY_JSON:\s*(\{.*\})\s*$")
_MEMORY_STATUS_LINE = re.compile(r"MEMORY_STATUS_JSON:\s*(\{.*\})\s*$")
_MODEL_RUNTIME_STATUS_LINE = re.compile(r"^MODEL_RUNTIME_STATUS_JSON:\s*(\{.*\})\s*$")
_ASYNC_OUTPUT_SAVE_STATUS_LINE = re.compile(r"^ASYNC_OUTPUT_SAVE_STATUS_JSON:\s*(\{.*\})\s*$")
_ASYNC_OUTPUT_SAVE_ERROR_LINE = re.compile(r"^ASYNC_OUTPUT_SAVE_ERROR_JSON:\s*(\{.*\})\s*$")
_ACTIVE_JOB_STATUSES = {
    "preparing_model",
    "warming_model",
    "running",
    "paused",
    "finalizing",
    "cancelling",
}
_CANCELLABLE_JOB_STATUSES = {"preparing_model", "warming_model", "running", "paused"}
_MISSING = object()
_SUBPROCESS_STREAM_LIMIT = 16 * 1024 * 1024


def _normalize_live_memory_status(value: Mapping[str, Any] | None) -> dict[str, Any]:
    status = dict(value or {})
    snapshot = dict(status.get("latest_snapshot") or {})
    snapshot["cuda"] = normalize_cuda_memory_payload(
        dict(snapshot.get("cuda") or {})
    )
    status["latest_snapshot"] = snapshot
    if status.get("job_peak_allocated_vram_bytes") is None:
        status["job_peak_allocated_vram_bytes"] = status.get(
            "peak_allocated_vram_bytes"
        )
    if status.get("job_peak_reserved_vram_bytes") is None:
        status["job_peak_reserved_vram_bytes"] = status.get(
            "peak_reserved_vram_bytes"
        )
    return status


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


def _coerce_boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _timestamp_from_iso(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _normalize_top_level_request(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(payload or {})
    normalized["width"] = _coerce_top_level_number(normalized.get("width"), integer=True, default=640)
    normalized["height"] = _coerce_top_level_number(normalized.get("height"), integer=True, default=960)
    normalized["steps"] = _coerce_top_level_number(normalized.get("steps"), integer=True, default=20)
    normalized["batch_size"] = _coerce_top_level_number(normalized.get("batch_size"), integer=True, default=1)
    normalized["batch_count"] = _coerce_top_level_number(normalized.get("batch_count"), integer=True, default=1)
    normalized["cfg_scale"] = _coerce_top_level_number(normalized.get("cfg_scale"), integer=False, default=7.0)
    normalized["cfg_rescale"] = _coerce_top_level_number(normalized.get("cfg_rescale"), integer=False, default=0.0)
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
    normalized["hires_cfg_rescale"] = _coerce_top_level_number(
        normalized.get("hires_cfg_rescale"), integer=False, default=None
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
    normalized["hires_strategy"] = str(
        normalized.get("hires_strategy") or "pixel_neural"
    ).strip().casefold()
    normalized["hires_upscaler"] = str(normalized.get("hires_upscaler") or "").strip()
    normalized["hires_upscaler_id"] = str(
        normalized.get("hires_upscaler_id") or normalized["hires_upscaler"]
    ).strip()
    if normalized["hires_strategy"] != "pixel_neural":
        if not normalized["hires_enabled"]:
            normalized["hires_strategy"] = "pixel_neural"
            normalized["hires_upscaler"] = ""
            normalized["hires_upscaler_id"] = ""
        else:
            raise ValueError("hires_strategy must be pixel_neural.")
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
    if normalized["outpaint_prototype_enabled"]:
        if normalized["hires_enabled"]:
            raise ValueError("Existing-image expansion cannot run with hires enabled.")
        if int(normalized.get("batch_size") or 1) != 1:
            raise ValueError("Existing-image expansion requires batch_size=1.")
        if not normalized["outpaint_source_image"]:
            raise ValueError("Choose a source image before enabling existing-image expansion.")
        if int(normalized.get("width") or 0) % 8 or int(normalized.get("height") or 0) % 8:
            raise ValueError("Existing-image expansion target width and height must be divisible by 8.")
    seed_value = normalized.get("seed")
    normalized["seed"] = _coerce_top_level_number(seed_value, integer=True, default=None) if seed_value not in (None, "") else None
    normalized["save_images"] = _coerce_boolean(normalized.get("save_images", True), default=True)
    normalized["save_txt"] = _coerce_boolean(normalized.get("save_txt", True), default=True)
    normalized["save_json"] = _coerce_boolean(normalized.get("save_json", True), default=True)
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GenerationJob:
    job_id: str
    request: dict[str, Any]
    status: str = "queued"
    worker_stage: str = "queued"
    execution_mode: str = "pending"
    model_runtime_diagnostics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str | None = None
    status_changed_at: str = field(default_factory=_utc_now)
    last_progress_at: str | None = None
    last_runtime_line_at: str | None = None
    return_code: int | None = None
    output_paths: list[str] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    error: str | None = None
    job_root: str | None = None
    console_log_path: str | None = None
    failure_bundle_path: str | None = None
    live_preview_root: str | None = None
    live_preview_latest_path: str | None = None
    live_preview_path: str | None = None
    live_preview_url: str | None = None
    live_preview_decode_mode: str | None = None
    live_preview_history: list[dict[str, Any]] = field(default_factory=list)
    live_cfg_step_series: dict[str, Any] = field(default_factory=lambda: {
        "schema_version": 1,
        "coordinate": "live_denoising_step",
        "source": "preview_stream",
        "supports_future_step_overrides": True,
        "points": [],
    })
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float | None = None
    resolved_seed: int | None = None
    resolved_seeds: list[int] = field(default_factory=list)
    final_output_url: str | None = None
    live_preview_metrics: dict[str, Any] = field(default_factory=dict)
    sampling_timing: dict[str, Any] = field(default_factory=dict)
    memory_status: dict[str, Any] = field(default_factory=dict)
    sse_clients_connected: int = 0
    sse_clients_peak: int = 0
    stale_preview_events_ignored: int = 0
    terminal_events_emitted: int = 0
    model_selection: dict[str, Any] = field(default_factory=dict)
    model_diagnostics: dict[str, Any] = field(default_factory=dict)
    prompt_parser_diagnostics: dict[str, Any] = field(default_factory=dict)
    output_quality_diagnostics: dict[str, Any] = field(default_factory=dict)
    prompt_preflight: dict[str, Any] = field(default_factory=dict)
    scheduler_settings_requested: dict[str, Any] = field(default_factory=dict)
    scheduler_settings_effective: dict[str, Any] = field(default_factory=dict)
    scheduler_validation_warnings: list[str] = field(default_factory=list)
    scheduler_compatibility_policy: dict[str, Any] = field(default_factory=dict)
    scheduler_preset_reference: dict[str, Any] = field(default_factory=dict)
    scheduler_requested_hash: str | None = None
    scheduler_effective_hash: str | None = None
    scheduler_step_count_source: str | None = None
    scheduler_warnings_acknowledged: bool = False
    output_save_status: dict[str, Any] = field(default_factory=dict)
    output_save_events: list[dict[str, Any]] = field(default_factory=list)
    pending_save_batches: int = 0
    completed_save_batches: int = 0
    failed_save_batches: int = 0
    pause_after_current_requested: bool = False
    pause_requested_at: str | None = None
    paused_at: str | None = None
    resumed_at: str | None = None
    resume_count: int = 0
    skip_current_requested: bool = False
    skip_requested_at: str | None = None
    skipped_images: int = 0
    skipped_image_seeds: list[int] = field(default_factory=list)
    skip_events: list[dict[str, Any]] = field(default_factory=list)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "request": self.request,
            "status": self.status,
            "worker_stage": self.worker_stage,
            "execution_mode": self.execution_mode,
            "model_runtime_diagnostics": dict(self.model_runtime_diagnostics),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "status_changed_at": self.status_changed_at,
            "last_progress_at": self.last_progress_at,
            "last_runtime_line_at": self.last_runtime_line_at,
            "return_code": self.return_code,
            "output_paths": list(self.output_paths),
            "log_lines": self.log_lines[-80:],
            "error": self.error,
            "failure_stage_code": (
                extract_outpaint_failure_stage(self.error)
                or extract_hires_failure_stage(self.error)
            ),
            "failure_stage_label": (
                outpaint_failure_label(extract_outpaint_failure_stage(self.error))
                if extract_outpaint_failure_stage(self.error)
                else (
                    hires_failure_stage_label(extract_hires_failure_stage(self.error))
                    if extract_hires_failure_stage(self.error)
                    else ""
                )
            ),
            "failure_stage_domain": (
                "outpaint" if extract_outpaint_failure_stage(self.error)
                else ("hires" if extract_hires_failure_stage(self.error) else "")
            ),
            "job_root": self.job_root,
            "console_log_path": self.console_log_path,
            "failure_bundle_path": self.failure_bundle_path,
            "live_preview_root": self.live_preview_root,
            "live_preview_latest_path": self.live_preview_latest_path,
            "live_preview_path": self.live_preview_path,
            "live_preview_url": self.live_preview_url,
            "live_preview_decode_mode": self.live_preview_decode_mode,
            "live_preview_history": self.live_preview_history[-40:],
            "live_cfg_step_series": {
                **dict(self.live_cfg_step_series or {}),
                "points": list((self.live_cfg_step_series or {}).get("points") or []),
            },
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "resolved_seed": self.resolved_seed,
            "resolved_seeds": list(self.resolved_seeds),
            "final_output_url": self.final_output_url,
            "live_preview_metrics": dict(self.live_preview_metrics),
            "sampling_timing": dict(self.sampling_timing),
            "memory_status": dict(self.memory_status),
            "prompt_parser_diagnostics": dict(self.prompt_parser_diagnostics),
            "output_quality_diagnostics": dict(self.output_quality_diagnostics),
            "prompt_preflight": dict(self.prompt_preflight or self.request.get("prompt_preflight") or {}),
            "sse_clients_connected": int(self.sse_clients_connected),
            "sse_clients_peak": int(self.sse_clients_peak),
            "stale_preview_events_ignored": int(self.stale_preview_events_ignored),
            "terminal_events_emitted": int(self.terminal_events_emitted),
            "scheduler_name": self.request.get("scheduler_name"),
            "scheduler_preset_reference": dict(self.scheduler_preset_reference),
            "scheduler_preset_name": self.scheduler_preset_reference.get("name"),
            "scheduler_validation_warning_count": len(self.scheduler_validation_warnings),
            "scheduler_validation_warnings": list(self.scheduler_validation_warnings),
            "scheduler_compatibility_policy": dict(self.scheduler_compatibility_policy),
            "scheduler_requested_hash": self.scheduler_requested_hash,
            "scheduler_effective_hash": self.scheduler_effective_hash,
            "scheduler_step_count_source": self.scheduler_step_count_source,
            "scheduler_warnings_acknowledged": self.scheduler_warnings_acknowledged,
            "output_save_status": dict(self.output_save_status),
            "output_save_events": list(self.output_save_events[-16:]),
            "pending_save_batches": int(self.pending_save_batches),
            "completed_save_batches": int(self.completed_save_batches),
            "failed_save_batches": int(self.failed_save_batches),
            "pause_after_current_requested": bool(self.pause_after_current_requested),
            "pause_requested_at": self.pause_requested_at,
            "paused_at": self.paused_at,
            "resumed_at": self.resumed_at,
            "resume_count": int(self.resume_count),
            "skip_current_requested": bool(self.skip_current_requested),
            "skip_requested_at": self.skip_requested_at,
            "skipped_images": int(self.skipped_images),
            "skipped_image_seeds": list(self.skipped_image_seeds),
            "skip_events": list(self.skip_events[-32:]),
            "model_selection": dict(self.model_selection),
            "model_diagnostics": dict(self.model_diagnostics),
        }


class GenerationJobManager:
    """One-GPU FIFO queue backed by the canonical resident model runtime."""

    def __init__(
        self,
        context: ProjectContext,
        *,
        settings_provider: Callable[[], Mapping[str, Any]] | None = None,
        recent_output_provider: Callable[[Path], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self.context = context
        self.settings_provider = settings_provider
        self.recent_output_provider = recent_output_provider
        self.registry = RuntimeRegistrySystem(project_context=context)
        self.selections = WebUISelectionResolver(self.registry)
        self.jobs: dict[str, GenerationJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._started = False
        self._event_subscribers: dict[str, set[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._live_preview_history_limit = 64
        self._terminal_events_emitted: set[str] = set()
        self._queue_resume_event = asyncio.Event()
        self._queue_resume_event.set()
        self._job_resume_events: dict[str, asyncio.Event] = {}
        self._queue_pause_requested_at: str | None = None
        self._queue_pause_owner_job_id: str | None = None
        self.runtime_startup_options: dict[str, Any] = {}
        self._last_cleanup_report: dict[str, Any] = {}
        self._last_job_cache_report: dict[str, Any] = {}
        self.model_runtime = ResidentModelRuntimeClient(context)
        self._watchdog_task: asyncio.Task[None] | None = None
        self._watchdog_report: dict[str, Any] = {
            "enabled": True,
            "running": False,
            "interval_seconds": 5,
            "running_stall_timeout_seconds": 180,
            "transition_stall_timeout_seconds": 120,
            "checks": 0,
            "recoveries": 0,
            "last_check_at": None,
            "last_recovery_at": None,
            "last_recovery_reason": None,
            "last_recovery_job_id": None,
        }

    def _runtime_request_values(self) -> dict[str, Any]:
        if self.runtime_startup_options:
            options = self.runtime_startup_options
        else:
            options = resolve_runtime_startup_options(
                environment={},
                settings=self._application_settings(),
            )
        values = runtime_request_settings(options)
        application_settings = self._application_settings()
        raw_overrides = application_settings.get("runtime_job_overrides")
        overrides = dict(raw_overrides) if isinstance(raw_overrides, Mapping) else {}
        if not overrides:
            return values

        normalized = runtime_request_settings(
            resolve_runtime_startup_options(environment={}, settings=overrides)
        )
        for key in RUNTIME_REPLAY_JOB_FIELDS:
            if key in overrides and key in normalized:
                values[key] = normalized[key]
        return values

    def _live_preview_request_values(self, job_root: Path) -> dict[str, Any]:
        application_settings = (
            dict(self.settings_provider() or {})
            if callable(self.settings_provider)
            else {}
        )
        live_preview_root = job_root / "live-preview"
        return {
            "live_preview_enabled": application_settings.get(
                "live_preview_enabled", True
            ),
            # Step/progress/CFG telemetry is independent of decoded image frames.
            # This keeps the graph and sampler timing alive when the image-preview
            # checkbox is disabled or image decoding is suspended.
            "live_preview_telemetry_enabled": True,
            "live_preview_mode": FORCED_LIVE_PREVIEW_MODE,
            "live_preview_interval": application_settings.get(
                "live_preview_interval", DEFAULT_FORCED_LIVE_PREVIEW_INTERVAL
            ),
            "live_preview_width": application_settings.get(
                "live_preview_width", 384
            ),
            "live_preview_format": application_settings.get(
                "live_preview_format", "webp"
            ),
            "live_preview_keep_history": application_settings.get(
                "live_preview_keep_history", "current_job"
            ),
            "live_preview_batch_index": application_settings.get(
                "live_preview_batch_index", 0
            ),
            "live_preview_quality": application_settings.get(
                "live_preview_quality", 78
            ),
            "live_preview_adaptive_throttle": application_settings.get(
                "live_preview_adaptive_throttle", True
            ),
            "live_preview_adaptive_target_ratio": application_settings.get(
                "live_preview_adaptive_target_ratio", 0.75
            ),
            "live_preview_adaptive_recovery_ratio": application_settings.get(
                "live_preview_adaptive_recovery_ratio", 0.40
            ),
            "live_preview_adaptive_max_interval": application_settings.get(
                "live_preview_adaptive_max_interval", 8
            ),
            "live_preview_adaptive_window": application_settings.get(
                "live_preview_adaptive_window", 6
            ),
            "live_preview_adaptive_suspend_on_overhead": application_settings.get(
                "live_preview_adaptive_suspend_on_overhead", False
            ),
            "cfg_lab_enabled": application_settings.get("cfg_lab_enabled", False),
            "live_preview_cfg_visual_enabled": bool(
                application_settings.get("cfg_lab_enabled", False)
                and application_settings.get("live_preview_cfg_visual_enabled", False)
            ),
            "diagnostics": self._diagnostics_request_settings(application_settings),
            "external_vae_override_enabled": True,
            "vae_mode": "checkpoint_embedded_auto",
            **self._runtime_request_values(),
            "memory_pinned_cpu_memory": application_settings.get(
                "memory_pinned_cpu_memory", False
            ),
            "memory_allow_tiled_vae_fallback": application_settings.get(
                "memory_allow_tiled_vae_fallback", True
            ),
            "memory_allow_preview_suspension_on_oom": application_settings.get(
                "memory_allow_preview_suspension_on_oom", True
            ),
            "live_preview_root": str(live_preview_root),
            "live_preview_clone_tensors": False,
            "live_preview_async": True,
            "progress_json": True,
        }

    @staticmethod
    def _merge_runtime_preview_values(
        request_payload: dict[str, Any],
        preview_values: Mapping[str, Any],
    ) -> None:
        replay_fields = set(RUNTIME_REPLAY_JOB_FIELDS)
        for key, value in preview_values.items():
            if key in replay_fields and key in request_payload:
                continue
            request_payload[key] = value

    def _application_settings(self) -> dict[str, Any]:
        return dict(self.settings_provider() or {}) if callable(self.settings_provider) else {}

    @staticmethod
    def _diagnostics_request_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
        mode = str(settings.get("diagnostics_mode") or "failures_only").strip().lower()
        diagnostic_decode_enabled = bool(settings.get("diagnostic_decode_enabled", False))
        if mode == "off":
            return {
                "mode": "off",
                "failure_bundles": False,
                "export_events": False,
                "tensor_summaries": False,
                "tensor_statistics": False,
                "capture_output_quality": False,
                "diagnostic_decode_enabled": diagnostic_decode_enabled,
            }
        if mode == "every_run":
            return {
                "mode": "every_run",
                "failure_bundles": True,
                "export_events": True,
                "tensor_summaries": True,
                "tensor_statistics": False,
                "capture_output_quality": False,
                "diagnostic_decode_enabled": diagnostic_decode_enabled,
            }
        if mode == "deep_tensor":
            return {
                "mode": "deep_tensor",
                "failure_bundles": True,
                "export_events": True,
                "tensor_summaries": True,
                "tensor_statistics": True,
                "capture_output_quality": True,
                "diagnostic_decode_enabled": diagnostic_decode_enabled,
                "sampler_trace": {
                    "enabled": True,
                    "export_json": True,
                    "export_csv": False,
                    "export_txt_summary": True,
                    "capture_latents": False,
                    "capture_latent_every_n": 0,
                },
            }
        return {
            "mode": "failures_only",
            "failure_bundles": True,
            "export_events": False,
            "tensor_summaries": True,
            "tensor_statistics": False,
            "capture_output_quality": False,
            "diagnostic_decode_enabled": diagnostic_decode_enabled,
        }

    def _model_runtime_settings(self) -> dict[str, Any]:
        settings = self._application_settings()
        return {
            **self._runtime_request_values(),
            "memory_pinned_cpu_memory": settings.get("memory_pinned_cpu_memory", False),
            "memory_allow_tiled_vae_fallback": settings.get("memory_allow_tiled_vae_fallback", True),
            "memory_allow_preview_suspension_on_oom": settings.get("memory_allow_preview_suspension_on_oom", True),
        }

    async def activate_model(
        self,
        model_path: str,
        *,
        selection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_path = str(model_path or "").strip()
        if not resolved_path:
            raise ValueError("A checkpoint model path is required for activation.")
        completion = await self.model_runtime.activate(
            resolved_path,
            runtime_settings=self._model_runtime_settings(),
        )
        if not completion.get("ok"):
            raise RuntimeError(str(completion.get("error") or "Model activation failed."))
        result = dict(completion.get("result") or {})
        status = self.model_runtime.status()
        current_model_path = str(status.get("current_model_path") or "").strip()
        if not current_model_path:
            raise RuntimeError("Model activation completed without a resident checkpoint path.")
        expected = os.path.normcase(str(Path(resolved_path).expanduser().resolve()))
        actual = os.path.normcase(str(Path(current_model_path).expanduser().resolve()))
        if actual != expected:
            raise RuntimeError(
                "Model activation completed for a different checkpoint than the dropdown selection."
            )
        if torch.cuda.is_available() and not bool(status.get("gpu_loaded")):
            devices = dict(status.get("component_devices") or {})
            raise RuntimeError(
                "The selected checkpoint did not become fully GPU-resident. "
                f"Component devices: {devices or 'unavailable'}"
            )
        if selection:
            result["selection"] = dict(selection)
        return {**completion, "status": status, "result": result}

    async def unload_model(self) -> dict[str, Any]:
        completion = await self.model_runtime.unload()
        if not completion.get("ok"):
            raise RuntimeError(str(completion.get("error") or "Model unload failed."))
        return dict(completion.get("result") or {})

    def model_runtime_status(self) -> dict[str, Any]:
        return self.model_runtime.status()


    @property
    def jobs_root(self) -> Path:
        return self.context.data_root / "webui" / "jobs"

    def clear_job_cache(
        self,
        *,
        preserve_active: bool = True,
        startup: bool = False,
    ) -> dict[str, Any]:
        """Delete session-only WebUI job data without touching final output images."""

        root = self.jobs_root
        root.mkdir(parents=True, exist_ok=True)
        active_statuses = {"queued", *_ACTIVE_JOB_STATUSES}
        preserved_ids = {
            job.job_id
            for job in self.jobs.values()
            if preserve_active and job.status in active_statuses
        }
        report: dict[str, Any] = {
            "startup": bool(startup),
            "root": str(root),
            "removed_job_ids": [],
            "removed_files": [],
            "removed_bytes": 0,
            "preserved_active": sorted(preserved_ids),
            "final_outputs_deleted": 0,
        }

        for item in list(root.iterdir()):
            if item.name in preserved_ids:
                continue
            size = self._directory_size(item) if item.is_dir() else 0
            if item.is_file():
                try:
                    size = int(item.stat().st_size)
                except OSError:
                    size = 0
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    report["removed_job_ids"].append(item.name)
                else:
                    item.unlink()
                    report["removed_files"].append(item.name)
            except OSError:
                continue
            report["removed_bytes"] += size

        for job_id, job in list(self.jobs.items()):
            if job_id in preserved_ids:
                continue
            subscribers = self._event_subscribers.pop(job_id, set())
            for queue in list(subscribers):
                self._offer_event(queue, None)
            self.jobs.pop(job_id, None)
            self._terminal_events_emitted.discard(job_id)

        report["removed_count"] = len(report["removed_job_ids"]) + len(report["removed_files"])
        report["remaining_job_directories"] = sum(1 for item in root.iterdir() if item.is_dir())
        self._last_job_cache_report = report
        return report

    @staticmethod
    def _directory_size(path: Path) -> int:
        total = 0
        if not path.exists():
            return total
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += int(item.stat().st_size)
            except OSError:
                continue
        return total

    def cleanup_preview_directories(self, *, now_timestamp: float | None = None) -> dict[str, Any]:
        """Remove only old job preview directories, never final txt2img outputs."""

        settings = self._application_settings()
        enabled = _coerce_boolean(settings.get("live_preview_cleanup_enabled", True), True)
        jobs_root = self.jobs_root
        report = {
            "enabled": enabled,
            "removed": [],
            "removed_bytes": 0,
            "preserved_active": [],
            "remaining_preview_directories": 0,
        }
        if not enabled or not jobs_root.exists():
            self._last_cleanup_report = report
            return report

        try:
            max_age_days = max(0, int(settings.get("live_preview_retention_days", 7)))
        except (TypeError, ValueError):
            max_age_days = 7
        try:
            max_jobs = max(1, int(settings.get("live_preview_retention_jobs", 24)))
        except (TypeError, ValueError):
            max_jobs = 24
        try:
            max_bytes = max(1, int(float(settings.get("live_preview_disk_budget_mb", 1024)) * 1024 * 1024))
        except (TypeError, ValueError):
            max_bytes = 1024 * 1024 * 1024

        now_value = float(now_timestamp if now_timestamp is not None else datetime.now(timezone.utc).timestamp())
        active_ids = {
            job.job_id for job in self.jobs.values()
            if job.status == "queued" or job.status in _ACTIVE_JOB_STATUSES
        }
        entries: list[dict[str, Any]] = []
        for job_root in jobs_root.iterdir():
            preview_root = job_root / "live-preview"
            if not preview_root.is_dir():
                continue
            try:
                modified = max(item.stat().st_mtime for item in [preview_root, *preview_root.glob("*")])
            except (OSError, ValueError):
                modified = preview_root.stat().st_mtime
            entries.append({
                "job_id": job_root.name,
                "path": preview_root,
                "modified": float(modified),
                "bytes": self._directory_size(preview_root),
            })

        entries.sort(key=lambda item: item["modified"], reverse=True)
        keep: list[dict[str, Any]] = []
        remove: list[dict[str, Any]] = []
        cutoff = now_value - (max_age_days * 86400) if max_age_days > 0 else None
        for entry in entries:
            if entry["job_id"] in active_ids:
                keep.append(entry)
                report["preserved_active"].append(entry["job_id"])
            elif cutoff is not None and entry["modified"] < cutoff:
                remove.append(entry)
            else:
                keep.append(entry)

        non_active_keep = [item for item in keep if item["job_id"] not in active_ids]
        for entry in non_active_keep[max_jobs:]:
            keep.remove(entry)
            remove.append(entry)

        total_bytes = sum(int(item["bytes"]) for item in keep)
        for entry in sorted(
            [item for item in keep if item["job_id"] not in active_ids],
            key=lambda item: item["modified"],
        ):
            if total_bytes <= max_bytes:
                break
            keep.remove(entry)
            remove.append(entry)
            total_bytes -= int(entry["bytes"])

        seen: set[Path] = set()
        for entry in remove:
            path = Path(entry["path"])
            if path in seen or entry["job_id"] in active_ids:
                continue
            seen.add(path)
            try:
                shutil.rmtree(path)
            except OSError:
                continue
            report["removed"].append(entry["job_id"])
            report["removed_bytes"] += int(entry["bytes"])

        report["remaining_preview_directories"] = sum(
            1 for job_root in jobs_root.iterdir() if (job_root / "live-preview").is_dir()
        )
        report["disk_budget_bytes"] = max_bytes
        report["retention_days"] = max_age_days
        report["retention_jobs"] = max_jobs
        self._last_cleanup_report = report
        return report

    def _touch_job_runtime(self, job: GenerationJob, *, progress: bool = False) -> str:
        now = _utc_now()
        job.updated_at = now
        job.last_runtime_line_at = now
        if progress:
            job.last_progress_at = now
        return now

    def _transition_job(
        self,
        job: GenerationJob,
        *,
        status: str | None = None,
        worker_stage: str | None = None,
    ) -> str:
        now = _utc_now()
        status_value = str(status or job.status)
        stage_value = str(worker_stage or job.worker_stage)
        if status_value != job.status:
            job.status = status_value
            job.status_changed_at = now
        if stage_value != job.worker_stage:
            job.worker_stage = stage_value
        job.updated_at = now
        return now

    def _recent_output_payload(self, image_path: str | Path) -> dict[str, Any] | None:
        try:
            resolved = Path(image_path).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

        payload: Mapping[str, Any] | None = None
        provider = self.recent_output_provider
        if callable(provider):
            try:
                payload = provider(resolved)
            except TypeError:
                payload = provider(Path(resolved))
            except Exception:
                payload = None
        else:
            try:
                from image_gen.webui.catalog import WebUICatalog

                payload = WebUICatalog(self.context).output_summary_from_path(resolved)
            except Exception:
                payload = None

        if isinstance(payload, Mapping):
            return dict(payload)
        return None

    def _record_job_output(
        self,
        job: GenerationJob,
        image_path: str | Path,
        *,
        seed_text: str | None = None,
    ) -> dict[str, Any] | None:
        image_value = str(image_path)
        self._transition_job(job, status="finalizing", worker_stage="saving_output")
        if image_value not in job.output_paths:
            job.output_paths.append(image_value)
        job.final_output_url = self._output_url_for_path(image_value)
        if job.resolved_seed is None and seed_text not in (None, ""):
            try:
                job.resolved_seed = int(str(seed_text))
            except (TypeError, ValueError):
                pass
        job.updated_at = _utc_now()
        self._persist_job(job)
        recent_output = self._recent_output_payload(image_value)
        payload = {
            "latest_output_path": image_value,
            "latest_output_url": job.final_output_url,
            "output_count": len(job.output_paths),
        }
        if recent_output is not None:
            payload["recent_output"] = recent_output
        self._publish_event(job, "job-output-produced", **payload)
        return recent_output

    def _watchdog_settings(self) -> dict[str, Any]:
        settings = self._application_settings()
        enabled = _coerce_boolean(settings.get("queue_watchdog_enabled", True), True)
        try:
            interval = max(2.0, float(settings.get("queue_watchdog_interval_seconds", 5) or 5.0))
        except (TypeError, ValueError):
            interval = 5.0
        try:
            running_timeout = max(30.0, float(settings.get("queue_watchdog_running_stall_timeout_seconds", 180) or 180.0))
        except (TypeError, ValueError):
            running_timeout = 180.0
        try:
            transition_timeout = max(20.0, float(settings.get("queue_watchdog_transition_stall_timeout_seconds", 120) or 120.0))
        except (TypeError, ValueError):
            transition_timeout = 120.0
        try:
            finalizing_timeout = max(60.0, float(settings.get("queue_watchdog_finalizing_stall_timeout_seconds", 600) or 600.0))
        except (TypeError, ValueError):
            finalizing_timeout = 600.0
        self._watchdog_report.update(
            {
                "enabled": enabled,
                "interval_seconds": interval,
                "running_stall_timeout_seconds": running_timeout,
                "transition_stall_timeout_seconds": transition_timeout,
                "finalizing_stall_timeout_seconds": finalizing_timeout,
            }
        )
        return {
            "enabled": enabled,
            "interval_seconds": interval,
            "running_stall_timeout_seconds": running_timeout,
            "transition_stall_timeout_seconds": transition_timeout,
            "finalizing_stall_timeout_seconds": finalizing_timeout,
        }

    def _job_last_activity_timestamp(self, job: GenerationJob) -> float:
        candidates = [
            _timestamp_from_iso(job.last_progress_at),
            _timestamp_from_iso(job.last_runtime_line_at),
            _timestamp_from_iso(job.updated_at),
            _timestamp_from_iso(job.status_changed_at),
            _timestamp_from_iso(job.started_at),
        ]
        if not any(value is not None for value in candidates):
            candidates.append(_timestamp_from_iso(job.created_at))
        return max((value for value in candidates if value is not None), default=datetime.now(timezone.utc).timestamp())

    def _job_stall_reason(
        self,
        job: GenerationJob,
        *,
        now_timestamp: float,
        runtime_status: Mapping[str, Any] | None,
        settings: Mapping[str, Any],
    ) -> str | None:
        if job.status not in _ACTIVE_JOB_STATUSES or job.status == "paused":
            return None
        if job.status == "running":
            timeout = float(settings.get("running_stall_timeout_seconds"))
        elif job.status == "finalizing":
            timeout = float(settings.get("finalizing_stall_timeout_seconds"))
        else:
            timeout = float(settings.get("transition_stall_timeout_seconds"))
        stale_for = now_timestamp - self._job_last_activity_timestamp(job)
        if job.execution_mode == "resident_model":
            worker_job_id = str((runtime_status or {}).get("current_job_id") or "").strip()
            worker_stage = str((runtime_status or {}).get("stage") or "idle").strip().lower()
            worker_online = bool((runtime_status or {}).get("online", True))
            if not worker_online and stale_for >= min(timeout, 15.0):
                return f"Model runtime went offline while {job.job_id} remained {job.status}."
            if worker_job_id and worker_job_id == job.job_id:
                return None if stale_for < timeout else (
                    f"Model runtime still reports {job.job_id} active, but no runtime activity was observed for {stale_for:.1f} seconds."
                )
            if stale_for >= min(timeout, 15.0) and not worker_job_id and worker_stage in {"idle", "ready"}:
                if job.status == "finalizing" and (job.output_paths or (job.total_steps and job.current_step >= job.total_steps)):
                    return None
                return (
                    "Model runtime no longer reports an active job, but the queue still marked "
                    f"{job.job_id} as {job.status}."
                )
            if stale_for >= min(timeout, 15.0) and worker_stage in {"failed", "offline"}:
                return f"Model runtime entered {worker_stage} while {job.job_id} remained {job.status}."
        elif job.process is None and stale_for >= min(timeout, 15.0):
            return f"The isolated generation process disappeared while {job.job_id} remained {job.status}."
        if stale_for >= timeout:
            return f"No runtime activity was observed for {stale_for:.1f} seconds while {job.job_id} remained {job.status}."
        return None

    async def _recover_terminal_job(
        self,
        job: GenerationJob,
        *,
        reason: str,
        source: str,
    ) -> dict[str, Any]:
        timestamp = _utc_now()
        entry = {
            "timestamp": timestamp,
            "source": source,
            "reason": reason,
        }
        job.log_lines.append(f"{source.upper()} RECOVERY: {reason}")
        job.model_runtime_diagnostics.setdefault("recovery_actions", []).append(entry)
        job.model_diagnostics.setdefault("recovery", []).append(entry)
        if job.execution_mode == "resident_model":
            try:
                await self.model_runtime.stop()
                entry["model_runtime_stopped"] = True
            except Exception as exc:  # pragma: no cover - best effort
                entry["model_runtime_stop_error"] = f"{type(exc).__name__}: {exc}"
        elif job.process is not None:
            try:
                job.process.terminate()
                entry["process_terminated"] = True
            except Exception as exc:  # pragma: no cover - best effort
                entry["process_terminate_error"] = f"{type(exc).__name__}: {exc}"
        job.error = reason
        job.return_code = 130 if job.status == "cancelling" else 1
        terminal_status = "cancelled" if job.status == "cancelling" else "failed"
        self._transition_job(job, status=terminal_status, worker_stage=terminal_status)
        job.completed_at = timestamp
        job.process = None
        self._watchdog_report["recoveries"] = int(self._watchdog_report.get("recoveries", 0) or 0) + 1
        self._watchdog_report["last_recovery_at"] = timestamp
        self._watchdog_report["last_recovery_reason"] = reason
        self._watchdog_report["last_recovery_job_id"] = job.job_id
        if job.execution_mode == "resident_model":
            self._finalize_resident_job(job)
        else:
            job.model_diagnostics["live_preview"] = self.diagnostics_payload(job)["phase09h_validation"]
            if job.job_root:
                (Path(job.job_root) / "model-diagnostics.json").write_text(
                    json.dumps(job.model_diagnostics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            self._persist_job(job)
            terminal_event = {
                "completed": "job-completed",
                "cancelled": "job-cancelled",
                "failed": "job-failed",
            }.get(job.status, "job-progress")
            self._publish_terminal_once(job, terminal_event)
            self.cleanup_preview_directories()
        return entry

    async def _run_watchdog_check(self) -> None:
        settings = self._watchdog_settings()
        self._watchdog_report["last_check_at"] = _utc_now()
        self._watchdog_report["checks"] = int(self._watchdog_report.get("checks", 0) or 0) + 1
        if not settings["enabled"]:
            return
        now_timestamp = datetime.now(timezone.utc).timestamp()
        runtime_status = self.model_runtime.status()
        for job in list(self.jobs.values()):
            if (
                job.execution_mode == "resident_model"
                and job.status == "finalizing"
                and not str(runtime_status.get("current_job_id") or "").strip()
                and str(runtime_status.get("stage") or "").lower() in {"ready", "idle"}
                and (job.output_paths or (job.total_steps and job.current_step >= job.total_steps))
            ):
                job.return_code = 0
                self._transition_job(job, status="completed", worker_stage="completed")
                self._finalize_resident_job(job)
                continue
            reason = self._job_stall_reason(job, now_timestamp=now_timestamp, runtime_status=runtime_status, settings=settings)
            if reason:
                await self._recover_terminal_job(job, reason=reason, source="watchdog")
                break

    async def _watchdog_loop(self) -> None:
        while not self._stopping:
            settings = self._watchdog_settings()
            self._watchdog_report["running"] = True
            try:
                await asyncio.sleep(float(settings["interval_seconds"]))
                await self._run_watchdog_check()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive watchdog logging
                self._watchdog_report["last_error"] = f"{type(exc).__name__}: {exc}"
        self._watchdog_report["running"] = False

    def clear_queued_jobs(self, *, reason: str = "Queued jobs were cleared from the WebUI.") -> dict[str, Any]:
        report = {
            "cleared_job_ids": [],
            "cleared_count": 0,
            "reason": reason,
        }
        for job in self.jobs.values():
            if job.status != "queued":
                continue
            timestamp = self._transition_job(job, status="cancelled", worker_stage="cancelled")
            job.completed_at = timestamp
            job.error = reason
            job.log_lines.append(f"MANUAL CLEAR: {reason}")
            self._persist_job(job)
            self._publish_terminal_once(job, "job-cancelled")
            report["cleared_job_ids"].append(job.job_id)
        report["cleared_count"] = len(report["cleared_job_ids"])
        return report

    def dismiss_terminal_jobs(self) -> dict[str, Any]:
        terminal = {"completed", "cancelled", "failed"}
        removed: list[str] = []
        for job_id, job in list(self.jobs.items()):
            if job.status not in terminal:
                continue
            subscribers = self._event_subscribers.pop(job_id, set())
            for queue in list(subscribers):
                self._offer_event(queue, None)
            self.jobs.pop(job_id, None)
            self._terminal_events_emitted.discard(job_id)
            removed.append(job_id)
        return {"removed_job_ids": removed, "removed_count": len(removed)}

    async def recover_worker(
        self,
        *,
        clear_active: bool = True,
        clear_queue: bool = False,
        reason: str = "Manual recovery requested from the WebUI.",
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "reason": reason,
            "clear_active": bool(clear_active),
            "clear_queue": bool(clear_queue),
            "active_job_id": None,
            "worker_stopped": False,
            "queue": None,
        }
        active = next((job for job in self.jobs.values() if job.status in _ACTIVE_JOB_STATUSES), None)
        if clear_active and active is not None:
            report["active_job_id"] = active.job_id
            await self._recover_terminal_job(active, reason=reason, source="manual")
            report["worker_stopped"] = True
        else:
            try:
                await self.model_runtime.stop()
                report["worker_stopped"] = True
            except Exception as exc:
                report["worker_stop_error"] = f"{type(exc).__name__}: {exc}"
        if clear_queue:
            report["queue"] = self.clear_queued_jobs(reason="Queued jobs were cleared during manual recovery.")
        return report

    async def start(self) -> None:
        self._started = True
        self.clear_job_cache(preserve_active=True, startup=True)
        self.cleanup_preview_directories()
        self._watchdog_settings()
        if self._worker_task is None or self._worker_task.done():
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_loop())
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def stop(self) -> None:
        self._started = False
        self._stopping = True
        self._queue_resume_event.set()
        for resume_event in self._job_resume_events.values():
            resume_event.set()
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
        active_resident_job = next(
            (
                job
                for job in self.jobs.values()
                if job.execution_mode == "resident_model" and job.status in _ACTIVE_JOB_STATUSES
            ),
            None,
        )
        if active_resident_job is not None:
            await self.model_runtime.cancel_active(active_resident_job.job_id)
        for job in self.jobs.values():
            if job.process is not None and job.status in _ACTIVE_JOB_STATUSES:
                job.process.terminate()
        await self.model_runtime.stop()
        for subscribers in self._event_subscribers.values():
            for queue in list(subscribers):
                self._offer_event(queue, None)
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._watchdog_task is not None:
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        self._watchdog_report["running"] = False

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

    def preflight_scheduler(self, request: dict[str, Any]) -> dict[str, Any]:
        normalized = self.normalize_generation_request(request)
        resolution = scheduler_resolution_from_payload(normalized)
        return {
            "ok": True,
            "scheduler_name": normalized.get("scheduler_name"),
            "steps": normalized.get("steps"),
            "scheduler_kwargs": dict(normalized.get("scheduler_kwargs") or {}),
            "requested_settings": dict(resolution.get("requested_settings") or {}),
            "effective_settings": dict(resolution.get("effective_settings") or {}),
            "compatibility_policy": dict(resolution.get("compatibility_policy") or {}),
            "validation_warnings": list(resolution.get("validation_warnings") or []),
            "validation_warning_count": int(resolution.get("validation_warning_count", 0) or 0),
            "preset_reference": dict(resolution.get("preset_reference") or {}),
            "step_count_source": resolution.get("step_count_source"),
            "requested_hash": resolution.get("requested_hash"),
            "effective_hash": resolution.get("effective_hash"),
            "fallback_applied": bool(resolution.get("fallback_applied", False)),
        }

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _write_scheduler_artifacts(
        self,
        *,
        job_root: Path,
        requested_generation: Mapping[str, Any],
        normalized_generation: Mapping[str, Any],
        resolution: Mapping[str, Any],
    ) -> None:
        self._write_json(job_root / "requested-generation.json", dict(requested_generation))
        self._write_json(job_root / "normalized-generation.json", dict(normalized_generation))
        self._write_json(
            job_root / "scheduler-settings-requested.json",
            dict(resolution.get("requested_settings") or {}),
        )
        self._write_json(
            job_root / "scheduler-settings-effective.json",
            dict(resolution.get("effective_settings") or {}),
        )
        self._write_json(
            job_root / "scheduler-validation-warnings.json",
            list(resolution.get("validation_warnings") or []),
        )
        self._write_json(
            job_root / "scheduler-preset-reference.json",
            dict(resolution.get("preset_reference") or {}),
        )

    async def submit(
        self,
        request: dict[str, Any],
        *,
        model_selection: Mapping[str, Any] | None = None,
    ) -> GenerationJob:
        job_id = uuid.uuid4().hex[:12]
        requested_generation = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
        normalized = self.normalize_generation_request(request)
        resolution = scheduler_resolution_from_payload(normalized)
        warnings = list(resolution.get("validation_warnings") or [])
        acknowledged = _coerce_boolean(
            request.get("_webui_scheduler_warnings_acknowledged", not warnings),
            default=not warnings,
        )
        acknowledgement_required = _coerce_boolean(
            request.get("_webui_scheduler_requires_warning_acknowledgement", False),
            default=False,
        )
        if warnings and acknowledgement_required and not acknowledged:
            raise ValueError(
                "Scheduler settings produced warnings that must be acknowledged before queueing: "
                + " | ".join(warnings)
            )
        selected = dict(model_selection or {})
        job_root = self.context.data_root / "webui" / "jobs" / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        job = GenerationJob(
            job_id=job_id,
            request=normalized,
            job_root=str(job_root),
            prompt_preflight=dict(normalized.get("prompt_preflight") or {}),
            model_selection=selected,
            scheduler_settings_requested=dict(resolution.get("requested_settings") or {}),
            scheduler_settings_effective=dict(resolution.get("effective_settings") or {}),
            scheduler_validation_warnings=warnings,
            scheduler_compatibility_policy=dict(resolution.get("compatibility_policy") or {}),
            scheduler_preset_reference=dict(resolution.get("preset_reference") or {}),
            scheduler_requested_hash=resolution.get("requested_hash"),
            scheduler_effective_hash=resolution.get("effective_hash"),
            scheduler_step_count_source=resolution.get("step_count_source"),
            scheduler_warnings_acknowledged=acknowledged,
            model_diagnostics={
                "submission": {
                    "browser_requested_path": request.get("_webui_model_requested_path")
                    or request.get("model_path"),
                    "browser_resolved_path": request.get("_webui_model_browser_resolved_path"),
                    "browser_matches_active": request.get("_webui_model_browser_matches_active"),
                    "browser_resolve_error": request.get("_webui_model_browser_resolve_error"),
                    "browser_selection_id": request.get("_webui_model_selection_id"),
                    "backend_active_path": selected.get("resolved_path"),
                    "backend_selection_id": selected.get("selection_id"),
                    "normalized_request_path": normalized.get("model_path"),
                    "server_python_executable": sys.executable,
                    "server_python_version": sys.version,
                    "server_cwd": os.getcwd(),
                    "project_root": str(self.context.project_root),
                    "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
                },
                "scheduler_settings": dict(resolution),
            },
        )
        prompt_preflight_payload = dict(normalized.get("prompt_preflight") or {})
        if prompt_preflight_payload:
            (job_root / "prompt-preflight.json").write_text(
                json.dumps(prompt_preflight_payload, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        self._write_scheduler_artifacts(
            job_root=job_root,
            requested_generation=requested_generation,
            normalized_generation=normalized,
            resolution=resolution,
        )
        self.jobs[job_id] = job
        self._persist_job(job)
        if self._started and (self._worker_task is None or self._worker_task.done()):
            self._stopping = False
            self._worker_task = asyncio.create_task(self._worker_loop())
        await self._queue.put(job_id)
        return job

    async def cancel(self, job_id: str) -> GenerationJob | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.status == "queued":
            completed = self._transition_job(job, status="cancelled", worker_stage="cancelled")
            job.completed_at = completed
            self._persist_job(job)
            self._publish_terminal_once(job, "job-cancelled")
        elif job.status in _CANCELLABLE_JOB_STATUSES:
            was_paused = job.status == "paused"
            self._transition_job(job, status="cancelling", worker_stage="cancelling")
            job.pause_after_current_requested = False
            job.skip_current_requested = False
            resume_event = self._job_resume_events.get(job.job_id)
            if resume_event is not None:
                resume_event.set()
            if job.execution_mode == "resident_model" and not was_paused:
                await self.model_runtime.cancel_active(job.job_id)
            elif job.process is not None:
                job.process.terminate()
            self._persist_job(job)
            self._publish_event(job, "job-progress")
        return job

    def _active_generation_job(self) -> GenerationJob | None:
        return next(
            (job for job in self.jobs.values() if job.status in _ACTIVE_JOB_STATUSES),
            None,
        )

    def _resume_event_for_job(self, job_id: str) -> asyncio.Event:
        event = self._job_resume_events.get(job_id)
        if event is None:
            event = asyncio.Event()
            event.set()
            self._job_resume_events[job_id] = event
        return event

    async def pause_after_current(self, job_id: str | None = None) -> dict[str, Any]:
        active = self._active_generation_job()
        if job_id and (active is None or active.job_id != str(job_id)):
            raise ValueError("The requested generation is not the active queue item.")
        if active is not None and active.status in {"finalizing", "cancelling"}:
            raise ValueError("The active generation can no longer be paused between images.")

        requested_at = _utc_now()
        self._queue_pause_requested_at = requested_at
        self._queue_pause_owner_job_id = active.job_id if active is not None else None
        self._queue_resume_event.clear()

        if active is not None:
            active.pause_after_current_requested = True
            active.pause_requested_at = requested_at
            active.resumed_at = None
            self._resume_event_for_job(active.job_id).clear()
            if active.status != "paused":
                self._transition_job(
                    active,
                    worker_stage="pause_after_current_requested",
                )
            self._persist_job(active)
            self._publish_event(
                active,
                "job-progress",
                pause_after_current_requested=True,
                queue_pause_requested=True,
            )

        return self.status()

    async def resume_queue(self) -> dict[str, Any]:
        resumed_at = _utc_now()
        active = self._active_generation_job()
        if active is not None:
            active.pause_after_current_requested = False
            active.pause_requested_at = None
            active.resumed_at = resumed_at
            active.resume_count += 1
            resume_event = self._resume_event_for_job(active.job_id)
            resume_event.set()
            if active.status == "paused":
                self._transition_job(active, status="running", worker_stage="resuming_queue")
            self._persist_job(active)
            self._publish_event(
                active,
                "job-progress",
                queue_pause_requested=False,
                resumed_at=resumed_at,
            )
        for resume_event in self._job_resume_events.values():
            resume_event.set()
        self._queue_pause_requested_at = None
        self._queue_pause_owner_job_id = None
        self._queue_resume_event.set()
        return self.status()

    async def skip_current(self, job_id: str) -> GenerationJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "running":
            raise ValueError("Skip is only available while an image is actively generating.")
        runtime_status = self.model_runtime.status()
        if str(runtime_status.get("current_job_id") or "") != job.job_id:
            raise ValueError("The resident runtime is not currently sampling this generation.")
        if job.skip_current_requested:
            return job

        job.skip_current_requested = True
        job.skip_requested_at = _utc_now()
        self._transition_job(job, worker_stage="skipping_current_image")
        self._persist_job(job)
        self._publish_event(
            job,
            "job-progress",
            skip_current_requested=True,
            skip_requested_at=job.skip_requested_at,
        )
        await self.model_runtime.cancel_active(job.job_id)
        return job

    def get_job(self, job_id: str) -> GenerationJob | None:
        return self.jobs.get(job_id)

    @staticmethod
    def _persist_job(job: GenerationJob) -> None:
        if not job.job_root:
            return
        root = Path(job.job_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "job.json").write_text(
            json.dumps(job.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if job.failure_bundle_path:
            bundle = Path(job.failure_bundle_path)
            (root / "failure-link.json").write_text(
                json.dumps(
                    {
                        "failure_bundle_path": str(bundle),
                        "exists": bundle.exists(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def list_jobs(self) -> list[dict[str, Any]]:
        values = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
        return [item.to_dict() for item in values]

    def status(self) -> dict[str, Any]:
        active_job = self._active_generation_job()
        active = active_job.job_id if active_job is not None else None
        queued = sum(1 for job in self.jobs.values() if job.status == "queued")
        queue_pause_requested = not self._queue_resume_event.is_set()
        return {
            "online": self._worker_task is not None and not self._worker_task.done(),
            "active_job_id": active,
            "queued": queued,
            "queue_pause_requested": queue_pause_requested,
            "queue_paused": bool(
                queue_pause_requested
                and (active_job is None or active_job.status == "paused")
            ),
            "queue_pause_requested_at": self._queue_pause_requested_at,
            "queue_pause_owner_job_id": self._queue_pause_owner_job_id,
            "sse_clients_connected": sum(job.sse_clients_connected for job in self.jobs.values()),
            "preview_cleanup": dict(self._last_cleanup_report),
            "job_cache_cleanup": dict(self._last_job_cache_report),
            "watchdog": dict(self._watchdog_report),
            "model_runtime": self.model_runtime.status(),
        }

    def _offer_event(self, queue: asyncio.Queue[dict[str, Any] | None], payload: dict[str, Any] | None) -> None:
        try:
            queue.put_nowait(payload)
            return
        except asyncio.QueueFull:
            pass
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def _build_event_payload(self, job: GenerationJob, event_type: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "type": event_type,
            "timestamp": _utc_now(),
            "job_id": job.job_id,
            "status": job.status,
            "job": job.to_dict(),
        }
        payload.update(extra)
        return payload

    def _publish_event(self, job: GenerationJob, event_type: str, **extra: Any) -> None:
        payload = self._build_event_payload(job, event_type, **extra)
        subscribers = self._event_subscribers.get(job.job_id, set())
        for queue in list(subscribers):
            self._offer_event(queue, payload)

    def _publish_terminal_once(self, job: GenerationJob, event_type: str) -> bool:
        if job.job_id in self._terminal_events_emitted:
            return False
        self._terminal_events_emitted.add(job.job_id)
        job.terminal_events_emitted += 1
        self._publish_event(job, event_type)
        return True

    async def subscribe(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=1)
        subscribers = self._event_subscribers.setdefault(job_id, set())
        subscribers.add(queue)
        job = self.jobs.get(job_id)
        if job is not None:
            job.sse_clients_connected = len(subscribers)
            job.sse_clients_peak = max(job.sse_clients_peak, job.sse_clients_connected)
            job.live_preview_metrics["sse_clients_connected"] = job.sse_clients_connected
            job.live_preview_metrics["sse_clients_peak"] = job.sse_clients_peak
            initial = self._build_event_payload(
                job,
                "job-progress",
                current_step=job.current_step,
                total_steps=job.total_steps,
                progress_percent=job.progress_percent,
                live_preview_url=job.live_preview_url,
                live_preview_path=job.live_preview_path,
                live_preview_decode_mode=job.live_preview_decode_mode,
            )
            self._offer_event(queue, initial)
        try:
            while True:
                payload = await queue.get()
                if payload is None:
                    break
                yield payload
        finally:
            subscribers.discard(queue)
            if job is not None:
                job.sse_clients_connected = len(subscribers)
                job.live_preview_metrics["sse_clients_connected"] = job.sse_clients_connected
            if not subscribers:
                self._event_subscribers.pop(job_id, None)

    async def _worker_loop(self) -> None:
        while not self._stopping:
            await self._queue_resume_event.wait()
            job_id = await self._queue.get()
            await self._queue_resume_event.wait()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                self._queue.task_done()
                continue
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Defensive boundary: never strand later queue items.
                job.error = f"Unhandled queue runtime error: {type(exc).__name__}: {exc}"
                job.return_code = 1
                job.log_lines.extend(traceback.format_exc().splitlines()[-60:])
                self._transition_job(job, status="failed", worker_stage="failed")
                if job.execution_mode == "resident_model":
                    self._finalize_resident_job(job)
                else:
                    job.completed_at = _utc_now()
                    job.updated_at = job.completed_at
                    self._persist_job(job)
                    self._publish_terminal_once(job, "job-failed")
            finally:
                self._queue.task_done()

    def _preview_step_url(self, job: GenerationJob, step_number: int, *, updated_at: str | None = None) -> str:
        version = updated_at or job.updated_at or _utc_now()
        return f"/api/jobs/{job.job_id}/preview/{int(step_number)}?v={version}"

    def _preview_latest_url(self, job: GenerationJob, *, updated_at: str | None = None) -> str:
        version = updated_at or job.updated_at or _utc_now()
        return f"/api/jobs/{job.job_id}/preview/latest?v={version}"

    def _output_url_for_path(self, value: str | Path) -> str | None:
        try:
            output_root = self.context.txt2img_output_root.resolve()
            resolved = Path(value).expanduser().resolve()
            relative = resolved.relative_to(output_root).as_posix()
        except Exception:
            return None
        return f"/outputs/{quote(relative, safe='/')}"

    def _safe_within(self, root: Path, path: Path) -> Path | None:
        try:
            resolved_root = root.resolve()
            resolved_path = path.resolve()
            resolved_path.relative_to(resolved_root)
            return resolved_path
        except Exception:
            return None

    def live_preview_root_path(self, job: GenerationJob) -> Path | None:
        if not job.live_preview_root:
            return None
        root = Path(job.live_preview_root)
        if not root.exists():
            return root
        return self._safe_within(root, root)

    def live_preview_step_path(self, job: GenerationJob, step_number: int) -> Path | None:
        root = self.live_preview_root_path(job)
        if root is None:
            return None
        for item in reversed(job.live_preview_history):
            if int(item.get("step", 0)) != int(step_number):
                continue
            preview_path = item.get("preview_path")
            if preview_path:
                candidate = self._safe_within(root, Path(preview_path))
                if candidate is not None and candidate.is_file():
                    return candidate
            filename = item.get("filename")
            if filename:
                candidate = self._safe_within(root, root / str(filename))
                if candidate is not None and candidate.is_file() and candidate.name.startswith(f"step_{int(step_number):03d}"):
                    return candidate
        for candidate in sorted(root.glob(f"step_{int(step_number):03d}.*")):
            safe = self._safe_within(root, candidate)
            if safe is not None and safe.is_file():
                return safe
        return None

    def live_preview_latest_file(self, job: GenerationJob) -> Path | None:
        root = self.live_preview_root_path(job)
        if root is None:
            return None
        latest_json = root / "latest.json"
        if latest_json.is_file():
            try:
                latest = json.loads(latest_json.read_text(encoding="utf-8"))
            except Exception:
                latest = {}
            filename = latest.get("filename")
            if filename:
                candidate = self._safe_within(root, root / str(filename))
                if candidate is not None and candidate.is_file():
                    return candidate
        if job.live_preview_path:
            candidate = self._safe_within(root, Path(job.live_preview_path))
            if candidate is not None and candidate.is_file():
                return candidate
        return None

    def _apply_step_progress_payload(self, job: GenerationJob, payload: Mapping[str, Any]) -> bool:
        if job.status in {"cancelling", "cancelled", "failed", "completed"}:
            return False

        step_number = max(
            0,
            int(_coerce_top_level_number(payload.get("step"), integer=True, default=0) or 0),
        )
        total_steps = max(
            step_number,
            int(
                _coerce_top_level_number(
                    payload.get("total_steps"), integer=True, default=step_number
                )
                or step_number
            ),
        )
        phase_index = max(
            0,
            int(_coerce_top_level_number(payload.get("phase_index"), integer=True, default=0) or 0),
        )
        previous_phase = int(job.sampling_timing.get("phase_index") or 0)
        if phase_index == previous_phase and step_number < int(job.current_step or 0):
            return False

        progress_percent = _coerce_top_level_number(
            payload.get("progress_percent"), integer=False, default=None
        )
        if progress_percent is None:
            progress_percent = (step_number / max(total_steps, 1)) * 100.0
        progress_percent = min(max(float(progress_percent), 0.0), 100.0)
        updated_at = _utc_now()

        if step_number > 0 and total_steps > 0 and step_number >= total_steps:
            self._transition_job(job, status="finalizing", worker_stage="sampling_complete")
        else:
            self._transition_job(job, status="running", worker_stage="sampling")

        job.current_step = step_number
        job.total_steps = total_steps
        job.progress_percent = progress_percent
        job.sampling_timing = {
            "schema_version": int(
                _coerce_top_level_number(payload.get("schema_version"), integer=True, default=1)
                or 1
            ),
            "phase_index": phase_index,
            "description": str(payload.get("description") or "Sampling"),
            "unit": str(payload.get("unit") or "step"),
            "step": step_number,
            "total_steps": total_steps,
            "step_duration_ms": _coerce_top_level_number(
                payload.get("step_duration_ms"), integer=False, default=None
            ),
            "average_step_ms": _coerce_top_level_number(
                payload.get("average_step_ms"), integer=False, default=None
            ),
            "rolling_average_step_ms": _coerce_top_level_number(
                payload.get("rolling_average_step_ms"), integer=False, default=None
            ),
            "sampling_elapsed_ms": _coerce_top_level_number(
                payload.get("sampling_elapsed_ms"), integer=False, default=0.0
            ),
            "estimated_remaining_ms": _coerce_top_level_number(
                payload.get("estimated_remaining_ms"), integer=False, default=None
            ),
            "timed_step_count": int(
                _coerce_top_level_number(payload.get("timed_step_count"), integer=True, default=0)
                or 0
            ),
            "updated_at": updated_at,
        }
        job.updated_at = updated_at
        job.last_runtime_line_at = updated_at
        job.last_progress_at = updated_at
        self._persist_job(job)
        self._publish_event(
            job,
            "job-progress",
            current_step=job.current_step,
            total_steps=job.total_steps,
            progress_percent=job.progress_percent,
            sampling_timing=dict(job.sampling_timing),
        )
        return True

    def _apply_step_preview_payload(self, job: GenerationJob, payload: Mapping[str, Any]) -> bool:
        step_number = max(1, int(_coerce_top_level_number(payload.get("step"), integer=True, default=1) or 1))
        is_final = bool(payload.get("is_final", False))
        incoming_filename = str(payload.get("filename") or "")
        incoming_preview_path = str(payload.get("preview_path") or "")
        incoming_preview_suspended = bool(payload.get("preview_image_suspended", False))
        incoming_has_preview_image = bool(
            not incoming_preview_suspended
            and (incoming_filename or incoming_preview_path)
        )
        if job.status in {"cancelling", "cancelled", "failed"}:
            job.stale_preview_events_ignored += 1
            return False
        # Per-step telemetry is emitted immediately, while asynchronous image
        # encoding may finish several sampler steps later. Reject stale
        # telemetry, but retain delayed image frames without rolling progress
        # backward.
        if (
            step_number < int(job.current_step or 0)
            and not is_final
            and not incoming_has_preview_image
        ):
            job.stale_preview_events_ignored += 1
            job.live_preview_metrics["stale_preview_events_ignored"] = job.stale_preview_events_ignored
            return False
        total_steps = max(step_number, int(_coerce_top_level_number(payload.get("total_steps"), integer=True, default=step_number) or step_number))
        progress_percent = float(payload.get("progress_percent") or (step_number / max(total_steps, 1)) * 100.0)
        progress_percent = min(max(progress_percent, 0.0), 100.0)
        updated_at = str(payload.get("updated_at") or _utc_now())
        record = {
            "step": step_number,
            "total_steps": total_steps,
            "progress_percent": progress_percent,
            "decode_mode": str(payload.get("decode_mode") or "fast"),
            "filename": incoming_filename,
            "is_final": is_final,
            "telemetry_only": bool(payload.get("telemetry_only", False)),
            "updated_at": updated_at,
            "sampler_name": payload.get("sampler_name"),
            "scheduler_name": payload.get("scheduler_name"),
            "preview_path": incoming_preview_path,
            "image_width": int(_coerce_top_level_number(payload.get("image_width"), integer=True, default=0) or 0),
            "image_height": int(_coerce_top_level_number(payload.get("image_height"), integer=True, default=0) or 0),
            "sigma": _coerce_top_level_number(payload.get("sigma"), integer=False, default=None),
            "timestep": _coerce_top_level_number(payload.get("model_timestep"), integer=False, default=None),
            "requested_cfg_scale": _coerce_top_level_number(payload.get("requested_cfg_scale"), integer=False, default=None),
            "effective_cfg_scale": _coerce_top_level_number(payload.get("effective_cfg_scale"), integer=False, default=None),
            "guidance_mode": str(payload.get("guidance_mode") or payload.get("cfg_guidance_mode") or "flat"),
            "cfg_rescale": _coerce_top_level_number(payload.get("cfg_rescale"), integer=False, default=0.0),
            "cfg_rescale_applied": bool(payload.get("cfg_rescale_applied", False)),
            "override_source": str(payload.get("override_source") or "base_request"),
            "transition_id": payload.get("transition_id"),
            "preview_image_suspended": incoming_preview_suspended,
            "preview_image_suspension_reason": str(payload.get("preview_image_suspension_reason") or ""),
            "preview_image_suspension_source": str(payload.get("preview_image_suspension_source") or ""),
            "preview_decoder_released": bool(payload.get("preview_decoder_released", False)),
            "cfg_telemetry_continues": bool(payload.get("cfg_telemetry_continues", False)),
        }
        root = self.live_preview_root_path(job)
        if root is not None:
            filename = record["filename"]
            if filename and not record["preview_path"]:
                record["preview_path"] = str(root / filename)
        keep_history = str(job.request.get("live_preview_keep_history") or "current_job").strip().lower()
        has_preview_image = bool(
            not record["preview_image_suspended"]
            and (record["filename"] or record["preview_path"])
        )
        record["preview_url"] = (
            (
                self._preview_latest_url(job, updated_at=updated_at)
                if keep_history == "latest_only"
                else self._preview_step_url(job, step_number, updated_at=updated_at)
            )
            if has_preview_image
            else ""
        )

        previous_step = int(job.current_step or 0)
        previous_total = int(job.total_steps or 0)
        previous_percent = float(job.progress_percent or 0.0)
        if is_final or (step_number >= total_steps and step_number >= previous_step):
            self._transition_job(job, status="finalizing", worker_stage="finalizing")
        elif job.status != "finalizing":
            self._transition_job(job, status="running", worker_stage="sampling")
        job.current_step = max(previous_step, step_number)
        job.total_steps = max(previous_total, total_steps, job.current_step)
        job.progress_percent = max(previous_percent, progress_percent)
        job.updated_at = updated_at
        job.last_runtime_line_at = updated_at
        job.last_progress_at = updated_at
        if has_preview_image:
            job.live_preview_decode_mode = record["decode_mode"]
            job.live_preview_path = record["preview_path"] or None
            job.live_preview_url = record["preview_url"]

            history = [item for item in job.live_preview_history if int(item.get("step", 0)) != step_number]
            history.append(record)
            history.sort(key=lambda item: int(item.get("step", 0)))
            if len(history) > self._live_preview_history_limit:
                history = history[-self._live_preview_history_limit:]
            job.live_preview_history = history
        elif record["preview_image_suspended"]:
            job.live_preview_metrics["image_decode_suspended"] = True
            job.live_preview_metrics["image_decode_suspension_reason"] = record[
                "preview_image_suspension_reason"
            ]
            job.live_preview_metrics["image_decode_suspension_source"] = record[
                "preview_image_suspension_source"
            ]
            job.live_preview_metrics["preview_decoder_released"] = bool(
                record["preview_decoder_released"]
            )
            job.live_preview_metrics["cfg_telemetry_continues_during_preview_suspension"] = True

        requested_cfg = record.get("requested_cfg_scale")
        effective_cfg = record.get("effective_cfg_scale")
        if bool(job.request.get("cfg_lab_enabled", False)) and (requested_cfg is not None or effective_cfg is not None):
            if requested_cfg is None:
                requested_cfg = effective_cfg
            if effective_cfg is None:
                effective_cfg = requested_cfg
            series = dict(job.live_cfg_step_series or {})
            points = [
                dict(item)
                for item in (series.get("points") or [])
                if int(item.get("step_index", -1)) != step_number - 1
            ]
            points.append({
                "step_index": step_number - 1,
                "requested_cfg_scale": float(requested_cfg),
                "effective_cfg_scale": float(effective_cfg),
                "sigma": record.get("sigma"),
                "timestep": record.get("timestep"),
                "guidance_mode": record.get("guidance_mode") or "flat",
                "cfg_rescale": float(record.get("cfg_rescale") or 0.0),
                "cfg_rescale_applied": bool(record.get("cfg_rescale_applied", False)),
                "override_source": record.get("override_source") or "base_request",
                "transition_id": record.get("transition_id"),
            })
            points.sort(key=lambda item: int(item.get("step_index", 0)))
            series.update({
                "schema_version": 1,
                "coordinate": "live_denoising_step",
                "source": "preview_stream",
                "supports_future_step_overrides": True,
                "points": points,
            })
            job.live_cfg_step_series = series

        self._persist_job(job)
        self._publish_event(
            job,
            "step-preview",
            step=step_number,
            total_steps=total_steps,
            progress_percent=progress_percent,
            live_preview_url=job.live_preview_url,
            live_preview_path=job.live_preview_path,
            live_preview_decode_mode=job.live_preview_decode_mode,
            requested_cfg_scale=record.get("requested_cfg_scale"),
            effective_cfg_scale=record.get("effective_cfg_scale"),
            guidance_mode=record.get("guidance_mode"),
            cfg_rescale=record.get("cfg_rescale"),
            live_cfg_step_series=job.live_cfg_step_series,
        )
        self._publish_event(
            job,
            "job-progress",
            current_step=job.current_step,
            total_steps=job.total_steps,
            progress_percent=job.progress_percent,
            live_preview_url=job.live_preview_url,
            live_preview_path=job.live_preview_path,
            live_preview_decode_mode=job.live_preview_decode_mode,
            requested_cfg_scale=record.get("requested_cfg_scale"),
            effective_cfg_scale=record.get("effective_cfg_scale"),
            guidance_mode=record.get("guidance_mode"),
            cfg_rescale=record.get("cfg_rescale"),
            live_cfg_step_series=job.live_cfg_step_series,
        )
        return True

    def _apply_output_save_status_payload(
        self,
        job: GenerationJob,
        payload: Mapping[str, Any],
    ) -> None:
        normalized = dict(payload or {})
        event = str(normalized.get("event") or "")
        if not event:
            return
        job.pending_save_batches = max(0, int(_coerce_top_level_number(normalized.get("pending_batches"), integer=True, default=0) or 0))
        job.completed_save_batches = max(0, int(_coerce_top_level_number(normalized.get("completed_batches"), integer=True, default=0) or 0))
        job.failed_save_batches = max(0, int(_coerce_top_level_number(normalized.get("failed_batches"), integer=True, default=0) or 0))
        job.output_save_status = normalized
        job.output_save_events.append({**normalized, "updated_at": _utc_now()})
        if len(job.output_save_events) > 24:
            job.output_save_events = job.output_save_events[-24:]
        if event in {"enqueued", "started", "completed", "failed"} and job.status not in {"completed", "cancelled", "failed"}:
            if job.status != "running" or event in {"started", "failed"}:
                self._transition_job(job, status="finalizing", worker_stage="saving_output")
        self._touch_job_runtime(job, progress=False)
        self._persist_job(job)
        self._publish_event(
            job,
            "job-progress",
            output_save_status=dict(job.output_save_status),
            pending_save_batches=job.pending_save_batches,
            completed_save_batches=job.completed_save_batches,
            failed_save_batches=job.failed_save_batches,
        )

    def _apply_runtime_line(self, job: GenerationJob, line: str) -> None:
        runtime_status_match = _MODEL_RUNTIME_STATUS_LINE.match(line)
        if runtime_status_match:
            try:
                status_payload = json.loads(runtime_status_match.group(1))
            except json.JSONDecodeError:
                status_payload = {}
            if isinstance(status_payload, dict):
                stage = str(status_payload.get("stage") or "preparing_model")
                batch_orchestration = dict(
                    job.model_runtime_diagnostics.get("batch_orchestration") or {}
                )
                job.model_runtime_diagnostics = dict(status_payload)
                if batch_orchestration:
                    job.model_runtime_diagnostics["batch_orchestration"] = batch_orchestration
                runtime_memory = dict(status_payload.get("memory") or {})
                component_devices = dict(status_payload.get("component_devices") or {})
                if runtime_memory or component_devices:
                    active_gpu_components = [
                        str(component)
                        for component, device in component_devices.items()
                        if str(device or "").lower().startswith("cuda")
                    ]
                    offloaded_components = [
                        str(component)
                        for component, device in component_devices.items()
                        if not str(device or "").lower().startswith("cuda")
                    ]
                    previous = dict(job.memory_status or {})
                    previous_snapshot = dict(previous.get("latest_snapshot") or {})
                    previous_cuda = dict(previous_snapshot.get("cuda") or {})
                    current_allocated = runtime_memory.get("allocated_bytes")
                    current_reserved = runtime_memory.get("reserved_bytes")
                    previous_peak_allocated = previous.get("peak_allocated_vram_bytes")
                    previous_peak_reserved = previous.get("peak_reserved_vram_bytes")
                    normalized_cuda = normalize_cuda_memory_payload(
                        {
                            **previous_cuda,
                            "available": bool(runtime_memory),
                            "device_name": runtime_memory.get("device_name"),
                            "allocated_vram_bytes": current_allocated,
                            "reserved_vram_bytes": current_reserved,
                            "free_vram_bytes": runtime_memory.get("free_bytes"),
                            "total_vram_bytes": runtime_memory.get("total_bytes"),
                        }
                    )
                    job.memory_status = {
                        **previous,
                        "event": "model_runtime_status",
                        "stage": stage,
                        "active_stage": stage,
                        "active_gpu_components": active_gpu_components,
                        "offloaded_components": offloaded_components,
                        "latest_snapshot": {
                            **previous_snapshot,
                            "pipeline_stage": stage,
                            "cuda": normalized_cuda,
                        },
                        "peak_allocated_vram_bytes": max(
                            int(previous_peak_allocated or 0),
                            int(current_allocated or 0),
                        ),
                        "peak_reserved_vram_bytes": max(
                            int(previous_peak_reserved or 0),
                            int(current_reserved or 0),
                        ),
                        "job_peak_allocated_vram_bytes": max(
                            int(previous.get("job_peak_allocated_vram_bytes") or 0),
                            int(previous_peak_allocated or 0),
                            int(current_allocated or 0),
                        ),
                        "job_peak_reserved_vram_bytes": max(
                            int(previous.get("job_peak_reserved_vram_bytes") or 0),
                            int(previous_peak_reserved or 0),
                            int(current_reserved or 0),
                        ),
                        "telemetry_source": "resident_model_runtime",
                        "updated_at": _utc_now(),
                    }
                if stage in {"preparing_model", "loading_tokenizer"}:
                    self._transition_job(job, status="preparing_model", worker_stage=stage)
                elif stage in {"loading_checkpoint", "moving_to_gpu", "reusing_checkpoint", "model_ready"}:
                    self._transition_job(job, status="warming_model", worker_stage=stage)
                elif stage in {"applying_retention_policy", "ready"} and job.status not in {"completed", "cancelled", "failed"}:
                    self._transition_job(job, status="finalizing", worker_stage=stage)
                elif stage == "running":
                    self._transition_job(job, status="running", worker_stage=stage)
                else:
                    self._transition_job(job, worker_stage=stage)
                self._touch_job_runtime(job, progress=stage == "running")
                if stage == "failed" and job.status not in {"cancelled", "cancelling"}:
                    job.error = str(status_payload.get("error") or status_payload.get("last_error") or "Model runtime failed.")
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    worker_stage=job.worker_stage,
                    model_runtime=dict(status_payload),
                    memory_status=dict(job.memory_status),
                )
            return

        output_save_status_match = _ASYNC_OUTPUT_SAVE_STATUS_LINE.match(line)
        if output_save_status_match:
            try:
                payload = json.loads(output_save_status_match.group(1))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                self._apply_output_save_status_payload(job, payload)
            return

        output_save_error_match = _ASYNC_OUTPUT_SAVE_ERROR_LINE.match(line)
        if output_save_error_match:
            try:
                payload = json.loads(output_save_error_match.group(1))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                merged = dict(job.output_save_status or {})
                merged.update(payload)
                merged["event"] = merged.get("event") or "failed"
                self._apply_output_save_status_payload(job, merged)
            return

        seed_match = _GENERATION_SEED_LINE.match(line)
        if seed_match:
            try:
                seed_payload = json.loads(seed_match.group(1))
            except json.JSONDecodeError:
                seed_payload = {}
            try:
                job.resolved_seed = int(seed_payload.get("base_seed"))
            except (TypeError, ValueError):
                job.resolved_seed = None
            job.resolved_seeds = []
            for value in seed_payload.get("image_seeds") or []:
                try:
                    job.resolved_seeds.append(int(value))
                except (TypeError, ValueError):
                    continue
            self._touch_job_runtime(job)
            self._persist_job(job)
            self._publish_event(
                job,
                "job-progress",
                resolved_seed=job.resolved_seed,
                resolved_seeds=list(job.resolved_seeds),
            )
            return

        preview_summary_match = _LIVE_PREVIEW_SUMMARY_LINE.match(line)
        if preview_summary_match:
            try:
                preview_summary = json.loads(preview_summary_match.group(1))
            except json.JSONDecodeError:
                preview_summary = {}
            if isinstance(preview_summary, dict):
                self._transition_job(job, status="finalizing", worker_stage="finalizing")
                job.live_preview_metrics.update(preview_summary)
                job.live_preview_metrics["sse_clients_connected"] = job.sse_clients_connected
                job.live_preview_metrics["sse_clients_peak"] = job.sse_clients_peak
                job.live_preview_metrics["stale_preview_events_ignored"] = job.stale_preview_events_ignored
                self._touch_job_runtime(job)
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    live_preview_metrics=dict(job.live_preview_metrics),
                )
            return

        memory_status_match = _MEMORY_STATUS_LINE.search(line)
        if memory_status_match:
            try:
                memory_payload = json.loads(memory_status_match.group(1))
            except json.JSONDecodeError:
                memory_payload = {}
            if isinstance(memory_payload, dict):
                status_payload = _normalize_live_memory_status(
                    memory_payload.get("status") or {}
                )
                updated_at = _utc_now()
                job.memory_status = {
                    **status_payload,
                    "event": memory_payload.get("event"),
                    "stage": memory_payload.get("stage"),
                    "active_stage": memory_payload.get("active_stage"),
                    "updated_at": updated_at,
                }
                job.updated_at = updated_at
                job.last_runtime_line_at = updated_at
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    memory_status=dict(job.memory_status),
                )
            return

        image_match = _IMAGE_LINE.match(line)
        if image_match:
            self._record_job_output(
                job,
                image_match.group("path"),
                seed_text=image_match.group("seed"),
            )
            return

        failure_match = _FAILURE_BUNDLE_LINE.search(line)
        if failure_match:
            job.failure_bundle_path = failure_match.group(1).strip()

        runtime_diag_match = _RUNTIME_DIAGNOSTIC_LINE.match(line)
        if runtime_diag_match:
            try:
                payload = json.loads(runtime_diag_match.group(1))
            except json.JSONDecodeError:
                payload = {"parse_error": line}
            if isinstance(payload, dict):
                job.model_diagnostics["runtime_environment"] = payload

        model_match = _MODEL_DIAGNOSTIC_LINE.match(line)
        if model_match:
            try:
                payload = json.loads(model_match.group(1))
            except json.JSONDecodeError:
                payload = {"parse_error": line}
            if isinstance(payload, dict):
                job.model_diagnostics["runtime"] = payload
                job.model_runtime_diagnostics["resident_reuse_benefited"] = bool(payload.get("cache_reused"))

        output_quality_match = _OUTPUT_QUALITY_DIAGNOSTIC_LINE.match(line)
        if output_quality_match:
            try:
                payload = json.loads(output_quality_match.group(1))
            except json.JSONDecodeError:
                payload = {"parse_error": line}
            if isinstance(payload, dict):
                job.output_quality_diagnostics = dict(payload)
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    output_quality_diagnostics=dict(job.output_quality_diagnostics),
                )

        prompt_parser_match = _PROMPT_PARSER_DIAGNOSTIC_LINE.match(line)
        if prompt_parser_match:
            try:
                payload = json.loads(prompt_parser_match.group(1))
            except json.JSONDecodeError:
                payload = {"parse_error": line}
            if isinstance(payload, dict):
                job.prompt_parser_diagnostics = dict(payload)
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    prompt_parser_diagnostics=dict(job.prompt_parser_diagnostics),
                )

        step_progress_match = _STEP_PROGRESS_LINE.search(line)
        if step_progress_match:
            try:
                payload = json.loads(step_progress_match.group(1))
            except json.JSONDecodeError:
                payload = {"parse_error": line}
            if isinstance(payload, dict):
                self._apply_step_progress_payload(job, payload)

        step_preview_match = _STEP_PREVIEW_LINE.search(line)
        if step_preview_match:
            try:
                payload = json.loads(step_preview_match.group(1))
            except json.JSONDecodeError:
                payload = {"parse_error": line}
            if isinstance(payload, dict):
                self._apply_step_preview_payload(job, payload)

    def _apply_model_parity(self, job: GenerationJob) -> None:
        runtime_model = dict(job.model_diagnostics.get("runtime") or {})
        expected_model = str(job.model_selection.get("resolved_path") or job.request.get("model_path") or "").strip()
        loaded_model = str(
            runtime_model.get("loaded_path")
            or runtime_model.get("resolved_path")
            or runtime_model.get("requested_path")
            or ""
        ).strip()
        model_paths_match: bool | None = None
        if expected_model and loaded_model:
            expected_token = os.path.normcase(str(Path(expected_model).expanduser().resolve()))
            loaded_token = os.path.normcase(str(Path(loaded_model).expanduser().resolve()))
            model_paths_match = expected_token == loaded_token
            if not model_paths_match:
                self._transition_job(job, status="failed", worker_stage="failed")
                job.error = (
                    "Model parity violation: the WebUI selected checkpoint was not the "
                    "checkpoint loaded by the runtime. "
                    f"Selected: {expected_model}. Loaded: {loaded_model}."
                )
        job.model_diagnostics["model_parity"] = {
            "selected_path": expected_model,
            "loaded_path": loaded_model,
            "matches": model_paths_match,
            "enforced": bool(expected_model),
        }

    def diagnostics_payload(self, job: GenerationJob) -> dict[str, Any]:
        payload = job.to_dict()
        metrics = dict(job.live_preview_metrics)
        payload["phase11d_scheduler"] = {
            "requested_settings": dict(job.scheduler_settings_requested),
            "effective_settings": dict(job.scheduler_settings_effective),
            "compatibility_policy": dict(job.scheduler_compatibility_policy),
            "validation_warnings": list(job.scheduler_validation_warnings),
            "validation_warning_count": len(job.scheduler_validation_warnings),
            "preset_reference": dict(job.scheduler_preset_reference),
            "requested_hash": job.scheduler_requested_hash,
            "effective_hash": job.scheduler_effective_hash,
            "step_count_source": job.scheduler_step_count_source,
            "warnings_acknowledged": job.scheduler_warnings_acknowledged,
        }
        live_cfg_points = list((job.live_cfg_step_series or {}).get("points") or [])
        payload["phase11h1_live_cfg_preview"] = {
            "visual_enabled": bool(job.request.get("live_preview_cfg_visual_enabled", False)),
            "series": {
                **dict(job.live_cfg_step_series or {}),
                "points": live_cfg_points,
            },
            "latest_requested_cfg_scale": (
                live_cfg_points[-1].get("requested_cfg_scale") if live_cfg_points else None
            ),
            "latest_effective_cfg_scale": (
                live_cfg_points[-1].get("effective_cfg_scale") if live_cfg_points else None
            ),
        }
        payload["phase13_memory"] = {
            **dict(job.memory_status or {}),
            "requested_policy": job.request.get("memory_policy", "auto"),
            "vram_safety_margin_mb": job.request.get("memory_vram_safety_margin_mb", 1024),
            "allow_preview_suspension_on_oom": job.request.get(
                "memory_allow_preview_suspension_on_oom", True
            ),
            "cfg_telemetry_continues_during_preview_suspension": True,
        }
        payload["phase13c_async_output_save_pipeline"] = {
            "pending_save_batches": int(job.pending_save_batches),
            "completed_save_batches": int(job.completed_save_batches),
            "failed_save_batches": int(job.failed_save_batches),
            "latest_status": dict(job.output_save_status),
            "recent_events": list(job.output_save_events[-8:]),
        }
        payload["phase14k7_preview_memory_policy"] = {
            "requested_policy": job.request.get("preview_policy", "normal"),
            "image_decode_suspended": bool(
                metrics.get("image_decode_suspended")
                or (job.memory_status or {}).get("preview_image_decode_suspended")
            ),
            "suspension_reason": str(
                metrics.get("image_decode_suspension_reason")
                or (job.memory_status or {}).get("preview_image_decode_suspension_reason")
                or ""
            ),
            "suspension_source": str(
                metrics.get("image_decode_suspension_source")
                or (job.memory_status or {}).get("preview_image_decode_suspension_source")
                or ""
            ),
            "preview_decoder_released": bool(
                metrics.get("preview_decoder_released")
                or (job.memory_status or {}).get("preview_decoder_released")
            ),
            "cfg_telemetry_continues": True,
        }
        payload["phase13c_prompt_parser"] = {
            "requested_parser": job.request.get("prompt_parser_name", "legacy"),
            "requested_options": dict(job.request.get("prompt_parser_kwargs") or {}),
            "shortcut_profile_name": job.request.get("prompt_shortcut_profile_name", "legacy_default"),
            "shortcut_profile_snapshot": dict(job.request.get("prompt_shortcut_profile_snapshot") or {}),
            "parser_preset_name": job.request.get("prompt_parser_preset_name", ""),
            "runtime": dict(job.prompt_parser_diagnostics),
            "available": default_prompt_parser_registry().has(
                job.request.get("prompt_parser_name", "legacy"),
                require_available=True,
            ),
        }
        payload["model_residency"] = {
            "execution_mode": job.execution_mode,
            "worker_stage": job.worker_stage,
            "model_runtime_status": self.model_runtime.status(),
            "job_runtime_diagnostics": dict(job.model_runtime_diagnostics),
            "watchdog": dict(self._watchdog_report),
            "resident_reuse_benefited": bool(
                (job.model_diagnostics.get("runtime") or {}).get("cache_reused")
                or job.model_runtime_diagnostics.get("resident_reuse_benefited")
            ),
        }
        payload["phase09h_validation"] = {
            "preview_enabled": bool(job.request.get("live_preview_enabled", False)),
            "preview_mode": job.request.get("live_preview_mode"),
            "preview_interval": job.request.get("live_preview_interval"),
            "preview_width": job.request.get("live_preview_width"),
            "preview_format": job.request.get("live_preview_format"),
            "preview_frames_emitted": metrics.get("preview_frames_emitted", metrics.get("frames_processed", 0)),
            "preview_frames_failed": metrics.get("preview_frames_failed", metrics.get("worker_failures", 0)),
            "preview_decode_time_total_ms": metrics.get("preview_decode_time_total_ms", 0.0),
            "preview_encode_time_total_ms": metrics.get("preview_encode_time_total_ms", 0.0),
            "preview_last_step": metrics.get("preview_last_step", job.current_step),
            "sse_clients_connected": job.sse_clients_connected,
            "sse_clients_peak": job.sse_clients_peak,
            "stale_preview_events_ignored": job.stale_preview_events_ignored,
            "runtime_python": platform.python_version(),
            "runtime_torch": torch.__version__,
            "model_path": job.request.get("model_path"),
            "model_architecture": dict(job.model_selection.get("architecture_contract") or {}),
            "model_architecture_summary": job.model_selection.get("architecture_summary"),
            "sampler": job.request.get("sampler_name"),
            "scheduler": job.request.get("scheduler_name"),
            "seed": job.resolved_seed if job.resolved_seed is not None else job.request.get("seed"),
        }
        return payload

    async def _run_job(self, job: GenerationJob) -> None:
        try:
            await self._run_job_resident(job)
            return
        except ModelRuntimeUnavailable as exc:
            if job.status == "cancelling":
                self._transition_job(job, status="cancelled", worker_stage="cancelled")
                self._finalize_resident_job(job)
                return
            safe_to_retry = (
                not job.output_paths
                and int(job.current_step or 0) == 0
                and not job.skip_events
            )
            if safe_to_retry:
                job.model_runtime_diagnostics.setdefault("recovery_actions", []).append({
                    "timestamp": _utc_now(),
                    "action": "restart_and_reactivate",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                try:
                    await self.model_runtime.stop()
                    model_path = str(job.model_selection.get("resolved_path") or job.request.get("model_path") or "").strip()
                    if model_path:
                        await self.activate_model(model_path, selection=job.model_selection)
                    await self._run_job_resident(job)
                    return
                except Exception as retry_exc:
                    job.model_runtime_diagnostics["retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
                    exc = ModelRuntimeUnavailable(str(retry_exc))
            self._transition_job(job, status="failed", worker_stage="failed")
            job.error = f"Model runtime unavailable: {exc}"
            self._finalize_resident_job(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._transition_job(job, status="failed", worker_stage="failed")
            job.error = f"Generation failed: {type(exc).__name__}: {exc}"
            job.return_code = 1
            job.log_lines.extend(traceback.format_exc().splitlines()[-60:])
            self._finalize_resident_job(job)

    def _prepare_job_request(
        self,
        job: GenerationJob,
    ) -> tuple[Path, Path, dict[str, Any]]:
        started = job.started_at or _utc_now()
        job.started_at = started
        job.updated_at = _utc_now()
        job.status_changed_at = job.status_changed_at or started
        job_root = self.context.data_root / "webui" / "jobs" / job.job_id
        job_root.mkdir(parents=True, exist_ok=True)
        job.job_root = str(job_root)
        request_path = job_root / "request.json"
        console_path = job_root / "console.log"
        job.console_log_path = str(console_path)

        request_payload = self.normalize_generation_request(job.request)
        request_model_path_before_lock = request_payload.get("model_path")
        authoritative_model_path = str(job.model_selection.get("resolved_path") or "").strip()
        if authoritative_model_path:
            request_payload["model_path"] = authoritative_model_path
        request_payload["save_images"] = True
        current_settings = (
            dict(self.settings_provider() or {}) if self.settings_provider is not None else {}
        )
        runtime_startup_status = build_runtime_startup_status(
            self.runtime_startup_options,
            current_settings,
            worker_ready=(self.model_runtime.status().get("ready") or None),
            worker_status=self.model_runtime.status(),
        )
        request_payload["runtime_startup_status"] = runtime_startup_status
        request_payload.setdefault("output_dir", str(self.context.txt2img_output_root))
        request_payload.setdefault("output_prefix", "{index:05d}-{seed}")

        preview_values = self._live_preview_request_values(job_root)
        live_preview_root = Path(preview_values["live_preview_root"])
        job.live_preview_root = str(live_preview_root)
        job.live_preview_latest_path = str(live_preview_root / "latest.json")
        self._merge_runtime_preview_values(request_payload, preview_values)
        job.request = dict(request_payload)
        job.model_diagnostics["diagnostics_mode"] = str(
            (request_payload.get("diagnostics") or {}).get("mode") or "failures_only"
        )
        job.model_diagnostics["preflight"] = {
            "authoritative_model_path": authoritative_model_path,
            "request_model_path_before_lock": request_model_path_before_lock,
            "request_model_path_after_lock": request_payload.get("model_path"),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "cwd": os.getcwd(),
            "project_root": str(self.context.project_root),
            "job_root": str(job_root),
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        }
        job.model_diagnostics["runtime_startup_status"] = runtime_startup_status
        job.model_diagnostics["request_file"] = {
            "model_path": request_payload.get("model_path"),
            "request_path": str(request_path),
        }
        request_path.write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (job_root / "model-selection.json").write_text(
            json.dumps(
                {
                    "selection": job.model_selection,
                    "diagnostics": job.model_diagnostics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return request_path, console_path, preview_values

    async def _restore_resident_runtime_after_skip(self, job: GenerationJob) -> None:
        model_path = str(
            job.model_selection.get("resolved_path")
            or job.request.get("model_path")
            or ""
        ).strip()
        if not model_path:
            raise ModelRuntimeUnavailable(
                "The current image was skipped, but no checkpoint path was available to restore the resident runtime."
            )
        self._transition_job(job, status="preparing_model", worker_stage="restoring_after_skip")
        self._persist_job(job)
        self._publish_event(job, "job-progress", worker_stage=job.worker_stage)
        await self.activate_model(model_path, selection=job.model_selection)
        if job.status not in {"cancelling", "cancelled"}:
            self._transition_job(job, status="running", worker_stage="skip_recovery_complete")
            self._touch_job_runtime(job, progress=True)
            self._persist_job(job)
            self._publish_event(job, "job-progress", worker_stage=job.worker_stage)

    async def _pause_between_images_if_requested(
        self,
        job: GenerationJob,
        *,
        has_more_images: bool,
    ) -> None:
        if self._queue_resume_event.is_set():
            return
        job.pause_after_current_requested = False
        if not has_more_images:
            self._persist_job(job)
            self._publish_event(
                job,
                "job-progress",
                queue_pause_requested=True,
                queue_paused_after_job=True,
            )
            return

        paused_at = _utc_now()
        job.paused_at = paused_at
        self._transition_job(job, status="paused", worker_stage="paused_between_images")
        self._touch_job_runtime(job, progress=True)
        self._persist_job(job)
        self._publish_event(
            job,
            "job-paused",
            paused_at=paused_at,
            queue_pause_requested=True,
        )
        resume_event = self._resume_event_for_job(job.job_id)
        await resume_event.wait()
        if job.status in {"cancelling", "cancelled"}:
            return
        if job.status == "paused":
            self._transition_job(job, status="running", worker_stage="resuming_queue")
        self._touch_job_runtime(job, progress=True)
        self._persist_job(job)
        self._publish_event(
            job,
            "job-progress",
            resumed_at=job.resumed_at,
            queue_pause_requested=not self._queue_resume_event.is_set(),
        )

    async def _run_job_resident(self, job: GenerationJob) -> None:
        """Run WebUI requests as one resident-runtime command per image.

        Splitting a user batch into image-scoped commands preserves the requested seed
        sequence while creating safe control boundaries for pause-after-current and
        skip-current-image. The checkpoint remains resident unless skip requires the
        active worker process to be restarted.
        """

        self._transition_job(job, status="preparing_model", worker_stage="preparing_model")
        job.execution_mode = "resident_model"
        request_path, console_path, preview_values = self._prepare_job_request(job)
        job_root = Path(job.job_root or request_path.parent)
        base_request = json.loads(request_path.read_text(encoding="utf-8"))
        requested_batch_count = max(
            1,
            int(_coerce_top_level_number(base_request.get("batch_count"), integer=True, default=1) or 1),
        )
        requested_batch_size = max(
            1,
            int(_coerce_top_level_number(base_request.get("batch_size"), integer=True, default=1) or 1),
        )
        requested_image_count = requested_batch_count * requested_batch_size
        unlimited = _coerce_boolean(base_request.get("unlimited", False), default=False)
        base_seed = base_request.get("seed")
        seed_iterator = iter_batch_base_seeds(base_seed, batch_size=1)
        job_resume_event = self._resume_event_for_job(job.job_id)
        if self._queue_resume_event.is_set():
            job_resume_event.set()
        else:
            job_resume_event.clear()

        job.model_diagnostics["pipeline_parity"] = {
            "shares_canonical_runner_with_run_bat": True,
            "run_bat_entrypoint": "python -m modules.txt2img.cli run --interactive --save",
            "webui_entrypoint": "resident modules.txt2img.model_runtime JSONL command",
            "execution_path": [
                "src/image_gen/webui/jobs.py::GenerationJobManager._run_job_resident",
                "src/image_gen/webui/model_runtime.py::ResidentModelRuntimeClient",
                "modules.txt2img.model_runtime",
                "src/image_gen.runtime.txt2img_runner",
            ],
            "request_path": str(request_path),
            "request_contains_live_preview_overlay": True,
            "live_preview_overlay_keys": sorted(preview_values.keys()),
            "selected_model_resident_until_replaced": True,
            "webui_batch_orchestration": "one resident-runtime command per image slot",
        }
        orchestration = {
            "mode": "unlimited" if unlimited else "batch_count",
            "requested_batch_count": requested_batch_count,
            "requested_batch_size": requested_batch_size,
            "requested_image_count": None if unlimited else requested_image_count,
            "attempted_images": 0,
            "completed_images": 0,
            "skipped_images": int(job.skipped_images),
            "completed_batches": 0,
            "current_batch": 0,
            "current_image": 0,
            "current_image_in_batch": 0,
            "command_completions": [],
        }
        job.model_runtime_diagnostics["batch_orchestration"] = orchestration
        (job_root / "command.txt").write_text(
            "resident model runtime: WebUI-managed image iteration commands\n"
            + json.dumps(
                {
                    "job_id": job.job_id,
                    "base_request_path": str(request_path),
                    "mode": orchestration["mode"],
                    "requested_batch_count": requested_batch_count,
                    "requested_batch_size": requested_batch_size,
                    "requested_image_count": orchestration["requested_image_count"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._persist_job(job)
        self._publish_event(
            job,
            "job-started",
            worker_stage=job.worker_stage,
            batch_number=0,
            batch_count=requested_batch_count,
            batch_size=requested_batch_size,
            image_number=0,
            image_count=orchestration["requested_image_count"],
            completed_images=0,
            unlimited=unlimited,
        )

        attempted_images = 0
        completed_images = 0
        last_completion: dict[str, Any] = {}
        with console_path.open("w", encoding="utf-8", newline="\n") as console:
            async def on_line(line: str) -> None:
                job.log_lines.append(line)
                console.write(line + "\n")
                console.flush()
                self._apply_runtime_line(job, line)

            while unlimited or attempted_images < requested_image_count:
                if job.status in {"cancelling", "cancelled"}:
                    break

                image_number = attempted_images + 1
                parent_batch_number = ((image_number - 1) // requested_batch_size) + 1
                image_in_batch = ((image_number - 1) % requested_batch_size) + 1
                image_seed = next(seed_iterator)
                iteration_request = json.loads(json.dumps(base_request, ensure_ascii=False))
                iteration_request["batch_count"] = 1
                iteration_request["batch_size"] = 1
                iteration_request["unlimited"] = False
                iteration_request["seed"] = image_seed
                iteration_request["_webui_parent_batch_number"] = parent_batch_number
                iteration_request["_webui_parent_batch_count"] = requested_batch_count
                iteration_request["_webui_parent_batch_size"] = requested_batch_size
                iteration_request["_webui_parent_image_in_batch"] = image_in_batch
                iteration_request["_webui_parent_image_number"] = image_number
                iteration_request["_webui_parent_image_count"] = None if unlimited else requested_image_count
                iteration_request["_webui_parent_unlimited"] = unlimited

                iteration_preview_root = job_root / "live-preview" / f"batch_{image_number:05d}"
                iteration_preview_root.mkdir(parents=True, exist_ok=True)
                iteration_request["live_preview_root"] = str(iteration_preview_root)
                iteration_request_path = job_root / f"request-batch-{image_number:05d}.json"
                iteration_request_path.write_text(
                    json.dumps(iteration_request, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                job.current_step = 0
                job.total_steps = max(
                    1,
                    int(_coerce_top_level_number(iteration_request.get("steps"), integer=True, default=1) or 1),
                )
                job.progress_percent = 0.0
                job.live_preview_url = None
                job.live_preview_path = None
                job.live_preview_decode_mode = None
                job.live_preview_history = []
                job.live_cfg_step_series = {}
                job.live_preview_root = str(iteration_preview_root)
                job.live_preview_latest_path = str(iteration_preview_root / "latest.json")
                job.skip_current_requested = False
                job.skip_requested_at = None
                self._transition_job(job, status="running", worker_stage="starting_image")

                orchestration.update(
                    {
                        "current_batch": parent_batch_number,
                        "current_image": image_number,
                        "current_image_in_batch": image_in_batch,
                        "attempted_images": attempted_images,
                        "completed_images": completed_images,
                        "skipped_images": int(job.skipped_images),
                        "current_seed": image_seed,
                        "current_request_path": str(iteration_request_path),
                    }
                )
                job.model_runtime_diagnostics["batch_orchestration"] = orchestration
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    worker_stage=job.worker_stage,
                    batch_number=parent_batch_number,
                    batch_count=requested_batch_count,
                    batch_size=requested_batch_size,
                    image_number=image_number,
                    image_in_batch=image_in_batch,
                    image_count=None if unlimited else requested_image_count,
                    completed_images=completed_images,
                    skipped_images=job.skipped_images,
                    unlimited=unlimited,
                    live_preview_url=None,
                    live_preview_path=None,
                    current_step=0,
                    total_steps=job.total_steps,
                    progress_percent=0.0,
                )
                console.write(
                    "WEBUI_IMAGE_START_JSON: "
                    + json.dumps(
                        {
                            "image_number": image_number,
                            "image_count": None if unlimited else requested_image_count,
                            "batch_number": parent_batch_number,
                            "image_in_batch": image_in_batch,
                            "batch_count": requested_batch_count,
                            "batch_size": requested_batch_size,
                            "unlimited": unlimited,
                            "seed": image_seed,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                console.flush()

                output_count_before = len(job.output_paths)
                try:
                    runtime_kwargs = {
                        "job_id": job.job_id,
                        "config_path": iteration_request_path,
                        "save_txt": bool(job.request.get("save_txt", True)),
                        "save_json": bool(job.request.get("save_json", True)),
                        "save_diagnostics_json": bool(
                            job.request.get("save_diagnostics_json", False)
                        ),
                        "on_line": on_line,
                    }
                    while True:
                        try:
                            completion = await self.model_runtime.run_job(**runtime_kwargs)
                            break
                        except TypeError as exc:
                            if (
                                "save_diagnostics_json" not in str(exc)
                                or "save_diagnostics_json" not in runtime_kwargs
                            ):
                                raise
                            runtime_kwargs.pop("save_diagnostics_json", None)
                except ModelRuntimeUnavailable as exc:
                    if job.skip_current_requested and job.status not in {"cancelling", "cancelled"}:
                        attempted_images += 1
                        output_completed_before_cancel = len(job.output_paths) > output_count_before
                        skip_event = {
                            "timestamp": _utc_now(),
                            "image_number": image_number,
                            "batch_number": parent_batch_number,
                            "image_in_batch": image_in_batch,
                            "seed": int(image_seed),
                            "runtime_error": f"{type(exc).__name__}: {exc}",
                            "output_completed_before_cancel": output_completed_before_cancel,
                        }
                        if output_completed_before_cancel:
                            completed_images += 1
                            skip_event["outcome"] = "completed_before_skip_reached_runtime"
                        else:
                            job.skipped_images += 1
                            job.skipped_image_seeds.append(int(image_seed))
                            skip_event["outcome"] = "skipped"
                        job.skip_events.append(skip_event)
                        job.skip_current_requested = False
                        job.skip_requested_at = None
                        orchestration["command_completions"].append(
                            {
                                "image_number": image_number,
                                "batch_number": parent_batch_number,
                                "ok": False,
                                "skipped": not output_completed_before_cancel,
                                "output_completed_before_cancel": output_completed_before_cancel,
                                "error": str(exc),
                            }
                        )
                        orchestration.update(
                            {
                                "attempted_images": attempted_images,
                                "completed_images": completed_images,
                                "skipped_images": int(job.skipped_images),
                                "completed_batches": attempted_images // requested_batch_size,
                                "last_skip_event": skip_event,
                            }
                        )
                        job.model_runtime_diagnostics["batch_orchestration"] = orchestration
                        self._touch_job_runtime(job, progress=True)
                        self._persist_job(job)
                        self._publish_event(
                            job,
                            "job-image-skipped",
                            skip_event=skip_event,
                            completed_images=completed_images,
                            skipped_images=job.skipped_images,
                        )
                        console.write(
                            "WEBUI_IMAGE_SKIP_JSON: "
                            + json.dumps(skip_event, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                        console.flush()
                        await self._restore_resident_runtime_after_skip(job)
                        has_more_images = unlimited or attempted_images < requested_image_count
                        await self._pause_between_images_if_requested(
                            job,
                            has_more_images=has_more_images,
                        )
                        continue
                    raise

                last_completion = dict(completion)
                completion_summary = {
                    "image_number": image_number,
                    "batch_number": parent_batch_number,
                    "image_in_batch": image_in_batch,
                    "ok": bool(completion.get("ok")),
                    "command_id": completion.get("command_id"),
                    "result": dict(completion.get("result") or {}),
                    "error": completion.get("error"),
                }
                orchestration["command_completions"].append(completion_summary)

                if not completion.get("ok"):
                    job.return_code = 1
                    self._transition_job(job, status="failed", worker_stage="failed")
                    job.error = str(completion.get("error") or "Model runtime generation failed.")
                    traceback_text = str(completion.get("traceback") or "").strip()
                    if traceback_text:
                        job.log_lines.extend(traceback_text.splitlines()[-40:])
                    break

                runtime_result = dict(completion.get("result") or {})
                runtime_completed = int(runtime_result.get("completed_batches") or 0)
                if runtime_completed != 1:
                    job.return_code = 1
                    self._transition_job(job, status="failed", worker_stage="failed")
                    job.error = (
                        "Resident runtime image contract violation: each WebUI image iteration must "
                        f"complete exactly one runtime batch, but reported {runtime_completed}."
                    )
                    break

                attempted_images += 1
                completed_images += 1
                orchestration.update(
                    {
                        "attempted_images": attempted_images,
                        "completed_images": completed_images,
                        "skipped_images": int(job.skipped_images),
                        "completed_batches": attempted_images // requested_batch_size,
                        "current_batch": parent_batch_number,
                        "current_image": image_number,
                        "last_completed_at": _utc_now(),
                    }
                )
                job.model_runtime_diagnostics["batch_orchestration"] = orchestration
                self._touch_job_runtime(job, progress=True)
                self._persist_job(job)
                self._publish_event(
                    job,
                    "job-progress",
                    worker_stage="image_completed",
                    batch_number=parent_batch_number,
                    batch_count=requested_batch_count,
                    batch_size=requested_batch_size,
                    image_number=image_number,
                    image_in_batch=image_in_batch,
                    image_count=None if unlimited else requested_image_count,
                    completed_images=completed_images,
                    skipped_images=job.skipped_images,
                    unlimited=unlimited,
                    output_count=len(job.output_paths),
                )
                console.write(
                    "WEBUI_IMAGE_COMPLETE_JSON: "
                    + json.dumps(
                        {
                            "image_number": image_number,
                            "completed_images": completed_images,
                            "skipped_images": job.skipped_images,
                            "output_count": len(job.output_paths),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                console.flush()

                has_more_images = unlimited or attempted_images < requested_image_count
                await self._pause_between_images_if_requested(
                    job,
                    has_more_images=has_more_images,
                )
                await asyncio.sleep(0)

        job.model_runtime_diagnostics["command_completion"] = last_completion
        orchestration["attempted_images"] = attempted_images
        orchestration["completed_images"] = completed_images
        orchestration["skipped_images"] = int(job.skipped_images)
        orchestration["completed_batches"] = attempted_images // requested_batch_size
        orchestration["finished_at"] = _utc_now()
        job.model_runtime_diagnostics["batch_orchestration"] = orchestration

        if job.status in {"cancelling", "cancelled"}:
            self._transition_job(job, status="cancelled", worker_stage="cancelled")
            job.return_code = 130
        elif job.status == "failed":
            job.return_code = 1
        else:
            job.return_code = 0
            self._transition_job(job, status="completed", worker_stage="completed")
            self._apply_model_parity(job)

        self._job_resume_events.pop(job.job_id, None)
        self._finalize_resident_job(job)

    def _finalize_resident_job(self, job: GenerationJob) -> None:
        job.process = None
        job.completed_at = _utc_now()
        job.updated_at = job.completed_at
        final_status = self.model_runtime.status()
        final_memory = dict(final_status.get("memory") or {})
        if final_memory:
            job.memory_status = {
                **final_memory,
                "event": "model_runtime_final_status",
                "stage": final_status.get("stage"),
                "active_stage": final_status.get("active_stage"),
                "updated_at": job.completed_at,
            }
        job.model_diagnostics["live_preview"] = self.diagnostics_payload(job)["phase09h_validation"]
        job.model_diagnostics["resident_model"] = {
            **dict(job.model_runtime_diagnostics),
            "final_status": final_status,
            "execution_mode": job.execution_mode,
        }
        if job.job_root:
            (Path(job.job_root) / "model-diagnostics.json").write_text(
                json.dumps(job.model_diagnostics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self._persist_job(job)
        terminal_event = {
            "completed": "job-completed",
            "cancelled": "job-cancelled",
            "failed": "job-failed",
        }.get(job.status, "job-progress")
        self._publish_terminal_once(job, terminal_event)
        self.cleanup_preview_directories()

    async def _run_job_isolated(self, job: GenerationJob) -> None:
        self._transition_job(job, status="preparing_model", worker_stage="loading_model")
        if job.execution_mode == "pending":
            job.execution_mode = "isolated_subprocess"
        job.started_at = _utc_now()
        job.updated_at = job.started_at
        job.status_changed_at = job.started_at
        job_root = self.context.data_root / "webui" / "jobs" / job.job_id
        job_root.mkdir(parents=True, exist_ok=True)
        job.job_root = str(job_root)
        request_path = job_root / "request.json"
        console_path = job_root / "console.log"
        job.console_log_path = str(console_path)

        request_payload = self.normalize_generation_request(job.request)
        request_model_path_before_lock = request_payload.get("model_path")
        authoritative_model_path = str(job.model_selection.get("resolved_path") or "").strip()
        if authoritative_model_path:
            request_payload["model_path"] = authoritative_model_path
        job.request = dict(request_payload)
        request_payload["save_images"] = True
        request_payload.setdefault("output_dir", str(self.context.txt2img_output_root))
        request_payload.setdefault("output_prefix", "{index:05d}-{seed}")

        preview_values = self._live_preview_request_values(job_root)
        live_preview_root = Path(preview_values["live_preview_root"])
        job.live_preview_root = str(live_preview_root)
        job.live_preview_latest_path = str(live_preview_root / "latest.json")
        self._merge_runtime_preview_values(request_payload, preview_values)
        job.request = dict(request_payload)
        job.model_diagnostics["preflight"] = {
            "authoritative_model_path": authoritative_model_path,
            "request_model_path_before_lock": request_model_path_before_lock,
            "request_model_path_after_lock": request_payload.get("model_path"),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "cwd": os.getcwd(),
            "project_root": str(self.context.project_root),
            "job_root": str(job_root),
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        }
        job.model_diagnostics["request_file"] = {
            "model_path": request_payload.get("model_path"),
            "request_path": str(request_path),
        }
        request_path.write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (job_root / "model-selection.json").write_text(
            json.dumps(
                {
                    "selection": job.model_selection,
                    "diagnostics": job.model_diagnostics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        source_root = str(self.context.project_root / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            [source_root, str(self.context.project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        command = [
            sys.executable,
            "-m",
            "modules.txt2img.cli",
            "run",
            "--project-root",
            str(self.context.project_root),
            "--config",
            str(request_path),
        ]
        if not bool(job.request.get("save_txt", True)):
            command.append("--no-txt")
        if not bool(job.request.get("save_json", True)):
            command.append("--no-json")
        if not bool(job.request.get("save_diagnostics_json", False)):
            command.append("--no-diagnostics-json")
        job.model_diagnostics["pipeline_parity"] = {
            "shares_cli_runner_with_run_bat": True,
            "run_bat_entrypoint": "python -m modules.txt2img.cli run --interactive --save",
            "webui_entrypoint": "python -m modules.txt2img.cli run --config <request.json>",
            "execution_path": [
                "src/image_gen/webui/jobs.py::GenerationJobManager._run_job",
                "modules.txt2img.cli:run",
                "modules.txt2img.txt2img_runner",
                "src/image_gen.runtime.txt2img_runner",
            ],
            "command": subprocess.list2cmdline(command),
            "request_path": str(request_path),
            "request_contains_live_preview_overlay": True,
            "live_preview_overlay_keys": sorted(preview_values.keys()),
            "webui_only_keys_do_not_change_final_decode": True,
        }
        (job_root / "command.txt").write_text(
            subprocess.list2cmdline(command),
            encoding="utf-8",
        )
        self._persist_job(job)
        self._publish_event(job, "job-started")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.context.project_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # Diagnostic JSONL lines can be much larger than asyncio's
                # default 64 KiB reader limit.
                limit=_SUBPROCESS_STREAM_LIMIT,
            )
            job.process = process
            assert process.stdout is not None
            with console_path.open("w", encoding="utf-8", newline="\n") as console:
                while True:
                    raw = await process.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    job.log_lines.append(line)
                    console.write(line + "\n")
                    console.flush()
                    seed_match = _GENERATION_SEED_LINE.match(line)
                    if seed_match:
                        try:
                            seed_payload = json.loads(seed_match.group(1))
                        except json.JSONDecodeError:
                            seed_payload = {}
                        self._transition_job(job, status="running", worker_stage="running")
                        try:
                            job.resolved_seed = int(seed_payload.get("base_seed"))
                        except (TypeError, ValueError):
                            job.resolved_seed = None
                        raw_seeds = seed_payload.get("image_seeds") or []
                        job.resolved_seeds = []
                        for value in raw_seeds:
                            try:
                                job.resolved_seeds.append(int(value))
                            except (TypeError, ValueError):
                                continue
                        job.updated_at = _utc_now()
                        self._persist_job(job)
                        self._publish_event(
                            job,
                            "job-progress",
                            resolved_seed=job.resolved_seed,
                            resolved_seeds=list(job.resolved_seeds),
                        )
                    preview_summary_match = _LIVE_PREVIEW_SUMMARY_LINE.match(line)
                    if preview_summary_match:
                        self._transition_job(job, status="finalizing", worker_stage="finalizing")
                        try:
                            preview_summary = json.loads(preview_summary_match.group(1))
                        except json.JSONDecodeError:
                            preview_summary = {}
                        if isinstance(preview_summary, dict):
                            job.live_preview_metrics.update(preview_summary)
                            job.live_preview_metrics["sse_clients_connected"] = job.sse_clients_connected
                            job.live_preview_metrics["sse_clients_peak"] = job.sse_clients_peak
                            job.live_preview_metrics["stale_preview_events_ignored"] = job.stale_preview_events_ignored
                            job.model_diagnostics["live_preview"] = self.diagnostics_payload(job)["phase09h_validation"]
                            self._persist_job(job)
                            self._publish_event(
                                job,
                                "job-progress",
                                live_preview_metrics=dict(job.live_preview_metrics),
                            )
                    memory_status_match = _MEMORY_STATUS_LINE.search(line)
                    if memory_status_match:
                        try:
                            memory_payload = json.loads(memory_status_match.group(1))
                        except json.JSONDecodeError:
                            memory_payload = {}
                        if isinstance(memory_payload, dict):
                            status_payload = _normalize_live_memory_status(
                                memory_payload.get("status") or {}
                            )
                            job.memory_status = {
                                **status_payload,
                                "event": memory_payload.get("event"),
                                "stage": memory_payload.get("stage"),
                                "active_stage": memory_payload.get("active_stage"),
                                "updated_at": _utc_now(),
                            }
                            self._persist_job(job)
                            self._publish_event(
                                job,
                                "job-progress",
                                memory_status=dict(job.memory_status),
                            )
                    image_match = _IMAGE_LINE.match(line)
                    if image_match:
                        self._transition_job(job, status="finalizing", worker_stage="saving_output")
                        image_path = image_match.group("path")
                        if image_path not in job.output_paths:
                            job.output_paths.append(image_path)
                        job.final_output_url = self._output_url_for_path(image_path)
                        if job.resolved_seed is None:
                            try:
                                job.resolved_seed = int(image_match.group("seed"))
                            except (TypeError, ValueError):
                                pass
                        job.updated_at = _utc_now()
                        self._persist_job(job)
                        self._publish_event(
                            job,
                            "job-output-produced",
                            latest_output_path=image_path,
                            latest_output_url=job.final_output_url,
                            output_count=len(job.output_paths),
                        )
                    failure_match = _FAILURE_BUNDLE_LINE.search(line)
                    if failure_match:
                        job.failure_bundle_path = failure_match.group(1).strip()
                    runtime_diag_match = _RUNTIME_DIAGNOSTIC_LINE.match(line)
                    if runtime_diag_match:
                        try:
                            payload = json.loads(runtime_diag_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            job.model_diagnostics["runtime_environment"] = payload
                    model_match = _MODEL_DIAGNOSTIC_LINE.match(line)
                    if model_match:
                        try:
                            payload = json.loads(model_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            job.model_diagnostics["runtime"] = payload
                    output_quality_match = _OUTPUT_QUALITY_DIAGNOSTIC_LINE.match(line)
                    if output_quality_match:
                        try:
                            payload = json.loads(output_quality_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            job.output_quality_diagnostics = dict(payload)
                            self._persist_job(job)
                            self._publish_event(
                                job,
                                "job-progress",
                                output_quality_diagnostics=dict(job.output_quality_diagnostics),
                            )

                    prompt_parser_match = _PROMPT_PARSER_DIAGNOSTIC_LINE.match(line)
                    if prompt_parser_match:
                        try:
                            payload = json.loads(prompt_parser_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            job.prompt_parser_diagnostics = dict(payload)
                            self._persist_job(job)
                            self._publish_event(
                                job,
                                "job-progress",
                                prompt_parser_diagnostics=dict(job.prompt_parser_diagnostics),
                            )
                    step_progress_match = _STEP_PROGRESS_LINE.search(line)
                    if step_progress_match:
                        try:
                            payload = json.loads(step_progress_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            self._apply_step_progress_payload(job, payload)

                    # tqdm-based samplers can write their carriage-return progress text
                    # immediately before the structured preview marker. Search the full
                    # console line instead of requiring the marker at column zero.
                    step_preview_match = _STEP_PREVIEW_LINE.search(line)
                    if step_preview_match:
                        try:
                            payload = json.loads(step_preview_match.group(1))
                        except json.JSONDecodeError:
                            payload = {"parse_error": line}
                        if isinstance(payload, dict):
                            self._apply_step_preview_payload(job, payload)
            job.return_code = await process.wait()
            if job.status == "cancelling":
                self._transition_job(job, status="cancelled", worker_stage="cancelled")
            elif job.return_code == 0:
                runtime_model = dict(job.model_diagnostics.get("runtime") or {})
                expected_model = str(job.model_selection.get("resolved_path") or "").strip()
                loaded_model = str(
                    runtime_model.get("loaded_path")
                    or runtime_model.get("resolved_path")
                    or runtime_model.get("requested_path")
                    or ""
                ).strip()
                model_paths_match: bool | None = None
                if expected_model and loaded_model:
                    expected_token = os.path.normcase(str(Path(expected_model).expanduser().resolve()))
                    loaded_token = os.path.normcase(str(Path(loaded_model).expanduser().resolve()))
                    model_paths_match = expected_token == loaded_token
                    if not model_paths_match:
                        self._transition_job(job, status="failed", worker_stage="failed")
                        job.error = (
                            "Model parity violation: the WebUI selected checkpoint was not the "
                            "checkpoint loaded by the canonical CLI runtime. "
                            f"Selected: {expected_model}. Loaded: {loaded_model}."
                        )
                    else:
                        self._transition_job(job, status="completed", worker_stage="completed")
                else:
                    self._transition_job(job, status="completed", worker_stage="completed")
                job.model_diagnostics["model_parity"] = {
                    "selected_path": expected_model,
                    "loaded_path": loaded_model,
                    "matches": model_paths_match,
                    "enforced": bool(expected_model),
                }
            else:
                self._transition_job(job, status="failed", worker_stage="failed")
                job.error = next(
                    (line for line in reversed(job.log_lines) if "ERROR" in line.upper()),
                    f"Generation process exited with code {job.return_code}.",
                )
            job.worker_stage = job.status
        except Exception as exc:
            self._transition_job(job, status="failed", worker_stage="failed")
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.process = None
            job.completed_at = _utc_now()
            job.updated_at = job.completed_at
            job.model_diagnostics["live_preview"] = self.diagnostics_payload(job)["phase09h_validation"]
            if job.job_root:
                (Path(job.job_root) / "model-diagnostics.json").write_text(
                    json.dumps(job.model_diagnostics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            self._persist_job(job)
            terminal_event = {
                "completed": "job-completed",
                "cancelled": "job-cancelled",
                "failed": "job-failed",
            }.get(job.status, "job-progress")
            self._publish_terminal_once(job, terminal_event)
            self.cleanup_preview_directories()

