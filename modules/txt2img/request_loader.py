from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from modules.contracts import (
    PROMPT_ASSET_CONTRACT_VERSION,
    GenerationRequest,
    normalize_prompt_asset_list,
)
from modules.txt2img.infotext_parser import parse_infotext
from modules.txt2img.manifest_adapters import manifest_to_request_kwargs
from modules.txt2img.manifest_io import load_manifest_json
from modules.prompt_parsers import PromptProcessingPreflight, default_prompt_parser_registry
from modules.prompt_shortcuts import PromptShortcutProfileDescriptor, default_prompt_shortcut_registry, validate_prompt_shortcut_profile
from image_gen.runtime.hires_sizing import apply_hires_dimensions

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


GENERATION_REQUEST_KEYS = {
    "positive_prompt",
    "negative_prompt",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "cfg_rescale",
    "batch_size",
    "seed",
    "prompt_asset_contract_version",
    "loras",
    "textual_inversions",
    "scheduler_name",
    "sampler_name",
    "scheduler_kwargs",
    "sampler_kwargs",
    "prompt_parser_name",
    "prompt_parser_kwargs",
    "prompt_cfg_pass_schedules",
    "prompt_cfg_recorded_schedules",
    "prompt_cfg_replay_mode",
    "prompt_expansion_record",
    "prompt_expansion_pass_records",
    "prompt_expansion_recorded",
    "prompt_expansion_replay_mode",
    "prompt_semantic_pass_records",
    "prompt_semantic_recorded",
    "prompt_semantic_replay_mode",
    "region_pass_records",
    "region_recorded",
    "region_replay_mode",
    "prompt_shortcut_profile_name",
    "prompt_shortcut_profile_snapshot",
    "prompt_parser_preset_name",
    "base_prompt_parser_name",
    "base_shortcut_profile_name",
    "hires_prompt_parser_mode",
    "hires_prompt_parser_name",
    "hires_prompt_parser_kwargs",
    "hires_shortcut_profile_mode",
    "hires_shortcut_profile_name",
    "hires_shortcut_profile_snapshot",
    "hires_positive_prompt",
    "hires_negative_prompt",
    "hires_size_mode",
    "hires_scale",
    "hires_width",
    "hires_height",
    "hires_dimension_plan_version",
    "hires_dimension_plan",
    "hires_axis_scale_width",
    "hires_axis_scale_height",
    "hires_uniform_scale",
    "hires_aspect_ratio_changed",
    "hires_enabled",
    "hires_steps",
    "hires_denoising_strength",
    "hires_step_policy",
    "hires_sampler_name",
    "hires_scheduler_name",
    "hires_cfg_scale",
    "hires_cfg_rescale",
    "hires_recorded_schedule_replay",
    "hires_recorded_schedule_fingerprint",
    "hires_schedule_conformance_source_replay",
    "hires_schedule_conformance_source_fingerprint",
    "hires_schedule_replay_mode",
    "hires_strategy",
    "hires_upscaler",
    "hires_upscaler_id",
    "hires_expected_upscaler_sha256",
    "hires_expected_native_scale",
    "hires_expected_vae_sha256",
    "hires_expected_vae_source_kind",
    "hires_tile_size",
    "hires_tile_overlap",
    "hires_tile_batch_size",
    "hires_exact_resize_filter",
    "hires_final_size_correction_filter",
    "hires_aspect_policy",
    "hires_padding_mode",
    "hires_recorded_target_correction",
    "hires_correction_fingerprint_enabled",
    "hires_recorded_correction_fingerprint",
    "hires_save_upscaled_pre_denoise",
    "hires_save_vae_roundtrip",
    "hires_diagnostic_vae_execution_fingerprint",
    "hires_save_lowres",
    "hires_memory_preflight",
    "hires_host_staging_policy",
    "hires_host_staging_cap_mb",
    "hires_artifact_disk_budget_mb",
    "outpaint_enabled",
    "outpaint_target_width",
    "outpaint_target_height",
    "outpaint_preservation_mode",
    "outpaint_mask_strategy",
    "outpaint_source_handoff_mode",
    "outpaint_prototype_enabled",
    "outpaint_source_image",
    "outpaint_anchor",
    "outpaint_source_x",
    "outpaint_source_y",
    "outpaint_feather_px",
    "outpaint_context_seed_mode",
    "outpaint_denoising_strength",
    "outpaint_latent_strategy",
    "outpaint_prompt_mode",
    "outpaint_overlay_positive_prompt",
    "outpaint_overlay_negative_prompt",
    "outpaint_diagnostic_artifacts",
    "outpaint_prototype_record",
    "outpaint_shape_expansion_enabled",
    "outpaint_shape_target_mode",
    "outpaint_shape_target_width",
    "outpaint_shape_target_height",
    "outpaint_shape_base_width",
    "outpaint_shape_base_height",
    "outpaint_shape_anchor",
    "outpaint_shape_context_seed_mode",
    "outpaint_shape_source_handoff",
    "outpaint_shape_prompt_mode",
    "outpaint_shape_overlay_positive_prompt",
    "outpaint_shape_overlay_negative_prompt",
    "outpaint_shape_denoising_strength",
    "outpaint_shape_save_base",
    "outpaint_shape_runtime_record",
    "prompt_preflight",
    "prompt_shadow_compare",
    "prompt_route_plan",
    "hires_prompt_route_plan",
    "parser_kwargs",
    "diagnostics",
    "return_latents",
    "save_images",
    "output_dir",
    "output_prefix",
}

PAYLOAD_ALIASES = {
    "prompt": "positive_prompt",
    "negative_prompt": "negative_prompt",

    # Runtime-facing names (used if explicitly provided)
    "sampler": "sampler_name",
    "scheduler": "scheduler_name",

    # A1111 compatibility → preserve labels, do NOT overwrite runtime name
    "schedule_type": "scheduler_label",
    "filename_pattern": "output_prefix",
    "output_filename_pattern": "output_prefix",
    "guidance_rescale": "cfg_rescale",
    "prompt_parser": "prompt_parser_name",
    "shortcut_profile": "prompt_shortcut_profile_name",
    "prompt_shortcut_profile": "prompt_shortcut_profile_name",
    "prompt_parser_preset": "prompt_parser_preset_name",
    "base_prompt_parser": "base_prompt_parser_name",
    "base_shortcut_profile": "base_shortcut_profile_name",
    "hires_prompt_parser": "hires_prompt_parser_name",
    "hires_shortcut_profile": "hires_shortcut_profile_name",

    "sampler_label": "sampler_label",
    "scheduler_label": "scheduler_label",
}

EXTRA_REQUEST_KEYS = {
    "model_path",
    "model_name",
    "model_hash",
    "model_version",

    # A1111 import fields
    "sampler_label",
    "scheduler_label",
    "infotext_source",
    "infotext_raw",
    
    
    "vae_path",
    "vae_name",
    "vae_hash",
    "compatibility_mode",
    "clip_skip",
    "guidance_rescale",
    "tiling",
    "lora_paths",
    
}


def _normalize_payload_keys(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        target_key = PAYLOAD_ALIASES.get(key, key)
        normalized[target_key] = value
    return normalized



def _load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")



def load_request_from_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML request files.")
    payload = yaml.safe_load(_load_text(path)) or {}
    if not isinstance(payload, dict):
        raise TypeError("YAML request payload must be a mapping.")
    return _normalize_payload_keys(payload)



def load_request_from_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_load_text(path))
    if not isinstance(payload, dict):
        raise TypeError("JSON request payload must be a mapping.")
    return _normalize_payload_keys(payload)



def load_request_from_manifest_json(path: str | Path) -> dict[str, Any]:
    manifest = load_manifest_json(path)
    payload = manifest_to_request_kwargs(manifest)
    # Phase 14M-3 migration rule: manifests created before policy metadata
    # existed replay with the historical proportional-tail semantics. New
    # config/CLI requests use the GenerationRequest fixed-step default.
    payload.setdefault("hires_step_policy", "proportional_tail_v1")
    return _normalize_payload_keys(payload)



def load_request_from_infotext(path: str | Path) -> dict[str, Any]:
    text = _load_text(path)
    payload = parse_infotext(text)

    # IMAGE_GEN's text sidecar historically recorded the final/internal hires
    # size in the A1111-style Size field.  When the sidecar also carries the
    # hires dimension plan, restore the UI-owned base dimensions before request
    # normalization so hires scaling is not applied a second time.
    if bool(payload.get("hires_enabled", False)):
        plan = payload.get("hires_dimension_plan")
        if isinstance(plan, dict):
            try:
                base_width = int(plan.get("base_width") or 0)
                base_height = int(plan.get("base_height") or 0)
            except (TypeError, ValueError):
                base_width = 0
                base_height = 0
            if base_width > 0 and base_height > 0:
                payload["width"] = base_width
                payload["height"] = base_height

    payload["infotext_source"] = str(path)
    payload["infotext_raw"] = text

    # Pre-normalized lookup helpers
    if "sampler_label" in payload:
        payload["_sampler_label_norm"] = str(payload["sampler_label"]).strip().lower()

    if "scheduler_label" in payload:
        payload["_scheduler_label_norm"] = str(payload["scheduler_label"]).strip().lower()
    
    return _normalize_payload_keys(payload)



def merge_cli_overrides(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    override_values = dict(overrides or {})
    for key, value in override_values.items():
        if value is None:
            continue
        if key in {"scheduler_kwargs", "sampler_kwargs", "prompt_parser_kwargs", "prompt_cfg_pass_schedules", "prompt_cfg_recorded_schedules", "prompt_expansion_record", "prompt_expansion_pass_records", "prompt_expansion_recorded", "prompt_semantic_pass_records", "prompt_semantic_recorded", "region_pass_records", "region_recorded", "prompt_shortcut_profile_snapshot", "hires_prompt_parser_kwargs", "hires_shortcut_profile_snapshot", "hires_recorded_schedule_replay", "hires_recorded_schedule_fingerprint", "hires_schedule_conformance_source_replay", "hires_schedule_conformance_source_fingerprint", "prompt_preflight", "prompt_route_plan", "hires_prompt_route_plan", "parser_kwargs", "diagnostics"}:
            current = dict(merged.get(key, {}) or {})
            current.update(dict(value or {}))
            merged[key] = current
        else:
            merged[key] = value

    schedule_sensitive = {
        "sampler_name",
        "scheduler_name",
        "scheduler_kwargs",
        "hires_sampler_name",
        "hires_scheduler_name",
        "hires_steps",
        "hires_denoising_strength",
        "hires_step_policy",
    }
    if any(override_values.get(key) is not None for key in schedule_sensitive):
        recorded_replay = dict(merged.get("hires_recorded_schedule_replay") or {})
        recorded_fingerprint = dict(merged.get("hires_recorded_schedule_fingerprint") or {})
        if recorded_replay and recorded_fingerprint:
            merged["hires_schedule_conformance_source_replay"] = recorded_replay
            merged["hires_schedule_conformance_source_fingerprint"] = recorded_fingerprint
        merged.pop("hires_recorded_schedule_replay", None)
        merged.pop("hires_recorded_schedule_fingerprint", None)
        merged["hires_schedule_replay_mode"] = "reconstruct"

    prompt_cfg_sensitive = {
        "positive_prompt",
        "steps",
        "cfg_scale",
        "prompt_parser_name",
        "prompt_parser_kwargs",
        "hires_positive_prompt",
        "hires_steps",
        "hires_cfg_scale",
        "hires_prompt_parser_name",
        "hires_prompt_parser_kwargs",
    }
    if any(override_values.get(key) is not None for key in prompt_cfg_sensitive):
        merged.pop("prompt_cfg_recorded_schedules", None)
        merged["prompt_cfg_replay_mode"] = "reconstruct"

    prompt_expansion_sensitive = {
        "positive_prompt",
        "negative_prompt",
        "seed",
        "prompt_parser_name",
        "prompt_parser_kwargs",
        "hires_positive_prompt",
        "hires_negative_prompt",
        "hires_enabled",
        "hires_prompt_parser_mode",
        "hires_prompt_parser_name",
        "hires_prompt_parser_kwargs",
    }
    if any(override_values.get(key) is not None for key in prompt_expansion_sensitive):
        merged.pop("prompt_expansion_recorded", None)
        merged["prompt_expansion_replay_mode"] = "reconstruct"
        merged.pop("prompt_semantic_recorded", None)
        merged["prompt_semantic_replay_mode"] = "reconstruct"

    region_sensitive = prompt_expansion_sensitive | {
        "width",
        "height",
        "steps",
        "hires_size_mode",
        "hires_scale",
        "hires_width",
        "hires_height",
        "hires_steps",
    }
    if any(override_values.get(key) is not None for key in region_sensitive):
        merged.pop("region_recorded", None)
        merged["region_replay_mode"] = "reconstruct"
    return merged



def _normalize_prompt_asset_request_fields(
    request_kwargs: dict[str, Any],
    extras: dict[str, Any],
) -> None:
    active_assets = extras.get("_webui_active_prompt_assets")
    if not isinstance(active_assets, list):
        active_assets = []

    lora_values = request_kwargs.get("loras")
    if not lora_values:
        lora_values = [
            item for item in active_assets
            if isinstance(item, dict) and str(item.get("asset_type") or "").strip().lower() in {"lora", "loras"}
        ]
    if not lora_values and isinstance(extras.get("lora_paths"), list):
        lora_values = list(extras.get("lora_paths") or [])

    textual_values = request_kwargs.get("textual_inversions")
    if not textual_values:
        textual_values = [
            item for item in active_assets
            if isinstance(item, dict) and str(item.get("asset_type") or "").strip().lower() in {
                "textual_inversion", "textual-inversion", "embedding", "ti"
            }
        ]

    request_kwargs["prompt_asset_contract_version"] = str(
        request_kwargs.get("prompt_asset_contract_version")
        or PROMPT_ASSET_CONTRACT_VERSION
    )
    request_kwargs["loras"] = normalize_prompt_asset_list(
        lora_values or [],
        asset_type="lora",
    )
    request_kwargs["textual_inversions"] = normalize_prompt_asset_list(
        textual_values or [],
        asset_type="textual_inversion",
    )

    # Keep the legacy path list available for older plugins while the canonical
    # request uses structured assets.
    extras["lora_paths"] = [
        asset.resolved_path or asset.path or asset.requested_path
        for asset in request_kwargs["loras"]
        if asset.resolved_path or asset.path or asset.requested_path
    ]


def split_request_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = _normalize_payload_keys(payload)
    request_kwargs = {
        k: v for k, v in normalized.items()
        if k in GENERATION_REQUEST_KEYS and k not in {"sampler_label", "scheduler_label"}
    }

    extras = {
        k: v for k, v in normalized.items()
        if k not in request_kwargs
    }
    return request_kwargs, extras



def payload_to_generation_request(payload: dict[str, Any]) -> tuple[GenerationRequest, dict[str, Any]]:
    request_kwargs, extras = split_request_payload(payload)
    _normalize_prompt_asset_request_fields(request_kwargs, extras)
    request_kwargs.setdefault("positive_prompt", "")
    legacy_parser_kwargs = dict(request_kwargs.get("parser_kwargs") or {})
    if not request_kwargs.get("prompt_parser_name"):
        request_kwargs["prompt_parser_name"] = str(
            legacy_parser_kwargs.get("prompt_parser")
            or legacy_parser_kwargs.get("prompt_parser_name")
            or "legacy"
        )
    request_kwargs.setdefault("prompt_parser_kwargs", {})
    parser_registry = default_prompt_parser_registry()
    parser_id = parser_registry.resolve_id(request_kwargs.get("prompt_parser_name") or "legacy")
    if not parser_registry.is_available(parser_id):
        _available, reason = parser_registry.availability(parser_id)
        raise ValueError(f"Prompt parser {parser_id!r} is unavailable: {reason}")
    request_kwargs["prompt_parser_name"] = parser_id
    snapshot = request_kwargs.get("prompt_shortcut_profile_snapshot")
    if isinstance(snapshot, dict) and snapshot:
        shortcut_profile = PromptShortcutProfileDescriptor.from_dict(snapshot, builtin=bool(snapshot.get("builtin", False)))
        validation = validate_prompt_shortcut_profile(shortcut_profile)
        if not validation.valid:
            raise ValueError("Embedded prompt shortcut profile is invalid: " + " | ".join(issue.message for issue in validation.errors))
    else:
        fallback_profile = "legacy_default" if parser_id == "legacy" else ("parser21_native" if parser_id == "parser21" else ("superhybrid_native" if parser_id == "superhybrid" else "canonical"))
        shortcut_profile = default_prompt_shortcut_registry().get(request_kwargs.get("prompt_shortcut_profile_name") or fallback_profile)
    compatible = parser_id in shortcut_profile.compatible_parsers or (
        parser_id == "combined" and any(item in shortcut_profile.compatible_parsers for item in ("legacy", "parser21", "superhybrid"))
    )
    if not compatible:
        raise ValueError(f"Prompt shortcut profile {shortcut_profile.profile_id!r} is not compatible with parser {parser_id!r}.")
    request_kwargs["prompt_shortcut_profile_name"] = shortcut_profile.profile_id
    request_kwargs["prompt_shortcut_profile_snapshot"] = shortcut_profile.snapshot()
    request_kwargs["prompt_parser_preset_name"] = str(request_kwargs.get("prompt_parser_preset_name") or "")
    report = PromptProcessingPreflight().validate(request_kwargs)
    if not report.get("valid"):
        messages = " | ".join(str(item.get("message") or "Prompt validation failed.") for item in report.get("blocking_errors") or [])
        raise ValueError(f"Prompt preflight failed: {messages}")
    request_kwargs.update(report.get("normalized_fields") or {})
    request_kwargs["prompt_cfg_pass_schedules"] = dict(
        request_kwargs.get("prompt_cfg_pass_schedules") or {}
    )
    request_kwargs["prompt_cfg_recorded_schedules"] = dict(
        request_kwargs.get("prompt_cfg_recorded_schedules") or {}
    )
    request_kwargs["prompt_cfg_replay_mode"] = str(
        request_kwargs.get("prompt_cfg_replay_mode") or "reconstruct"
    ).strip().lower()
    if request_kwargs["prompt_cfg_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError("prompt_cfg_replay_mode must be reconstruct or recorded_exact.")
    request_kwargs["prompt_expansion_record"] = dict(
        request_kwargs.get("prompt_expansion_record") or {}
    )
    request_kwargs["prompt_expansion_pass_records"] = dict(
        request_kwargs.get("prompt_expansion_pass_records") or {}
    )
    request_kwargs["prompt_expansion_recorded"] = dict(
        request_kwargs.get("prompt_expansion_recorded") or {}
    )
    request_kwargs["prompt_expansion_replay_mode"] = str(
        request_kwargs.get("prompt_expansion_replay_mode") or "reconstruct"
    ).strip().lower()
    if request_kwargs["prompt_expansion_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError("prompt_expansion_replay_mode must be reconstruct or recorded_exact.")
    request_kwargs["prompt_semantic_pass_records"] = dict(
        request_kwargs.get("prompt_semantic_pass_records") or {}
    )
    request_kwargs["prompt_semantic_recorded"] = dict(
        request_kwargs.get("prompt_semantic_recorded") or {}
    )
    request_kwargs["prompt_semantic_replay_mode"] = str(
        request_kwargs.get("prompt_semantic_replay_mode") or "reconstruct"
    ).strip().lower()
    if request_kwargs["prompt_semantic_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError("prompt_semantic_replay_mode must be reconstruct or recorded_exact.")
    request_kwargs["region_pass_records"] = dict(
        request_kwargs.get("region_pass_records") or {}
    )
    request_kwargs["region_recorded"] = dict(
        request_kwargs.get("region_recorded") or {}
    )
    request_kwargs["region_replay_mode"] = str(
        request_kwargs.get("region_replay_mode") or "reconstruct"
    ).strip().lower()
    if request_kwargs["region_replay_mode"] not in {"reconstruct", "recorded_exact"}:
        raise ValueError("region_replay_mode must be reconstruct or recorded_exact.")
    apply_hires_dimensions(request_kwargs)
    request_kwargs["prompt_preflight"] = report
    request = GenerationRequest(**request_kwargs)
    return request, extras



def request_to_payload(request: GenerationRequest, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    if is_dataclass(request):
        payload = asdict(request)
    else:
        payload = dict(vars(request))
    payload.update(dict(extras or {}))
    return payload



def load_request_payload(
    *,
    config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    infotext_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    base_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources = [config_path, manifest_path, infotext_path]
    active_count = sum(1 for item in sources if item)
    if active_count > 1:
        raise ValueError("Choose only one of config_path, manifest_path, or infotext_path.")

    payload: dict[str, Any] = _normalize_payload_keys(dict(base_payload or {}))
    source_payload: dict[str, Any] = {}
    if config_path:
        suffix = Path(config_path).suffix.lower()
        if suffix in {".yaml", ".yml"}:
            source_payload = load_request_from_yaml(config_path)
        elif suffix == ".json":
            source_payload = load_request_from_json(config_path)
        else:
            raise ValueError(f"Unsupported config extension: {suffix}")
    elif manifest_path:
        source_payload = load_request_from_manifest_json(manifest_path)
    elif infotext_path:
        source_payload = load_request_from_infotext(infotext_path)

    payload = merge_cli_overrides(payload, source_payload)
    return merge_cli_overrides(payload, cli_overrides)
