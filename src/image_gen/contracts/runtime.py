from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from modules.component_placement import component_placement_report



PROMPT_ASSET_CONTRACT_VERSION = "image-gen-prompt-assets-v1"

_PROMPT_ASSET_SOURCE_ALIASES = {
    "visual": "visual_selection",
    "visual_selection": "visual_selection",
    "active": "visual_selection",
    "inline": "inline_syntax",
    "inline_syntax": "inline_syntax",
    "model": "model_default",
    "model_specific": "model_default",
    "model_default": "model_default",
    "global": "global_default",
    "global_default": "global_default",
    "default": "global_default",
    "replay": "replay",
    "import": "imported",
    "imported": "imported",
    "api": "api_request",
    "api_request": "api_request",
}


def canonical_prompt_asset_source(value: Any, default: str = "visual_selection") -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return default
    return _PROMPT_ASSET_SOURCE_ALIASES.get(token, token)


def canonical_prompt_asset_type(value: Any, default: str = "lora") -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"lora", "loras"}:
        return "lora"
    if token in {"textual_inversion", "textual_inversions", "embedding", "embeddings", "ti"}:
        return "textual_inversion"
    return token or default


def _coerce_prompt_asset_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in {"0", "false", "no", "off", "disabled"}:
        return False
    if token in {"1", "true", "yes", "on", "enabled"}:
        return True
    return bool(value)


@dataclass
class PromptAssetSelection:
    """Structured prompt-asset request shared by the UI, runtime, and replay."""

    asset_type: str = "lora"
    asset_id: str = ""
    catalog_asset_id: str = ""
    name: str = ""
    path: str = ""
    requested_path: str = ""
    resolved_path: str = ""
    requested_hash: str = ""
    resolved_hash: str = ""
    weight: float = 1.0
    enabled: bool = True
    polarity: str = "positive"
    activation_text: str = ""
    model_family: str = ""
    source_url: str = ""
    source: str = "visual_selection"
    original_source: str = ""
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.asset_type = canonical_prompt_asset_type(self.asset_type)
        self.asset_id = str(self.asset_id or "").strip()
        self.catalog_asset_id = str(self.catalog_asset_id or self.asset_id or "").strip()
        self.name = str(self.name or "").strip()
        self.path = str(self.path or self.resolved_path or self.requested_path or "").strip()
        self.requested_path = str(self.requested_path or self.path or "").strip()
        self.resolved_path = str(self.resolved_path or "").strip()
        self.requested_hash = str(self.requested_hash or "").strip()
        self.resolved_hash = str(self.resolved_hash or "").strip()
        try:
            self.weight = float(self.weight)
        except (TypeError, ValueError):
            self.weight = 1.0
        self.enabled = _coerce_prompt_asset_bool(self.enabled, True)
        polarity = str(self.polarity or "positive").strip().lower()
        self.polarity = polarity if polarity in {"positive", "negative"} else "positive"
        self.activation_text = str(self.activation_text or "").strip()
        self.model_family = str(self.model_family or "").strip()
        self.source_url = str(self.source_url or "").strip()
        self.source = canonical_prompt_asset_source(self.source)
        self.original_source = canonical_prompt_asset_source(self.original_source, default="") if self.original_source else ""
        try:
            self.order = int(self.order)
        except (TypeError, ValueError):
            self.order = 0
        self.metadata = dict(self.metadata or {})

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        asset_type: str = "lora",
        order: int = 0,
        default_source: str = "visual_selection",
    ) -> "PromptAssetSelection":
        if isinstance(value, cls):
            payload = value.to_serializable_dict()
        elif isinstance(value, Mapping):
            payload = dict(value)
        elif isinstance(value, (str, Path)):
            raw_path = str(value)
            payload = {"name": Path(raw_path).stem, "path": raw_path}
        else:
            raise TypeError(f"Unsupported prompt asset value: {type(value).__name__}")
        payload.setdefault("asset_type", asset_type)
        payload.setdefault("order", order)
        payload["name"] = payload.get("name") or payload.get("requested_name") or payload.get("display_name") or ""
        payload["asset_id"] = payload.get("asset_id") or payload.get("catalog_asset_id") or ""
        payload["catalog_asset_id"] = payload.get("catalog_asset_id") or payload.get("asset_id") or ""
        payload["source"] = payload.get("source") or payload.get("selection_source") or payload.get("selection_origin") or payload.get("source_scope") or payload.get("origin") or default_source
        payload["original_source"] = payload.get("original_source") or payload.get("replay_source") or ""
        payload["requested_path"] = payload.get("requested_path") or payload.get("path") or ""
        payload["resolved_path"] = payload.get("resolved_path") or ""
        payload["requested_hash"] = payload.get("requested_hash") or payload.get("file_hash") or ""
        payload["resolved_hash"] = payload.get("resolved_hash") or payload.get("file_hash") or ""
        known = {field_name for field_name in cls.__dataclass_fields__}
        metadata = dict(payload.get("metadata") or {})
        for key, item in payload.items():
            if key not in known and key not in {"selection_source", "selection_origin", "source_scope", "replay_source", "file_hash"}:
                metadata.setdefault(key, item)
        payload["metadata"] = metadata
        return cls(**{key: value for key, value in payload.items() if key in known})

    def identity_key(self) -> str:
        return str(
            self.resolved_hash
            or self.requested_hash
            or self.resolved_path
            or self.path
            or self.catalog_asset_id
            or self.asset_id
            or self.name
        ).strip().casefold()

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "asset_id": self.asset_id,
            "catalog_asset_id": self.catalog_asset_id,
            "name": self.name,
            "path": self.path,
            "requested_path": self.requested_path,
            "resolved_path": self.resolved_path,
            "requested_hash": self.requested_hash,
            "resolved_hash": self.resolved_hash,
            "weight": float(self.weight),
            "enabled": bool(self.enabled),
            "polarity": self.polarity,
            "activation_text": self.activation_text,
            "model_family": self.model_family,
            "source_url": self.source_url,
            "source": self.source,
            "original_source": self.original_source,
            "order": int(self.order),
            "metadata": _json_safe(self.metadata),
        }


def normalize_prompt_asset_list(
    values: Any,
    *,
    asset_type: str,
    default_source: str = "visual_selection",
) -> list[PromptAssetSelection]:
    if values in (None, ""):
        return []
    source_values = values if isinstance(values, (list, tuple)) else [values]
    output: list[PromptAssetSelection] = []
    index_by_identity: dict[str, int] = {}
    for order, value in enumerate(source_values):
        asset = PromptAssetSelection.from_value(
            value,
            asset_type=asset_type,
            order=order,
            default_source=default_source,
        )
        identity = asset.identity_key()
        if identity and identity in index_by_identity:
            output[index_by_identity[identity]] = asset
            continue
        if identity:
            index_by_identity[identity] = len(output)
        output.append(asset)
    for order, asset in enumerate(output):
        asset.order = order
    return output

_CANONICAL_SCHEDULE_KEYS = {
    "requested_steps",
    "effective_steps",
    "scheduler_step_override_applied",
    "compatibility_mode",
}


def _tensor_summary(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _json_safe(value: Any) -> Any:
    """Convert runtime values to JSON-safe metadata without retaining live objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device) or isinstance(value, torch.dtype):
        return str(value)
    if torch.is_tensor(value):
        return _tensor_summary(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_serializable_dict") and callable(value.to_serializable_dict):
        return value.to_serializable_dict()
    return {
        "runtime_type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


@dataclass
class GenerationRequest:
    positive_prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    cfg_rescale: float = 0.0
    prompt_cfg_schedule: dict[str, Any] = field(default_factory=dict)
    prompt_cfg_pass_schedules: dict[str, Any] = field(default_factory=dict)
    prompt_cfg_recorded_schedules: dict[str, Any] = field(default_factory=dict)
    prompt_cfg_replay_mode: str = "reconstruct"
    prompt_expansion_record: dict[str, Any] = field(default_factory=dict)
    prompt_expansion_pass_records: dict[str, Any] = field(default_factory=dict)
    prompt_expansion_recorded: dict[str, Any] = field(default_factory=dict)
    prompt_expansion_replay_mode: str = "reconstruct"
    prompt_semantic_pass_records: dict[str, Any] = field(default_factory=dict)
    prompt_semantic_recorded: dict[str, Any] = field(default_factory=dict)
    prompt_semantic_replay_mode: str = "reconstruct"
    region_pass_records: dict[str, Any] = field(default_factory=dict)
    region_recorded: dict[str, Any] = field(default_factory=dict)
    region_replay_mode: str = "reconstruct"
    batch_size: int = 1
    seed: Optional[int] = None
    resolved_seeds: list[int] = field(default_factory=list)
    prompt_asset_contract_version: str = PROMPT_ASSET_CONTRACT_VERSION
    loras: list[PromptAssetSelection] = field(default_factory=list)
    textual_inversions: list[PromptAssetSelection] = field(default_factory=list)

    device: Optional[torch.device | str] = None
    dtype: Optional[torch.dtype] = None

    scheduler_name: Optional[str] = None
    sampler_name: Optional[str] = None

    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    prompt_parser_name: str = "legacy"
    prompt_parser_kwargs: dict[str, Any] = field(default_factory=dict)
    prompt_shortcut_profile_name: str = "legacy_default"
    prompt_shortcut_profile_snapshot: dict[str, Any] = field(default_factory=dict)
    prompt_parser_preset_name: str = ""
    base_prompt_parser_name: str = "legacy"
    base_shortcut_profile_name: str = "legacy_default"
    hires_prompt_parser_mode: str = "same_as_base"
    hires_prompt_parser_name: str = "legacy"
    hires_prompt_parser_kwargs: dict[str, Any] = field(default_factory=dict)
    hires_shortcut_profile_mode: str = "same_as_base"
    hires_shortcut_profile_name: str = "legacy_default"
    hires_shortcut_profile_snapshot: dict[str, Any] = field(default_factory=dict)
    hires_positive_prompt: str = ""
    hires_negative_prompt: str = ""
    hires_size_mode: str = "same_as_base"
    hires_scale: Optional[float] = 2.0
    hires_width: int = 0
    hires_height: int = 0
    hires_dimension_plan_version: str = ""
    hires_dimension_plan: dict[str, Any] = field(default_factory=dict)
    hires_axis_scale_width: float = 1.0
    hires_axis_scale_height: float = 1.0
    hires_uniform_scale: Optional[float] = None
    hires_aspect_ratio_changed: bool = False
    hires_enabled: bool = False
    hires_steps: int = 20
    hires_denoising_strength: float = 0.45
    # Phase 14M-1 freezes the existing arithmetic under a versioned name.
    # Later interactive defaults may change without changing legacy replay.
    hires_step_policy: str = "a1111_fixed_steps_v1"
    hires_sampler_name: str = ""
    hires_scheduler_name: str = ""
    hires_cfg_scale: Optional[float] = None
    hires_cfg_rescale: Optional[float] = None
    hires_recorded_schedule_replay: dict[str, Any] = field(default_factory=dict)
    hires_recorded_schedule_fingerprint: dict[str, Any] = field(default_factory=dict)
    hires_schedule_conformance_source_replay: dict[str, Any] = field(default_factory=dict)
    hires_schedule_conformance_source_fingerprint: dict[str, Any] = field(default_factory=dict)
    hires_schedule_replay_mode: str = "reconstruct"
    hires_strategy: str = "pixel_neural"
    hires_upscaler: str = ""
    hires_upscaler_id: str = ""
    hires_expected_upscaler_sha256: str = ""
    hires_expected_native_scale: int = 0
    hires_expected_vae_sha256: str = ""
    hires_expected_vae_source_kind: str = ""
    hires_tile_size: int = 0
    hires_tile_overlap: int = 16
    hires_tile_batch_size: int = 1
    hires_exact_resize_filter: str = "bicubic"
    hires_final_size_correction_filter: str = "auto"
    hires_aspect_policy: str = "stretch"
    hires_padding_mode: str = "reflect"
    hires_recorded_target_correction: dict[str, Any] = field(default_factory=dict)
    hires_correction_fingerprint_enabled: bool = False
    hires_recorded_correction_fingerprint: dict[str, Any] = field(default_factory=dict)
    hires_save_upscaled_pre_denoise: bool = False
    hires_save_vae_roundtrip: bool = False
    hires_diagnostic_vae_execution_fingerprint: bool = False
    hires_save_lowres: bool = False
    hires_memory_preflight: bool = True
    hires_host_staging_policy: str = "pageable"
    hires_host_staging_cap_mb: int = 1024
    hires_artifact_disk_budget_mb: int = 0
    outpaint_prototype_enabled: bool = False
    outpaint_source_image: str = ""
    outpaint_anchor: str = "center"
    outpaint_source_x: int = -1
    outpaint_source_y: int = -1
    outpaint_feather_px: int = 24
    outpaint_context_seed_mode: str = "edge_pad_v1"
    outpaint_denoising_strength: float = 0.70
    outpaint_latent_strategy: str = "noise_only_new_regions_v1"
    outpaint_prompt_mode: str = "source_prompt_v1"
    outpaint_overlay_positive_prompt: str = ""
    outpaint_overlay_negative_prompt: str = ""
    outpaint_diagnostic_artifacts: bool = False
    outpaint_prototype_record: dict[str, Any] = field(default_factory=dict)
    outpaint_shape_expansion_enabled: bool = False
    outpaint_shape_target_mode: str = "square"
    outpaint_shape_target_width: int = 0
    outpaint_shape_target_height: int = 0
    outpaint_shape_base_width: int = 0
    outpaint_shape_base_height: int = 0
    outpaint_shape_anchor: str = "center"
    outpaint_shape_context_seed_mode: str = "edge_pad_v1"
    outpaint_shape_source_handoff: str = "auto"
    outpaint_shape_prompt_mode: str = "overlay_only_v1"
    outpaint_shape_overlay_positive_prompt: str = ""
    outpaint_shape_overlay_negative_prompt: str = ""
    outpaint_shape_denoising_strength: float = 0.40
    outpaint_shape_save_base: bool = False
    outpaint_shape_runtime_record: dict[str, Any] = field(default_factory=dict)
    prompt_preflight: dict[str, Any] = field(default_factory=dict)
    prompt_shadow_compare: bool = False
    prompt_route_plan: dict[str, Any] = field(default_factory=dict)
    hires_prompt_route_plan: dict[str, Any] = field(default_factory=dict)
    parser_kwargs: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    return_latents: bool = False

    save_images: bool = False
    output_dir: Optional[str] = None
    output_prefix: str = "img"

    def __post_init__(self) -> None:
        self.prompt_asset_contract_version = str(
            self.prompt_asset_contract_version or PROMPT_ASSET_CONTRACT_VERSION
        )
        self.loras = normalize_prompt_asset_list(
            self.loras,
            asset_type="lora",
        )
        self.textual_inversions = normalize_prompt_asset_list(
            self.textual_inversions,
            asset_type="textual_inversion",
        )

    def to_serializable_dict(self) -> dict[str, Any]:
        payload = {
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "width": int(self.width),
            "height": int(self.height),
            "steps": int(self.steps),
            "cfg_scale": float(self.cfg_scale),
            "cfg_rescale": float(self.cfg_rescale),
            "prompt_cfg_schedule": _json_safe(self.prompt_cfg_schedule),
            "prompt_cfg_pass_schedules": _json_safe(self.prompt_cfg_pass_schedules),
            "prompt_cfg_recorded_schedules": _json_safe(self.prompt_cfg_recorded_schedules),
            "prompt_cfg_replay_mode": str(self.prompt_cfg_replay_mode or "reconstruct"),
            "prompt_expansion_record": _json_safe(self.prompt_expansion_record),
            "prompt_expansion_pass_records": _json_safe(self.prompt_expansion_pass_records),
            "prompt_expansion_recorded": _json_safe(self.prompt_expansion_recorded),
            "prompt_expansion_replay_mode": str(self.prompt_expansion_replay_mode or "reconstruct"),
            "prompt_semantic_pass_records": _json_safe(self.prompt_semantic_pass_records),
            "prompt_semantic_recorded": _json_safe(self.prompt_semantic_recorded),
            "prompt_semantic_replay_mode": str(self.prompt_semantic_replay_mode or "reconstruct"),
            "region_pass_records": _json_safe(self.region_pass_records),
            "region_recorded": _json_safe(self.region_recorded),
            "region_replay_mode": str(self.region_replay_mode or "reconstruct"),
            "batch_size": int(self.batch_size),
            "seed": self.seed,
            "resolved_seeds": [int(seed) for seed in self.resolved_seeds],
            "prompt_asset_contract_version": str(self.prompt_asset_contract_version or PROMPT_ASSET_CONTRACT_VERSION),
            "loras": [asset.to_serializable_dict() for asset in self.loras],
            "textual_inversions": [asset.to_serializable_dict() for asset in self.textual_inversions],
            "device": str(self.device) if self.device is not None else None,
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "scheduler_name": self.scheduler_name,
            "sampler_name": self.sampler_name,
            "scheduler_kwargs": _json_safe(self.scheduler_kwargs),
            "sampler_kwargs": _json_safe(self.sampler_kwargs),
            "prompt_parser_name": str(self.prompt_parser_name or "legacy"),
            "prompt_parser_kwargs": _json_safe(self.prompt_parser_kwargs),
            "prompt_shortcut_profile_name": str(self.prompt_shortcut_profile_name or "legacy_default"),
            "prompt_shortcut_profile_snapshot": _json_safe(self.prompt_shortcut_profile_snapshot),
            "prompt_parser_preset_name": str(self.prompt_parser_preset_name or ""),
            "base_prompt_parser_name": str(self.base_prompt_parser_name or self.prompt_parser_name or "legacy"),
            "base_shortcut_profile_name": str(self.base_shortcut_profile_name or self.prompt_shortcut_profile_name or "legacy_default"),
            "hires_prompt_parser_mode": str(self.hires_prompt_parser_mode or "same_as_base"),
            "hires_prompt_parser_name": str(self.hires_prompt_parser_name or self.prompt_parser_name or "legacy"),
            "hires_prompt_parser_kwargs": _json_safe(self.hires_prompt_parser_kwargs),
            "hires_shortcut_profile_mode": str(self.hires_shortcut_profile_mode or "same_as_base"),
            "hires_shortcut_profile_name": str(self.hires_shortcut_profile_name or self.prompt_shortcut_profile_name or "legacy_default"),
            "hires_shortcut_profile_snapshot": _json_safe(self.hires_shortcut_profile_snapshot),
            "hires_positive_prompt": str(self.hires_positive_prompt or self.positive_prompt),
            "hires_negative_prompt": str(self.hires_negative_prompt or self.negative_prompt),
            "hires_size_mode": str(self.hires_size_mode or "same_as_base"),
            "hires_scale": (float(self.hires_scale) if self.hires_scale is not None else None),
            "hires_width": int(self.hires_width or 0),
            "hires_height": int(self.hires_height or 0),
            "hires_dimension_plan_version": str(self.hires_dimension_plan_version or ""),
            "hires_dimension_plan": _json_safe(self.hires_dimension_plan),
            "hires_axis_scale_width": float(self.hires_axis_scale_width),
            "hires_axis_scale_height": float(self.hires_axis_scale_height),
            "hires_uniform_scale": (
                float(self.hires_uniform_scale) if self.hires_uniform_scale is not None else None
            ),
            "hires_aspect_ratio_changed": bool(self.hires_aspect_ratio_changed),
            "hires_enabled": bool(self.hires_enabled),
            "hires_steps": int(self.hires_steps or 20),
            "hires_denoising_strength": float(self.hires_denoising_strength or 0.45),
            "hires_step_policy": str(self.hires_step_policy or "a1111_fixed_steps_v1"),
            "hires_sampler_name": str(self.hires_sampler_name or ""),
            "hires_scheduler_name": str(self.hires_scheduler_name or ""),
            "hires_cfg_scale": (
                float(self.hires_cfg_scale) if self.hires_cfg_scale is not None else None
            ),
            "hires_cfg_rescale": (
                float(self.hires_cfg_rescale) if self.hires_cfg_rescale is not None else None
            ),
            "hires_recorded_schedule_replay": _json_safe(self.hires_recorded_schedule_replay),
            "hires_recorded_schedule_fingerprint": _json_safe(
                self.hires_recorded_schedule_fingerprint
            ),
            "hires_schedule_replay_mode": str(
                self.hires_schedule_replay_mode or "reconstruct"
            ),
            "hires_strategy": str(self.hires_strategy or "pixel_neural"),
            "hires_upscaler": str(self.hires_upscaler or ""),
            "hires_upscaler_id": str(self.hires_upscaler_id or ""),
            "hires_expected_upscaler_sha256": str(
                self.hires_expected_upscaler_sha256 or ""
            ),
            "hires_expected_native_scale": int(self.hires_expected_native_scale or 0),
            "hires_expected_vae_sha256": str(self.hires_expected_vae_sha256 or ""),
            "hires_expected_vae_source_kind": str(
                self.hires_expected_vae_source_kind or ""
            ),
            "hires_tile_size": int(self.hires_tile_size or 0),
            "hires_tile_overlap": int(self.hires_tile_overlap or 0),
            "hires_tile_batch_size": int(self.hires_tile_batch_size or 1),
            "hires_exact_resize_filter": str(self.hires_exact_resize_filter or "bicubic"),
            "hires_final_size_correction_filter": str(self.hires_final_size_correction_filter or "auto"),
            "hires_aspect_policy": str(self.hires_aspect_policy or "stretch"),
            "hires_padding_mode": str(self.hires_padding_mode or "reflect"),
            "hires_recorded_target_correction": _json_safe(self.hires_recorded_target_correction),
            "hires_correction_fingerprint_enabled": bool(self.hires_correction_fingerprint_enabled),
            "hires_recorded_correction_fingerprint": _json_safe(self.hires_recorded_correction_fingerprint),
            "hires_save_upscaled_pre_denoise": bool(self.hires_save_upscaled_pre_denoise),
            "hires_save_vae_roundtrip": bool(self.hires_save_vae_roundtrip),
            "hires_diagnostic_vae_execution_fingerprint": bool(
                self.hires_diagnostic_vae_execution_fingerprint
            ),
            "hires_save_lowres": bool(self.hires_save_lowres),
            "hires_memory_preflight": bool(self.hires_memory_preflight),
            "hires_host_staging_policy": str(self.hires_host_staging_policy or "pageable"),
            "hires_host_staging_cap_mb": int(self.hires_host_staging_cap_mb or 0),
            "hires_artifact_disk_budget_mb": int(self.hires_artifact_disk_budget_mb or 0),
            "outpaint_prototype_enabled": bool(self.outpaint_prototype_enabled),
            "outpaint_source_image": str(self.outpaint_source_image or ""),
            "outpaint_anchor": str(self.outpaint_anchor or "center"),
            "outpaint_source_x": int(self.outpaint_source_x),
            "outpaint_source_y": int(self.outpaint_source_y),
            "outpaint_feather_px": int(self.outpaint_feather_px),
            "outpaint_context_seed_mode": str(self.outpaint_context_seed_mode or "edge_pad_v1"),
            "outpaint_denoising_strength": float(self.outpaint_denoising_strength),
            "outpaint_latent_strategy": str(self.outpaint_latent_strategy or "noise_only_new_regions_v1"),
            "outpaint_prompt_mode": str(self.outpaint_prompt_mode or "source_prompt_v1"),
            "outpaint_overlay_positive_prompt": str(self.outpaint_overlay_positive_prompt or ""),
            "outpaint_overlay_negative_prompt": str(self.outpaint_overlay_negative_prompt or ""),
            "outpaint_diagnostic_artifacts": bool(self.outpaint_diagnostic_artifacts),
            "outpaint_prototype_record": _json_safe(self.outpaint_prototype_record),
            "outpaint_shape_expansion_enabled": bool(self.outpaint_shape_expansion_enabled),
            "outpaint_shape_target_mode": str(self.outpaint_shape_target_mode or "square"),
            "outpaint_shape_target_width": int(self.outpaint_shape_target_width or 0),
            "outpaint_shape_target_height": int(self.outpaint_shape_target_height or 0),
            "outpaint_shape_base_width": int(self.outpaint_shape_base_width or 0),
            "outpaint_shape_base_height": int(self.outpaint_shape_base_height or 0),
            "outpaint_shape_anchor": str(self.outpaint_shape_anchor or "center"),
            "outpaint_shape_context_seed_mode": str(self.outpaint_shape_context_seed_mode or "edge_pad_v1"),
            "outpaint_shape_source_handoff": str(self.outpaint_shape_source_handoff or "auto"),
            "outpaint_shape_prompt_mode": str(self.outpaint_shape_prompt_mode or "overlay_only_v1"),
            "outpaint_shape_overlay_positive_prompt": str(self.outpaint_shape_overlay_positive_prompt or ""),
            "outpaint_shape_overlay_negative_prompt": str(self.outpaint_shape_overlay_negative_prompt or ""),
            "outpaint_shape_denoising_strength": float(self.outpaint_shape_denoising_strength),
            "outpaint_shape_save_base": bool(self.outpaint_shape_save_base),
            "outpaint_shape_runtime_record": _json_safe(self.outpaint_shape_runtime_record),
            "prompt_preflight": _json_safe(self.prompt_preflight),
            "prompt_shadow_compare": bool(self.prompt_shadow_compare),
            "prompt_route_plan": _json_safe(self.prompt_route_plan),
            "hires_prompt_route_plan": _json_safe(self.hires_prompt_route_plan),
            "parser_kwargs": _json_safe(self.parser_kwargs),
            "diagnostics": _json_safe(self.diagnostics),
            "return_latents": bool(self.return_latents),
            "save_images": bool(self.save_images),
            "output_dir": self.output_dir,
            "output_prefix": self.output_prefix,
        }
        generation_width = getattr(self, "generation_width", None)
        generation_height = getattr(self, "generation_height", None)
        dimension_plan = getattr(self, "dimension_plan", None)
        if generation_width is not None:
            payload["generation_width"] = int(generation_width)
        if generation_height is not None:
            payload["generation_height"] = int(generation_height)
        if dimension_plan is not None:
            payload["dimension_plan"] = _json_safe(dimension_plan)
        return payload


@dataclass
class PipelineComponents:
    unet: torch.nn.Module
    vae: torch.nn.Module
    text_encoder: torch.nn.Module
    tokenizer: Any = None
    prediction_type: str = "epsilon"
    prediction_type_source: str = "pipeline_components"
    model_identity: str = ""
    model_hash: str = ""
    vae_provenance: dict[str, Any] = field(default_factory=dict)

    def placement_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            "unet": component_placement_report(self.unet),
            "vae": component_placement_report(self.vae),
            "text_encoder": component_placement_report(self.text_encoder),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "unet": f"{type(self.unet).__module__}.{type(self.unet).__qualname__}",
            "vae": f"{type(self.vae).__module__}.{type(self.vae).__qualname__}",
            "text_encoder": f"{type(self.text_encoder).__module__}.{type(self.text_encoder).__qualname__}",
            "prediction_type": self.prediction_type,
            "prediction_type_source": self.prediction_type_source,
            "model_identity": self.model_identity,
            "model_hash": self.model_hash,
            "vae_provenance": _json_safe(self.vae_provenance),
            "component_placement": self.placement_metadata(),
            "tokenizer": (
                f"{type(self.tokenizer).__module__}.{type(self.tokenizer).__qualname__}"
                if self.tokenizer is not None
                else None
            ),
        }


@dataclass
class ConditioningOutput:
    cond: torch.Tensor
    uncond: torch.Tensor
    prompt_schedules: dict[str, Any] = field(default_factory=dict)
    pooled_cond: Optional[torch.Tensor] = None
    pooled_uncond: Optional[torch.Tensor] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "cond": _tensor_summary(self.cond),
            "uncond": _tensor_summary(self.uncond),
            "pooled_cond": _tensor_summary(self.pooled_cond),
            "pooled_uncond": _tensor_summary(self.pooled_uncond),
            "prompt_schedules": _json_safe(self.prompt_schedules),
            "extra": _json_safe(self.extra),
        }


@dataclass(init=False)
class SchedulerOutput:
    """Canonical scheduler result with explicit step-count ownership.

    ``extra`` remains as a compatibility property for older callers. Canonical
    step fields are stored only on the object and are projected into ``extra``
    when legacy code reads it.
    """

    sigmas: torch.Tensor
    timesteps: Optional[torch.Tensor]
    requested_steps: int
    effective_steps: int
    scheduler_step_override_applied: bool
    compatibility_mode: Optional[str]
    metadata: dict[str, Any]

    def __init__(
        self,
        sigmas: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        *,
        requested_steps: Optional[int] = None,
        effective_steps: Optional[int] = None,
        scheduler_step_override_applied: Optional[bool] = None,
        compatibility_mode: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not torch.is_tensor(sigmas):
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
        if timesteps is not None and not torch.is_tensor(timesteps):
            timesteps = torch.as_tensor(timesteps, dtype=torch.float32)

        merged = dict(metadata or {})
        merged.update(dict(extra or {}))

        transitions = max(int(sigmas.numel()) - 1, 0)
        requested = requested_steps if requested_steps is not None else merged.pop("requested_steps", transitions)
        effective = effective_steps if effective_steps is not None else merged.pop("effective_steps", transitions)
        compatibility = (
            compatibility_mode
            if compatibility_mode is not None
            else merged.pop("compatibility_mode", None)
        )
        changed = scheduler_step_override_applied
        if changed is None:
            changed = merged.pop("scheduler_step_override_applied", None)
        if changed is None:
            changed = int(effective) != int(requested)

        for key in _CANONICAL_SCHEDULE_KEYS:
            merged.pop(key, None)

        self.sigmas = sigmas
        self.timesteps = timesteps
        self.requested_steps = int(requested)
        self.effective_steps = int(effective)
        self.scheduler_step_override_applied = bool(changed)
        self.compatibility_mode = compatibility
        self.metadata = merged
        self.validate()

    @property
    def extra(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "requested_steps": self.requested_steps,
            "effective_steps": self.effective_steps,
            "scheduler_step_override_applied": self.scheduler_step_override_applied,
            "compatibility_mode": self.compatibility_mode,
        }

    @extra.setter
    def extra(self, value: Mapping[str, Any]) -> None:
        merged = dict(value or {})
        if "requested_steps" in merged:
            self.requested_steps = int(merged.pop("requested_steps"))
        if "effective_steps" in merged:
            self.effective_steps = int(merged.pop("effective_steps"))
        if "scheduler_step_override_applied" in merged:
            self.scheduler_step_override_applied = bool(
                merged.pop("scheduler_step_override_applied")
            )
        else:
            self.scheduler_step_override_applied = self.effective_steps != self.requested_steps
        if "compatibility_mode" in merged:
            self.compatibility_mode = merged.pop("compatibility_mode")
        self.metadata = merged

    @property
    def sigma_transitions(self) -> int:
        return max(int(self.sigmas.numel()) - 1, 0)

    @property
    def initial_sigma(self) -> float:
        return float(self.sigmas[0].detach().cpu().item())

    def timestep_for_step(self, step_index: int) -> torch.Tensor:
        """Return the model timestep associated with a sigma transition.

        Schedules may expose either one timestep per sigma value or one
        timestep per transition. Both representations are accepted, but every
        denoising transition must have an explicit timestep.
        """
        if self.timesteps is None:
            raise ValueError("SchedulerOutput must provide timesteps for denoising.")
        index = int(step_index)
        if index < 0 or index >= self.sigma_transitions:
            raise IndexError(
                f"Step index {index} is outside the {self.sigma_transitions} schedule transitions."
            )
        return self.timesteps[index]

    def validate(self) -> None:
        if self.sigmas.ndim != 1:
            raise ValueError("SchedulerOutput.sigmas must be a one-dimensional tensor.")
        if self.sigmas.numel() < 2:
            raise ValueError("SchedulerOutput.sigmas must contain at least two values.")
        if self.requested_steps < 1:
            raise ValueError("SchedulerOutput.requested_steps must be at least 1.")
        if self.effective_steps < 1:
            raise ValueError("SchedulerOutput.effective_steps must be at least 1.")
        if self.timesteps is not None and self.timesteps.ndim != 1:
            raise ValueError("SchedulerOutput.timesteps must be one-dimensional when supplied.")
        if self.effective_steps != self.sigma_transitions:
            raise ValueError(
                "SchedulerOutput.effective_steps must equal the number of sigma transitions."
            )
        if self.timesteps is None:
            raise ValueError("SchedulerOutput.timesteps must be supplied.")
        valid_lengths = {self.sigma_transitions, int(self.sigmas.numel())}
        if int(self.timesteps.numel()) not in valid_lengths:
            raise ValueError(
                "SchedulerOutput.timesteps must contain either one value per sigma "
                "or one value per sigma transition."
            )
        if not torch.isfinite(self.sigmas).all():
            raise ValueError("SchedulerOutput.sigmas contains non-finite values.")
        if not torch.isfinite(self.timesteps).all():
            raise ValueError("SchedulerOutput.timesteps contains non-finite values.")
        if torch.any(self.sigmas < 0):
            raise ValueError("SchedulerOutput.sigmas cannot contain negative values.")
        if torch.any(self.sigmas[1:] > self.sigmas[:-1]):
            raise ValueError("SchedulerOutput.sigmas must be monotonically non-increasing.")

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "requested_steps": self.requested_steps,
            "effective_steps": self.effective_steps,
            "sigma_transitions": self.sigma_transitions,
            "scheduler_step_override_applied": self.scheduler_step_override_applied,
            "compatibility_mode": self.compatibility_mode,
            "sigmas": [float(value) for value in self.sigmas.detach().cpu().flatten()],
            "timesteps": (
                [float(value) for value in self.timesteps.detach().cpu().flatten()]
                if self.timesteps is not None
                else None
            ),
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class SamplerOutput:
    latents: torch.Tensor
    extra: dict[str, Any] = field(default_factory=dict)

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "latents": _tensor_summary(self.latents),
            "extra": _json_safe(self.extra),
        }


@dataclass
class GenerationResult:
    request: Optional[GenerationRequest] = None
    images: Any = None
    latents: Optional[torch.Tensor] = None
    conditioning: Optional[ConditioningOutput] = None
    schedule: Optional[SchedulerOutput] = None
    sampler: Optional[SamplerOutput] = None
    trace_exports: dict[str, Any] = field(default_factory=dict)
    auxiliary_images: dict[str, Any] = field(default_factory=dict)
    saved_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def schedule_extra(self) -> dict[str, Any]:
        return self.schedule.extra if self.schedule is not None else {}

    @property
    def sampler_extra(self) -> dict[str, Any]:
        return dict(self.sampler.extra) if self.sampler is not None else {}

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Any:
        mapping = self.to_legacy_dict()
        if key not in mapping:
            raise KeyError(key)
        return mapping[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "saved_paths":
            self.saved_paths = list(value or [])
            return
        if key == "images":
            self.images = value
            return
        if key == "latents":
            self.latents = value
            return
        self.metadata[key] = value

    def to_legacy_dict(self) -> dict[str, Any]:
        result = {
            "latents": self.latents,
            "conditioning": self.conditioning,
            "schedule": self.schedule,
            "schedule_extra": self.schedule_extra,
            "sampler_extra": self.sampler_extra,
            "trace_exports": dict(self.trace_exports),
            "auxiliary_images": {
                str(name): (_tensor_summary(value) if torch.is_tensor(value) else _json_safe(value))
                for name, value in self.auxiliary_images.items()
            },
        }
        if self.images is not None:
            result["images"] = self.images
        if self.saved_paths:
            result["saved_paths"] = list(self.saved_paths)
        result.update(self.metadata)
        return result

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_serializable_dict() if self.request is not None else None,
            "images": _tensor_summary(self.images) if torch.is_tensor(self.images) else _json_safe(self.images),
            "latents": _tensor_summary(self.latents),
            "conditioning": (
                self.conditioning.to_serializable_dict()
                if self.conditioning is not None
                else None
            ),
            "schedule": self.schedule.to_serializable_dict() if self.schedule is not None else None,
            "sampler": self.sampler.to_serializable_dict() if self.sampler is not None else None,
            "trace_exports": _json_safe(self.trace_exports),
            "auxiliary_images": {
                str(name): (_tensor_summary(value) if torch.is_tensor(value) else _json_safe(value))
                for name, value in self.auxiliary_images.items()
            },
            "saved_paths": list(self.saved_paths),
            "metadata": _json_safe(self.metadata),
        }
