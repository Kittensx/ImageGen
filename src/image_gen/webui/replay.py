from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from image_gen.contracts import (
    PROMPT_ASSET_CONTRACT_VERSION,
    normalize_prompt_asset_list,
)
from image_gen.runtime_options import (
    RUNTIME_REPLAY_JOB_FIELDS,
    extract_runtime_execution_record,
    runtime_execution_fingerprint,
    runtime_replay_assessment,
    runtime_replay_warnings,
)
from image_gen.webui.jobs import GenerationJobManager
from image_gen.webui.model_selection import ActiveModelSelection, WebUIModelSelectionState
from image_gen.webui.output_details import OutputMetadataDetails, load_output_details
from image_gen.webui.prompt_assets import extract_inline_loras_from_prompts, merge_replay_loras
from image_gen.runtime.lora_runtime import LoRAResolver
from image_gen.webui.schema_utils import normalize_config_schema
from modules.project_context import ProjectContext
from modules.txt2img.model_selector import MODEL_EXTENSIONS


_EDITABLE_FIELDS = {
    "positive_prompt",
    "negative_prompt",
    "model_path",
    "vae_path",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "seed",
    "batch_size",
    "batch_count",
    "sampler_name",
    "scheduler_name",
    "sampler_kwargs",
    "scheduler_kwargs",
    "prompt_parser_name",
    "prompt_parser_kwargs",
    "prompt_cfg_recorded_schedules",
    "prompt_cfg_replay_mode",
    "prompt_expansion_recorded",
    "prompt_expansion_replay_mode",
    "prompt_semantic_recorded",
    "prompt_semantic_replay_mode",
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
    "hires_strategy",
    "hires_upscaler",
    "hires_upscaler_id",
    "hires_tile_size",
    "hires_tile_overlap",
    "hires_tile_batch_size",
    "hires_exact_resize_filter",
    "hires_final_size_correction_filter",
    "hires_aspect_policy",
    "hires_padding_mode",
    "hires_blurred_edge_method",
    "hires_blurred_edge_compare_diagnostics",
    "hires_save_upscaled_pre_denoise",
    "hires_save_vae_roundtrip",
    "hires_diagnostic_vae_execution_fingerprint",
    "hires_save_lowres",
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
}
_BATCH_OVERRIDE_FIELDS = {
    "model_path",
    "vae_path",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "sampler_name",
    "scheduler_name",
    "seed",
    "batch_size",
    "batch_count",
    "sampler_kwargs",
    "scheduler_kwargs",
}
_OPERATIONAL_FIELDS = {
    "output_dir",
    "output_prefix",
    "save_images",
    "live_preview_enabled",
    "live_preview_mode",
    "live_preview_interval",
    "live_preview_width",
    "live_preview_format",
    "live_preview_history",
    "live_preview_batch_index",
    "live_preview_adaptive_throttle",
    "live_preview_adaptive_target",
    "live_preview_adaptive_max_interval",
}
_PRESERVABLE_BACKEND_FIELDS = {
    "cfg_rescale",
    "compatibility_mode",
    "clip_skip",
    "tiling",
    "prompt_parser_name",
    "prompt_parser_kwargs",
    "prompt_cfg_recorded_schedules",
    "prompt_cfg_replay_mode",
    "prompt_expansion_recorded",
    "prompt_expansion_replay_mode",
    "prompt_semantic_recorded",
    "prompt_semantic_replay_mode",
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
    "hires_strategy",
    "hires_upscaler",
    "hires_upscaler_id",
    "hires_expected_native_scale",
    "hires_tile_size",
    "hires_tile_overlap",
    "hires_tile_batch_size",
    "hires_exact_resize_filter",
    "hires_final_size_correction_filter",
    "hires_aspect_policy",
    "hires_padding_mode",
    "hires_blurred_edge_method",
    "hires_blurred_edge_compare_diagnostics",
    "hires_recorded_target_correction",
    "hires_correction_fingerprint_enabled",
    "hires_recorded_correction_fingerprint",
    "hires_save_upscaled_pre_denoise",
    "hires_save_vae_roundtrip",
    "hires_diagnostic_vae_execution_fingerprint",
    "hires_save_lowres",
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
    "canonical_prompt_contract",
    "lora_paths",
    "prompt_asset_contract_version",
    "loras",
    "textual_inversions",
    "vae_name",
    "vae_hash",
    *RUNTIME_REPLAY_JOB_FIELDS,
}
_CORE_REQUIRED_FIELDS = {
    "positive_prompt",
    "negative_prompt",
    "seed",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "sampler_name",
    "scheduler_name",
    "model_path",
}
_MANIFEST_COMPLETENESS_FIELDS = (
    "required_for_rerun.prompt",
    "required_for_rerun.negative_prompt",
    "required_for_rerun.seed",
    "required_for_rerun.width",
    "required_for_rerun.height",
    "required_for_rerun.steps",
    "required_for_rerun.cfg_scale",
    "required_for_rerun.batch_size",
    "required_for_rerun.batch_count",
    "required_for_rerun.sampler_name",
    "required_for_rerun.scheduler_name",
    "required_for_rerun.model_path",
    "optional_for_rerun.sampler_kwargs",
    "optional_for_rerun.scheduler_kwargs",
    "optional_for_rerun.guidance_rescale",
    "optional_for_rerun.extra.vae_path",
    "extra.model_provenance.loaded_path",
    "extra.model_provenance.sha256",
)
_TOKEN_TTL_SECONDS = 15 * 60


def _canonical_model_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    compact = text.replace("_", "").replace("-", "").replace(" ", "")
    if "sdxl" in compact or "stablediffusionxl" in compact:
        return "sdxl"
    if "sd3" in compact or "stablediffusion3" in compact:
        return "sd3"
    if "sd2" in compact or "stablediffusion2" in compact:
        return "sd2"
    if "sd1" in compact or compact in {"15", "14", "sd15", "sd14"}:
        return "sd1"
    if "flux" in compact:
        return "flux"
    if "pony" in compact:
        return "pony"
    return compact


def _manifest_prompt_assets(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest or {})
    optional = dict(payload.get("optional_for_rerun") or {})
    extra = dict(optional.get("extra") or {})
    contract = dict(extra.get("prompt_assets") or {})

    def _from_references(values: Any, asset_type: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if not isinstance(values, list):
            return output
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                continue
            source = dict(item)
            metadata = dict(source.get("extra") or {})
            output.append({
                "asset_type": asset_type,
                "asset_id": metadata.get("asset_id") or source.get("requested_identifier") or source.get("resolved_identifier") or "",
                "catalog_asset_id": metadata.get("catalog_asset_id") or source.get("requested_identifier") or source.get("resolved_identifier") or "",
                "name": source.get("requested_display_name") or source.get("resolved_display_name") or "",
                "path": source.get("resolved_path") or source.get("requested_path") or "",
                "requested_path": source.get("requested_path") or source.get("resolved_path") or "",
                "resolved_path": source.get("resolved_path") or "",
                "requested_hash": source.get("requested_hash") or "",
                "resolved_hash": source.get("resolved_hash") or "",
                "weight": metadata.get("weight", 1.0),
                "enabled": metadata.get("enabled", True),
                "polarity": metadata.get("polarity", "positive"),
                "activation_text": metadata.get("activation_text") or "",
                "model_family": metadata.get("model_family") or "",
                "source_url": source.get("source_url") or "",
                "source": "replay",
                "original_source": metadata.get("source") or metadata.get("original_source") or "",
                "order": metadata.get("order", index),
                "metadata": metadata.get("metadata") or {},
            })
        return output

    raw_loras = contract.get("loras") if isinstance(contract.get("loras"), list) else extra.get("loras")
    raw_textual = contract.get("textual_inversions") if isinstance(contract.get("textual_inversions"), list) else extra.get("textual_inversions")
    if not isinstance(raw_loras, list) or not raw_loras:
        raw_loras = _from_references(payload.get("loras"), "lora")
    if not isinstance(raw_textual, list) or not raw_textual:
        raw_textual = _from_references(payload.get("embeddings"), "textual_inversion")

    prompt_sources = [
        dict(payload.get("required_for_rerun") or {}).get("prompt"),
        extra.get("hires_positive_prompt"),
    ]
    canonical_contract = dict(extra.get("canonical_prompt_contract") or payload.get("extra", {}).get("prompt_contract") or {})
    positive_structure = dict(canonical_contract.get("canonical_positive_structure") or {})
    if positive_structure.get("lossless_source") is not None:
        prompt_sources[0] = str(positive_structure.get("lossless_source") or "")
    inline_loras = extract_inline_loras_from_prompts(*prompt_sources)
    raw_loras = merge_replay_loras(raw_loras or [], inline_loras)

    def _replay(values: Any, asset_type: str) -> list[dict[str, Any]]:
        assets = normalize_prompt_asset_list(values or [], asset_type=asset_type, default_source="replay")
        output: list[dict[str, Any]] = []
        for asset in assets:
            original_source = asset.original_source or asset.source
            asset.source = "replay"
            asset.original_source = "" if original_source == "replay" else original_source
            output.append(asset.to_serializable_dict())
        return output

    return {
        "contract_version": str(contract.get("contract_version") or extra.get("prompt_asset_contract_version") or PROMPT_ASSET_CONTRACT_VERSION),
        "loras": _replay(raw_loras, "lora"),
        "textual_inversions": _replay(raw_textual, "textual_inversion"),
    }


@dataclass
class ReplayPreflight:
    valid: bool
    request: dict[str, Any]
    field_results: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    missing_assets: list[dict[str, Any]]
    preserved_settings: dict[str, Any]
    unsupported_settings: dict[str, Any]
    completeness: dict[str, Any]
    summary: dict[str, Any]
    preflight_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _StoredPreflight:
    token: str
    specification: dict[str, Any]
    created_monotonic: float = field(default_factory=time.monotonic)


class ReplayService:
    """Server-authoritative single-output replay preflight and submission."""

    def __init__(
        self,
        context: ProjectContext,
        jobs: GenerationJobManager,
        model_selection: WebUIModelSelectionState,
        *,
        upscaler_catalog: Any | None = None,
    ) -> None:
        self.context = context
        self.jobs = jobs
        self.model_selection = model_selection
        self.upscaler_catalog = upscaler_catalog
        self._tokens: dict[str, _StoredPreflight] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _restore_recorded_hires_base_dimensions(request: dict[str, Any]) -> None:
        """Restore first-pass dimensions for exact replay of hires outputs."""

        if not bool(request.get("hires_enabled", False)):
            return
        plan = request.get("hires_dimension_plan")
        if not isinstance(plan, Mapping):
            return
        try:
            base_width = int(plan.get("base_width") or 0)
            base_height = int(plan.get("base_height") or 0)
        except (TypeError, ValueError):
            return
        if base_width > 0 and base_height > 0:
            request["width"] = base_width
            request["height"] = base_height

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _recorded_prompt_cfg_schedules(
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        optional = manifest.get("optional_for_rerun") if isinstance(manifest, Mapping) else {}
        extra = (optional or {}).get("extra") if isinstance(optional, Mapping) else {}
        schedules = (extra or {}).get("prompt_cfg_pass_schedules") if isinstance(extra, Mapping) else {}
        return dict(schedules) if isinstance(schedules, Mapping) else {}


    @staticmethod
    def _recorded_prompt_expansions(
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        optional = manifest.get("optional_for_rerun") if isinstance(manifest, Mapping) else {}
        extra = (optional or {}).get("extra") if isinstance(optional, Mapping) else {}
        records = (extra or {}).get("prompt_expansion_pass_records") if isinstance(extra, Mapping) else {}
        return dict(records) if isinstance(records, Mapping) else {}

    @staticmethod
    def _recorded_prompt_semantics(
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        optional = manifest.get("optional_for_rerun") if isinstance(manifest, Mapping) else {}
        extra = (optional or {}).get("extra") if isinstance(optional, Mapping) else {}
        records = (extra or {}).get("prompt_semantic_pass_records") if isinstance(extra, Mapping) else {}
        return dict(records) if isinstance(records, Mapping) else {}

    @staticmethod
    def _recorded_regions(
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        optional = manifest.get("optional_for_rerun") if isinstance(manifest, Mapping) else {}
        extra = (optional or {}).get("extra") if isinstance(optional, Mapping) else {}
        records = (extra or {}).get("region_pass_records") if isinstance(extra, Mapping) else {}
        return dict(records) if isinstance(records, Mapping) else {}

    @staticmethod
    def _recorded_hires_schedule(
        manifest: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        optional = manifest.get("optional_for_rerun") if isinstance(manifest, Mapping) else {}
        optional_extra = (optional or {}).get("extra") if isinstance(optional, Mapping) else {}
        if isinstance(optional_extra, Mapping):
            replay = optional_extra.get("hires_recorded_schedule_replay")
            fingerprint = optional_extra.get("hires_recorded_schedule_fingerprint")
            if isinstance(replay, Mapping) and replay and isinstance(fingerprint, Mapping) and fingerprint:
                return dict(replay), dict(fingerprint)
        extra = manifest.get("extra") if isinstance(manifest, Mapping) else {}
        pipeline_metadata = (extra or {}).get("pipeline_metadata") if isinstance(extra, Mapping) else {}
        hires = (pipeline_metadata or {}).get("hires_fix") if isinstance(pipeline_metadata, Mapping) else {}
        if not isinstance(hires, Mapping):
            return {}, {}
        replay = hires.get("schedule_replay")
        fingerprint = hires.get("schedule_fingerprint")
        return (
            dict(replay) if isinstance(replay, Mapping) else {},
            dict(fingerprint) if isinstance(fingerprint, Mapping) else {},
        )

    @staticmethod
    def _has_path(source: Mapping[str, Any], dotted: str) -> bool:
        value: Any = source
        for part in dotted.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return False
            value = value[part]
        return True

    @staticmethod
    def _value_at(source: Mapping[str, Any], dotted: str) -> Any:
        value: Any = source
        for part in dotted.split("."):
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        return value

    @staticmethod
    def _set_path(target: dict[str, Any], dotted: str, value: Any) -> None:
        parts = [part for part in dotted.split(".") if part]
        if not parts:
            return
        cursor: dict[str, Any] = target
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = copy.deepcopy(value)

    @staticmethod
    def _flatten(source: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in source.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                if not value:
                    output[path] = {}
                else:
                    output.update(ReplayService._flatten(value, path))
            else:
                output[path] = value
        return output

    @staticmethod
    def _same(left: Any, right: Any) -> bool:
        return json.dumps(left, sort_keys=True, default=str) == json.dumps(
            right, sort_keys=True, default=str
        )

    @staticmethod
    def _safe_specification(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        source = dict(payload or {})
        mode = str(source.get("mode") or "exact").strip().lower()
        if mode not in {"exact", "selected"}:
            raise ValueError("Replay mode must be 'exact' or 'selected'.")
        seed_mode = str(source.get("seed_mode") or "original").strip().lower()
        if seed_mode not in {"original", "random"}:
            raise ValueError("Seed mode must be 'original' or 'random'.")
        model_mode = str(source.get("model_mode") or "original").strip().lower()
        if model_mode not in {"original", "current"}:
            raise ValueError("Model mode must be 'original' or 'current'.")
        prompt_mode = str(source.get("prompt_mode") or "raw_original").strip().lower()
        if prompt_mode not in {"raw_original", "canonical_recorded", "best_available"}:
            raise ValueError("Prompt mode must be 'raw_original', 'canonical_recorded', or 'best_available'.")
        output_id = str(source.get("output_id") or "").strip()
        if not output_id:
            raise ValueError("A replay output identifier is required.")
        override_fields = [str(item) for item in source.get("override_fields") or []]
        unknown_overrides = sorted(set(override_fields) - _BATCH_OVERRIDE_FIELDS)
        if unknown_overrides:
            raise ValueError("Unsupported replay override field(s): " + ", ".join(unknown_overrides))
        request_overrides = copy.deepcopy(dict(source.get("request_overrides") or {}))
        return {
            "output_id": output_id,
            "mode": mode,
            "selected_fields": [str(item) for item in source.get("selected_fields") or []],
            "current_values": copy.deepcopy(dict(source.get("current_values") or {})),
            "seed_mode": seed_mode,
            "model_mode": model_mode,
            "prompt_mode": prompt_mode,
            "remap": copy.deepcopy(dict(source.get("remap") or {})),
            "request_overrides": request_overrides,
            "override_fields": override_fields,
        }

    def _cleanup_tokens(self) -> None:
        cutoff = time.monotonic() - _TOKEN_TTL_SECONDS
        with self._lock:
            stale = [token for token, item in self._tokens.items() if item.created_monotonic < cutoff]
            for token in stale:
                self._tokens.pop(token, None)

    def _issue_token(self, specification: dict[str, Any]) -> str:
        self._cleanup_tokens()
        token = uuid.uuid4().hex
        with self._lock:
            self._tokens[token] = _StoredPreflight(
                token=token,
                specification=copy.deepcopy(specification),
            )
        return token

    def _consume_specification(self, token: str) -> dict[str, Any]:
        self._cleanup_tokens()
        with self._lock:
            stored = self._tokens.get(str(token or ""))
        if stored is None:
            raise ValueError("Replay preflight expired or was not found. Run preflight again.")
        return copy.deepcopy(stored.specification)

    def _completeness(self, details: OutputMetadataDetails) -> dict[str, Any]:
        manifest = details.manifest
        recorded = [path for path in _MANIFEST_COMPLETENESS_FIELDS if self._has_path(manifest, path)]
        missing = [path for path in _MANIFEST_COMPLETENESS_FIELDS if path not in recorded]
        required_missing = [field for field in _CORE_REQUIRED_FIELDS if field not in details.replay]
        quality = "exact_request" if not missing and not required_missing else "best_available"
        return {
            "contract_version": 1,
            "quality": quality,
            "label": "Exact Request Replay" if quality == "exact_request" else "Best Available Replay",
            "recorded_fields": recorded,
            "missing_fields": missing,
            "missing_core_fields": required_missing,
        }

    def _plugin_backend_only_settings(self, request: Mapping[str, Any]) -> dict[str, Any]:
        preserved: dict[str, Any] = {}
        for kind, name_key, kwargs_key in (
            ("sampler", "sampler_name", "sampler_kwargs"),
            ("scheduler", "scheduler_name", "scheduler_kwargs"),
        ):
            descriptor = self.jobs.registry.resolve_descriptor(request.get(name_key), kind=kind)
            values = self._mapping(request.get(kwargs_key))
            if descriptor is None or not values:
                continue
            schema = normalize_config_schema(descriptor.config_schema, kind=kind)
            properties = self._mapping(schema.get("properties"))
            if not schema.get("additionalProperties", False):
                continue
            for key, value in values.items():
                if key not in properties:
                    preserved[f"{kwargs_key}.{key}"] = value
        return preserved

    @staticmethod
    def _remove_unsupported_paths(request: dict[str, Any], unsupported: Mapping[str, Any]) -> None:
        for path in unsupported:
            parts = str(path).split(".")
            if len(parts) != 2 or parts[0] not in {"sampler_kwargs", "scheduler_kwargs"}:
                continue
            container = request.get(parts[0])
            if isinstance(container, dict):
                container.pop(parts[1], None)

    def _resolve_vae(self, value: Any) -> Path:
        text = str(value or "").strip()
        if not text:
            raise ValueError("A VAE replacement path is required.")
        candidates = [self.context.resolve_project_path(text).expanduser().resolve()]
        tail = Path(text.replace("\\", "/")).name
        if tail:
            candidates.append((self.context.vae_dir / tail).expanduser().resolve())
        seen: set[str] = set()
        for candidate in candidates:
            token = str(candidate).casefold()
            if token in seen:
                continue
            seen.add(token)
            if candidate.is_file() and candidate.suffix.lower() in MODEL_EXTENSIONS:
                return candidate
        raise ValueError(f"VAE file could not be resolved: {text}")

    def _build_raw_request(
        self,
        details: OutputMetadataDetails,
        specification: Mapping[str, Any],
    ) -> tuple[dict[str, Any], set[str]]:
        original = copy.deepcopy(details.replay)
        current = copy.deepcopy(dict(specification.get("current_values") or {}))
        mode = specification["mode"]
        replaced: set[str] = set()

        if mode == "selected":
            request = current
            for field_name in specification.get("selected_fields") or []:
                if field_name in original:
                    request[field_name] = copy.deepcopy(original[field_name])
                    continue
                if "." in field_name and self._has_path(original, field_name):
                    self._set_path(request, field_name, self._value_at(original, field_name))
        else:
            request = original
            for name in _OPERATIONAL_FIELDS:
                if name in current:
                    request[name] = copy.deepcopy(current[name])

        if mode == "exact":
            self._restore_recorded_hires_base_dimensions(request)

        # Phase 14K-11 runtime controls are not creative request fields. Restore
        # them for both exact and selected replay modes whenever the source
        # image recorded them, while still allowing explicit future replay UI
        # controls to supply a value first.
        for name in RUNTIME_REPLAY_JOB_FIELDS:
            if name in original:
                request.setdefault(name, copy.deepcopy(original[name]))

        prompt_mode = str(specification.get("prompt_mode") or "raw_original")
        prompt_contract = self._mapping(self._mapping(details.manifest.get("extra")).get("prompt_contract"))
        prompt_mode_fallback = ""
        if prompt_mode == "raw_original":
            if "raw_positive" in prompt_contract or "raw_negative" in prompt_contract:
                request["positive_prompt"] = str(prompt_contract.get("raw_positive") or "")
                request["negative_prompt"] = str(prompt_contract.get("raw_negative") or "")
            else:
                prompt_mode_fallback = "Raw original prompts were unavailable; replay used the best available recorded prompt."
        elif prompt_mode == "canonical_recorded":
            positive_structure = self._mapping(prompt_contract.get("canonical_positive_structure"))
            negative_structure = self._mapping(prompt_contract.get("canonical_negative_structure"))
            positive = positive_structure.get("lossless_source") or prompt_contract.get("translated_positive") or prompt_contract.get("canonical_positive")
            negative = negative_structure.get("lossless_source") or prompt_contract.get("translated_negative") or prompt_contract.get("canonical_negative")
            if positive is not None or negative is not None:
                request["positive_prompt"] = str(positive or "")
                request["negative_prompt"] = str(negative or "")
                request["prompt_shortcut_profile_name"] = "canonical"
                request.pop("prompt_shortcut_profile_snapshot", None)
                request["base_shortcut_profile_name"] = "canonical"
            else:
                prompt_mode_fallback = "Canonical recorded prompts were unavailable; replay used the best available recorded prompt."

        request_overrides = dict(specification.get("request_overrides") or {})
        for name in specification.get("override_fields") or []:
            if name in request_overrides:
                request[name] = copy.deepcopy(request_overrides[name])
                replaced.add(name)

        remap = dict(specification.get("remap") or {})
        for name in ("model_path", "vae_path", "sampler_name", "scheduler_name"):
            if remap.get(name) not in (None, ""):
                request[name] = remap[name]
                replaced.add(name)

        if specification.get("seed_mode") == "random":
            request["seed"] = -1
            replaced.add("seed")

        if specification.get("model_mode") == "current":
            current_model = self.model_selection.current()
            if current_model is None:
                request.pop("model_path", None)
            else:
                request["model_path"] = current_model.resolved_path
            replaced.add("model_path")

        schedule_sensitive_fields = {
            "sampler_name",
            "scheduler_name",
            "hires_sampler_name",
            "hires_scheduler_name",
            "hires_steps",
            "hires_denoising_strength",
            "hires_step_policy",
        }
        recorded_replay, recorded_fingerprint = self._recorded_hires_schedule(
            details.manifest
        )
        use_recorded_schedule = bool(
            mode == "exact"
            and recorded_replay
            and recorded_fingerprint
            and not (replaced & schedule_sensitive_fields)
        )
        if recorded_replay and recorded_fingerprint:
            request["hires_schedule_conformance_source_replay"] = recorded_replay
            request["hires_schedule_conformance_source_fingerprint"] = recorded_fingerprint
        if use_recorded_schedule:
            request["hires_recorded_schedule_replay"] = recorded_replay
            request["hires_recorded_schedule_fingerprint"] = recorded_fingerprint
            request["hires_schedule_replay_mode"] = "recorded_exact"
        else:
            request.pop("hires_recorded_schedule_replay", None)
            request.pop("hires_recorded_schedule_fingerprint", None)
            request["hires_schedule_replay_mode"] = "reconstruct"

        prompt_cfg_sensitive_fields = {
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
        recorded_prompt_cfg_schedules = self._recorded_prompt_cfg_schedules(
            details.manifest
        )
        use_recorded_prompt_cfg = bool(
            mode == "exact"
            and recorded_prompt_cfg_schedules
            and not (replaced & prompt_cfg_sensitive_fields)
        )
        if use_recorded_prompt_cfg:
            request["prompt_cfg_recorded_schedules"] = recorded_prompt_cfg_schedules
            request["prompt_cfg_replay_mode"] = "recorded_exact"
        else:
            request.pop("prompt_cfg_recorded_schedules", None)
            request["prompt_cfg_replay_mode"] = "reconstruct"

        prompt_expansion_sensitive_fields = {
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
        recorded_prompt_expansions = self._recorded_prompt_expansions(details.manifest)
        use_recorded_prompt_expansion = bool(
            mode == "exact"
            and recorded_prompt_expansions
            and not (replaced & prompt_expansion_sensitive_fields)
        )
        if use_recorded_prompt_expansion:
            request["prompt_expansion_recorded"] = recorded_prompt_expansions
            request["prompt_expansion_replay_mode"] = "recorded_exact"
        else:
            request.pop("prompt_expansion_recorded", None)
            request["prompt_expansion_replay_mode"] = "reconstruct"

        recorded_prompt_semantics = self._recorded_prompt_semantics(details.manifest)
        use_recorded_prompt_semantics = bool(
            mode == "exact"
            and recorded_prompt_semantics
            and use_recorded_prompt_expansion
            and not (replaced & prompt_expansion_sensitive_fields)
        )
        if use_recorded_prompt_semantics:
            request["prompt_semantic_recorded"] = recorded_prompt_semantics
            request["prompt_semantic_replay_mode"] = "recorded_exact"
        else:
            request.pop("prompt_semantic_recorded", None)
            request["prompt_semantic_replay_mode"] = "reconstruct"

        region_sensitive_fields = prompt_expansion_sensitive_fields | {
            "width",
            "height",
            "steps",
            "hires_size_mode",
            "hires_scale",
            "hires_width",
            "hires_height",
            "hires_steps",
        }
        recorded_regions = self._recorded_regions(details.manifest)
        use_recorded_regions = bool(
            mode == "exact"
            and recorded_regions
            and use_recorded_prompt_expansion
            and not (replaced & region_sensitive_fields)
        )
        if use_recorded_regions:
            request["region_recorded"] = recorded_regions
            request["region_replay_mode"] = "recorded_exact"
        else:
            request.pop("region_recorded", None)
            request["region_replay_mode"] = "reconstruct"

        prompt_assets = _manifest_prompt_assets(details.manifest)
        request["prompt_asset_contract_version"] = prompt_assets["contract_version"]
        request["loras"] = [dict(item) for item in prompt_assets["loras"]]
        request["textual_inversions"] = [dict(item) for item in prompt_assets["textual_inversions"]]
        request["lora_paths"] = [
            str(item.get("resolved_path") or item.get("path") or item.get("requested_path") or "")
            for item in prompt_assets["loras"]
            if item.get("resolved_path") or item.get("path") or item.get("requested_path")
        ]
        self._remove_unsupported_paths(request, details.unsupported)
        request.setdefault("save_images", True)
        request.setdefault("batch_size", 1)
        request.setdefault("batch_count", 1)
        recorded_execution = extract_runtime_execution_record(details.manifest)
        if recorded_execution:
            request["runtime_replay_conformance_source"] = runtime_execution_fingerprint(
                recorded_execution
            )

        diagnostics = self._mapping(request.get("diagnostics"))
        diagnostics["replay"] = {
            "source_output_id": details.output_id,
            "mode": mode,
            "seed_mode": specification.get("seed_mode"),
            "model_mode": specification.get("model_mode"),
            "prompt_mode": prompt_mode,
            "prompt_mode_fallback": prompt_mode_fallback,
            "metadata_source": details.metadata_source,
            "schedule_replay_mode": request.get("hires_schedule_replay_mode", "reconstruct"),
            "recorded_schedule_available": bool(recorded_replay and recorded_fingerprint),
            "prompt_cfg_replay_mode": request.get("prompt_cfg_replay_mode", "reconstruct"),
            "recorded_prompt_cfg_available": bool(recorded_prompt_cfg_schedules),
            "prompt_expansion_replay_mode": request.get("prompt_expansion_replay_mode", "reconstruct"),
            "recorded_prompt_expansion_available": bool(recorded_prompt_expansions),
            "region_replay_mode": request.get("region_replay_mode", "reconstruct"),
            "recorded_region_available": bool(recorded_regions),
        }
        request["diagnostics"] = diagnostics
        return request, replaced

    @classmethod
    def _recorded_hires_runtime_identity(
        cls,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        optional = cls._mapping(manifest.get("optional_for_rerun"))
        optional_extra = cls._mapping(optional.get("extra"))
        compact = {
            "upscaler_id": str(
                optional_extra.get("hires_upscaler_id")
                or optional_extra.get("hires_upscaler")
                or ""
            ),
            "upscaler_sha256": str(
                optional_extra.get("hires_expected_upscaler_sha256") or ""
            ).casefold(),
            "upscaler_display_name": "",
            "vae_sha256": str(
                optional_extra.get("hires_expected_vae_sha256") or ""
            ).casefold(),
            "vae_source_kind": str(
                optional_extra.get("hires_expected_vae_source_kind") or ""
            ),
        }
        if any(compact.values()):
            return compact
        extra = cls._mapping(manifest.get("extra"))
        pipeline = cls._mapping(extra.get("pipeline_metadata"))
        hires = cls._mapping(pipeline.get("hires_fix"))
        source = cls._mapping(hires.get("pixel_source_preparation"))
        upscale = cls._mapping(source.get("upscale_metadata"))
        vae_encode = cls._mapping(source.get("vae_encode"))
        vae = cls._mapping(vae_encode.get("vae"))
        plan = cls._mapping(hires.get("upscale_plan"))
        descriptor = cls._mapping(plan.get("descriptor"))
        return {
            "upscaler_id": str(
                upscale.get("upscaler_id")
                or plan.get("upscaler_id")
                or descriptor.get("upscaler_id")
                or ""
            ),
            "upscaler_sha256": str(
                upscale.get("upscaler_sha256")
                or descriptor.get("sha256")
                or ""
            ).casefold(),
            "upscaler_display_name": str(
                upscale.get("upscaler_display_name")
                or descriptor.get("display_name")
                or ""
            ),
            "vae_sha256": str(vae.get("sha256") or "").casefold(),
            "vae_source_kind": str(vae.get("source_kind") or ""),
        }

    def _validate_hires_replay_identity(
        self,
        raw_request: dict[str, Any],
        manifest: Mapping[str, Any],
    ) -> list[str]:
        if not bool(raw_request.get("hires_enabled", False)):
            return []
        selected = str(
            raw_request.get("hires_upscaler_id")
            or raw_request.get("hires_upscaler")
            or ""
        ).strip()
        strategy = str(raw_request.get("hires_strategy") or "pixel_neural").strip().casefold()
        if strategy != "pixel_neural":
            return [
                f"Recorded neural upscaler ID {selected!r} requires hires_strategy='pixel_neural'; replay will not fall back."
            ]
        if self.upscaler_catalog is None:
            return [
                "Exact pixel-neural replay cannot validate the current upscaler catalog."
            ]
        identity = self._recorded_hires_runtime_identity(manifest)
        expected_id = str(identity.get("upscaler_id") or selected)
        expected_hash = str(identity.get("upscaler_sha256") or "").casefold()
        descriptor = self.upscaler_catalog.descriptor(expected_id)
        if descriptor is None:
            return [
                f"Recorded neural upscaler {expected_id!r} is missing. Refresh the catalog or restore the exact model; replay will not substitute another file."
            ]
        if not descriptor.selectable:
            return [
                f"Recorded neural upscaler {expected_id!r} is present but not selectable: {descriptor.load_status}."
            ]
        if len(expected_hash) != 64:
            return [
                f"Recorded neural upscaler {expected_id!r} lacks the full SHA-256 required for exact replay."
            ]
        if descriptor.sha256.casefold() != expected_hash:
            return [
                f"Recorded neural upscaler hash mismatch for {expected_id!r}: expected {expected_hash}, found {descriptor.sha256.casefold()}. The same filename with different content is rejected."
            ]
        raw_request["hires_strategy"] = "pixel_neural"
        raw_request["hires_upscaler"] = descriptor.upscaler_id
        raw_request["hires_upscaler_id"] = descriptor.upscaler_id

        expected_vae_hash = str(identity.get("vae_sha256") or "").casefold()
        if len(expected_vae_hash) != 64:
            return [
                "Recorded pixel-neural job lacks the full VAE SHA-256 required for exact replay."
            ]
        vae_path = str(raw_request.get("vae_path") or "").strip()
        source_kind = str(identity.get("vae_source_kind") or "").strip().casefold()
        candidate_path = Path(vae_path) if vae_path else Path(str(raw_request.get("model_path") or ""))
        if not candidate_path.is_file():
            label = "external VAE" if vae_path else "checkpoint containing the embedded VAE"
            return [f"The {label} required for exact VAE hash validation is unavailable: {candidate_path}."]
        if not vae_path and source_kind not in {
            "embedded", "embedded_checkpoint", "checkpoint_embedded"
        }:
            return [
                "The recorded VAE was not checkpoint-embedded, but replay metadata does not resolve an external VAE path."
            ]
        actual_vae_hash = self._sha256_file(candidate_path)
        if actual_vae_hash != expected_vae_hash:
            return [
                f"Recorded VAE hash mismatch: expected {expected_vae_hash}, found {actual_vae_hash}. Exact pixel-neural replay is blocked."
            ]
        return []

    def _validate_assets_and_plugins(
        self,
        raw_request: dict[str, Any],
        specification: Mapping[str, Any],
        manifest: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], ActiveModelSelection | None]:
        missing: list[dict[str, Any]] = []
        errors: list[str] = []
        authorized_model: ActiveModelSelection | None = None

        model_path = raw_request.get("model_path")
        if not model_path:
            message = (
                "No current WebUI model is active." if specification.get("model_mode") == "current"
                else "The original checkpoint path was not recorded."
            )
            missing.append({"kind": "model", "field": "model_path", "requested": model_path, "reason": message})
            errors.append(message)
        else:
            try:
                authorized_model = self.model_selection.authorize(model_path, source="replay_preflight")
                raw_request["model_path"] = authorized_model.resolved_path
            except (OSError, ValueError) as exc:
                missing.append({"kind": "model", "field": "model_path", "requested": model_path, "reason": str(exc)})
                errors.append(f"Checkpoint unavailable: {exc}")

        vae_path = raw_request.get("vae_path")
        if vae_path not in (None, ""):
            try:
                raw_request["vae_path"] = str(self._resolve_vae(vae_path))
            except (OSError, ValueError) as exc:
                missing.append({"kind": "vae", "field": "vae_path", "requested": vae_path, "reason": str(exc)})
                errors.append(f"VAE unavailable: {exc}")

        errors.extend(self._validate_hires_replay_identity(raw_request, manifest or {}))

        checkpoint_family = _canonical_model_family(
            (authorized_model.architecture if authorized_model is not None else "")
            or (authorized_model.architecture_summary if authorized_model is not None else "")
            or (authorized_model.checkpoint_kind if authorized_model is not None else "")
        )
        lora_resolver = LoRAResolver(self.context)
        for asset_type, field_name in (("lora", "loras"), ("textual_inversion", "textual_inversions")):
            structured_assets = raw_request.get(field_name)
            if not isinstance(structured_assets, list):
                continue
            validated_assets: list[dict[str, Any]] = []
            for index, item in enumerate(structured_assets):
                if not isinstance(item, Mapping):
                    continue
                candidate = dict(item)
                if candidate.get("enabled") is False:
                    validated_assets.append(candidate)
                    continue
                resolved_path = str(
                    candidate.get("resolved_path")
                    or candidate.get("path")
                    or candidate.get("requested_path")
                    or ""
                ).strip()
                display = str(
                    candidate.get("name")
                    or candidate.get("requested_name")
                    or Path(resolved_path).stem
                    or f"{asset_type.replace('_', ' ').title()} {index + 1}"
                )
                if not resolved_path and asset_type == "lora":
                    try:
                        resolved_file = lora_resolver.resolve(display, resolved_path)
                    except ValueError:
                        resolved_file = None
                    if resolved_file is not None:
                        resolved_path = str(resolved_file)
                        metadata = lora_resolver.metadata(resolved_file)
                        compatibility = lora_resolver.compatibility_hash(resolved_file, sidecar_metadata=metadata)
                        candidate["name"] = candidate.get("name") or resolved_file.stem
                        candidate["path"] = resolved_path
                        candidate["requested_path"] = candidate.get("requested_path") or resolved_path
                        candidate["resolved_path"] = resolved_path
                        candidate["requested_hash"] = candidate.get("requested_hash") or lora_resolver.file_hash(resolved_file)
                        candidate["resolved_hash"] = candidate.get("resolved_hash") or lora_resolver.file_hash(resolved_file)
                        candidate["model_family"] = candidate.get("model_family") or metadata.get("model_family") or ""
                        candidate["activation_text"] = candidate.get("activation_text") or metadata.get("activation_text") or ""
                        candidate["source_url"] = candidate.get("source_url") or metadata.get("source_url") or ""
                        if compatibility.get("a1111_hash"):
                            candidate.setdefault("metadata", {})
                            candidate["metadata"]["a1111_hash"] = compatibility.get("a1111_hash")
                            candidate["metadata"]["a1111_short_hash"] = compatibility.get("a1111_short_hash")
                if not resolved_path:
                    reason = f"Recorded {asset_type.replace('_', ' ')} path is missing for {display!r}."
                    missing.append({"kind": asset_type, "field": f"{field_name}[{index}]", "requested": display, "reason": reason})
                    errors.append(reason)
                    continue
                try:
                    resolved_file = self.context.resolve_project_path(resolved_path).expanduser().resolve()
                except Exception:
                    resolved_file = Path(resolved_path).expanduser()
                if not resolved_file.is_file() and asset_type == "lora":
                    try:
                        resolved_retry = lora_resolver.resolve(display, resolved_path)
                    except ValueError:
                        resolved_retry = None
                    if resolved_retry is not None:
                        resolved_file = Path(resolved_retry)
                        resolved_path = str(resolved_retry)
                        metadata = lora_resolver.metadata(resolved_file)
                        candidate["name"] = candidate.get("name") or resolved_file.stem
                        candidate["path"] = resolved_path
                        candidate["requested_path"] = candidate.get("requested_path") or resolved_path
                        candidate["resolved_path"] = resolved_path
                        candidate["requested_hash"] = candidate.get("requested_hash") or lora_resolver.file_hash(resolved_file)
                        candidate["resolved_hash"] = candidate.get("resolved_hash") or lora_resolver.file_hash(resolved_file)
                        candidate["model_family"] = candidate.get("model_family") or metadata.get("model_family") or ""
                        candidate["activation_text"] = candidate.get("activation_text") or metadata.get("activation_text") or ""
                        candidate["source_url"] = candidate.get("source_url") or metadata.get("source_url") or ""
                if not resolved_file.is_file():
                    reason = f"Recorded {asset_type.replace('_', ' ')} is not installed: {display!r} ({resolved_path})."
                    missing.append({"kind": asset_type, "field": f"{field_name}[{index}]", "requested": resolved_path, "reason": reason})
                    errors.append(reason)
                    continue
                asset_family = _canonical_model_family(candidate.get("model_family"))
                if asset_family and checkpoint_family and asset_family != checkpoint_family:
                    reason = (
                        f"Recorded {asset_type.replace('_', ' ')} {display!r} targets model family "
                        f"'{asset_family}', but the replay checkpoint family is '{checkpoint_family}'."
                    )
                    missing.append({"kind": asset_type, "field": f"{field_name}[{index}]", "requested": display, "reason": reason})
                    errors.append(reason)
                    continue
                candidate["resolved_path"] = str(resolved_file)
                candidate["path"] = str(resolved_file)
                candidate["original_source"] = candidate.get("original_source") or (
                    candidate.get("source") if candidate.get("source") != "replay" else ""
                )
                candidate["source"] = "replay"
                candidate.setdefault("order", index)
                validated_assets.append(candidate)
            raw_request[field_name] = validated_assets
            if asset_type == "lora":
                raw_request["lora_paths"] = [
                    str(item.get("resolved_path") or item.get("path") or "")
                    for item in validated_assets
                    if item.get("enabled") is not False
                ]

        for kind, field_name in (("sampler", "sampler_name"), ("scheduler", "scheduler_name")):
            requested = raw_request.get(field_name)
            if not requested or self.jobs.registry.resolve_descriptor(requested, kind=kind) is None:
                reason = f"Recorded {kind} plugin is not installed: {requested!r}."
                missing.append({"kind": kind, "field": field_name, "requested": requested, "reason": reason})
                errors.append(reason)

        from modules.prompt_parsers import default_prompt_parser_registry

        requested_parser = raw_request.get("prompt_parser_name") or "legacy"
        if not default_prompt_parser_registry().has(requested_parser, require_available=True):
            reason = f"Recorded prompt parser is not installed: {requested_parser!r}."
            missing.append({
                "kind": "prompt_parser",
                "field": "prompt_parser_name",
                "requested": requested_parser,
                "reason": reason,
            })
            errors.append(reason)

        from modules.prompt_shortcuts import PromptShortcutProfileDescriptor, default_prompt_shortcut_registry, validate_prompt_shortcut_profile

        profile_name = raw_request.get("prompt_shortcut_profile_name") or ("legacy_default" if requested_parser == "legacy" else ("parser21_native" if requested_parser == "parser21" else ("superhybrid_native" if requested_parser == "superhybrid" else "canonical")))
        snapshot = raw_request.get("prompt_shortcut_profile_snapshot")
        if isinstance(snapshot, Mapping) and snapshot:
            profile = PromptShortcutProfileDescriptor.from_dict(dict(snapshot), builtin=bool(snapshot.get("builtin", False)))
            validation = validate_prompt_shortcut_profile(profile)
            if not validation.valid:
                reason = "Recorded prompt shortcut profile snapshot is invalid."
                missing.append({"kind": "prompt_shortcut_profile", "field": "prompt_shortcut_profile_snapshot", "requested": profile_name, "reason": reason})
                errors.append(reason)
        elif not default_prompt_shortcut_registry().has(profile_name):
            reason = f"Recorded prompt shortcut profile is not installed: {profile_name!r}."
            missing.append({"kind": "prompt_shortcut_profile", "field": "prompt_shortcut_profile_name", "requested": profile_name, "reason": reason})
            errors.append(reason)

        return missing, errors, authorized_model

    def _field_results(
        self,
        *,
        original: Mapping[str, Any],
        outgoing: Mapping[str, Any],
        replaced: set[str],
        preserved: Mapping[str, Any],
        unsupported: Mapping[str, Any],
        completeness: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        original_flat = self._flatten(original)
        outgoing_flat = self._flatten(outgoing)
        results: list[dict[str, Any]] = []
        keys = sorted(set(original_flat) | set(outgoing_flat))
        for path in keys:
            top = path.split(".", 1)[0]
            original_value = original_flat.get(path)
            outgoing_value = outgoing_flat.get(path)
            if path in preserved or top in _PRESERVABLE_BACKEND_FIELDS:
                status = "preserved_backend_only"
            elif top in replaced:
                status = "replaced"
            elif path not in original_flat:
                status = "normalized"
            elif self._same(original_value, outgoing_value):
                status = "exact"
            else:
                status = "normalized"
            results.append(
                {
                    "field": path,
                    "original": original_value,
                    "outgoing": outgoing_value,
                    "status": status,
                }
            )

        for path, entry in sorted(unsupported.items()):
            results.append(
                {
                    "field": path,
                    "original": entry.get("value") if isinstance(entry, Mapping) else None,
                    "outgoing": None,
                    "status": "unsupported",
                    "reason": entry.get("reason") if isinstance(entry, Mapping) else str(entry),
                }
            )
        for path in completeness.get("missing_fields") or []:
            results.append(
                {
                    "field": path,
                    "original": None,
                    "outgoing": None,
                    "status": "missing",
                    "reason": "Not recorded under the Phase 10B manifest-completeness contract.",
                }
            )
        return results

    def _evaluate(self, specification: dict[str, Any], *, issue_token: bool) -> ReplayPreflight:
        details = load_output_details(self.context, specification["output_id"])
        completeness = self._completeness(details)
        raw_request, replaced = self._build_raw_request(details, specification)
        missing_assets, errors, _ = self._validate_assets_and_plugins(
            raw_request, specification, details.manifest
        )
        warnings = list(details.warnings)
        enabled_textual_inversions = [
            item for item in raw_request.get("textual_inversions") or []
            if isinstance(item, Mapping) and item.get("enabled") is not False
        ]
        if enabled_textual_inversions:
            warnings.append(
                "Textual-inversion selections are preserved by the Phase UI-6 replay contract, "
                "but textual-inversion runtime application is still planned."
            )
        manifest = self._mapping(details.manifest)
        recorded_execution = extract_runtime_execution_record(manifest)
        if not recorded_execution:
            recorded_runtime = self._mapping(
                self._mapping(manifest.get("extra")).get("runtime_startup_options")
            )
            if not recorded_runtime:
                recorded_runtime = self._mapping(
                    self._mapping(
                        self._mapping(manifest.get("optional_for_rerun")).get("extra")
                    ).get("runtime_startup_options")
                )
            warnings.extend(
                runtime_replay_warnings(
                    recorded_runtime,
                    self.jobs.runtime_startup_options,
                )
            )
        prompt_replay_diagnostics = self._mapping(self._mapping(raw_request.get("diagnostics")).get("replay"))
        if prompt_replay_diagnostics.get("prompt_mode_fallback"):
            warnings.append(str(prompt_replay_diagnostics["prompt_mode_fallback"]))
        if specification.get("prompt_mode") == "canonical_recorded":
            warnings.append("Canonical Recorded Prompt mode replays the stored parser input through the canonical shortcut profile rather than the original user-facing aliases.")
        elif specification.get("prompt_mode") == "best_available":
            warnings.append("Best Available Prompt mode may not preserve the original user-facing shortcut text exactly.")

        normalized: dict[str, Any] = {}
        if not errors:
            try:
                strict_source = dict(raw_request)
                strict_source["_webui_scheduler_user_selected"] = True
                strict_source["_webui_selection_version"] = 2
                strict = self.jobs.selections.normalize(
                    strict_source,
                    fallback_payload={},
                    migrate_legacy_auto_fallback=False,
                    reject_unknown=True,
                )
                normalized = self.jobs.normalize_generation_request(strict.payload)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))

        runtime_assessment = runtime_replay_assessment(
            recorded_execution,
            self.jobs.runtime_startup_options,
            outgoing_request=normalized or raw_request,
        )
        warnings.extend(runtime_assessment.get("warnings") or [])
        completeness["runtime_replay"] = runtime_assessment
        if (
            runtime_assessment.get("recorded_runtime_available", False)
            and not runtime_assessment.get("exact_replay_supported", False)
        ):
            completeness["quality"] = "best_available"
            completeness["label"] = "Best Available Replay"

        for field_name in sorted(_CORE_REQUIRED_FIELDS):
            if field_name not in raw_request:
                errors.append(f"Required replay field is missing: {field_name}.")

        if completeness["quality"] != "exact_request":
            warnings.append(
                "This output predates or does not satisfy the complete Phase 10B replay manifest. "
                "It is labeled Best Available Replay rather than Exact Request Replay."
            )
        if details.image.get("model", {}).get("hash"):
            warnings.append(
                "The stored checkpoint SHA-256 is reported for comparison; preflight validates the file path, "
                "extension, size, and modification snapshot without re-hashing a potentially multi-gigabyte file."
            )

        preserved = {
            key: copy.deepcopy(value)
            for key, value in normalized.items()
            if key in _PRESERVABLE_BACKEND_FIELDS
        }
        preserved.update(self._plugin_backend_only_settings(normalized))
        field_results = self._field_results(
            original=details.replay,
            outgoing=normalized,
            replaced=replaced,
            preserved=preserved,
            unsupported=details.unsupported,
            completeness=completeness,
        )

        summary = {
            "output_id": details.output_id,
            "metadata_source": details.metadata_source,
            "replay_label": completeness["label"],
            "prompt": normalized.get("positive_prompt", raw_request.get("positive_prompt", "")),
            "negative_prompt": normalized.get("negative_prompt", raw_request.get("negative_prompt", "")),
            "seed": normalized.get("seed", raw_request.get("seed")),
            "model_path": normalized.get("model_path", raw_request.get("model_path")),
            "vae_path": normalized.get("vae_path", raw_request.get("vae_path")),
            "sampler_name": normalized.get("sampler_name", raw_request.get("sampler_name")),
            "scheduler_name": normalized.get("scheduler_name", raw_request.get("scheduler_name")),
            "width": normalized.get("width", raw_request.get("width")),
            "height": normalized.get("height", raw_request.get("height")),
            "steps": normalized.get("steps", raw_request.get("steps")),
            "cfg_scale": normalized.get("cfg_scale", raw_request.get("cfg_scale")),
            "batch_size": normalized.get("batch_size", raw_request.get("batch_size")),
            "batch_count": normalized.get("batch_count", raw_request.get("batch_count")),
            "advanced_setting_count": sum(
                len(self._mapping(normalized.get(key)))
                for key in ("sampler_kwargs", "scheduler_kwargs")
            ),
            "preserved_setting_count": len(preserved),
            "unsupported_setting_count": len(details.unsupported),
            "runtime_exact_replay_supported": bool(
                runtime_assessment.get("exact_replay_supported", False)
            ),
            "runtime_substitution_count": len(
                runtime_assessment.get("substitutions") or []
            ),
        }

        result = ReplayPreflight(
            valid=not errors,
            request=normalized,
            field_results=field_results,
            warnings=list(dict.fromkeys(warnings)),
            errors=list(dict.fromkeys(errors)),
            missing_assets=missing_assets,
            preserved_settings=preserved,
            unsupported_settings=copy.deepcopy(details.unsupported),
            completeness=completeness,
            summary=summary,
        )
        if issue_token:
            result.preflight_token = self._issue_token(specification)
        return result

    def evaluate_specification(
        self,
        payload: Mapping[str, Any] | None,
        *,
        issue_token: bool = False,
    ) -> ReplayPreflight:
        """Validate a replay specification through the canonical replay path.

        Phase 10C uses this public entry point for independently validated batch
        items while retaining the same normalization, asset authorization, and
        plugin checks as single-output replay.
        """
        specification = self._safe_specification(payload)
        return self._evaluate(specification, issue_token=issue_token)

    def preflight(self, payload: Mapping[str, Any] | None) -> ReplayPreflight:
        return self.evaluate_specification(payload, issue_token=True)

    async def submit(self, preflight_token: str) -> tuple[ReplayPreflight, Any]:
        specification = self._consume_specification(preflight_token)
        result = self._evaluate(specification, issue_token=False)
        if not result.valid:
            raise ValueError("Replay preflight is no longer valid: " + "; ".join(result.errors))
        selection = self.model_selection.authorize(
            result.request.get("model_path"), source="replay_submission"
        )
        request = dict(result.request)
        request["model_path"] = selection.resolved_path
        job = await self.jobs.submit(request, model_selection=selection.to_dict())
        with self._lock:
            self._tokens.pop(preflight_token, None)
        return result, job


def request_fingerprint(request: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(request), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["ReplayPreflight", "ReplayService", "request_fingerprint", "_BATCH_OVERRIDE_FIELDS"]
