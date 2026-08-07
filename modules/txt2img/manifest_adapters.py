from __future__ import annotations

from typing import Any

from image_gen.contracts import (
    PROMPT_ASSET_CONTRACT_VERSION,
    normalize_prompt_asset_list,
)
from modules.txt2img.generation_manifest import GenerationManifest


def _recorded_hires_runtime_identity(manifest: GenerationManifest) -> dict[str, str]:
    payload = manifest.to_dict()
    optional = payload.get("optional_for_rerun") if isinstance(payload, dict) else {}
    optional_extra = (optional or {}).get("extra") if isinstance(optional, dict) else {}
    if isinstance(optional_extra, dict):
        compact_identity = {
            "strategy": str(optional_extra.get("hires_strategy") or ""),
            "upscaler_id": str(optional_extra.get("hires_upscaler_id") or optional_extra.get("hires_upscaler") or ""),
            "upscaler_sha256": str(optional_extra.get("hires_expected_upscaler_sha256") or "").casefold(),
            "native_scale": int(optional_extra.get("hires_expected_native_scale") or 0),
            "aspect_policy": str(optional_extra.get("hires_aspect_policy") or ""),
            "vae_sha256": str(optional_extra.get("hires_expected_vae_sha256") or "").casefold(),
            "vae_source_kind": str(optional_extra.get("hires_expected_vae_source_kind") or ""),
        }
        if any(compact_identity.values()):
            return compact_identity
    extra = payload.get("extra") if isinstance(payload, dict) else {}
    pipeline = (extra or {}).get("pipeline_metadata") if isinstance(extra, dict) else {}
    hires = (pipeline or {}).get("hires_fix") if isinstance(pipeline, dict) else {}
    if not isinstance(hires, dict):
        return {}
    source = hires.get("pixel_source_preparation")
    source = source if isinstance(source, dict) else {}
    upscale = source.get("upscale_metadata")
    upscale = upscale if isinstance(upscale, dict) else {}
    vae_encode = source.get("vae_encode")
    vae_encode = vae_encode if isinstance(vae_encode, dict) else {}
    vae = vae_encode.get("vae")
    vae = vae if isinstance(vae, dict) else {}
    plan = hires.get("upscale_plan")
    plan = plan if isinstance(plan, dict) else {}
    descriptor = plan.get("descriptor")
    descriptor = descriptor if isinstance(descriptor, dict) else {}
    diagnostics = hires.get("phase14n7_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    diagnostic_upscaler = diagnostics.get("upscaler")
    diagnostic_upscaler = diagnostic_upscaler if isinstance(diagnostic_upscaler, dict) else {}
    diagnostic_vae = diagnostics.get("vae")
    diagnostic_vae = diagnostic_vae if isinstance(diagnostic_vae, dict) else {}
    return {
        "strategy": str(diagnostics.get("strategy") or plan.get("strategy") or ""),
        "upscaler_id": str(
            upscale.get("upscaler_id")
            or plan.get("upscaler_id")
            or descriptor.get("upscaler_id")
            or diagnostic_upscaler.get("id")
            or ""
        ),
        "upscaler_sha256": str(
            upscale.get("upscaler_sha256")
            or descriptor.get("sha256")
            or diagnostic_upscaler.get("sha256")
            or ""
        ).casefold(),
        "vae_sha256": str(
            vae.get("sha256") or diagnostic_vae.get("sha256") or ""
        ).casefold(),
        "vae_source_kind": str(
            vae.get("source_kind") or diagnostic_vae.get("source_kind") or ""
        ),
    }


def _restore_recorded_hires_base_dimensions(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore first-pass dimensions for legacy hires manifests.

    Compact replay manifests record base dimensions directly in
    ``required_for_rerun.width`` / ``height``.  Older hires manifests instead
    stored the final/internal output size there while keeping the original base
    dimensions in ``hires_dimension_plan``.  Keep this compatibility repair so
    those older manifests still replay without collapsing or mis-scaling the
    hires target.
    """

    if not bool(payload.get("hires_enabled", False)):
        return payload
    plan = payload.get("hires_dimension_plan")
    if not isinstance(plan, dict):
        return payload
    try:
        base_width = int(plan.get("base_width") or 0)
        base_height = int(plan.get("base_height") or 0)
    except (TypeError, ValueError):
        return payload
    if base_width > 0 and base_height > 0:
        payload["width"] = base_width
        payload["height"] = base_height
    return payload


def _recorded_hires_schedule(manifest: GenerationManifest) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = manifest.to_dict()
    optional = payload.get("optional_for_rerun") if isinstance(payload, dict) else {}
    optional_extra = (optional or {}).get("extra") if isinstance(optional, dict) else {}
    if isinstance(optional_extra, dict):
        replay = optional_extra.get("hires_recorded_schedule_replay")
        fingerprint = optional_extra.get("hires_recorded_schedule_fingerprint")
        if isinstance(replay, dict) and replay and isinstance(fingerprint, dict) and fingerprint:
            return dict(replay), dict(fingerprint)
    extra = payload.get("extra") if isinstance(payload, dict) else {}
    pipeline_metadata = (extra or {}).get("pipeline_metadata") if isinstance(extra, dict) else {}
    hires = (pipeline_metadata or {}).get("hires_fix") if isinstance(pipeline_metadata, dict) else {}
    if not isinstance(hires, dict):
        return {}, {}
    replay = hires.get("schedule_replay")
    fingerprint = hires.get("schedule_fingerprint")
    return (
        dict(replay) if isinstance(replay, dict) else {},
        dict(fingerprint) if isinstance(fingerprint, dict) else {},
    )




def _recorded_prompt_cfg_schedules(manifest: GenerationManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    optional = payload.get("optional_for_rerun") if isinstance(payload, dict) else {}
    extra = (optional or {}).get("extra") if isinstance(optional, dict) else {}
    schedules = (extra or {}).get("prompt_cfg_pass_schedules") if isinstance(extra, dict) else {}
    return dict(schedules) if isinstance(schedules, dict) else {}



def _recorded_prompt_expansions(manifest: GenerationManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    optional = payload.get("optional_for_rerun") if isinstance(payload, dict) else {}
    extra = (optional or {}).get("extra") if isinstance(optional, dict) else {}
    records = (extra or {}).get("prompt_expansion_pass_records") if isinstance(extra, dict) else {}
    return dict(records) if isinstance(records, dict) else {}

def _recorded_regions(manifest: GenerationManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    optional = payload.get("optional_for_rerun") if isinstance(payload, dict) else {}
    extra = (optional or {}).get("extra") if isinstance(optional, dict) else {}
    records = (extra or {}).get("region_pass_records") if isinstance(extra, dict) else {}
    return dict(records) if isinstance(records, dict) else {}

def _recorded_prompt_semantics(manifest: GenerationManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    optional = payload.get("optional_for_rerun") if isinstance(payload, dict) else {}
    extra = (optional or {}).get("extra") if isinstance(optional, dict) else {}
    records = (extra or {}).get("prompt_semantic_pass_records") if isinstance(extra, dict) else {}
    return dict(records) if isinstance(records, dict) else {}

def _recorded_prompt_assets(manifest: GenerationManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    optional = payload.get("optional_for_rerun") if isinstance(payload, dict) else {}
    extra = (optional or {}).get("extra") if isinstance(optional, dict) else {}
    contract = (extra or {}).get("prompt_assets") if isinstance(extra, dict) else {}
    if not isinstance(contract, dict):
        contract = {}

    loras = contract.get("loras")
    textual = contract.get("textual_inversions")
    if not isinstance(loras, list):
        loras = (extra or {}).get("loras") if isinstance(extra, dict) else []
    if not isinstance(textual, list):
        textual = (extra or {}).get("textual_inversions") if isinstance(extra, dict) else []

    def _from_asset_refs(values: Any, asset_type: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if not isinstance(values, list):
            return output
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                continue
            metadata = dict(value.get("extra") or {})
            output.append({
                "asset_type": asset_type,
                "asset_id": metadata.get("asset_id") or value.get("requested_identifier") or value.get("resolved_identifier") or "",
                "catalog_asset_id": metadata.get("catalog_asset_id") or value.get("requested_identifier") or value.get("resolved_identifier") or "",
                "name": value.get("requested_display_name") or value.get("resolved_display_name") or "",
                "path": value.get("resolved_path") or value.get("requested_path") or "",
                "requested_path": value.get("requested_path") or value.get("resolved_path") or "",
                "resolved_path": value.get("resolved_path") or "",
                "requested_hash": value.get("requested_hash") or "",
                "resolved_hash": value.get("resolved_hash") or "",
                "weight": metadata.get("weight", 1.0),
                "enabled": metadata.get("enabled", True),
                "polarity": metadata.get("polarity", "positive"),
                "activation_text": metadata.get("activation_text") or "",
                "model_family": metadata.get("model_family") or "",
                "source_url": value.get("source_url") or "",
                "source": "replay",
                "original_source": metadata.get("source") or metadata.get("original_source") or "",
                "order": metadata.get("order", index),
                "metadata": metadata.get("metadata") or {},
            })
        return output

    if not isinstance(loras, list) or not loras:
        loras = _from_asset_refs(payload.get("loras"), "lora")
    if not isinstance(textual, list) or not textual:
        textual = _from_asset_refs(payload.get("embeddings"), "textual_inversion")

    def _mark_replay(values: Any, asset_type: str) -> list[dict[str, Any]]:
        normalized = normalize_prompt_asset_list(values or [], asset_type=asset_type, default_source="replay")
        output: list[dict[str, Any]] = []
        for asset in normalized:
            original = asset.original_source or asset.source
            asset.original_source = original if original != "replay" else asset.original_source
            asset.source = "replay"
            output.append(asset.to_serializable_dict())
        return output

    return {
        "contract_version": str(contract.get("contract_version") or (extra or {}).get("prompt_asset_contract_version") or PROMPT_ASSET_CONTRACT_VERSION),
        "loras": _mark_replay(loras, "lora"),
        "textual_inversions": _mark_replay(textual, "textual_inversion"),
    }


def manifest_to_request_kwargs(
    manifest: GenerationManifest,
    include_optional_for_rerun: bool = True,
    ui_model_path: str | None = None,
) -> dict[str, Any]:
    payload = manifest.to_rerun_payload(
        include_optional_for_rerun=include_optional_for_rerun
    )
    _restore_recorded_hires_base_dimensions(payload)
    payload.setdefault("hires_step_policy", "proportional_tail_v1")
    prompt_cfg_schedules = _recorded_prompt_cfg_schedules(manifest)
    if prompt_cfg_schedules:
        payload["prompt_cfg_recorded_schedules"] = prompt_cfg_schedules
        payload["prompt_cfg_replay_mode"] = "recorded_exact"
    prompt_expansions = _recorded_prompt_expansions(manifest)
    if prompt_expansions:
        payload["prompt_expansion_recorded"] = prompt_expansions
        payload["prompt_expansion_replay_mode"] = "recorded_exact"
    prompt_semantics = _recorded_prompt_semantics(manifest)
    if prompt_semantics:
        payload["prompt_semantic_recorded"] = prompt_semantics
        payload["prompt_semantic_replay_mode"] = "recorded_exact"
    regions = _recorded_regions(manifest)
    if regions:
        payload["region_recorded"] = regions
        payload["region_replay_mode"] = "recorded_exact"
    replay, fingerprint = _recorded_hires_schedule(manifest)
    if replay and fingerprint:
        payload["hires_recorded_schedule_replay"] = replay
        payload["hires_recorded_schedule_fingerprint"] = fingerprint
        payload["hires_schedule_conformance_source_replay"] = replay
        payload["hires_schedule_conformance_source_fingerprint"] = fingerprint
        payload["hires_schedule_replay_mode"] = "recorded_exact"
    hires_identity = _recorded_hires_runtime_identity(manifest)
    if hires_identity.get("strategy") == "pixel_neural":
        payload["hires_expected_upscaler_sha256"] = hires_identity.get(
            "upscaler_sha256", ""
        )
        payload["hires_expected_vae_sha256"] = hires_identity.get("vae_sha256", "")
        payload["hires_expected_vae_source_kind"] = hires_identity.get(
            "vae_source_kind", ""
        )

    prompt_assets = _recorded_prompt_assets(manifest)
    payload["prompt_asset_contract_version"] = prompt_assets["contract_version"]
    payload["loras"] = prompt_assets["loras"]
    payload["textual_inversions"] = prompt_assets["textual_inversions"]
    payload["lora_paths"] = [
        item.get("resolved_path") or item.get("path") or item.get("requested_path") or ""
        for item in prompt_assets["loras"]
        if item.get("resolved_path") or item.get("path") or item.get("requested_path")
    ]

    # Local fallback policy for base model:
    # prefer manifest requested model path if present,
    # otherwise fall back to UI-selected model.
    model_path = payload.get("model_path") or ""
    if not model_path and ui_model_path:
        payload["model_path"] = ui_model_path

    return payload


def apply_manifest_to_existing_request(
    request: Any,
    manifest: GenerationManifest,
    include_optional_for_rerun: bool = True,
) -> Any:
    payload = manifest.to_rerun_payload(
        include_optional_for_rerun=include_optional_for_rerun
    )
    _restore_recorded_hires_base_dimensions(payload)
    payload.setdefault("hires_step_policy", "proportional_tail_v1")
    prompt_cfg_schedules = _recorded_prompt_cfg_schedules(manifest)
    if prompt_cfg_schedules:
        payload["prompt_cfg_recorded_schedules"] = prompt_cfg_schedules
        payload["prompt_cfg_replay_mode"] = "recorded_exact"
    prompt_expansions = _recorded_prompt_expansions(manifest)
    if prompt_expansions:
        payload["prompt_expansion_recorded"] = prompt_expansions
        payload["prompt_expansion_replay_mode"] = "recorded_exact"
    prompt_semantics = _recorded_prompt_semantics(manifest)
    if prompt_semantics:
        payload["prompt_semantic_recorded"] = prompt_semantics
        payload["prompt_semantic_replay_mode"] = "recorded_exact"
    regions = _recorded_regions(manifest)
    if regions:
        payload["region_recorded"] = regions
        payload["region_replay_mode"] = "recorded_exact"
    replay, fingerprint = _recorded_hires_schedule(manifest)
    if replay and fingerprint:
        payload["hires_recorded_schedule_replay"] = replay
        payload["hires_recorded_schedule_fingerprint"] = fingerprint
        payload["hires_schedule_conformance_source_replay"] = replay
        payload["hires_schedule_conformance_source_fingerprint"] = fingerprint
        payload["hires_schedule_replay_mode"] = "recorded_exact"
    hires_identity = _recorded_hires_runtime_identity(manifest)
    if hires_identity.get("strategy") == "pixel_neural":
        payload["hires_expected_upscaler_sha256"] = hires_identity.get(
            "upscaler_sha256", ""
        )
        payload["hires_expected_vae_sha256"] = hires_identity.get("vae_sha256", "")
        payload["hires_expected_vae_source_kind"] = hires_identity.get(
            "vae_source_kind", ""
        )

    prompt_assets = _recorded_prompt_assets(manifest)
    payload["prompt_asset_contract_version"] = prompt_assets["contract_version"]
    payload["loras"] = prompt_assets["loras"]
    payload["textual_inversions"] = prompt_assets["textual_inversions"]

    for key, value in payload.items():
        if not hasattr(request, key):
            continue
        if key == "loras":
            value = normalize_prompt_asset_list(value, asset_type="lora", default_source="replay")
        elif key == "textual_inversions":
            value = normalize_prompt_asset_list(value, asset_type="textual_inversion", default_source="replay")
        setattr(request, key, value)

    return request