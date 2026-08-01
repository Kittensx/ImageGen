from __future__ import annotations

from typing import Any

from modules.txt2img.generation_manifest import GenerationManifest


def _recorded_hires_schedule(manifest: GenerationManifest) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = manifest.to_dict()
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


def manifest_to_request_kwargs(
    manifest: GenerationManifest,
    include_optional_for_rerun: bool = True,
    ui_model_path: str | None = None,
) -> dict[str, Any]:
    payload = manifest.to_rerun_payload(
        include_optional_for_rerun=include_optional_for_rerun
    )
    payload.setdefault("hires_step_policy", "proportional_tail_v1")
    replay, fingerprint = _recorded_hires_schedule(manifest)
    if replay and fingerprint:
        payload["hires_recorded_schedule_replay"] = replay
        payload["hires_recorded_schedule_fingerprint"] = fingerprint
        payload["hires_schedule_conformance_source_replay"] = replay
        payload["hires_schedule_conformance_source_fingerprint"] = fingerprint
        payload["hires_schedule_replay_mode"] = "recorded_exact"

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
    payload.setdefault("hires_step_policy", "proportional_tail_v1")
    replay, fingerprint = _recorded_hires_schedule(manifest)
    if replay and fingerprint:
        payload["hires_recorded_schedule_replay"] = replay
        payload["hires_recorded_schedule_fingerprint"] = fingerprint
        payload["hires_schedule_conformance_source_replay"] = replay
        payload["hires_schedule_conformance_source_fingerprint"] = fingerprint
        payload["hires_schedule_replay_mode"] = "recorded_exact"

    for key, value in payload.items():
        if hasattr(request, key):
            setattr(request, key, value)

    return request