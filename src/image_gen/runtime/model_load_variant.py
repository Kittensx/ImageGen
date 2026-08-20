from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, MutableMapping


MODEL_LOAD_VARIANT_FIELDS = (
    "sd2_dedicated_generation",
    "sd2_runtime_profile_override",
    "sdxl_runtime_profile_override",
    "sd3_runtime_profile_override",
    "sd3_text_encoder_source",
    "sd3_clip_l_source",
    "sd3_clip_g_source",
    "sd3_t5_enabled",
    "sd3_t5_source",
    "advanced_model_composition_sha256",
    "vae_path",
)

_AUTO_SOURCE_FIELDS = {
    "sd3_text_encoder_source",
    "sd3_clip_l_source",
    "sd3_clip_g_source",
    "sd3_t5_source",
}


_ADVANCED_INTERNAL_FIELDS = (
    "_advanced_model_resolved",
    "advanced_model_composition_sha256",
)


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "on", "enabled", "enable"}


def _normalize_source(value: Any) -> str:
    token = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "shared": "external",
        "standalone": "external",
        "separate": "external",
        "checkpoint": "embedded",
    }
    return aliases.get(token, token)


def sanitize_model_load_runtime_settings(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return load settings with disabled Advanced Models made inert.

    Advanced-model selections are intentionally sticky in the browser so users can
    toggle the feature back on without rebuilding their composition. Those sticky
    preferences must never leak into the resident runtime after the toggle is off.
    In particular, a stale resolved component composition from a previous SD1.x job
    must not participate in a later whole-checkpoint SDXL/SD2/SD3 activation.
    """

    payload = dict(values or {})
    if not _normalize_bool(payload.get("advanced_models_enabled")):
        payload.pop("_advanced_model_resolved", None)
        payload["advanced_model_composition_sha256"] = ""
    return payload


def update_model_load_runtime_settings(
    target: MutableMapping[str, Any],
    values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge runtime load settings while clearing stale Advanced Models identity.

    ``dict.update`` alone is unsafe for the long-lived model worker because an
    omitted internal key otherwise survives from the previous job. This helper
    performs the negative transition (Advanced Models on -> off) explicitly.
    """

    incoming = sanitize_model_load_runtime_settings(values)
    if not _normalize_bool(incoming.get("advanced_models_enabled")):
        for key in _ADVANCED_INTERNAL_FIELDS:
            target.pop(key, None)
    target.update(incoming)
    return dict(target)


def model_load_variant_payload(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the canonical identity of composition-affecting load choices.

    The fingerprint intentionally ignores ordinary generation controls and
    normalizes semantically equivalent source defaults. This keeps resident
    reuse stable while still forcing a reload when component composition can
    actually change (for example T5 off -> on or embedded -> shared).
    """

    source = sanitize_model_load_runtime_settings(values)
    payload: dict[str, Any] = {}
    for key in MODEL_LOAD_VARIANT_FIELDS:
        if key == "sd3_t5_enabled":
            payload[key] = _normalize_bool(source[key]) if key in source else "auto"
        elif key == "sd2_dedicated_generation":
            payload[key] = _normalize_bool(source.get(key))
        elif key in _AUTO_SOURCE_FIELDS:
            payload[key] = _normalize_source(source.get(key))
        else:
            payload[key] = str(source.get(key) or "").strip()
    return payload


def model_load_variant_fingerprint(values: Mapping[str, Any] | None) -> str:
    payload = model_load_variant_payload(values)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_load_variant_matches_resident(
    values: Mapping[str, Any] | None,
    resident_status: Mapping[str, Any] | None,
) -> bool:
    """Compare a request against resident composition identity.

    New runtimes always publish ``runtime_load_variant_fingerprint``. For an
    older/fake resident status that predates the fingerprint, permit reuse only
    for the canonical default load variant. Any explicit composition-affecting
    choice fails closed to a reload.
    """

    resident = dict(resident_status or {})
    current = str(resident.get("runtime_load_variant_fingerprint") or "").strip()
    if current:
        return model_load_variant_fingerprint(values) == current
    return model_load_variant_payload(values) == model_load_variant_payload(None)


__all__ = [
    "MODEL_LOAD_VARIANT_FIELDS",
    "model_load_variant_payload",
    "model_load_variant_fingerprint",
    "model_load_variant_matches_resident",
    "sanitize_model_load_runtime_settings",
    "update_model_load_runtime_settings",
]
