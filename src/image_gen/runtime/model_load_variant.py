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

_AUTO_PROFILE_FIELDS = (
    "sd2_runtime_profile_override",
    "sdxl_runtime_profile_override",
    "sd3_runtime_profile_override",
)

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


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    """Return the normalized *requested* identity of load-affecting choices.

    This payload deliberately preserves automatic/default instructions such as a
    blank runtime profile or ``source=auto``. It remains the compatibility/raw
    identity used by existing loader cache contracts. Resident reuse should prefer
    :func:`model_load_variant_comparison` when the runtime publishes an effective
    identity, because automatic instructions can resolve to concrete runtime state.
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


def resolved_model_load_variant_payload(
    values: Mapping[str, Any] | None,
    *,
    profile_ids: Mapping[str, Any] | None = None,
    sd3_text_encoder_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the effective load identity from authoritative post-load evidence.

    ``profile_ids`` maps any profile-bearing load field to the profile actually
    selected by the model contract. ``sd3_text_encoder_sources`` is the evidence
    produced by ``prepare_sd3_text_encoder_states`` and converts source policies
    such as ``auto`` into the source kind that was actually instantiated.

    The function has no architecture-loader imports on purpose: callers pass the
    already-resolved evidence from the canonical load path, keeping one small
    identity contract shared by SD2, SDXL, SD3, and future profile-bearing families.
    """

    payload = model_load_variant_payload(values)
    for key, value in dict(profile_ids or {}).items():
        if key not in _AUTO_PROFILE_FIELDS:
            continue
        token = str(value or "").strip()
        if token:
            payload[key] = token

    evidence = dict(sd3_text_encoder_sources or {})
    roles = dict(evidence.get("roles") or {})
    if roles:
        clip_l = _normalize_source(dict(roles.get("clip_l") or {}).get("source_kind"))
        clip_g = _normalize_source(dict(roles.get("clip_g") or {}).get("source_kind"))
        if clip_l != "auto":
            payload["sd3_clip_l_source"] = clip_l
        if clip_g != "auto":
            payload["sd3_clip_g_source"] = clip_g
        if clip_l != "auto" and clip_g != "auto":
            payload["sd3_text_encoder_source"] = clip_l if clip_l == clip_g else "mixed"

        if "t5_enabled" in evidence:
            payload["sd3_t5_enabled"] = bool(evidence.get("t5_enabled"))
        t5_source = _normalize_source(dict(roles.get("t5xxl") or {}).get("source_kind"))
        if payload.get("sd3_t5_enabled") is False:
            payload["sd3_t5_source"] = "disabled"
        elif t5_source != "auto":
            payload["sd3_t5_source"] = t5_source
    elif payload.get("sd3_t5_enabled") is False:
        # Source selection is inert when T5 is explicitly disabled. Treat all raw
        # source spellings as one effective disabled state.
        payload["sd3_t5_source"] = "disabled"

    return payload




def model_load_variant_payload_fingerprint(payload: Mapping[str, Any]) -> str:
    """Fingerprint an already-canonical load-variant payload without re-normalizing it."""

    return _fingerprint_payload(payload)


def model_load_variant_fingerprint(values: Mapping[str, Any] | None) -> str:
    return _fingerprint_payload(model_load_variant_payload(values))


def resolved_model_load_variant_fingerprint(
    values: Mapping[str, Any] | None,
    *,
    profile_ids: Mapping[str, Any] | None = None,
    sd3_text_encoder_sources: Mapping[str, Any] | None = None,
) -> str:
    return _fingerprint_payload(
        resolved_model_load_variant_payload(
            values,
            profile_ids=profile_ids,
            sd3_text_encoder_sources=sd3_text_encoder_sources,
        )
    )


def _canonicalize_request_against_effective_resident(
    values: Mapping[str, Any] | None,
    resident_effective: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve automatic request instructions against proven resident state.

    This function is only used after the caller has established the same resident
    checkpoint/composition candidate. Automatic values are therefore allowed to
    inherit the resident's concrete result; explicit values are never rewritten to
    match resident state.
    """

    source = sanitize_model_load_runtime_settings(values)
    raw = model_load_variant_payload(source)
    effective = dict(raw)
    resident = dict(resident_effective or {})
    differences: list[dict[str, Any]] = []

    def replace_if_changed(field: str, value: Any, reason: str) -> None:
        before = effective.get(field)
        if before == value:
            return
        effective[field] = value
        differences.append(
            {
                "field": field,
                "raw": before,
                "effective": value,
                "reason": reason,
            }
        )

    for field in _AUTO_PROFILE_FIELDS:
        if not str(source.get(field) or "").strip():
            resident_value = str(resident.get(field) or "").strip()
            if resident_value:
                replace_if_changed(field, resident_value, "automatic_profile_resolved_to_resident_effective")

    global_sd3_source = _normalize_source(source.get("sd3_text_encoder_source"))
    if global_sd3_source == "auto":
        resident_value = _normalize_source(resident.get("sd3_text_encoder_source"))
        if resident_value != "auto":
            replace_if_changed(
                "sd3_text_encoder_source",
                resident_value,
                "automatic_source_resolved_to_resident_effective",
            )

    for field in ("sd3_clip_l_source", "sd3_clip_g_source"):
        requested = _normalize_source(source.get(field))
        if requested != "auto":
            continue
        if global_sd3_source != "auto":
            replace_if_changed(field, global_sd3_source, "role_source_inherits_explicit_global_policy")
            continue
        resident_value = _normalize_source(resident.get(field))
        if resident_value != "auto":
            replace_if_changed(field, resident_value, "automatic_source_resolved_to_resident_effective")

    if "sd3_t5_enabled" not in source:
        resident_enabled = resident.get("sd3_t5_enabled")
        if isinstance(resident_enabled, bool):
            replace_if_changed(
                "sd3_t5_enabled",
                resident_enabled,
                "automatic_t5_enabled_resolved_to_resident_effective",
            )

    if effective.get("sd3_t5_enabled") is False:
        replace_if_changed("sd3_t5_source", "disabled", "t5_source_inert_while_disabled")
    elif _normalize_source(source.get("sd3_t5_source")) == "auto":
        resident_value = _normalize_source(resident.get("sd3_t5_source"))
        if resident_value != "auto":
            replace_if_changed(
                "sd3_t5_source",
                resident_value,
                "automatic_source_resolved_to_resident_effective",
            )

    return effective, differences


def model_load_variant_comparison(
    values: Mapping[str, Any] | None,
    resident_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare requested load state with the resident's canonical effective state.

    New CNRR-aware runtimes publish ``runtime_effective_load_variant``. When that
    is available, automatic/default request instructions are resolved against the
    concrete resident result before comparison. Older/fake runtimes retain the
    previous fingerprint behavior so rollback and focused legacy tests remain safe.
    """

    resident_status_payload = dict(resident_status or {})
    raw_requested = model_load_variant_payload(values)
    raw_requested_fingerprint = _fingerprint_payload(raw_requested)
    resident_raw = dict(resident_status_payload.get("runtime_load_variant") or {})
    resident_raw_fingerprint = str(
        resident_status_payload.get("runtime_load_variant_fingerprint") or ""
    ).strip()
    resident_effective = dict(
        resident_status_payload.get("runtime_effective_load_variant") or {}
    )
    resident_effective_fingerprint = str(
        resident_status_payload.get("runtime_effective_load_variant_fingerprint") or ""
    ).strip()

    normalization_differences: list[dict[str, Any]] = []
    if resident_effective:
        requested_effective, normalization_differences = _canonicalize_request_against_effective_resident(
            values,
            resident_effective,
        )
        requested_effective_fingerprint = _fingerprint_payload(requested_effective)
        expected_fingerprint = resident_effective_fingerprint or _fingerprint_payload(resident_effective)
        resident_for_diff = resident_effective
        matches = requested_effective == resident_effective
        comparison_mode = "effective"
    else:
        requested_effective = dict(raw_requested)
        requested_effective_fingerprint = raw_requested_fingerprint
        expected_fingerprint = resident_raw_fingerprint
        resident_for_diff = resident_raw
        matches = bool(expected_fingerprint and requested_effective_fingerprint == expected_fingerprint)
        comparison_mode = "legacy_raw"

    mismatch_fields: list[dict[str, Any]] = []
    if resident_for_diff:
        for key in MODEL_LOAD_VARIANT_FIELDS:
            requested = requested_effective.get(key)
            resident = resident_for_diff.get(key)
            if requested != resident:
                mismatch_fields.append(
                    {
                        "field": key,
                        "requested": requested,
                        "resident": resident,
                    }
                )
    elif requested_effective_fingerprint != expected_fingerprint:
        mismatch_fields.append(
            {
                "field": "runtime_load_variant_fingerprint",
                "requested": requested_effective_fingerprint,
                "resident": expected_fingerprint,
            }
        )

    return {
        "matches": bool(matches),
        "comparison_mode": comparison_mode,
        "requested_fingerprint": requested_effective_fingerprint,
        "resident_fingerprint": expected_fingerprint,
        "requested": requested_effective,
        "resident": resident_for_diff,
        "raw_requested": raw_requested,
        "raw_requested_fingerprint": raw_requested_fingerprint,
        "resident_raw": resident_raw,
        "resident_raw_fingerprint": resident_raw_fingerprint,
        "normalization_differences": normalization_differences,
        "mismatch_fields": mismatch_fields,
    }


def model_load_variant_matches_resident(
    values: Mapping[str, Any] | None,
    resident_status: Mapping[str, Any] | None,
) -> bool:
    """Return whether the request's load contract is resident-compatible.

    A truly legacy resident without either raw or effective fingerprints may only
    reuse the canonical default raw variant. This preserves the previous fail-closed
    behavior for explicit composition-affecting choices.
    """

    resident = dict(resident_status or {})
    if resident.get("runtime_effective_load_variant") or resident.get("runtime_load_variant_fingerprint"):
        return bool(model_load_variant_comparison(values, resident).get("matches"))
    return model_load_variant_payload(values) == model_load_variant_payload(None)


__all__ = [
    "MODEL_LOAD_VARIANT_FIELDS",
    "model_load_variant_payload",
    "model_load_variant_payload_fingerprint",
    "model_load_variant_fingerprint",
    "resolved_model_load_variant_payload",
    "resolved_model_load_variant_fingerprint",
    "model_load_variant_comparison",
    "model_load_variant_matches_resident",
    "sanitize_model_load_runtime_settings",
    "update_model_load_runtime_settings",
]
