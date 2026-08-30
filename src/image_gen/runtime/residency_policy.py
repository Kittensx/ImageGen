from __future__ import annotations

from typing import Any, Mapping

MODEL_RESIDENCY_MODE_MANAGED = "managed"
MODEL_RESIDENCY_MODE_HOT = "hot"
MODEL_RESIDENCY_MODES = frozenset({MODEL_RESIDENCY_MODE_MANAGED, MODEL_RESIDENCY_MODE_HOT})

RESIDENCY_STATE_EMPTY = "empty"
RESIDENCY_STATE_MANAGED = "managed_resident"
RESIDENCY_STATE_HOT_GPU = "hot_gpu"
RESIDENCY_STATE_HOT_STAGED = "hot_staged"
RESIDENCY_STATE_SWITCHING = "switching"
RESIDENCY_STATE_RECOVERING = "recovering"
RESIDENCY_STATES = frozenset(
    {
        RESIDENCY_STATE_EMPTY,
        RESIDENCY_STATE_MANAGED,
        RESIDENCY_STATE_HOT_GPU,
        RESIDENCY_STATE_HOT_STAGED,
        RESIDENCY_STATE_SWITCHING,
        RESIDENCY_STATE_RECOVERING,
    }
)

GENERATION_RESIDENCY_COLD_LOAD = "cold_load"
GENERATION_RESIDENCY_MODEL_SWITCH = "model_switch"
GENERATION_RESIDENCY_MANAGED_REUSE = "resident_managed_reuse"
GENERATION_RESIDENCY_HOT_REUSE = "hot_reuse"
GENERATION_RESIDENCY_HOT_STAGED_REUSE = "hot_staged_reuse"
POST_JOB_RESIDENCY_MANAGED_RETENTION = "managed_retention"
POST_JOB_RESIDENCY_HOT_HOLD = "hot_hold"
POST_JOB_RESIDENCY_HOT_RESTORE = "hot_restore"
POST_JOB_RESIDENCY_HOT_STAGED_HOLD = "hot_staged_hold"
POST_JOB_RESIDENCY_FORCED_RELEASE = "forced_release"
POST_JOB_RESIDENCY_ACTIONS = frozenset(
    {
        POST_JOB_RESIDENCY_MANAGED_RETENTION,
        POST_JOB_RESIDENCY_HOT_HOLD,
        POST_JOB_RESIDENCY_HOT_RESTORE,
        POST_JOB_RESIDENCY_HOT_STAGED_HOLD,
        POST_JOB_RESIDENCY_FORCED_RELEASE,
    }
)


GENERATION_RESIDENCY_CLASSIFICATIONS = frozenset(
    {
        GENERATION_RESIDENCY_COLD_LOAD,
        GENERATION_RESIDENCY_MODEL_SWITCH,
        GENERATION_RESIDENCY_MANAGED_REUSE,
        GENERATION_RESIDENCY_HOT_REUSE,
        GENERATION_RESIDENCY_HOT_STAGED_REUSE,
    }
)


def normalize_model_residency_mode(value: Any) -> str:
    """Return the persisted HMR policy value.

    HMR-01 deliberately fails legacy/unknown values closed to Managed so an
    existing installation is never opted into higher sustained VRAM use merely
    because a stale or malformed setting is present.
    """

    token = str(value or "").strip().lower().replace("-", "_")
    return MODEL_RESIDENCY_MODE_HOT if token == MODEL_RESIDENCY_MODE_HOT else MODEL_RESIDENCY_MODE_MANAGED


def resolve_effective_residency_state(
    *,
    requested_mode: Any,
    stage: Any,
    resident: bool,
    gpu_loaded: bool = False,
    staged_runtime: bool = False,
    hot_residency_active: bool = False,
    hot_gpu_ready: bool | None = None,
) -> str:
    """Resolve normalized runtime residency from observed lifecycle state.

    ``hot_residency_active`` must be true only after the resident runtime has
    suppressed Managed post-job retention and established the architecture-
    appropriate reusable working set. Merely requesting Hot is never enough.
    """

    normalized_mode = normalize_model_residency_mode(requested_mode)
    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage == RESIDENCY_STATE_RECOVERING:
        return RESIDENCY_STATE_RECOVERING
    if normalized_stage in {RESIDENCY_STATE_SWITCHING, "unloading", "model_switch"}:
        return RESIDENCY_STATE_SWITCHING
    if not resident:
        return RESIDENCY_STATE_EMPTY
    if hot_residency_active and normalized_mode == MODEL_RESIDENCY_MODE_HOT:
        gpu_ready = bool(gpu_loaded) if hot_gpu_ready is None else bool(hot_gpu_ready)
        if staged_runtime or not gpu_ready:
            return RESIDENCY_STATE_HOT_STAGED
        return RESIDENCY_STATE_HOT_GPU
    return RESIDENCY_STATE_MANAGED


def build_residency_diagnostics(
    *,
    requested_mode: Any,
    effective_state: Any,
    resident_status: Mapping[str, Any] | None = None,
    last_residency_transition: Any = None,
    last_residency_reason: Any = None,
    retention_suppressed_for_hot: bool = False,
    hot_reuse_count: int = 0,
    cold_or_switch_load_count: int = 0,
    last_generation_residency_classification: Any = None,
    hot_since: Any = None,
) -> dict[str, Any]:
    """Build the shared status/diagnostic contract introduced by HMR-01."""

    resident = dict(resident_status or {})
    mode = normalize_model_residency_mode(requested_mode)
    state = str(effective_state or RESIDENCY_STATE_EMPTY).strip().lower()
    if state not in RESIDENCY_STATES:
        state = RESIDENCY_STATE_EMPTY
    hot_effective = state in {RESIDENCY_STATE_HOT_GPU, RESIDENCY_STATE_HOT_STAGED}
    classification = str(last_generation_residency_classification or "").strip().lower() or None
    if classification not in GENERATION_RESIDENCY_CLASSIFICATIONS:
        classification = None
    return {
        "residency_mode_requested": mode,
        "residency_state_effective": state,
        "hot_model_path": resident.get("model_path") if hot_effective else None,
        "hot_model_identity": resident.get("model_identity") if hot_effective else None,
        "hot_composition_sha256": resident.get("composition_sha256") if hot_effective else None,
        "hot_load_variant_fingerprint": (
            resident.get("runtime_effective_load_variant_fingerprint")
            or resident.get("runtime_load_variant_fingerprint")
        ) if hot_effective else None,
        "hot_since": hot_since if hot_effective else None,
        "last_residency_transition": last_residency_transition,
        "last_residency_reason": str(last_residency_reason or "") or None,
        "retention_suppressed_for_hot": bool(retention_suppressed_for_hot and hot_effective),
        "hot_reuse_count": max(0, int(hot_reuse_count or 0)),
        "cold_or_switch_load_count": max(0, int(cold_or_switch_load_count or 0)),
        "last_generation_residency_classification": classification,
    }


__all__ = [
    "MODEL_RESIDENCY_MODE_MANAGED",
    "MODEL_RESIDENCY_MODE_HOT",
    "MODEL_RESIDENCY_MODES",
    "RESIDENCY_STATE_EMPTY",
    "RESIDENCY_STATE_MANAGED",
    "RESIDENCY_STATE_HOT_GPU",
    "RESIDENCY_STATE_HOT_STAGED",
    "RESIDENCY_STATE_SWITCHING",
    "RESIDENCY_STATE_RECOVERING",
    "RESIDENCY_STATES",
    "GENERATION_RESIDENCY_COLD_LOAD",
    "GENERATION_RESIDENCY_MODEL_SWITCH",
    "GENERATION_RESIDENCY_MANAGED_REUSE",
    "GENERATION_RESIDENCY_HOT_REUSE",
    "GENERATION_RESIDENCY_HOT_STAGED_REUSE",
    "GENERATION_RESIDENCY_CLASSIFICATIONS",
    "POST_JOB_RESIDENCY_MANAGED_RETENTION",
    "POST_JOB_RESIDENCY_HOT_HOLD",
    "POST_JOB_RESIDENCY_HOT_RESTORE",
    "POST_JOB_RESIDENCY_HOT_STAGED_HOLD",
    "POST_JOB_RESIDENCY_FORCED_RELEASE",
    "POST_JOB_RESIDENCY_ACTIONS",
    "normalize_model_residency_mode",
    "resolve_effective_residency_state",
    "build_residency_diagnostics",
]
