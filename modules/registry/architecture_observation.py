from __future__ import annotations

from typing import Any


ARCHITECTURE_STATE_SUPPORTED = "supported"
ARCHITECTURE_STATE_RECOGNIZED_UNSUPPORTED = "recognized_unsupported"
ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED = "observed_unclassified"
ARCHITECTURE_STATE_NON_COMPONENT_ASSET = "non_component_asset"
ARCHITECTURE_STATE_INVALID = "invalid"

ARCHITECTURE_STATES = {
    ARCHITECTURE_STATE_SUPPORTED,
    ARCHITECTURE_STATE_RECOGNIZED_UNSUPPORTED,
    ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED,
    ARCHITECTURE_STATE_NON_COMPONENT_ASSET,
    ARCHITECTURE_STATE_INVALID,
}

_PLACEHOLDER_ARCHITECTURES = {
    "",
    "unknown",
    "unknown_model",
    "unknown-model",
    "unclassified",
    "unidentified",
    "none",
    "null",
    "n/a",
}

_PLACEHOLDER_ASSET_TYPES = {
    "",
    "unknown",
    "unknown_model",
    "unknown-model",
    "unclassified",
    "unidentified",
}


def normalize_architecture_identifier(value: Any) -> str:
    """Return a real architecture identifier or an empty string.

    Ambiguous detector outputs such as ``sd1.x_or_sd2.x`` are evidence about the
    observation, not architecture identities, so they are deliberately not stored
    in the canonical architecture column.
    """

    token = str(value or "").strip().lower()
    if token in _PLACEHOLDER_ARCHITECTURES or "_or_" in token:
        return ""
    return token


def is_placeholder_architecture(value: Any) -> bool:
    return not bool(normalize_architecture_identifier(value))


def normalize_asset_type(value: Any, *, format_type: Any = "") -> str:
    token = str(value or "").strip().lower()
    if token not in _PLACEHOLDER_ASSET_TYPES:
        return token
    if str(format_type or "").strip().lower() == "safetensors":
        return "safetensors_asset"
    return "unclassified_asset"


def normalize_architecture_state(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token if token in ARCHITECTURE_STATES else ""


def derive_architecture_state(
    *,
    architecture: Any,
    provider_supported: bool,
    format_type: Any = "",
    explicit_state: Any = "",
) -> str:
    explicit = normalize_architecture_state(explicit_state)
    if explicit:
        return explicit
    canonical = normalize_architecture_identifier(architecture)
    if provider_supported and canonical:
        return ARCHITECTURE_STATE_SUPPORTED
    if canonical:
        return ARCHITECTURE_STATE_RECOGNIZED_UNSUPPORTED
    if str(format_type or "").strip().lower() == "safetensors":
        return ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED
    return ARCHITECTURE_STATE_NON_COMPONENT_ASSET


__all__ = [
    "ARCHITECTURE_STATE_SUPPORTED",
    "ARCHITECTURE_STATE_RECOGNIZED_UNSUPPORTED",
    "ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED",
    "ARCHITECTURE_STATE_NON_COMPONENT_ASSET",
    "ARCHITECTURE_STATE_INVALID",
    "ARCHITECTURE_STATES",
    "normalize_architecture_identifier",
    "is_placeholder_architecture",
    "normalize_asset_type",
    "normalize_architecture_state",
    "derive_architecture_state",
]
