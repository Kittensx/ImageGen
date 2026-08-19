from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


HIRES_PROFILE_SCHEMA_VERSION = "image-gen-hires-profile-v1"
HIRES_DEFAULT_ASSIGNMENTS_SCHEMA_VERSION = "image-gen-hires-default-assignments-v1"
HIRES_SETTING_DESCRIPTOR_VERSION = "image-gen-hires-setting-descriptor-v1"
HIRES_PROFILE_SAVE_MANIFEST_VERSION = "image-gen-hires-profile-save-manifest-v1"

SUPPORTED_HIRES_DEFAULT_SCOPES = frozenset(
    {
        "global",
        "model_family",
        "checkpoint",
        "upscaler",
        "model_family_upscaler",
        "checkpoint_upscaler",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)


def _normalized_sha256(value: Any, *, field_name: str) -> str:
    digest = str(value or "").strip().casefold()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field_name} must be a full 64-character SHA-256 digest.")
    return digest


@dataclass(frozen=True)
class HiresProfile:
    profile_id: str
    name: str
    values: dict[str, Any]
    included_fields: tuple[str, ...]
    description: str = ""
    source: str = "user"
    read_only: bool = False
    schema_version: str = HIRES_PROFILE_SCHEMA_VERSION
    compatibility: dict[str, Any] = field(default_factory=dict)
    baseline_profile_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not str(self.profile_id or "").strip():
            raise ValueError("Hires profile_id is required.")
        if not str(self.name or "").strip():
            raise ValueError("Hires profile name is required.")
        if str(self.schema_version) != HIRES_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported hires profile schema: {self.schema_version!r}")
        value_keys = tuple(sorted(str(key) for key in self.values))
        included = tuple(sorted({str(key) for key in self.included_fields}))
        if value_keys != included:
            raise ValueError("Hires profile included_fields must exactly match values keys.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "read_only": bool(self.read_only),
            "included_fields": list(self.included_fields),
            "values": _json_safe(self.values),
            "compatibility": _json_safe(self.compatibility),
            "baseline_profile_id": self.baseline_profile_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HiresProfile":
        values = dict(value.get("values") or {})
        included = value.get("included_fields")
        if not isinstance(included, (list, tuple)):
            included = list(values)
        return cls(
            profile_id=str(value.get("profile_id") or "").strip(),
            name=str(value.get("name") or "").strip(),
            description=str(value.get("description") or ""),
            source=str(value.get("source") or "user"),
            read_only=bool(value.get("read_only", False)),
            schema_version=str(value.get("schema_version") or HIRES_PROFILE_SCHEMA_VERSION),
            included_fields=tuple(str(key) for key in included),
            values=values,
            compatibility=dict(value.get("compatibility") or {}),
            baseline_profile_id=str(value.get("baseline_profile_id") or ""),
            created_at=str(value.get("created_at") or ""),
            updated_at=str(value.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class HiresDefaultAssignment:
    scope: str
    profile_id: str
    model_family: str = ""
    checkpoint_sha256: str = ""
    upscaler_sha256: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        scope = str(self.scope or "").strip().casefold()
        if scope not in SUPPORTED_HIRES_DEFAULT_SCOPES:
            raise ValueError(
                f"Unsupported hires default scope {self.scope!r}; expected one of "
                f"{sorted(SUPPORTED_HIRES_DEFAULT_SCOPES)}."
            )
        if not str(self.profile_id or "").strip():
            raise ValueError("Hires default assignment profile_id is required.")
        if scope in {"model_family", "model_family_upscaler"} and not str(self.model_family or "").strip():
            raise ValueError(f"{scope} assignments require model_family.")
        if scope in {"checkpoint", "checkpoint_upscaler"}:
            _normalized_sha256(self.checkpoint_sha256, field_name="checkpoint_sha256")
        if scope in {"upscaler", "model_family_upscaler", "checkpoint_upscaler"}:
            _normalized_sha256(self.upscaler_sha256, field_name="upscaler_sha256")

    @property
    def assignment_key(self) -> str:
        scope = str(self.scope).casefold()
        parts = [scope]
        if scope in {"model_family", "model_family_upscaler"}:
            parts.append(str(self.model_family).strip().casefold())
        if scope in {"checkpoint", "checkpoint_upscaler"}:
            parts.append(str(self.checkpoint_sha256).strip().casefold())
        if scope in {"upscaler", "model_family_upscaler", "checkpoint_upscaler"}:
            parts.append(str(self.upscaler_sha256).strip().casefold())
        return ":".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": str(self.scope).casefold(),
            "profile_id": self.profile_id,
            "model_family": str(self.model_family or "").strip(),
            "checkpoint_sha256": str(self.checkpoint_sha256 or "").strip().casefold(),
            "upscaler_sha256": str(self.upscaler_sha256 or "").strip().casefold(),
            "assignment_key": self.assignment_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HiresDefaultAssignment":
        return cls(
            scope=str(value.get("scope") or ""),
            profile_id=str(value.get("profile_id") or ""),
            model_family=str(value.get("model_family") or ""),
            checkpoint_sha256=str(value.get("checkpoint_sha256") or ""),
            upscaler_sha256=str(value.get("upscaler_sha256") or ""),
            created_at=str(value.get("created_at") or ""),
            updated_at=str(value.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class HiresSettingDescriptor:
    key: str
    label: str
    group: str
    description: str
    value_type: str
    current_value: Any
    baseline_value: Any
    allowed_values: tuple[dict[str, Any], ...] = ()
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    asset_kind: str = ""
    editor_kind: str = "read_only"
    available: bool = True
    included: bool = False
    modified: bool = False
    editable: bool = True
    source: str = "profile_schema"
    persistence_eligibility: str = "eligible"
    schema_version: str = HIRES_SETTING_DESCRIPTOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": self.key,
            "label": self.label,
            "group": self.group,
            "description": self.description,
            "value_type": self.value_type,
            "current_value": _json_safe(self.current_value),
            "baseline_value": _json_safe(self.baseline_value),
            "allowed_values": [_json_safe(item) for item in self.allowed_values],
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "asset_kind": self.asset_kind,
            "editor_kind": self.editor_kind,
            "available": bool(self.available),
            "included": bool(self.included),
            "modified": bool(self.modified),
            "editable": bool(self.editable),
            "source": self.source,
            "persistence_eligibility": self.persistence_eligibility,
        }


@dataclass(frozen=True)
class HiresProfileSaveManifest:
    profile_id: str
    profile_name: str
    baseline_profile_id: str
    descriptors: tuple[HiresSettingDescriptor, ...]
    included_fields: tuple[str, ...]
    excluded_fields: tuple[str, ...]
    modified_fields: tuple[str, ...]
    schema_excluded_fields: tuple[str, ...]
    unclassified_fields: tuple[str, ...]
    rejected_fields: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    schema_version: str = HIRES_PROFILE_SAVE_MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "baseline_profile_id": self.baseline_profile_id,
            "descriptors": [item.to_dict() for item in self.descriptors],
            "included_fields": list(self.included_fields),
            "excluded_fields": list(self.excluded_fields),
            "modified_fields": list(self.modified_fields),
            "schema_excluded_fields": list(self.schema_excluded_fields),
            "unclassified_fields": list(self.unclassified_fields),
            "rejected_fields": list(self.rejected_fields),
            "warnings": list(self.warnings),
        }


class HiresProfileValidationError(ValueError):
    """Validation error that retains the serializer-driven inspection manifest."""

    def __init__(self, message: str, *, manifest: HiresProfileSaveManifest | None = None) -> None:
        super().__init__(message)
        self.manifest = manifest


__all__ = [
    "HIRES_DEFAULT_ASSIGNMENTS_SCHEMA_VERSION",
    "HIRES_PROFILE_SAVE_MANIFEST_VERSION",
    "HIRES_PROFILE_SCHEMA_VERSION",
    "HIRES_SETTING_DESCRIPTOR_VERSION",
    "SUPPORTED_HIRES_DEFAULT_SCOPES",
    "HiresDefaultAssignment",
    "HiresProfile",
    "HiresProfileSaveManifest",
    "HiresProfileValidationError",
    "HiresSettingDescriptor",
]
