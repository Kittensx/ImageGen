from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping, Protocol, Sequence


SOURCE_FORM_PHYSICAL_COMPONENT = "physical_component"
SOURCE_FORM_STANDALONE_SHARED = "standalone_shared"
SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT = "digital_checkpoint_component"
SOURCE_FORM_RECONSTRUCTED_EXPORT = "reconstructed_export"
SOURCE_FORM_UNKNOWN = "unknown"

SELECTION_OFF = "off"
SELECTION_AUTO = "auto"
SELECTION_EXPLICIT = "explicit"

AVAILABILITY_AVAILABLE = "available"
AVAILABILITY_MISSING = "missing"
AVAILABILITY_UNKNOWN = "unknown"

LOAD_STRATEGY_PHYSICAL = "physical_component"
LOAD_STRATEGY_STANDALONE = "standalone_shared"
LOAD_STRATEGY_DIGITAL = "digital_checkpoint_component"
LOAD_STRATEGY_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ComponentRoleDefinition:
    canonical_role_id: str
    display_label: str
    required: bool
    off_allowed: bool
    auto_allowed: bool
    expected_source_kinds: tuple[str, ...]
    base_weight_role: bool = False
    structural_constraints: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_role_id": self.canonical_role_id,
            "display_label": self.display_label,
            "required": self.required,
            "off_allowed": self.off_allowed,
            "auto_allowed": self.auto_allowed,
            "expected_source_kinds": list(self.expected_source_kinds),
            "base_weight_role": self.base_weight_role,
            "structural_constraints": dict(self.structural_constraints),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentRoleDefinition":
        return cls(
            canonical_role_id=str(payload.get("canonical_role_id") or ""),
            display_label=str(payload.get("display_label") or ""),
            required=bool(payload.get("required")),
            off_allowed=bool(payload.get("off_allowed")),
            auto_allowed=bool(payload.get("auto_allowed")),
            expected_source_kinds=tuple(str(item) for item in (payload.get("expected_source_kinds") or ())),
            base_weight_role=bool(payload.get("base_weight_role")),
            structural_constraints=dict(payload.get("structural_constraints") or {}),
        )


class ArchitectureFamilyProvider(Protocol):
    family_id: str
    display_label: str
    architecture_aliases: tuple[str, ...]
    required_roles: tuple[ComponentRoleDefinition, ...]
    optional_roles: tuple[ComponentRoleDefinition, ...]
    base_weight_role: str
    version: str

    def role_definitions(self) -> tuple[ComponentRoleDefinition, ...]: ...
    def role_definition(self, role: str) -> ComponentRoleDefinition | None: ...
    def structurally_compatible(self, *, family: str, role: str) -> bool: ...
    def evaluate_structural_compatibility(self, *, role: str, evidence: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def component_extraction_rules(self, role: str) -> Mapping[str, Any]: ...
    def supports_digital_hydration(self, role: str) -> bool: ...
    def supports_runtime_composition(self) -> bool: ...
    def placement_capabilities(self, role: str) -> tuple[str, ...]: ...
    def blueprint_capabilities(self) -> Mapping[str, Any]: ...
    def supports_analysis_layout(self, role: str) -> bool: ...
    def analysis_layout_version(self, role: str) -> int | None: ...
    def describe_analysis_layout(self, role: str) -> Any | None: ...
    def resolve_analysis_nodes(self, role: str, tensor_names: Sequence[str]) -> Mapping[str, tuple[str, ...]]: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ComponentIdentity:
    component_sha256: str
    structure_sha256: str
    byte_count: int
    tensor_count: int
    normalized_role: str
    provider_family_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        digest = self.component_sha256.strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("ComponentIdentity.component_sha256 must be a 64-character hexadecimal SHA-256.")
        object.__setattr__(self, "component_sha256", digest)
        structure = self.structure_sha256.strip().lower()
        if structure and (len(structure) != 64 or any(ch not in "0123456789abcdef" for ch in structure)):
            raise ValueError("ComponentIdentity.structure_sha256 must be empty or a 64-character hexadecimal SHA-256.")
        object.__setattr__(self, "structure_sha256", structure)
        if self.byte_count < 0 or self.tensor_count < 0:
            raise ValueError("ComponentIdentity byte_count and tensor_count must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_sha256": self.component_sha256,
            "structure_sha256": self.structure_sha256,
            "byte_count": int(self.byte_count),
            "tensor_count": int(self.tensor_count),
            "normalized_role": self.normalized_role,
            "provider_family_evidence": list(self.provider_family_evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentIdentity":
        return cls(
            component_sha256=str(payload.get("component_sha256") or ""),
            structure_sha256=str(payload.get("structure_sha256") or ""),
            byte_count=int(payload.get("byte_count") or 0),
            tensor_count=int(payload.get("tensor_count") or 0),
            normalized_role=str(payload.get("normalized_role") or ""),
            provider_family_evidence=tuple(str(item) for item in (payload.get("provider_family_evidence") or ())),
        )


@dataclass(frozen=True)
class ComponentOccurrence:
    component_sha256: str
    asset_id: int | None
    asset_path: str
    source_form: str
    embedded_state: str
    role: str
    source_prefixes: tuple[str, ...] = ()
    availability_state: str = AVAILABILITY_UNKNOWN
    locator: Mapping[str, Any] = field(default_factory=dict)
    scan_timestamp: str = ""
    scanner_version: str = ""
    provider_family: str = ""
    provider_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_sha256": self.component_sha256,
            "asset_id": self.asset_id,
            "asset_path": self.asset_path,
            "source_form": self.source_form,
            "embedded_state": self.embedded_state,
            "role": self.role,
            "source_prefixes": list(self.source_prefixes),
            "availability_state": self.availability_state,
            "locator": dict(self.locator),
            "scan_timestamp": self.scan_timestamp,
            "scanner_version": self.scanner_version,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentOccurrence":
        return cls(
            component_sha256=str(payload.get("component_sha256") or ""),
            asset_id=(int(payload["asset_id"]) if payload.get("asset_id") is not None else None),
            asset_path=str(payload.get("asset_path") or ""),
            source_form=str(payload.get("source_form") or SOURCE_FORM_UNKNOWN),
            embedded_state=str(payload.get("embedded_state") or "unknown"),
            role=str(payload.get("role") or ""),
            source_prefixes=tuple(str(item) for item in (payload.get("source_prefixes") or ())),
            availability_state=str(payload.get("availability_state") or AVAILABILITY_UNKNOWN),
            locator=dict(payload.get("locator") or {}),
            scan_timestamp=str(payload.get("scan_timestamp") or ""),
            scanner_version=str(payload.get("scanner_version") or ""),
            provider_family=str(payload.get("provider_family") or ""),
            provider_version=str(payload.get("provider_version") or ""),
        )


@dataclass(frozen=True)
class ComponentSelection:
    family: str
    role: str
    selector_mode: str
    explicit_fingerprint: str = ""
    placement_policy: str = ""
    source_override: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        mode = self.selector_mode.strip().lower()
        if mode not in {SELECTION_OFF, SELECTION_AUTO, SELECTION_EXPLICIT}:
            raise ValueError(f"Unsupported component selector mode: {self.selector_mode!r}")
        object.__setattr__(self, "selector_mode", mode)
        digest = self.explicit_fingerprint.strip().lower()
        if mode == SELECTION_EXPLICIT:
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("Explicit component selection requires a 64-character hexadecimal fingerprint.")
        elif digest:
            raise ValueError("explicit_fingerprint may only be set for explicit component selections.")
        object.__setattr__(self, "explicit_fingerprint", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "role": self.role,
            "selector_mode": self.selector_mode,
            "explicit_fingerprint": self.explicit_fingerprint or None,
            "placement_policy": self.placement_policy or None,
            "source_override": dict(self.source_override or {}) or None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentSelection":
        return cls(
            family=str(payload.get("family") or ""),
            role=str(payload.get("role") or ""),
            selector_mode=str(payload.get("selector_mode") or SELECTION_AUTO),
            explicit_fingerprint=str(payload.get("explicit_fingerprint") or ""),
            placement_policy=str(payload.get("placement_policy") or ""),
            source_override=(dict(payload.get("source_override") or {}) or None),
        )


@dataclass(frozen=True)
class ResolvedComponent:
    component_sha256: str
    role: str
    source: ComponentOccurrence
    provider_family: str
    provider_version: str
    availability_evidence: Mapping[str, Any] = field(default_factory=dict)
    exclusion_state: str = "eligible"
    validation_state: str = "untested"
    load_strategy: str = LOAD_STRATEGY_UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_sha256": self.component_sha256,
            "role": self.role,
            "source": self.source.to_dict(),
            "source_form": self.source.source_form,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "availability_evidence": dict(self.availability_evidence),
            "exclusion_state": self.exclusion_state,
            "validation_state": self.validation_state,
            "load_strategy": self.load_strategy,
        }


@dataclass(frozen=True)
class CompositionIdentity:
    composition_sha256: str
    family: str
    provider_version: str
    components: tuple[tuple[str, str], ...]
    behavior_choices: tuple[tuple[str, str], ...]
    identity_version: str = "component-composition-v2"

    @classmethod
    def derive(
        cls,
        *,
        family: str,
        provider_version: str,
        components: Mapping[str, str] | Sequence[tuple[str, str]],
        behavior_choices: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        identity_version: str = "component-composition-v2",
    ) -> "CompositionIdentity":
        component_items = tuple(sorted((str(role), str(digest).strip().lower()) for role, digest in dict(components).items()))
        behavior_items = tuple(sorted((str(key), cls._normalize_behavior_value(value)) for key, value in dict(behavior_choices).items()))
        payload = {
            "identity_version": identity_version,
            "family": str(family),
            "provider_version": str(provider_version),
            "components": component_items,
            "behavior_choices": behavior_items,
        }
        digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return cls(
            composition_sha256=digest,
            family=str(family),
            provider_version=str(provider_version),
            components=component_items,
            behavior_choices=behavior_items,
            identity_version=identity_version,
        )

    @staticmethod
    def _normalize_behavior_value(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        if value is None:
            return ""
        return str(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_version": self.identity_version,
            "composition_sha256": self.composition_sha256,
            "composition_short_hash": self.composition_sha256[:12],
            "family": self.family,
            "provider_version": self.provider_version,
            "components": {role: digest for role, digest in self.components},
            "behavior_choices": {key: value for key, value in self.behavior_choices},
        }


def source_form_for_asset_type(asset_type: str, *, metadata: Mapping[str, Any] | None = None) -> str:
    token = str(asset_type or "").strip().lower()
    meta = dict(metadata or {})
    explicit = str(meta.get("component_source_form") or "").strip().lower()
    if explicit in {
        SOURCE_FORM_PHYSICAL_COMPONENT,
        SOURCE_FORM_STANDALONE_SHARED,
        SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT,
        SOURCE_FORM_RECONSTRUCTED_EXPORT,
    }:
        return explicit
    if token == "checkpoint":
        return SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT
    if token in {"vae", "text_encoder", "component"}:
        return SOURCE_FORM_STANDALONE_SHARED
    return SOURCE_FORM_UNKNOWN


def load_strategy_for_source_form(source_form: str) -> str:
    token = str(source_form or "").strip().lower()
    if token == SOURCE_FORM_PHYSICAL_COMPONENT:
        return LOAD_STRATEGY_PHYSICAL
    if token == SOURCE_FORM_STANDALONE_SHARED:
        return LOAD_STRATEGY_STANDALONE
    if token == SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT:
        return LOAD_STRATEGY_DIGITAL
    return LOAD_STRATEGY_UNKNOWN


__all__ = [
    "ArchitectureFamilyProvider",
    "ComponentIdentity",
    "ComponentOccurrence",
    "ComponentRoleDefinition",
    "ComponentSelection",
    "CompositionIdentity",
    "ResolvedComponent",
    "AVAILABILITY_AVAILABLE",
    "AVAILABILITY_MISSING",
    "AVAILABILITY_UNKNOWN",
    "SELECTION_AUTO",
    "SELECTION_EXPLICIT",
    "SELECTION_OFF",
    "SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT",
    "SOURCE_FORM_PHYSICAL_COMPONENT",
    "SOURCE_FORM_RECONSTRUCTED_EXPORT",
    "SOURCE_FORM_STANDALONE_SHARED",
    "SOURCE_FORM_UNKNOWN",
    "load_strategy_for_source_form",
    "source_form_for_asset_type",
]
