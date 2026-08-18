from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping


RELATIONSHIP_EVIDENCE_CONTRACT_VERSION = "component-relationship-evidence-v1"
RELATIONSHIP_SOURCE_RECORDED = "recorded"
RELATIONSHIP_SOURCE_EXACT_ANALYSIS = "exact_analysis"
RELATIONSHIP_SOURCE_INFERRED = "inferred"
RELATIONSHIP_SOURCE_RUNTIME_VALIDATED = "runtime_validated"
RELATIONSHIP_EVIDENCE_SOURCES = {
    RELATIONSHIP_SOURCE_RECORDED,
    RELATIONSHIP_SOURCE_EXACT_ANALYSIS,
    RELATIONSHIP_SOURCE_INFERRED,
    RELATIONSHIP_SOURCE_RUNTIME_VALIDATED,
}

RELATIONSHIP_STATUS_ACTIVE = "active"
RELATIONSHIP_STATUS_STALE = "stale"
RELATIONSHIP_STATUS_SUPERSEDED = "superseded"
RELATIONSHIP_STATUSES = {
    RELATIONSHIP_STATUS_ACTIVE,
    RELATIONSHIP_STATUS_STALE,
    RELATIONSHIP_STATUS_SUPERSEDED,
}

POLICY_SCOPE_GLOBAL = "global"
POLICY_SCOPE_BASE = "base"
POLICY_ACTION_DISABLE = "disable"
POLICY_SOURCE_USER = "user"
POLICY_SOURCE_ADMIN = "admin"
POLICY_SOURCE_SYSTEM = "system"
POLICY_SOURCES = {POLICY_SOURCE_USER, POLICY_SOURCE_ADMIN, POLICY_SOURCE_SYSTEM}

VALIDATION_STAGE_STRUCTURAL = "structural"
VALIDATION_STAGE_HYDRATION = "hydration"
VALIDATION_STAGE_RUNTIME_INTERFACE = "runtime_interface"
VALIDATION_STAGE_CONDITIONING = "conditioning"
VALIDATION_STAGE_GENERATION = "generation"
VALIDATION_STAGE_PARITY = "parity"
VALIDATION_STAGES = (
    VALIDATION_STAGE_STRUCTURAL,
    VALIDATION_STAGE_HYDRATION,
    VALIDATION_STAGE_RUNTIME_INTERFACE,
    VALIDATION_STAGE_CONDITIONING,
    VALIDATION_STAGE_GENERATION,
    VALIDATION_STAGE_PARITY,
)
VALIDATION_RESULT_PASS = "pass"
VALIDATION_RESULT_FAIL = "fail"
VALIDATION_RESULT_ERROR = "error"
VALIDATION_RESULTS = {VALIDATION_RESULT_PASS, VALIDATION_RESULT_FAIL, VALIDATION_RESULT_ERROR}
VALIDATION_BLOCKING = "blocking"
VALIDATION_ADVISORY = "advisory"
VALIDATION_BLOCKING_STATES = {VALIDATION_BLOCKING, VALIDATION_ADVISORY}

SPLIT_ELIGIBILITY_ELIGIBLE = "eligible"
SPLIT_ELIGIBILITY_BLOCKED = "blocked"
SPLIT_ELIGIBILITY_INCONCLUSIVE = "inconclusive"
SPLIT_ELIGIBILITY_UNTESTED = "untested"
SPLIT_ELIGIBILITY_STATES = {
    SPLIT_ELIGIBILITY_ELIGIBLE,
    SPLIT_ELIGIBILITY_BLOCKED,
    SPLIT_ELIGIBILITY_INCONCLUSIVE,
    SPLIT_ELIGIBILITY_UNTESTED,
}
SPLIT_GATE_RECOMMENDED = "recommended"
SPLIT_GATE_INFORMATIONAL = "informational"
SPLIT_GATE_MODES = {SPLIT_GATE_RECOMMENDED, SPLIT_GATE_INFORMATIONAL}

TRANSIENT_VALIDATION_ERROR_CATEGORIES = {
    "oom",
    "out_of_memory",
    "memory_pressure",
    "cancelled",
    "interrupted",
    "watchdog_timeout",
    "io_transient",
}


def _sha256_hex(value: str, *, field_name: str, allow_empty: bool = False) -> str:
    digest = str(value or "").strip().lower()
    if allow_empty and not digest:
        return ""
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256.")
    return digest


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class RelationshipParticipant:
    component_sha256: str = ""
    participant_role: str = "component"
    position: int = 0
    composition_id: str = ""
    blueprint_id: str = ""
    weight: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        component = _sha256_hex(
            self.component_sha256,
            field_name="RelationshipParticipant.component_sha256",
            allow_empty=True,
        )
        if not component and not str(self.composition_id or "").strip() and not str(self.blueprint_id or "").strip():
            raise ValueError("Relationship participants require a component, composition, or blueprint identity.")
        object.__setattr__(self, "component_sha256", component)
        object.__setattr__(self, "participant_role", str(self.participant_role or "component").strip() or "component")
        object.__setattr__(self, "position", int(self.position))
        object.__setattr__(self, "composition_id", str(self.composition_id or "").strip())
        object.__setattr__(self, "blueprint_id", str(self.blueprint_id or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "position": int(self.position),
            "participant_role": self.participant_role,
            "component_sha256": self.component_sha256 or None,
            "composition_id": self.composition_id or None,
            "blueprint_id": self.blueprint_id or None,
            "weight": self.weight,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "metadata": dict(self.metadata)}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, default_position: int = 0) -> "RelationshipParticipant":
        return cls(
            component_sha256=str(payload.get("component_sha256") or ""),
            participant_role=str(payload.get("participant_role") or payload.get("role") or "component"),
            position=int(payload.get("position", default_position)),
            composition_id=str(payload.get("composition_id") or ""),
            blueprint_id=str(payload.get("blueprint_id") or ""),
            weight=(float(payload["weight"]) if payload.get("weight") is not None else None),
            metadata=dict(payload.get("metadata") or {}),
        )


def normalize_relationship_participants(
    participants: Iterable[RelationshipParticipant | Mapping[str, Any]],
) -> tuple[RelationshipParticipant, ...]:
    normalized: list[RelationshipParticipant] = []
    for index, item in enumerate(participants):
        if isinstance(item, RelationshipParticipant):
            participant = item
        else:
            participant = RelationshipParticipant.from_mapping(item, default_position=index)
        normalized.append(participant)
    if len(normalized) < 2:
        raise ValueError("A relationship requires at least two participants.")
    positions = [item.position for item in normalized]
    if len(positions) != len(set(positions)):
        raise ValueError("Relationship participant positions must be unique.")
    normalized.sort(
        key=lambda item: (
            item.position,
            item.participant_role,
            item.component_sha256,
            item.composition_id,
            item.blueprint_id,
        )
    )
    return tuple(normalized)


def relationship_key(
    *,
    relationship_type: str,
    participants: Iterable[RelationshipParticipant | Mapping[str, Any]],
    family_id: str = "",
    provider_id: str = "",
) -> str:
    normalized = normalize_relationship_participants(participants)
    payload = {
        "contract_version": RELATIONSHIP_EVIDENCE_CONTRACT_VERSION,
        "relationship_type": str(relationship_type or "").strip(),
        "family_id": str(family_id or "").strip(),
        "provider_id": str(provider_id or "").strip(),
        "participants": [item.identity_payload() for item in normalized],
    }
    if not payload["relationship_type"]:
        raise ValueError("relationship_type is required.")
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def normalized_blocking_state(*, result: str, requested: str | None, error_category: str = "") -> str:
    value = str(requested or "").strip().lower()
    if value in VALIDATION_BLOCKING_STATES:
        if str(error_category or "").strip().lower() in TRANSIENT_VALIDATION_ERROR_CATEGORIES:
            return VALIDATION_ADVISORY
        return value
    if str(result or "").strip().lower() == VALIDATION_RESULT_FAIL and str(error_category or "").strip().lower() not in TRANSIENT_VALIDATION_ERROR_CATEGORIES:
        return VALIDATION_ADVISORY
    return VALIDATION_ADVISORY


__all__ = [
    "POLICY_ACTION_DISABLE",
    "POLICY_SCOPE_BASE",
    "POLICY_SCOPE_GLOBAL",
    "POLICY_SOURCE_ADMIN",
    "POLICY_SOURCE_SYSTEM",
    "POLICY_SOURCE_USER",
    "RELATIONSHIP_EVIDENCE_CONTRACT_VERSION",
    "RELATIONSHIP_SOURCE_EXACT_ANALYSIS",
    "RELATIONSHIP_SOURCE_INFERRED",
    "RELATIONSHIP_SOURCE_RECORDED",
    "RELATIONSHIP_SOURCE_RUNTIME_VALIDATED",
    "RELATIONSHIP_STATUS_ACTIVE",
    "RELATIONSHIP_STATUS_STALE",
    "RELATIONSHIP_STATUS_SUPERSEDED",
    "RelationshipParticipant",
    "TRANSIENT_VALIDATION_ERROR_CATEGORIES",
    "SPLIT_ELIGIBILITY_BLOCKED",
    "SPLIT_ELIGIBILITY_ELIGIBLE",
    "SPLIT_ELIGIBILITY_INCONCLUSIVE",
    "SPLIT_ELIGIBILITY_STATES",
    "SPLIT_ELIGIBILITY_UNTESTED",
    "SPLIT_GATE_INFORMATIONAL",
    "SPLIT_GATE_MODES",
    "SPLIT_GATE_RECOMMENDED",
    "VALIDATION_ADVISORY",
    "VALIDATION_BLOCKING",
    "VALIDATION_RESULT_ERROR",
    "VALIDATION_RESULT_FAIL",
    "VALIDATION_RESULT_PASS",
    "VALIDATION_STAGE_CONDITIONING",
    "VALIDATION_STAGE_GENERATION",
    "VALIDATION_STAGE_HYDRATION",
    "VALIDATION_STAGE_PARITY",
    "VALIDATION_STAGE_RUNTIME_INTERFACE",
    "VALIDATION_STAGE_STRUCTURAL",
    "VALIDATION_STAGES",
    "canonical_json",
    "normalize_relationship_participants",
    "normalized_blocking_state",
    "relationship_key",
]
