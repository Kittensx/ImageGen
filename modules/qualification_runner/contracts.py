from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


QUALIFICATION_RUNNER_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1

REVIEW_CHOICES: tuple[str, ...] = (
    "pending",
    "accepted",
    "output_poor",
    "component_compatibility_suspect",
    "execution_profile_suspect",
    "technical_failure",
    "inconclusive",
    "parity_match",
    "parity_mismatch",
)

RETEST_DEFAULT_TOKEN = "default"
RETEST_SCALAR_FIELDS: tuple[str, ...] = (
    "steps",
    "cfg_scale",
    "seed",
    "width",
    "height",
    "sampler_name",
    "scheduler_name",
)


@dataclass(frozen=True)
class QualificationPattern:
    pattern_id: str
    label: str
    mutation_kind: str
    component_role: str = ""
    values: tuple[Any, ...] = ()
    include_control: bool = True
    request_overrides: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_dict(cls, pattern_id: str, payload: Mapping[str, Any]) -> "QualificationPattern":
        values = payload.get("values") or ()
        if isinstance(values, (str, bytes)):
            values = (values,)
        return cls(
            pattern_id=str(pattern_id).strip(),
            label=str(payload.get("label") or pattern_id).strip(),
            mutation_kind=str(payload.get("mutation_kind") or "control").strip().lower(),
            component_role=str(payload.get("component_role") or "").strip(),
            values=tuple(values),
            include_control=bool(payload.get("include_control", True)),
            request_overrides=dict(payload.get("request_overrides") or {}),
            description=str(payload.get("description") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "label": self.label,
            "mutation_kind": self.mutation_kind,
            "component_role": self.component_role or None,
            "values": list(self.values),
            "include_control": self.include_control,
            "request_overrides": dict(self.request_overrides),
            "description": self.description,
        }


@dataclass(frozen=True)
class BlueprintSnapshot:
    asset_id: int
    model_path: str
    model_filename: str
    model_sha256: str
    family: str
    family_label: str
    base_weight_role: str
    components: Mapping[str, str]
    component_details: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "model_path": self.model_path,
            "model_filename": self.model_filename,
            "model_sha256": self.model_sha256,
            "family": self.family,
            "family_label": self.family_label,
            "base_weight_role": self.base_weight_role,
            "components": dict(self.components),
            "component_details": {
                key: dict(value) for key, value in self.component_details.items()
            },
        }


@dataclass(frozen=True)
class QualificationCase:
    case_id: str
    label: str
    mutation_kind: str
    mutation: Mapping[str, Any]
    request_payload: Mapping[str, Any]
    resolved_composition: Mapping[str, Any]
    parent_case_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "mutation_kind": self.mutation_kind,
            "mutation": dict(self.mutation),
            "parent_case_id": self.parent_case_id or None,
            "request_payload": dict(self.request_payload),
            "resolved_composition": dict(self.resolved_composition),
        }


__all__ = [
    "BlueprintSnapshot",
    "QUALIFICATION_RUNNER_SCHEMA_VERSION",
    "QualificationCase",
    "QualificationPattern",
    "RETEST_DEFAULT_TOKEN",
    "RETEST_SCALAR_FIELDS",
    "REVIEW_CHOICES",
    "REVIEW_SCHEMA_VERSION",
]
