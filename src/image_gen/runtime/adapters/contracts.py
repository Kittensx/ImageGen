from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

ADAPTER_INSPECTION_CONTRACT_VERSION = "image-gen-adapter-inspection-v1"
ADAPTER_COMPATIBILITY_CONTRACT_VERSION = "image-gen-adapter-compatibility-v1"
ADAPTER_RUNTIME_PLAN_CONTRACT_VERSION = "image-gen-adapter-runtime-plan-v1"

ADAPTER_FORMATS = frozenset({
    "standard_kohya_lora",
    "standard_diffusers_peft_lora",
    "standard_lora_up_down",
    "lycoris_loha",
    "lycoris_lokr",
    "lycoris_locon",
    "lycoris_other",
    "non_adapter_full_model",
    "unknown_adapter",
    "invalid",
})

SUPPORT_STATES = frozenset({
    "supported",
    "supported_with_warning",
    "partial",
    "unsupported",
    "misclassified",
    "invalid",
})


def _tuple_text(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


@dataclass(frozen=True)
class AdapterInspectionRecord:
    contract_version: str = ADAPTER_INSPECTION_CONTRACT_VERSION
    asset_type: str = "lora"
    source_path: str = ""
    file_signature: Mapping[str, Any] = field(default_factory=dict)
    sha256: str = ""
    model_family: str = ""
    model_family_evidence: tuple[str, ...] = ()
    adapter_format: str = "unknown_adapter"
    adapter_format_evidence: tuple[str, ...] = ()
    adapter_extensions: tuple[str, ...] = ()
    network_type: str = "Unknown"
    tensor_key_count: int = 0
    target_scopes: tuple[str, ...] = ()
    target_counts: Mapping[str, int] = field(default_factory=dict)
    source_rank: Any = None
    source_alpha: Any = None
    inspection_warnings: tuple[str, ...] = ()
    inspection_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "asset_type": self.asset_type,
            "source_path": self.source_path,
            "file_signature": dict(self.file_signature),
            "sha256": self.sha256,
            "model_family": self.model_family,
            "model_family_evidence": list(self.model_family_evidence),
            "adapter_format": self.adapter_format,
            "adapter_format_evidence": list(self.adapter_format_evidence),
            "adapter_extensions": list(self.adapter_extensions),
            "network_type": self.network_type,
            "tensor_key_count": int(self.tensor_key_count),
            "target_scopes": list(self.target_scopes),
            "target_counts": {str(key): int(value) for key, value in dict(self.target_counts).items()},
            "source_rank": self.source_rank,
            "source_alpha": self.source_alpha,
            "inspection_warnings": list(self.inspection_warnings),
            "inspection_errors": list(self.inspection_errors),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AdapterInspectionRecord":
        payload = dict(value or {})
        adapter_format = str(payload.get("adapter_format") or "unknown_adapter")
        if adapter_format not in ADAPTER_FORMATS:
            adapter_format = "unknown_adapter"
        return cls(
            contract_version=str(payload.get("contract_version") or ADAPTER_INSPECTION_CONTRACT_VERSION),
            asset_type=str(payload.get("asset_type") or "lora"),
            source_path=str(payload.get("source_path") or payload.get("path") or ""),
            file_signature=dict(payload.get("file_signature") or {}),
            sha256=str(payload.get("sha256") or ""),
            model_family=str(payload.get("model_family") or payload.get("detected_model_family") or ""),
            model_family_evidence=_tuple_text(payload.get("model_family_evidence")),
            adapter_format=adapter_format,
            adapter_format_evidence=_tuple_text(payload.get("adapter_format_evidence")),
            adapter_extensions=_tuple_text(payload.get("adapter_extensions")),
            network_type=str(payload.get("network_type") or "Unknown"),
            tensor_key_count=int(payload.get("tensor_key_count") or 0),
            target_scopes=_tuple_text(payload.get("target_scopes")),
            target_counts={
                str(key): int(item or 0)
                for key, item in dict(payload.get("target_counts") or {}).items()
            },
            source_rank=payload.get("source_rank", payload.get("network_dimension")),
            source_alpha=payload.get("source_alpha", payload.get("network_alpha")),
            inspection_warnings=_tuple_text(payload.get("inspection_warnings")),
            inspection_errors=_tuple_text(payload.get("inspection_errors")),
        )


@dataclass(frozen=True)
class AdapterCompatibilityResult:
    contract_version: str = ADAPTER_COMPATIBILITY_CONTRACT_VERSION
    family_status: str = "unknown"
    format_status: str = "unsupported"
    target_status: str = "unknown"
    overall_support_state: str = "unsupported"
    runtime_loadable: bool = False
    blocking_reason: str = ""
    warnings: tuple[str, ...] = ()
    loader_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "family_status": self.family_status,
            "format_status": self.format_status,
            "target_status": self.target_status,
            "overall_support_state": self.overall_support_state,
            "runtime_loadable": bool(self.runtime_loadable),
            "blocking_reason": self.blocking_reason,
            "warnings": list(self.warnings),
            "loader_id": self.loader_id,
        }


@dataclass(frozen=True)
class AdapterRuntimePlan:
    contract_version: str = ADAPTER_RUNTIME_PLAN_CONTRACT_VERSION
    adapter_identity: str = ""
    asset_id: str = ""
    requested_name: str = ""
    resolved_path: str = ""
    file_hash: str = ""
    inspection_contract_version: str = ADAPTER_INSPECTION_CONTRACT_VERSION
    adapter_format: str = "unknown_adapter"
    model_family: str = ""
    active_checkpoint_family: str = ""
    compatibility: Mapping[str, Any] = field(default_factory=dict)
    loader_id: str = ""
    requested_weight: float = 1.0
    effective_weight: float = 1.0
    weight_semantics: str = "user multiplier after loader-native normalization"
    expected_component_targets: tuple[str, ...] = ()
    blocking_reason: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "adapter_identity": self.adapter_identity,
            "asset_id": self.asset_id,
            "requested_name": self.requested_name,
            "resolved_path": self.resolved_path,
            "file_hash": self.file_hash,
            "inspection_contract_version": self.inspection_contract_version,
            "adapter_format": self.adapter_format,
            "model_family": self.model_family,
            "active_checkpoint_family": self.active_checkpoint_family,
            "compatibility": dict(self.compatibility),
            "loader_id": self.loader_id,
            "requested_weight": float(self.requested_weight),
            "effective_weight": float(self.effective_weight),
            "weight_semantics": self.weight_semantics,
            "expected_component_targets": list(self.expected_component_targets),
            "blocking_reason": self.blocking_reason,
            "warnings": list(self.warnings),
        }
