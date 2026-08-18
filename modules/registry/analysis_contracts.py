from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping


ANALYSIS_LAYOUT_CONTRACT_VERSION = "component-analysis-layout-v1"
TENSOR_HASH_MANIFEST_VERSION = 2
TENSOR_HASH_ALGORITHM_VERSION = "sha256-canonical-tensor-node-v1-resolver-v1"


def _validate_sha256(value: str, *, field_name: str, allow_empty: bool = False) -> str:
    digest = str(value or "").strip().lower()
    if not digest and allow_empty:
        return ""
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256.")
    return digest


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ComponentAnalysisNode:
    node_id: str
    node_kind: str
    ordinal: int
    parent_node_id: str = ""
    tensor_prefixes: tuple[str, ...] = ()
    tensor_names: tuple[str, ...] = ()
    exact_hash: str = ""
    tensor_count: int = 0
    byte_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_id = str(self.node_id or "").strip()
        if not node_id:
            raise ValueError("ComponentAnalysisNode.node_id must not be empty.")
        if any(ch.isspace() for ch in node_id):
            raise ValueError("ComponentAnalysisNode.node_id must be a stable token without whitespace.")
        kind = str(self.node_kind or "").strip().lower()
        if kind not in {"block", "tensor_group", "component_root"}:
            raise ValueError(f"Unsupported analysis node kind: {self.node_kind!r}")
        if int(self.ordinal) < 0:
            raise ValueError("ComponentAnalysisNode.ordinal must be non-negative.")
        if int(self.tensor_count) < 0 or int(self.byte_count) < 0:
            raise ValueError("ComponentAnalysisNode tensor_count and byte_count must be non-negative.")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "node_kind", kind)
        object.__setattr__(self, "parent_node_id", str(self.parent_node_id or "").strip())
        object.__setattr__(self, "tensor_prefixes", tuple(dict.fromkeys(str(v) for v in self.tensor_prefixes if str(v))))
        object.__setattr__(self, "tensor_names", tuple(dict.fromkeys(str(v) for v in self.tensor_names if str(v))))
        object.__setattr__(self, "exact_hash", _validate_sha256(self.exact_hash, field_name="ComponentAnalysisNode.exact_hash", allow_empty=True))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def matches_tensor(self, tensor_name: str) -> bool:
        name = str(tensor_name)
        if name in self.tensor_names:
            return True
        return any(name.startswith(prefix) for prefix in self.tensor_prefixes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "parent_node_id": self.parent_node_id,
            "ordinal": int(self.ordinal),
            "tensor_prefixes": list(self.tensor_prefixes),
            "tensor_names": list(self.tensor_names),
            "exact_hash": self.exact_hash,
            "tensor_count": int(self.tensor_count),
            "byte_count": int(self.byte_count),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentAnalysisNode":
        return cls(
            node_id=str(payload.get("node_id") or ""),
            node_kind=str(payload.get("node_kind") or ""),
            parent_node_id=str(payload.get("parent_node_id") or ""),
            ordinal=int(payload.get("ordinal") or 0),
            tensor_prefixes=tuple(str(v) for v in (payload.get("tensor_prefixes") or ())),
            tensor_names=tuple(str(v) for v in (payload.get("tensor_names") or ())),
            exact_hash=str(payload.get("exact_hash") or ""),
            tensor_count=int(payload.get("tensor_count") or 0),
            byte_count=int(payload.get("byte_count") or 0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ComponentAnalysisLayout:
    provider_id: str
    family_id: str
    component_role: str
    layout_version: int
    nodes: tuple[ComponentAnalysisNode, ...]
    grouping_rules: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.provider_id or "").strip():
            raise ValueError("ComponentAnalysisLayout.provider_id must not be empty.")
        if not str(self.family_id or "").strip():
            raise ValueError("ComponentAnalysisLayout.family_id must not be empty.")
        if not str(self.component_role or "").strip():
            raise ValueError("ComponentAnalysisLayout.component_role must not be empty.")
        if int(self.layout_version) < 1:
            raise ValueError("ComponentAnalysisLayout.layout_version must be >= 1.")
        ordered = tuple(sorted(self.nodes, key=lambda n: (n.ordinal, n.node_id)))
        ids = [node.node_id for node in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("ComponentAnalysisLayout node IDs must be unique.")
        known = set(ids)
        for node in ordered:
            if node.parent_node_id and node.parent_node_id not in known:
                raise ValueError(f"Unknown parent node ID {node.parent_node_id!r} for {node.node_id!r}.")
        object.__setattr__(self, "provider_id", str(self.provider_id).strip())
        object.__setattr__(self, "family_id", str(self.family_id).strip())
        object.__setattr__(self, "component_role", str(self.component_role).strip())
        object.__setattr__(self, "nodes", ordered)
        object.__setattr__(self, "grouping_rules", dict(self.grouping_rules or {}))

    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    def resolve_tensor_names(self, tensor_names: tuple[str, ...] | list[str]) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for tensor_name in sorted(str(item) for item in tensor_names):
            matches = [node for node in self.nodes if node.matches_tensor(tensor_name)]
            if len(matches) > 1:
                raise ValueError(f"Tensor {tensor_name!r} matches multiple analysis nodes: {[n.node_id for n in matches]}")
            if matches:
                result[matches[0].node_id].append(tensor_name)
        return {node_id: tuple(values) for node_id, values in result.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": ANALYSIS_LAYOUT_CONTRACT_VERSION,
            "provider_id": self.provider_id,
            "family_id": self.family_id,
            "component_role": self.component_role,
            "layout_version": int(self.layout_version),
            "nodes": [node.to_dict() for node in self.nodes],
            "grouping_rules": dict(self.grouping_rules),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentAnalysisLayout":
        return cls(
            provider_id=str(payload.get("provider_id") or ""),
            family_id=str(payload.get("family_id") or ""),
            component_role=str(payload.get("component_role") or ""),
            layout_version=int(payload.get("layout_version") or 0),
            nodes=tuple(ComponentAnalysisNode.from_dict(item) for item in (payload.get("nodes") or ())),
            grouping_rules=dict(payload.get("grouping_rules") or {}),
        )


@dataclass(frozen=True)
class TensorHashManifestNode:
    exact_hash: str
    tensor_count: int
    byte_count: int
    tensors: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exact_hash", _validate_sha256(self.exact_hash, field_name="TensorHashManifestNode.exact_hash"))
        if int(self.tensor_count) < 0 or int(self.byte_count) < 0:
            raise ValueError("TensorHashManifestNode counts must be non-negative.")
        normalized: dict[str, str] = {}
        for name, digest in dict(self.tensors or {}).items():
            key = str(name or "")
            if not key:
                raise ValueError("TensorHashManifestNode tensor names must not be empty.")
            normalized[key] = _validate_sha256(str(digest), field_name=f"tensor hash for {key}")
        if int(self.tensor_count) != len(normalized):
            raise ValueError("TensorHashManifestNode.tensor_count must match tensors length.")
        object.__setattr__(self, "tensors", dict(sorted(normalized.items())))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_hash": self.exact_hash,
            "tensor_count": int(self.tensor_count),
            "byte_count": int(self.byte_count),
            "tensors": dict(self.tensors),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorHashManifestNode":
        return cls(
            exact_hash=str(payload.get("exact_hash") or ""),
            tensor_count=int(payload.get("tensor_count") or 0),
            byte_count=int(payload.get("byte_count") or 0),
            tensors=dict(payload.get("tensors") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TensorHashManifest:
    provider_id: str
    family_id: str
    component_role: str
    layout_version: int
    component_sha256: str
    nodes: Mapping[str, TensorHashManifestNode]
    algorithm_version: str = TENSOR_HASH_ALGORITHM_VERSION
    manifest_version: int = TENSOR_HASH_MANIFEST_VERSION
    analysis_manifest_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_sha256", _validate_sha256(self.component_sha256, field_name="TensorHashManifest.component_sha256"))
        if int(self.manifest_version) != TENSOR_HASH_MANIFEST_VERSION:
            raise ValueError(f"Unsupported tensor hash manifest version: {self.manifest_version}")
        if int(self.layout_version) < 1:
            raise ValueError("TensorHashManifest.layout_version must be >= 1.")
        nodes: dict[str, TensorHashManifestNode] = {}
        for node_id, node in dict(self.nodes or {}).items():
            key = str(node_id or "").strip()
            if not key:
                raise ValueError("TensorHashManifest node IDs must not be empty.")
            if key in nodes:
                raise ValueError(f"Duplicate TensorHashManifest node ID: {key}")
            nodes[key] = node if isinstance(node, TensorHashManifestNode) else TensorHashManifestNode.from_dict(node)
        object.__setattr__(self, "nodes", dict(sorted(nodes.items())))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        computed = self.compute_manifest_sha256()
        provided = _validate_sha256(self.analysis_manifest_sha256, field_name="TensorHashManifest.analysis_manifest_sha256", allow_empty=True)
        if provided and provided != computed:
            raise ValueError("TensorHashManifest.analysis_manifest_sha256 does not match canonical manifest content.")
        object.__setattr__(self, "analysis_manifest_sha256", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "manifest_version": int(self.manifest_version),
            "algorithm_version": self.algorithm_version,
            "provider_id": self.provider_id,
            "family_id": self.family_id,
            "component_role": self.component_role,
            "layout_version": int(self.layout_version),
            "component_sha256": self.component_sha256,
            "nodes": {key: value.to_dict() for key, value in self.nodes.items()},
            "metadata": dict(self.metadata),
        }

    def compute_manifest_sha256(self) -> str:
        return sha256(_canonical_json(self._identity_payload())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "analysis_manifest_sha256": self.analysis_manifest_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorHashManifest":
        return cls(
            manifest_version=int(payload.get("manifest_version") or 0),
            algorithm_version=str(payload.get("algorithm_version") or ""),
            provider_id=str(payload.get("provider_id") or ""),
            family_id=str(payload.get("family_id") or ""),
            component_role=str(payload.get("component_role") or ""),
            layout_version=int(payload.get("layout_version") or 0),
            component_sha256=str(payload.get("component_sha256") or ""),
            nodes={str(key): TensorHashManifestNode.from_dict(value) for key, value in dict(payload.get("nodes") or {}).items()},
            analysis_manifest_sha256=str(payload.get("analysis_manifest_sha256") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )


__all__ = [
    "ANALYSIS_LAYOUT_CONTRACT_VERSION",
    "TENSOR_HASH_ALGORITHM_VERSION",
    "TENSOR_HASH_MANIFEST_VERSION",
    "ComponentAnalysisLayout",
    "ComponentAnalysisNode",
    "TensorHashManifest",
    "TensorHashManifestNode",
]
