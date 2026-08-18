from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from .analysis_contracts import ComponentAnalysisLayout


ANALYSIS_LAYER_RESOLVER_VERSION = "analysis-layer-resolver-v1"
STRUCTURAL_VARIANT_VERSION = "analysis-structural-variant-v1"

RESOLUTION_RESOLVED = "resolved"
RESOLUTION_PROPOSED = "proposed"
RESOLUTION_AMBIGUOUS = "ambiguous"
RESOLUTION_UNRESOLVED = "unresolved"
RESOLUTION_CONTRADICTION = "contradiction"

EVIDENCE_EXPLICIT_PROVIDER_RULE = "explicit_provider_rule"
EVIDENCE_STRUCTURALLY_CONFIRMED = "structurally_confirmed"
EVIDENCE_CROSS_MODEL_CONFIRMED = "cross_model_confirmed"

ASSIGNMENT_EXPLICIT_PROVIDER_RULE = "explicit_provider_rule"
ASSIGNMENT_AUTOMATIC_STRUCTURAL_VARIANT = "automatic_structural_variant"

_SAFE_NODE_RE = re.compile(r"^[A-Za-z0-9_:-]+$")


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class AnalysisTensorDescriptor:
    key: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    payload_sha256: str = ""

    def __post_init__(self) -> None:
        key = str(self.key or "").strip()
        if not key:
            raise ValueError("AnalysisTensorDescriptor.key must not be empty.")
        if int(self.byte_count) < 0:
            raise ValueError("AnalysisTensorDescriptor.byte_count must be non-negative.")
        digest = str(self.payload_sha256 or "").strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("AnalysisTensorDescriptor.payload_sha256 must be empty or a SHA-256 digest.")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "dtype", str(self.dtype or ""))
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "byte_count", int(self.byte_count))
        object.__setattr__(self, "payload_sha256", digest)

    def structural_dict(self, *, relative_key: str | None = None) -> dict[str, Any]:
        return {
            "key": str(self.key if relative_key is None else relative_key),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "byte_count": self.byte_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.structural_dict(),
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class AnalysisNodeAssignment:
    node_id: str
    status: str
    evidence_level: str
    assignment_source: str
    structural_variant_id: str
    tensor_names: tuple[str, ...]
    namespace_root: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "assignment_source": self.assignment_source,
            "structural_variant_id": self.structural_variant_id,
            "tensor_names": list(self.tensor_names),
            "namespace_root": self.namespace_root,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class AnalysisResolutionReport:
    provider_id: str
    family_id: str
    component_role: str
    layout_version: int
    resolver_version: str
    assignments: tuple[AnalysisNodeAssignment, ...]
    unmapped_tensor_names: tuple[str, ...] = ()

    @property
    def resolved_tensor_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for item in self.assignments:
            if item.status == RESOLUTION_RESOLVED:
                names.update(item.tensor_names)
        return tuple(sorted(names))

    @property
    def complete(self) -> bool:
        return not self.unmapped_tensor_names and all(item.status == RESOLUTION_RESOLVED for item in self.assignments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "family_id": self.family_id,
            "component_role": self.component_role,
            "layout_version": int(self.layout_version),
            "resolver_version": self.resolver_version,
            "complete": self.complete,
            "resolved_tensor_count": len(self.resolved_tensor_names),
            "unmapped_tensor_names": list(self.unmapped_tensor_names),
            "assignments": [item.to_dict() for item in self.assignments],
        }


class AnalyticalLayerResolver:
    """Resolve provider-declared and newly discovered analytical nodes.

    Explicit provider rules always win. Tensors not covered by those rules may be
    grouped by a provider-enabled top-level namespace strategy. Each discovered
    group receives a deterministic structural-variant ID based on names relative to
    the namespace root, shapes, dtypes, and byte counts. Payload values are excluded
    from structural variant identity.

    A singleton group is not treated as suspicious merely because only one model has
    been observed. It is a valid structural variant when the namespace assignment is
    internally unambiguous. Cross-model confirmation is reported separately by
    ``summarize_variant_observations``.
    """

    def resolve(
        self,
        *,
        layout: ComponentAnalysisLayout,
        tensors: Sequence[AnalysisTensorDescriptor],
    ) -> AnalysisResolutionReport:
        ordered = tuple(sorted(tensors, key=lambda item: item.key))
        by_key = {item.key: item for item in ordered}
        explicit = layout.resolve_tensor_names([item.key for item in ordered])
        claimed: set[str] = set()
        assignments: list[AnalysisNodeAssignment] = []

        for node in layout.nodes:
            names = tuple(explicit.get(node.node_id, ()))
            if not names:
                continue
            claimed.update(names)
            descriptors = tuple(by_key[name] for name in names)
            variant_id = self.structural_variant_id(node.node_id, descriptors)
            assignments.append(
                AnalysisNodeAssignment(
                    node_id=node.node_id,
                    status=RESOLUTION_RESOLVED,
                    evidence_level=EVIDENCE_EXPLICIT_PROVIDER_RULE,
                    assignment_source=ASSIGNMENT_EXPLICIT_PROVIDER_RULE,
                    structural_variant_id=variant_id,
                    tensor_names=names,
                    namespace_root=self._common_namespace_root(names),
                    evidence={
                        "resolver_version": ANALYSIS_LAYER_RESOLVER_VERSION,
                        "provider_rule": True,
                        "node_kind": node.node_kind,
                        "tensor_prefixes": list(node.tensor_prefixes),
                        "tensor_names": list(node.tensor_names),
                        "provider_node_metadata": dict(node.metadata),
                        "structural_schema_sha256": variant_id,
                    },
                )
            )

        unmatched = tuple(item for item in ordered if item.key not in claimed)
        auto_policy = dict(layout.grouping_rules.get("auto_resolution") or {})
        auto_enabled = bool(auto_policy.get("enabled", False))
        if auto_enabled and unmatched:
            groups: dict[str, list[AnalysisTensorDescriptor]] = {}
            for tensor in unmatched:
                root = self._namespace_root(tensor.key)
                groups.setdefault(root, []).append(tensor)

            for root, values in sorted(groups.items()):
                descriptors = tuple(sorted(values, key=lambda item: item.key))
                safe_node = bool(root and _SAFE_NODE_RE.fullmatch(root))
                collides = root in set(layout.node_ids())
                status = RESOLUTION_RESOLVED if safe_node and not collides else RESOLUTION_PROPOSED
                evidence_level = EVIDENCE_STRUCTURALLY_CONFIRMED if status == RESOLUTION_RESOLVED else "namespace_candidate"
                variant_id = self.structural_variant_id(root or "unresolved", descriptors)
                assignments.append(
                    AnalysisNodeAssignment(
                        node_id=root or "unresolved",
                        status=status,
                        evidence_level=evidence_level,
                        assignment_source=ASSIGNMENT_AUTOMATIC_STRUCTURAL_VARIANT,
                        structural_variant_id=variant_id,
                        tensor_names=tuple(item.key for item in descriptors),
                        namespace_root=root,
                        evidence={
                            "resolver_version": ANALYSIS_LAYER_RESOLVER_VERSION,
                            "strategy": "top_level_namespace_structural_variant",
                            "namespace_root": root,
                            "safe_node_token": safe_node,
                            "explicit_node_collision": collides,
                            "singleton_variant_allowed": bool(auto_policy.get("allow_singleton_variant", True)),
                            "structural_schema": [
                                item.structural_dict(relative_key=self._relative_to_root(item.key, root))
                                for item in descriptors
                            ],
                            "structural_schema_sha256": variant_id,
                            "payload_values_used_for_variant_identity": False,
                        },
                    )
                )

        resolved_names = {
            name
            for assignment in assignments
            if assignment.status == RESOLUTION_RESOLVED
            for name in assignment.tensor_names
        }
        unmapped = tuple(sorted(item.key for item in ordered if item.key not in resolved_names))
        return AnalysisResolutionReport(
            provider_id=layout.provider_id,
            family_id=layout.family_id,
            component_role=layout.component_role,
            layout_version=layout.layout_version,
            resolver_version=ANALYSIS_LAYER_RESOLVER_VERSION,
            assignments=tuple(sorted(assignments, key=lambda item: (item.node_id, item.structural_variant_id))),
            unmapped_tensor_names=unmapped,
        )

    @staticmethod
    def _namespace_root(key: str) -> str:
        value = str(key or "").strip()
        if not value:
            return ""
        return value.split(".", 1)[0]

    @staticmethod
    def _common_namespace_root(names: Iterable[str]) -> str:
        roots = {AnalyticalLayerResolver._namespace_root(name) for name in names}
        return next(iter(roots)) if len(roots) == 1 else ""

    @staticmethod
    def _relative_to_root(key: str, root: str) -> str:
        if key == root:
            return "@self"
        prefix = f"{root}."
        if root and key.startswith(prefix):
            return key[len(prefix):]
        return key

    @classmethod
    def structural_variant_id(
        cls,
        node_id: str,
        tensors: Sequence[AnalysisTensorDescriptor],
    ) -> str:
        root = cls._common_namespace_root(item.key for item in tensors)
        # The assigned node ID is intentionally excluded. The variant describes
        # structural evidence, so equivalent structure assigned to two different
        # nodes remains detectable as a true assignment contradiction.
        payload = {
            "version": STRUCTURAL_VARIANT_VERSION,
            "namespace_root": root,
            "tensors": [
                item.structural_dict(relative_key=cls._relative_to_root(item.key, root))
                for item in sorted(tensors, key=lambda value: value.key)
            ],
        }
        return _sha256_payload(payload)


def summarize_variant_observations(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize real observations without treating rare variants as failures.

    A contradiction is only reported when the same family/role/namespace/structural
    variant is assigned to different node IDs. Different structural-variant IDs are
    valid variants, regardless of how many checkpoints contain them.
    """

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    signature_assignments: dict[tuple[str, str, str, str], set[str]] = {}
    for raw in observations:
        family = str(raw.get("family_id") or raw.get("family") or "")
        role = str(raw.get("component_role") or raw.get("role") or "")
        node_id = str(raw.get("node_id") or "")
        namespace_root = str(raw.get("namespace_root") or node_id)
        variant_id = str(raw.get("structural_variant_id") or "")
        component_sha = str(raw.get("component_sha256") or "")
        if not node_id or not variant_id:
            continue
        key = (family, role, node_id)
        group = groups.setdefault(key, {
            "family_id": family,
            "component_role": role,
            "node_id": node_id,
            "variants": {},
        })
        variant = group["variants"].setdefault(variant_id, {
            "structural_variant_id": variant_id,
            "observation_count": 0,
            "component_sha256": [],
            "assignment_sources": set(),
            "evidence_levels": set(),
        })
        variant["observation_count"] += 1
        if component_sha and component_sha not in variant["component_sha256"]:
            variant["component_sha256"].append(component_sha)
        variant["assignment_sources"].add(str(raw.get("assignment_source") or ""))
        variant["evidence_levels"].add(str(raw.get("evidence_level") or ""))
        signature_assignments.setdefault((family, role, namespace_root, variant_id), set()).add(node_id)

    normalized_groups: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        variants = []
        for variant_id, value in sorted(group["variants"].items()):
            count = int(value["observation_count"])
            variants.append({
                "structural_variant_id": variant_id,
                "observation_count": count,
                "component_sha256": sorted(value["component_sha256"]),
                "assignment_sources": sorted(item for item in value["assignment_sources"] if item),
                "evidence_levels": sorted(item for item in value["evidence_levels"] if item),
                "cross_model_confirmed": count > 1,
                "rare_variant": count == 1,
            })
        normalized_groups.append({
            "family_id": group["family_id"],
            "component_role": group["component_role"],
            "node_id": group["node_id"],
            "variant_count": len(variants),
            "variants": variants,
        })

    contradictions = []
    for (family, role, namespace_root, variant_id), node_ids in sorted(signature_assignments.items()):
        if len(node_ids) > 1:
            contradictions.append({
                "family_id": family,
                "component_role": role,
                "namespace_root": namespace_root,
                "structural_variant_id": variant_id,
                "node_ids": sorted(node_ids),
                "status": RESOLUTION_CONTRADICTION,
            })

    return {
        "resolver_version": ANALYSIS_LAYER_RESOLVER_VERSION,
        "node_groups": normalized_groups,
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
    }


__all__ = [
    "ANALYSIS_LAYER_RESOLVER_VERSION",
    "STRUCTURAL_VARIANT_VERSION",
    "ASSIGNMENT_AUTOMATIC_STRUCTURAL_VARIANT",
    "ASSIGNMENT_EXPLICIT_PROVIDER_RULE",
    "EVIDENCE_CROSS_MODEL_CONFIRMED",
    "EVIDENCE_EXPLICIT_PROVIDER_RULE",
    "EVIDENCE_STRUCTURALLY_CONFIRMED",
    "RESOLUTION_AMBIGUOUS",
    "RESOLUTION_CONTRADICTION",
    "RESOLUTION_PROPOSED",
    "RESOLUTION_RESOLVED",
    "RESOLUTION_UNRESOLVED",
    "AnalysisNodeAssignment",
    "AnalysisResolutionReport",
    "AnalysisTensorDescriptor",
    "AnalyticalLayerResolver",
    "summarize_variant_observations",
]
