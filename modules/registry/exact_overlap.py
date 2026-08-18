from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from itertools import combinations
from typing import Any, Iterable, Mapping

from .analysis_contracts import TensorHashManifest
from .asset_registry import AssetRegistry


EXACT_OVERLAP_ENGINE_VERSION = "model-lab-exact-overlap-v1"
EXACT_OVERLAP_EVIDENCE_KIND = "exact_analysis_manifest_comparison"
RELATIONSHIP_SHARED_ANALYSIS_LAYOUT = "shared_analysis_layout"
RELATIONSHIP_EXACT_NODE_OVERLAP = "exact_node_overlap"
RELATIONSHIP_EXACT_TENSOR_OVERLAP = "exact_tensor_overlap"
RELATIONSHIP_SUBSET_EXACT_MATCH = "subset_exact_match"
EXACT_OVERLAP_RELATIONSHIP_TYPES = (
    RELATIONSHIP_SHARED_ANALYSIS_LAYOUT,
    RELATIONSHIP_EXACT_NODE_OVERLAP,
    RELATIONSHIP_EXACT_TENSOR_OVERLAP,
    RELATIONSHIP_SUBSET_EXACT_MATCH,
)


@dataclass(frozen=True)
class ExactOverlapComparison:
    left_component_sha256: str
    right_component_sha256: str
    status: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": EXACT_OVERLAP_ENGINE_VERSION,
            "left_component_sha256": self.left_component_sha256,
            "right_component_sha256": self.right_component_sha256,
            "status": self.status,
            "evidence": dict(self.evidence),
        }


class ExactOverlapService:
    """Compare ML-F02 manifests and persist exact, descriptive overlap evidence.

    ML-F03 relationships are keyed only by immutable component fingerprints. They do
    not infer runtime compatibility, model quality, ancestry, fine-tuning, or merge
    provenance. File paths and source availability are deliberately absent from exact
    relationship identity so moves/missing occurrences do not rewrite historical
    content evidence.
    """

    def __init__(self, registry: AssetRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _compatibility_key(manifest: TensorHashManifest) -> tuple[str, str, str, int, str]:
        return (
            manifest.provider_id,
            manifest.family_id,
            manifest.component_role,
            int(manifest.layout_version),
            manifest.algorithm_version,
        )

    @staticmethod
    def _evidence_version(manifest: TensorHashManifest) -> str:
        return (
            f"{EXACT_OVERLAP_ENGINE_VERSION}|provider={manifest.provider_id}"
            f"|role={manifest.component_role}|layout={int(manifest.layout_version)}"
            f"|algorithm={manifest.algorithm_version}"
        )

    @staticmethod
    def _load_manifest(record: Mapping[str, Any]) -> TensorHashManifest | None:
        try:
            payload = json.loads(str(record.get("manifest_json") or "{}"))
            if not isinstance(payload, dict):
                return None
            return TensorHashManifest.from_dict(payload)
        except Exception:
            return None

    def _latest_manifests(
        self,
        *,
        component_sha256s: Iterable[str] | None = None,
    ) -> list[TensorHashManifest]:
        wanted = {
            str(value or "").strip().lower()
            for value in (component_sha256s or ())
            if str(value or "").strip()
        }
        rows = self.registry.list_component_analysis_manifests(limit=1_000_000)
        seen: set[tuple[str, str, str, int, str]] = set()
        manifests: list[TensorHashManifest] = []
        for row in rows:
            component_sha = str(row.get("component_sha256") or "").strip().lower()
            if wanted and component_sha not in wanted:
                continue
            manifest = self._load_manifest(row)
            if manifest is None:
                continue
            key = (
                manifest.component_sha256,
                manifest.provider_id,
                manifest.component_role,
                int(manifest.layout_version),
                manifest.algorithm_version,
            )
            if key in seen:
                continue
            seen.add(key)
            manifests.append(manifest)
        return manifests

    @classmethod
    def _select_best_compatible_pair(
        cls,
        left: Iterable[TensorHashManifest],
        right: Iterable[TensorHashManifest],
    ) -> tuple[TensorHashManifest, TensorHashManifest] | None:
        candidates: list[tuple[TensorHashManifest, TensorHashManifest]] = []
        for left_manifest in left:
            for right_manifest in right:
                if cls._compatibility_key(left_manifest) == cls._compatibility_key(right_manifest):
                    candidates.append((left_manifest, right_manifest))
        if not candidates:
            return None
        candidates.sort(
            key=lambda pair: (
                int(pair[0].layout_version),
                pair[0].algorithm_version,
                pair[0].analysis_manifest_sha256,
                pair[1].analysis_manifest_sha256,
            ),
            reverse=True,
        )
        return candidates[0]

    def _best_compatible_pair(
        self,
        left_component_sha256: str,
        right_component_sha256: str,
    ) -> tuple[TensorHashManifest, TensorHashManifest] | None:
        left_sha = str(left_component_sha256 or "").strip().lower()
        right_sha = str(right_component_sha256 or "").strip().lower()
        manifests = self._latest_manifests(component_sha256s=(left_sha, right_sha))
        return self._select_best_compatible_pair(
            (item for item in manifests if item.component_sha256 == left_sha),
            (item for item in manifests if item.component_sha256 == right_sha),
        )

    @staticmethod
    def _tensor_index(manifest: TensorHashManifest) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for node_id, node in manifest.nodes.items():
            for tensor_name, tensor_hash in node.tensors.items():
                result[tensor_name] = {
                    "tensor_sha256": tensor_hash,
                    "node_id": node_id,
                }
        return result

    @classmethod
    def compare_manifests(
        cls,
        left: TensorHashManifest,
        right: TensorHashManifest,
        *,
        include_tensor_evidence: bool = False,
    ) -> ExactOverlapComparison:
        left_sha, right_sha = sorted((left.component_sha256, right.component_sha256))
        if left.component_sha256 == right.component_sha256:
            return ExactOverlapComparison(
                left_component_sha256=left_sha,
                right_component_sha256=right_sha,
                status="identical_component_identity",
                evidence={
                    "component_sha256": left.component_sha256,
                    "redundant_pairwise_relationship": False,
                    "reason": "Exact identical components collapse to one canonical component identity.",
                },
            )
        if cls._compatibility_key(left) != cls._compatibility_key(right):
            return ExactOverlapComparison(
                left_component_sha256=left_sha,
                right_component_sha256=right_sha,
                status="incompatible_analysis_layout",
                evidence={
                    "left_layout": {
                        "provider_id": left.provider_id,
                        "family_id": left.family_id,
                        "component_role": left.component_role,
                        "layout_version": int(left.layout_version),
                        "algorithm_version": left.algorithm_version,
                    },
                    "right_layout": {
                        "provider_id": right.provider_id,
                        "family_id": right.family_id,
                        "component_role": right.component_role,
                        "layout_version": int(right.layout_version),
                        "algorithm_version": right.algorithm_version,
                    },
                    "compatibility_claim": False,
                    "ancestry_claim": False,
                },
            )

        # Normalize left/right ordering so evidence is deterministic for the same pair.
        if left.component_sha256 != left_sha:
            left, right = right, left

        left_nodes = left.nodes
        right_nodes = right.nodes
        node_ids = sorted(set(left_nodes) | set(right_nodes))
        matching_node_ids: list[str] = []
        node_evidence: list[dict[str, Any]] = []
        for node_id in node_ids:
            left_node = left_nodes.get(node_id)
            right_node = right_nodes.get(node_id)
            exact_match = bool(
                left_node is not None
                and right_node is not None
                and left_node.exact_hash == right_node.exact_hash
            )
            if exact_match:
                matching_node_ids.append(node_id)
            node_evidence.append({
                "node_id": node_id,
                "left_present": left_node is not None,
                "right_present": right_node is not None,
                "left_exact_sha256": left_node.exact_hash if left_node is not None else "",
                "right_exact_sha256": right_node.exact_hash if right_node is not None else "",
                "left_structural_variant_id": str((left_node.metadata if left_node else {}).get("structural_variant_id") or ""),
                "right_structural_variant_id": str((right_node.metadata if right_node else {}).get("structural_variant_id") or ""),
                "exact_match": exact_match,
            })

        left_tensors = cls._tensor_index(left)
        right_tensors = cls._tensor_index(right)
        tensor_names = sorted(set(left_tensors) | set(right_tensors))
        matching_tensor_names: list[str] = []
        tensor_evidence: list[dict[str, Any]] = []
        for tensor_name in tensor_names:
            left_tensor = left_tensors.get(tensor_name)
            right_tensor = right_tensors.get(tensor_name)
            exact_match = bool(
                left_tensor is not None
                and right_tensor is not None
                and left_tensor["tensor_sha256"] == right_tensor["tensor_sha256"]
            )
            if exact_match:
                matching_tensor_names.append(tensor_name)
            if include_tensor_evidence:
                tensor_evidence.append({
                    "tensor_name": tensor_name,
                    "left_present": left_tensor is not None,
                    "right_present": right_tensor is not None,
                    "left_node_id": str((left_tensor or {}).get("node_id") or ""),
                    "right_node_id": str((right_tensor or {}).get("node_id") or ""),
                    "left_exact_sha256": str((left_tensor or {}).get("tensor_sha256") or ""),
                    "right_exact_sha256": str((right_tensor or {}).get("tensor_sha256") or ""),
                    "exact_match": exact_match,
                })

        comparable_nodes = len(node_ids)
        comparable_tensors = len(tensor_names)
        matching_nodes = len(matching_node_ids)
        matching_tensors = len(matching_tensor_names)
        left_tensor_pairs = {(key, value["tensor_sha256"]) for key, value in left_tensors.items()}
        right_tensor_pairs = {(key, value["tensor_sha256"]) for key, value in right_tensors.items()}
        left_subset_right = bool(left_tensor_pairs and left_tensor_pairs < right_tensor_pairs)
        right_subset_left = bool(right_tensor_pairs and right_tensor_pairs < left_tensor_pairs)

        evidence: dict[str, Any] = {
            "evidence_kind": EXACT_OVERLAP_EVIDENCE_KIND,
            "evidence_version": cls._evidence_version(left),
            "provider_id": left.provider_id,
            "family_id": left.family_id,
            "component_role": left.component_role,
            "layout_version": int(left.layout_version),
            "algorithm_version": left.algorithm_version,
            "left_manifest_sha256": left.analysis_manifest_sha256,
            "right_manifest_sha256": right.analysis_manifest_sha256,
            "left_node_count": len(left_nodes),
            "right_node_count": len(right_nodes),
            "comparable_nodes": comparable_nodes,
            "matching_nodes": matching_nodes,
            "matching_node_ratio": (matching_nodes / comparable_nodes) if comparable_nodes else 0.0,
            "matching_node_ids": matching_node_ids,
            "left_tensor_count": len(left_tensors),
            "right_tensor_count": len(right_tensors),
            "comparable_tensors": comparable_tensors,
            "matching_tensors": matching_tensors,
            "matching_tensor_ratio": (matching_tensors / comparable_tensors) if comparable_tensors else 0.0,
            "matching_tensor_names_materialized": bool(include_tensor_evidence),
            "matching_tensor_names": matching_tensor_names if include_tensor_evidence else [],
            "node_evidence": node_evidence,
            "tensor_evidence": tensor_evidence,
            "left_is_strict_exact_subset": left_subset_right,
            "right_is_strict_exact_subset": right_subset_left,
            "compatibility_claim": False,
            "ancestry_claim": False,
            "fine_tune_claim": False,
            "merge_claim": False,
            "semantics": "Exact descriptive overlap only; ratios are not compatibility, quality, or ancestry scores.",
        }
        return ExactOverlapComparison(
            left_component_sha256=left.component_sha256,
            right_component_sha256=right.component_sha256,
            status="comparable",
            evidence=evidence,
        )

    def compare_components(
        self,
        left_component_sha256: str,
        right_component_sha256: str,
        *,
        include_tensor_evidence: bool = False,
    ) -> dict[str, Any]:
        left_sha = str(left_component_sha256 or "").strip().lower()
        right_sha = str(right_component_sha256 or "").strip().lower()
        if left_sha == right_sha and left_sha:
            return ExactOverlapComparison(
                left_component_sha256=left_sha,
                right_component_sha256=right_sha,
                status="identical_component_identity",
                evidence={
                    "component_sha256": left_sha,
                    "redundant_pairwise_relationship": False,
                    "reason": "Exact identical components collapse to one canonical component identity.",
                },
            ).to_dict()
        pair = self._best_compatible_pair(left_sha, right_sha)
        if pair is None:
            return ExactOverlapComparison(
                left_component_sha256=left_sha,
                right_component_sha256=right_sha,
                status="no_compatible_manifest_pair",
                evidence={
                    "compatibility_claim": False,
                    "ancestry_claim": False,
                    "reason": "No current ML-F02 manifests share provider, family, role, layout version, and hash algorithm.",
                },
            ).to_dict()
        return self.compare_manifests(
            pair[0],
            pair[1],
            include_tensor_evidence=include_tensor_evidence,
        ).to_dict()

    def _candidate_pairs(
        self,
        manifests: Iterable[TensorHashManifest],
        *,
        include_tensor_only: bool,
        requested_components: set[str],
    ) -> set[tuple[str, str]]:
        groups: dict[tuple[str, str, str, int, str], list[TensorHashManifest]] = defaultdict(list)
        for manifest in manifests:
            groups[self._compatibility_key(manifest)].append(manifest)

        pairs: set[tuple[str, str]] = set()
        for group_manifests in groups.values():
            node_index: dict[str, set[str]] = defaultdict(set)
            tensor_index: dict[str, set[str]] = defaultdict(set)
            for manifest in group_manifests:
                for node in manifest.nodes.values():
                    node_index[node.exact_hash].add(manifest.component_sha256)
                    if include_tensor_only:
                        for tensor_hash in node.tensors.values():
                            tensor_index[tensor_hash].add(manifest.component_sha256)
            for bucket in node_index.values():
                for left, right in combinations(sorted(bucket), 2):
                    if not requested_components or left in requested_components or right in requested_components:
                        pairs.add((left, right))
            if include_tensor_only:
                for bucket in tensor_index.values():
                    for left, right in combinations(sorted(bucket), 2):
                        if not requested_components or left in requested_components or right in requested_components:
                            pairs.add((left, right))
        return pairs

    def refresh_relationships(
        self,
        *,
        component_sha256s: Iterable[str] | None = None,
        persist: bool = True,
        min_matching_nodes: int = 1,
        include_tensor_only: bool = False,
        min_matching_tensors: int = 1,
    ) -> dict[str, Any]:
        requested = {
            str(value or "").strip().lower()
            for value in (component_sha256s or ())
            if str(value or "").strip()
        }
        manifests = self._latest_manifests()
        by_component: dict[str, list[TensorHashManifest]] = defaultdict(list)
        for manifest in manifests:
            by_component[manifest.component_sha256].append(manifest)

        candidates = sorted(self._candidate_pairs(
            manifests,
            include_tensor_only=include_tensor_only,
            requested_components=requested,
        ))
        comparisons: list[dict[str, Any]] = []
        stored_records = 0
        stored_pairs = 0

        for left_sha, right_sha in candidates:
            pair = self._select_best_compatible_pair(
                by_component.get(left_sha, ()),
                by_component.get(right_sha, ()),
            )
            if pair is None:
                continue
            comparison = self.compare_manifests(pair[0], pair[1], include_tensor_evidence=False)
            if comparison.status != "comparable":
                continue
            payload = comparison.to_dict()
            evidence = dict(comparison.evidence)
            matching_nodes = int(evidence.get("matching_nodes") or 0)
            matching_tensors = int(evidence.get("matching_tensors") or 0)
            qualifies_node = matching_nodes >= max(1, int(min_matching_nodes))
            qualifies_tensor_only = bool(
                include_tensor_only
                and matching_nodes < max(1, int(min_matching_nodes))
                and matching_tensors >= max(1, int(min_matching_tensors))
            )
            if not (qualifies_node or qualifies_tensor_only):
                continue
            comparisons.append(payload)
            if not persist:
                continue

            evidence_version = str(evidence["evidence_version"])
            self.registry.upsert_component_relationship(
                source_component_sha256=left_sha,
                target_component_sha256=right_sha,
                relationship_type=RELATIONSHIP_SHARED_ANALYSIS_LAYOUT,
                evidence_kind=EXACT_OVERLAP_EVIDENCE_KIND,
                evidence_version=evidence_version,
                evidence_json=evidence,
            )
            stored_records += 1
            if matching_nodes:
                self.registry.upsert_component_relationship(
                    source_component_sha256=left_sha,
                    target_component_sha256=right_sha,
                    relationship_type=RELATIONSHIP_EXACT_NODE_OVERLAP,
                    evidence_kind=EXACT_OVERLAP_EVIDENCE_KIND,
                    evidence_version=evidence_version,
                    evidence_json=evidence,
                )
                stored_records += 1
            if matching_tensors:
                self.registry.upsert_component_relationship(
                    source_component_sha256=left_sha,
                    target_component_sha256=right_sha,
                    relationship_type=RELATIONSHIP_EXACT_TENSOR_OVERLAP,
                    evidence_kind=EXACT_OVERLAP_EVIDENCE_KIND,
                    evidence_version=evidence_version,
                    evidence_json=evidence,
                )
                stored_records += 1
            if evidence.get("left_is_strict_exact_subset") or evidence.get("right_is_strict_exact_subset"):
                self.registry.upsert_component_relationship(
                    source_component_sha256=left_sha,
                    target_component_sha256=right_sha,
                    relationship_type=RELATIONSHIP_SUBSET_EXACT_MATCH,
                    evidence_kind=EXACT_OVERLAP_EVIDENCE_KIND,
                    evidence_version=evidence_version,
                    evidence_json=evidence,
                )
                stored_records += 1
            stored_pairs += 1

        return {
            "engine_version": EXACT_OVERLAP_ENGINE_VERSION,
            "manifest_count": len(manifests),
            "unique_component_count": len(by_component),
            "candidate_pair_count": len(candidates),
            "qualifying_pair_count": len(comparisons),
            "stored_pair_count": stored_pairs if persist else 0,
            "stored_record_count": stored_records if persist else 0,
            "persisted": bool(persist),
            "min_matching_nodes": max(1, int(min_matching_nodes)),
            "include_tensor_only": bool(include_tensor_only),
            "min_matching_tensors": max(1, int(min_matching_tensors)),
            "comparisons": comparisons,
        }

    def rank_candidates_by_node_overlap(
        self,
        component_sha256: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        digest = str(component_sha256 or "").strip().lower()
        rows = self.registry.list_component_relationships(
            component_sha256=digest,
            relationship_type=RELATIONSHIP_EXACT_NODE_OVERLAP,
            limit=max(1, int(limit)) * 20,
        )
        by_candidate: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                evidence = json.loads(str(row.get("evidence_json") or "{}"))
            except Exception:
                evidence = {}
            other = (
                str(row.get("target_component_sha256") or "")
                if str(row.get("source_component_sha256") or "") == digest
                else str(row.get("source_component_sha256") or "")
            )
            candidate = {
                "component_sha256": digest,
                "candidate_component_sha256": other,
                "matching_nodes": int(evidence.get("matching_nodes") or 0),
                "comparable_nodes": int(evidence.get("comparable_nodes") or 0),
                "matching_node_ratio": float(evidence.get("matching_node_ratio") or 0.0),
                "matching_tensors": int(evidence.get("matching_tensors") or 0),
                "comparable_tensors": int(evidence.get("comparable_tensors") or 0),
                "matching_tensor_ratio": float(evidence.get("matching_tensor_ratio") or 0.0),
                "provider_id": str(evidence.get("provider_id") or ""),
                "component_role": str(evidence.get("component_role") or ""),
                "layout_version": int(evidence.get("layout_version") or 0),
                "evidence_version": str(row.get("evidence_version") or ""),
                "compatibility_claim": False,
                "ancestry_claim": False,
            }
            existing = by_candidate.get(other)
            candidate_key = (
                candidate["layout_version"],
                candidate["matching_nodes"],
                candidate["matching_node_ratio"],
                candidate["matching_tensors"],
                candidate["matching_tensor_ratio"],
                candidate["evidence_version"],
            )
            existing_key = (
                existing["layout_version"],
                existing["matching_nodes"],
                existing["matching_node_ratio"],
                existing["matching_tensors"],
                existing["matching_tensor_ratio"],
                existing["evidence_version"],
            ) if existing is not None else None
            if existing is None or candidate_key > existing_key:
                by_candidate[other] = candidate
        ranked = list(by_candidate.values())
        ranked.sort(
            key=lambda item: (
                item["matching_nodes"],
                item["matching_node_ratio"],
                item["matching_tensors"],
                item["matching_tensor_ratio"],
                item["candidate_component_sha256"],
            ),
            reverse=True,
        )
        return ranked[: max(1, int(limit))]



__all__ = [
    "EXACT_OVERLAP_ENGINE_VERSION",
    "EXACT_OVERLAP_EVIDENCE_KIND",
    "EXACT_OVERLAP_RELATIONSHIP_TYPES",
    "RELATIONSHIP_SHARED_ANALYSIS_LAYOUT",
    "RELATIONSHIP_EXACT_NODE_OVERLAP",
    "RELATIONSHIP_EXACT_TENSOR_OVERLAP",
    "RELATIONSHIP_SUBSET_EXACT_MATCH",
    "ExactOverlapComparison",
    "ExactOverlapService",
]
