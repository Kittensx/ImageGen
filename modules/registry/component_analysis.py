from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

from .analysis_contracts import (
    TENSOR_HASH_ALGORITHM_VERSION,
    TensorHashManifest,
    TensorHashManifestNode,
)
from .analysis_resolution import (
    ANALYSIS_LAYER_RESOLVER_VERSION,
    RESOLUTION_RESOLVED,
    AnalysisResolutionReport,
    AnalysisTensorDescriptor,
    AnalyticalLayerResolver,
)
from .asset_registry import AssetRegistry
from .family_providers import ArchitectureFamilyProviderRegistry, DEFAULT_FAMILY_PROVIDER_REGISTRY, canonicalize_family
from .models import (
    ANALYSIS_STRENGTH_EXACT,
    ANALYSIS_STRENGTH_LAYOUT,
    ANALYSIS_STRENGTH_NONE,
    ComponentSnapshotRecord,
)


COMPONENT_ANALYSIS_ENGINE_VERSION = "component-analysis-engine-v1"


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class ComponentAnalysisResult:
    asset_id: int
    component_sha256: str
    family_id: str
    component_role: str
    analysis_strength: str
    status: str
    cache_hit: bool
    resolution: Mapping[str, Any]
    manifest: Mapping[str, Any] | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": COMPONENT_ANALYSIS_ENGINE_VERSION,
            "asset_id": int(self.asset_id),
            "component_sha256": self.component_sha256,
            "family_id": self.family_id,
            "component_role": self.component_role,
            "analysis_strength": self.analysis_strength,
            "status": self.status,
            "cache_hit": bool(self.cache_hit),
            "reason": self.reason,
            "resolution": dict(self.resolution),
            "manifest": dict(self.manifest) if self.manifest is not None else None,
        }


class ComponentAnalysisEngine:
    """Build exact, compact analytical manifests from existing component evidence.

    ML-F02 intentionally reuses ``asset_components.tensor_manifest_json``. That
    snapshot already contains exact per-tensor payload SHA-256 values gathered while
    establishing the canonical component fingerprint, so analytical hashes can be
    derived without rereading multi-gigabyte payloads when that evidence is current.
    """

    def __init__(
        self,
        registry: AssetRegistry,
        *,
        providers: ArchitectureFamilyProviderRegistry | None = None,
        resolver: AnalyticalLayerResolver | None = None,
    ) -> None:
        self.registry = registry
        self.providers = providers or DEFAULT_FAMILY_PROVIDER_REGISTRY
        self.resolver = resolver or AnalyticalLayerResolver()

    def analyze_asset(
        self,
        asset_id: int,
        *,
        analysis_strength: str = ANALYSIS_STRENGTH_LAYOUT,
        persist: bool = True,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        strength = self._normalize_strength(analysis_strength)
        if strength == ANALYSIS_STRENGTH_NONE:
            return []
        asset = self.registry.get_asset_by_id(int(asset_id))
        if asset is None:
            return [{
                "engine_version": COMPONENT_ANALYSIS_ENGINE_VERSION,
                "asset_id": int(asset_id),
                "status": "asset_not_found",
                "analysis_strength": strength,
            }]
        snapshots = self.registry.get_component_snapshots(int(asset_id))
        if not snapshots:
            return [{
                "engine_version": COMPONENT_ANALYSIS_ENGINE_VERSION,
                "asset_id": int(asset_id),
                "status": "no_component_snapshot",
                "analysis_strength": strength,
                "reason": "Run a component structural/full refresh before analytical hashing.",
            }]
        results = []
        for snapshot in snapshots:
            family = self._family_for_snapshot(asset_id=int(asset_id), snapshot=snapshot, fallback=asset.architecture)
            results.append(
                self.analyze_snapshot(
                    snapshot,
                    family=family,
                    analysis_strength=strength,
                    persist=persist,
                    force=force,
                ).to_dict()
            )
        return results

    def analyze_assets(
        self,
        asset_ids: Iterable[int],
        *,
        analysis_strength: str = ANALYSIS_STRENGTH_LAYOUT,
        persist: bool = True,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for asset_id in sorted({int(value) for value in asset_ids}):
            rows.extend(self.analyze_asset(asset_id, analysis_strength=analysis_strength, persist=persist, force=force))
        return rows

    def analyze_snapshot(
        self,
        snapshot: ComponentSnapshotRecord,
        *,
        family: str,
        analysis_strength: str = ANALYSIS_STRENGTH_EXACT,
        persist: bool = True,
        force: bool = False,
    ) -> ComponentAnalysisResult:
        strength = self._normalize_strength(analysis_strength)
        canonical_family = canonicalize_family(family, providers=self.providers.providers())
        provider = self.providers.get(canonical_family)
        if provider is None:
            return ComponentAnalysisResult(
                asset_id=snapshot.asset_id,
                component_sha256=snapshot.component_sha256,
                family_id=canonical_family or str(family or ""),
                component_role=snapshot.component_role,
                analysis_strength=strength,
                status="provider_unresolved",
                cache_hit=False,
                resolution={},
                reason="No architecture provider is supported by the available structural evidence.",
            )
        layout = provider.describe_analysis_layout(snapshot.component_role)
        if layout is None:
            return ComponentAnalysisResult(
                asset_id=snapshot.asset_id,
                component_sha256=snapshot.component_sha256,
                family_id=provider.family_id,
                component_role=snapshot.component_role,
                analysis_strength=strength,
                status="analysis_layout_not_supported",
                cache_hit=False,
                resolution={},
                reason="Provider does not expose an analytical layout for this component role.",
            )

        descriptors = self._snapshot_descriptors(snapshot)
        resolution = self.resolver.resolve(layout=layout, tensors=descriptors)
        resolution_payload = resolution.to_dict()
        if strength == ANALYSIS_STRENGTH_LAYOUT:
            return ComponentAnalysisResult(
                asset_id=snapshot.asset_id,
                component_sha256=snapshot.component_sha256,
                family_id=provider.family_id,
                component_role=snapshot.component_role,
                analysis_strength=strength,
                status="resolved" if resolution.complete else "resolution_incomplete",
                cache_hit=False,
                resolution=resolution_payload,
                reason="" if resolution.complete else "One or more tensors could not be promoted to a resolved analytical node.",
            )

        if not resolution.complete:
            return ComponentAnalysisResult(
                asset_id=snapshot.asset_id,
                component_sha256=snapshot.component_sha256,
                family_id=provider.family_id,
                component_role=snapshot.component_role,
                analysis_strength=strength,
                status="exact_hash_blocked_by_resolution",
                cache_hit=False,
                resolution=resolution_payload,
                reason="Exact analytical hashing requires every tensor to have a resolved analytical assignment.",
            )

        if not force:
            cached = self.registry.get_component_analysis_manifest(
                component_sha256=snapshot.component_sha256,
                provider_id=provider.family_id,
                component_role=snapshot.component_role,
                layout_version=layout.layout_version,
                algorithm_version=TENSOR_HASH_ALGORITHM_VERSION,
            )
            if cached is not None:
                try:
                    cached_manifest = TensorHashManifest.from_dict(json.loads(cached["manifest_json"]))
                    return ComponentAnalysisResult(
                        asset_id=snapshot.asset_id,
                        component_sha256=snapshot.component_sha256,
                        family_id=provider.family_id,
                        component_role=snapshot.component_role,
                        analysis_strength=strength,
                        status="exact_manifest_cached",
                        cache_hit=True,
                        resolution=resolution_payload,
                        manifest=cached_manifest.to_dict(),
                    )
                except Exception:
                    # Corrupt/stale cache evidence is replaced deterministically below.
                    pass

        manifest = self._build_exact_manifest(
            snapshot=snapshot,
            family_id=provider.family_id,
            layout_version=layout.layout_version,
            descriptors=descriptors,
            resolution=resolution,
        )
        if persist:
            self.registry.store_component_analysis_manifest(manifest)
        return ComponentAnalysisResult(
            asset_id=snapshot.asset_id,
            component_sha256=snapshot.component_sha256,
            family_id=provider.family_id,
            component_role=snapshot.component_role,
            analysis_strength=strength,
            status="exact_manifest_created",
            cache_hit=False,
            resolution=resolution_payload,
            manifest=manifest.to_dict(),
        )

    def _build_exact_manifest(
        self,
        *,
        snapshot: ComponentSnapshotRecord,
        family_id: str,
        layout_version: int,
        descriptors: tuple[AnalysisTensorDescriptor, ...],
        resolution: AnalysisResolutionReport,
    ) -> TensorHashManifest:
        by_key = {item.key: item for item in descriptors}
        nodes: dict[str, TensorHashManifestNode] = {}
        for assignment in resolution.assignments:
            if assignment.status != RESOLUTION_RESOLVED:
                continue
            node_tensors = [by_key[name] for name in assignment.tensor_names]
            tensor_hashes = {
                item.key: self.tensor_exact_hash(item)
                for item in sorted(node_tensors, key=lambda value: value.key)
            }
            node_hash = self.node_exact_hash(tensor_hashes)
            nodes[assignment.node_id] = TensorHashManifestNode(
                exact_hash=node_hash,
                tensor_count=len(node_tensors),
                byte_count=sum(item.byte_count for item in node_tensors),
                tensors=tensor_hashes,
                metadata={
                    "assignment_source": assignment.assignment_source,
                    "evidence_level": assignment.evidence_level,
                    "structural_variant_id": assignment.structural_variant_id,
                    "namespace_root": assignment.namespace_root,
                },
            )
        return TensorHashManifest(
            provider_id=family_id,
            family_id=family_id,
            component_role=snapshot.component_role,
            layout_version=int(layout_version),
            component_sha256=snapshot.component_sha256,
            nodes=nodes,
            metadata={
                "engine_version": COMPONENT_ANALYSIS_ENGINE_VERSION,
                "resolver_version": ANALYSIS_LAYER_RESOLVER_VERSION,
                "component_snapshot_version": snapshot.snapshot_version,
                "component_structure_sha256": snapshot.structure_sha256,
                "source_tensor_manifest_reused": True,
                "resolution_complete": resolution.complete,
            },
        )

    @staticmethod
    def tensor_exact_hash(tensor: AnalysisTensorDescriptor) -> str:
        """Hash canonical tensor identity plus the exact payload SHA-256 chain."""
        return _sha256_payload({
            "algorithm_version": TENSOR_HASH_ALGORITHM_VERSION,
            "key": tensor.key,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "byte_count": tensor.byte_count,
            "payload_sha256": tensor.payload_sha256,
        })

    @staticmethod
    def node_exact_hash(tensor_hashes: Mapping[str, str]) -> str:
        """Hash ordered tensor identities/content; node display labels are excluded."""
        return _sha256_payload({
            "algorithm_version": TENSOR_HASH_ALGORITHM_VERSION,
            "tensors": [
                {"key": key, "tensor_sha256": digest}
                for key, digest in sorted((str(key), str(value)) for key, value in tensor_hashes.items())
            ],
        })

    @staticmethod
    def _snapshot_descriptors(snapshot: ComponentSnapshotRecord) -> tuple[AnalysisTensorDescriptor, ...]:
        try:
            rows = json.loads(snapshot.tensor_manifest_json or "[]")
        except Exception as exc:
            raise ValueError(f"Invalid component tensor manifest JSON for component {snapshot.component_sha256}: {exc}") from exc
        if not isinstance(rows, list):
            raise ValueError("Component tensor manifest must be a list.")
        descriptors = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            descriptors.append(
                AnalysisTensorDescriptor(
                    key=str(row.get("key") or ""),
                    dtype=str(row.get("dtype") or ""),
                    shape=tuple(int(value) for value in (row.get("shape") or ())),
                    byte_count=int(row.get("byte_count") or 0),
                    payload_sha256=str(row.get("payload_sha256") or ""),
                )
            )
        return tuple(sorted(descriptors, key=lambda item: item.key))

    def _family_for_snapshot(self, *, asset_id: int, snapshot: ComponentSnapshotRecord, fallback: str) -> str:
        sources = self.registry.list_component_sources(
            component_sha256=snapshot.component_sha256,
            asset_id=asset_id,
            role=snapshot.component_role,
            limit=100,
        )
        families = {
            canonicalize_family(item.provider_family or "", providers=self.providers.providers())
            for item in sources
        }
        families.discard("")
        if len(families) == 1:
            return next(iter(families))
        return canonicalize_family(fallback, providers=self.providers.providers()) or str(fallback or "")

    @staticmethod
    def _normalize_strength(value: str) -> str:
        strength = str(value or ANALYSIS_STRENGTH_NONE).strip().lower()
        if strength not in {ANALYSIS_STRENGTH_NONE, ANALYSIS_STRENGTH_LAYOUT, ANALYSIS_STRENGTH_EXACT}:
            raise ValueError(f"Unsupported analytical strength: {value!r}")
        return strength


__all__ = [
    "COMPONENT_ANALYSIS_ENGINE_VERSION",
    "ComponentAnalysisEngine",
    "ComponentAnalysisResult",
]
