from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import json

from modules.checkpoint_inspector import CheckpointInspector
from modules.project_context import ProjectContext
from image_gen.runtime.lora_inspector import inspect_lora_file
from .asset_registry import AssetRegistry
from .component_refresh import ComponentRegistryRefresher
from .component_snapshot import SafetensorsComponentSnapshotter
from .models import AssetRecord, ComponentSnapshotRecord
from .architecture_observation import (
    ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED,
    normalize_architecture_identifier,
)


INVENTORY_FORMAT = "image-gen-component-inventory-v1"
SAFETENSORS_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class InventoryCandidate:
    path: Path
    asset_kind: str
    source_root: Path | None = None
    source_scope: str = "project_models"


class ComponentInventoryScanner:
    """Fingerprint reusable model components across the project-wide models library.

    Discovery starts at ``<project_root>/models`` rather than the StableDiffusion
    subtree. The scan is intentionally non-destructive. Embedded components remain
    embedded. The registry stores observations about their exact content identity;
    it does not replace checkpoint payloads with references or physically decompose
    files.
    """

    def __init__(
        self,
        context: ProjectContext,
        *,
        registry: AssetRegistry | None = None,
        inspector: CheckpointInspector | None = None,
        snapshotter: SafetensorsComponentSnapshotter | None = None,
    ) -> None:
        self.context = context
        self.registry = registry or AssetRegistry(str(Path(context.registry_db_path).resolve()))
        self.inspector = inspector or CheckpointInspector()
        self.snapshotter = snapshotter or SafetensorsComponentSnapshotter()
        self.refresher = ComponentRegistryRefresher(
            context,
            registry=self.registry,
            inspector=self.inspector,
            snapshotter=self.snapshotter,
        )

    def models_scan_root(self) -> Path:
        project_root = getattr(self.context, "project_root", None)
        if project_root is not None:
            return (Path(project_root).resolve() / "models").resolve()
        configured = Path(self.context.models_root).resolve()
        if configured.name.casefold() == "stablediffusion":
            return configured.parent.resolve()
        return configured

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _classify_relative_repository_path(path: Path, root: Path) -> str | None:
        """Classify conventional component folders inside an explicitly supplied repository.

        This is deliberately conservative. Folder names can establish an asset *role* for
        standalone VAE/text-encoder libraries or excluded add-on classes, but never establish
        content identity. Unknown layouts remain ``auto`` and are inspected before hashing.
        """
        try:
            relative = path.relative_to(root)
        except ValueError:
            return None
        parts = {part.casefold().replace("-", "_").replace(" ", "_") for part in relative.parts[:-1]}
        excluded = {
            "controlnet": "controlnet",
            "controlnets": "controlnet",
            "vae_approx": "vae_approx",
            "esrgan": "esrgan",
            "realesrgan": "realesrgan",
            "gfpgan": "gfpgan",
            "codeformer": "codeformer",
            "embeddings": "embeddings",
            "hypernetworks": "hypernetworks",
            "upscalers": "upscalers",
            "upscaler": "upscalers",
        }
        # LoRA-looking folders are still structurally inspected. Folder names
        # are hints only and never establish that a file is an adapter.
        if parts & {"lora", "loras"}:
            return "auto"
        for token, label in excluded.items():
            if token in parts:
                return f"excluded:{label}"
        if parts & {"textencoders", "text_encoders", "textencoder", "text_encoder"}:
            return "text_encoder"
        if parts & {"vae", "vaes"}:
            return "vae"
        if parts & {"checkpoint", "checkpoints", "check_points"}:
            # A conventional folder name is not classification evidence.
            return "auto"
        return None

    def _classify_discovered_path(self, path: Path, *, repository_root: Path | None = None) -> str:
        positive = (
            (Path(self.context.checkpoints_dir).resolve(), "checkpoint"),
            *self.standalone_component_roots(),
        )
        for root, kind in positive:
            if self._is_under(path, root):
                return kind

        excluded_attributes = (
            ("lora_dir", "lora"),
            ("controlnet_dir", "controlnet"),
            ("vae_approx_dir", "vae_approx"),
            ("blip_dir", "blip"),
            ("codeformer_dir", "codeformer"),
            ("esrgan_dir", "esrgan"),
            ("gfpgan_dir", "gfpgan"),
            ("realesrgan_dir", "realesrgan"),
            ("embeddings_dir", "embeddings"),
            ("hypernetworks_dir", "hypernetworks"),
        )
        for attribute, label in excluded_attributes:
            raw = getattr(self.context, attribute, None)
            if raw is not None and self._is_under(path, Path(raw).resolve()):
                return f"excluded:{label}"

        if repository_root is not None:
            repository_kind = self._classify_relative_repository_path(path, repository_root)
            if repository_kind is not None:
                return repository_kind

        # Unknown folders under models\ or a supplied repository are deliberately
        # inspected before any component assumptions are made. If they are full
        # checkpoints, normal checkpoint routing is used. Otherwise they are reported
        # as unclassified rather than mislabeled.
        return "auto"

    def standalone_component_roots(self) -> tuple[tuple[Path, str], ...]:
        """Return configured standalone component libraries used as candidate sources.

        These roots classify source intent only. They never establish architecture,
        subtype, or compatibility; those are derived from fingerprinted tensor evidence.
        """
        return (
            (Path(self.context.vae_dir).resolve(), "vae"),
            ((Path(self.context.models_root) / "TextEncoders").resolve(), "text_encoder"),
        )

    def discover(self, *, repository_roots: Iterable[str | Path] = ()) -> list[InventoryCandidate]:
        """Discover Safetensors from project ``models/`` plus supplied local repositories."""
        candidates: dict[str, InventoryCandidate] = {}

        def add(
            path: Path,
            kind: str | None = None,
            *,
            source_root: Path | None = None,
            source_scope: str = "project_models",
        ) -> None:
            try:
                resolved = path.expanduser().resolve()
            except OSError:
                return
            if not resolved.is_file() or resolved.suffix.lower() != SAFETENSORS_SUFFIX:
                return
            normalized_root = source_root.expanduser().resolve() if source_root is not None else None
            resolved_kind = kind or self._classify_discovered_path(resolved, repository_root=normalized_root)
            token = str(resolved).casefold()
            existing = candidates.get(token)
            incoming = InventoryCandidate(
                path=resolved,
                asset_kind=resolved_kind,
                source_root=normalized_root,
                source_scope=source_scope,
            )
            if existing is None or self._kind_priority(resolved_kind) > self._kind_priority(existing.asset_kind):
                candidates[token] = incoming

        scan_root = self.models_scan_root()
        if scan_root.is_dir():
            try:
                for path in scan_root.rglob("*.safetensors"):
                    add(path, source_root=scan_root, source_scope="project_models")
            except OSError:
                pass

        supplied_roots: list[Path] = []
        seen_roots: set[str] = set()
        for raw_root in repository_roots:
            repo_root = Path(raw_root).expanduser().resolve()
            token = str(repo_root).casefold()
            if token in seen_roots or repo_root == scan_root:
                continue
            seen_roots.add(token)
            supplied_roots.append(repo_root)
            if not repo_root.is_dir():
                continue
            try:
                for path in repo_root.rglob("*.safetensors"):
                    add(path, source_root=repo_root, source_scope="supplied_repository")
            except OSError:
                pass

        # Include already-registered external checkpoint/VAE/text-encoder assets when
        # they still exist. This preserves user-managed libraries outside models\.
        for asset in self.registry.list_assets(limit=1_000_000):
            if asset.extension.lower() != SAFETENSORS_SUFFIX:
                continue
            if asset.asset_type in {"checkpoint", "vae", "text_encoder"}:
                add(
                    Path(asset.path),
                    asset.asset_type,
                    source_root=None,
                    source_scope="registered_external",
                )

        return sorted(candidates.values(), key=lambda item: str(item.path).casefold())

    def scan(
        self,
        *,
        force: bool = False,
        repository_roots: Iterable[str | Path] = (),
        progress_callback: Callable[[int, int, InventoryCandidate], None] | None = None,
        strength: str = "structural",
    ) -> dict[str, Any]:
        normalized_roots: list[Path] = []
        seen_roots: set[str] = set()
        for item in repository_roots:
            root = Path(item).expanduser().resolve()
            token = str(root).casefold()
            if token in seen_roots:
                continue
            seen_roots.add(token)
            if not root.exists():
                raise FileNotFoundError(f"Supplied component-inventory repository root does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"Supplied component-inventory repository root is not a directory: {root}")
            normalized_roots.append(root)
        normalized_repository_roots = tuple(normalized_roots)
        candidates = self.discover(repository_roots=normalized_repository_roots)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        counters: Counter[str] = Counter()

        for index, candidate in enumerate(candidates, start=1):
            if progress_callback is not None:
                progress_callback(index, len(candidates), candidate)
            try:
                result = self._scan_candidate(candidate, force=force, strength=strength)
                result["ordinal"] = index
                result["total_candidates"] = len(candidates)
                results.append(result)
                counters[result["status"]] += 1
                counters[f"kind:{result['asset_kind']}"] += 1
                for role in result.get("component_roles") or []:
                    counters[f"role:{role}"] += 1
            except Exception as exc:
                errors.append(
                    {
                        "path": str(candidate.path),
                        "asset_kind": candidate.asset_kind,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                counters["error"] += 1

        self.registry.mark_missing_assets()
        reconciliation = self.registry.reconcile_asset_locations()
        match_report = self.build_match_report(repository_roots=normalized_repository_roots)
        return {
            "format": INVENTORY_FORMAT,
            "requested_strength": strength,
            "scope": {
                "models_root": str(self.models_scan_root()),
                "additional_repository_roots": [str(path) for path in normalized_repository_roots],
                "checkpoints": str(Path(self.context.checkpoints_dir).resolve()),
                "vae": str(Path(self.context.vae_dir).resolve()),
                "text_encoders": str((Path(self.context.models_root) / "TextEncoders").resolve()),
                "standalone_component_roots": [
                    {"path": str(root), "candidate_kind": kind}
                    for root, kind in self.standalone_component_roots()
                ],
                "included_extensions": [SAFETENSORS_SUFFIX],
                "excluded_asset_classes": [
                    "LoRA",
                    "ControlNet",
                    "upscalers",
                    "embeddings",
                    "hypernetworks",
                    "legacy .ckpt/.pt/.pth/.bin component decomposition",
                ],
                "destructive_changes": False,
            },
            "candidate_count": len(candidates),
            "results": results,
            "errors": errors,
            "counters": dict(sorted(counters.items())),
            "reconciliation": reconciliation,
            "matches": match_report,
        }

    def _managed_roots(self) -> tuple[tuple[Path, str], ...]:
        # Retained for compatibility with callers that inspect scanner policy.
        return ((self.models_scan_root(), "models_root"),)

    @staticmethod
    def _kind_priority(kind: str) -> int:
        if kind == "checkpoint":
            return 4
        if kind in {"vae", "text_encoder"}:
            return 3
        if kind == "auto":
            return 1
        if kind.startswith("excluded:"):
            return 0
        return 0

    @staticmethod
    def _snapshot_source_quick_fingerprint(record: ComponentSnapshotRecord) -> str:
        try:
            payload = json.loads(record.metadata_json or "{}")
        except Exception:
            return ""
        return str(payload.get("source_quick_fingerprint") or "").strip().lower()

    def _snapshots_are_current(self, asset: AssetRecord) -> bool:
        current = self.registry.get_component_snapshots(asset.id)
        quick = str(asset.quick_fingerprint or "").strip().lower()
        return bool(current and quick) and all(
            self._snapshot_source_quick_fingerprint(item) == quick for item in current
        )

    def _scan_candidate(self, candidate: InventoryCandidate, *, force: bool, strength: str = "structural") -> dict[str, Any]:
        path = candidate.path
        # Import is the authoritative first read of a newly discovered file.
        # Persist its strong whole-file identity now so later classification,
        # duplicate detection, and component reuse never hash it twice.
        asset = self.registry.register_file(str(path), compute_sha256=True)

        if candidate.asset_kind.startswith("excluded:"):
            return self._result_payload(
                candidate,
                asset,
                (),
                status="skipped_excluded_asset_class",
            )

        report = None
        if candidate.asset_kind == "auto":
            lora_report = inspect_lora_file(path, include_compatibility_hash=False)
            adapter_format = str(lora_report.get("adapter_format") or "")
            if adapter_format not in {"", "invalid", "unknown_adapter", "non_adapter_full_model"}:
                snapshot = self.snapshotter.snapshot_standalone_component(
                    path,
                    component_role="lora_adapter",
                )
                stored = self.registry.store_component_snapshots(
                    asset.id,
                    {"lora_adapter": snapshot},
                    source_file_sha256=asset.sha256,
                    source_quick_fingerprint=asset.quick_fingerprint,
                    metadata_extra={
                        "classification_basis": "adapter_tensor_structure",
                        "adapter_format": adapter_format,
                        "model_family": str(lora_report.get("detected_model_family") or ""),
                    },
                )
                self.registry.store_inspection(
                    asset.id,
                    {
                        "asset_type": "lora",
                        "format_type": "safetensors",
                        "architecture": str(lora_report.get("detected_model_family") or ""),
                        "checkpoint_kind": "adapter",
                        "key_count": int(lora_report.get("tensor_key_count") or 0),
                        "prefix_summary": {"target_scopes": list(lora_report.get("target_scopes") or [])},
                        "example_keys": [],
                        "dtype_summary": {},
                        "tensor_shape_summary": {},
                        "metadata": {
                            "classification_basis": "adapter_tensor_structure",
                            "adapter_format": adapter_format,
                            "adapter_format_evidence": list(lora_report.get("adapter_format_evidence") or []),
                            "network_type": str(lora_report.get("network_type") or ""),
                        },
                        "inspector_version": "component_inventory_lora_v1",
                    },
                )
                refreshed = self.registry.get_asset_by_id(asset.id) or asset
                return self._result_payload(
                    InventoryCandidate(path, "lora", candidate.source_root, candidate.source_scope),
                    refreshed,
                    stored,
                    status="hashed",
                    architecture=str(lora_report.get("detected_model_family") or ""),
                    checkpoint_kind="adapter",
                )
            report = self.inspector.inspect(str(path), compute_sha256=False)
            eligibility = self.refresher.checkpoint_snapshot_eligibility(path, report=report)
            if eligibility["eligible"]:
                candidate = InventoryCandidate(
                    path=path,
                    asset_kind="checkpoint",
                    source_root=candidate.source_root,
                    source_scope=candidate.source_scope,
                )
            else:
                self._store_unknown_inspection(asset.id, path, report)
                observed_asset = self.registry.get_asset_by_id(asset.id) or asset
                payload = self._result_payload(
                    candidate,
                    observed_asset,
                    (),
                    status="observed_unclassified_safetensors",
                    architecture=observed_asset.architecture,
                    checkpoint_kind=report.checkpoint_kind,
                )
                payload["observation_reason"] = f"checkpoint_kind={report.checkpoint_kind or 'unknown'}"
                payload["structural_summary"] = {
                    "key_count": report.total_keys,
                    "key_prefixes": list(report.key_prefixes),
                    "architecture_evidence": list(report.architecture_evidence),
                }
                return payload

        if candidate.asset_kind == "checkpoint":
            report = report or self.inspector.inspect(str(path), compute_sha256=False)
            eligibility = self.refresher.checkpoint_snapshot_eligibility(path, report=report)
            if not eligibility["eligible"]:
                payload = self._result_payload(
                    candidate,
                    asset,
                    (),
                    status="skipped_ineligible_checkpoint_container",
                    architecture=report.architecture,
                    checkpoint_kind=report.checkpoint_kind,
                )
                payload["checkpoint_snapshot_eligibility"] = eligibility
                return payload

        if candidate.asset_kind not in {"checkpoint", "vae", "text_encoder"}:
            raise ValueError(f"Unsupported inventory asset kind: {candidate.asset_kind!r}")

        refresh = self.refresher.ensure_path(
            path,
            explicit_kind=candidate.asset_kind,
            force=force,
            source="component_inventory",
            precomputed_report=report,
            strength=strength,
        )
        refreshed_asset = self.registry.get_asset_by_path(str(path)) or asset
        snapshots = self.registry.get_component_snapshots(refreshed_asset.id)
        refresh_status = str(refresh.get("status") or "")
        compatibility_status = "cached" if refresh_status == "cached_complete" else "hashed"
        payload = self._result_payload(
            candidate,
            refreshed_asset,
            snapshots,
            status=compatibility_status,
            architecture=(report.architecture if report is not None else None),
            checkpoint_kind=(report.checkpoint_kind if report is not None else None),
        )
        payload["registry_refresh_status"] = refresh_status
        payload["registry_refresh"] = refresh
        if candidate.asset_kind == "checkpoint" and report is not None:
            payload["checkpoint_snapshot_eligibility"] = self.refresher.checkpoint_snapshot_eligibility(
                path,
                report=report,
            )
        return payload

    def _result_payload(
        self,
        candidate: InventoryCandidate,
        asset: AssetRecord,
        snapshots: Iterable[ComponentSnapshotRecord],
        *,
        status: str,
        architecture: str | None = None,
        checkpoint_kind: str | None = None,
    ) -> dict[str, Any]:
        snapshot_list = list(snapshots)
        return {
            "path": str(candidate.path),
            "source_root": str(candidate.source_root) if candidate.source_root is not None else None,
            "source_scope": candidate.source_scope,
            "asset_id": asset.id,
            "asset_kind": candidate.asset_kind,
            "status": status,
            "architecture": normalize_architecture_identifier(architecture or asset.architecture) or None,
            "architecture_state": asset.architecture_state,
            "checkpoint_kind": checkpoint_kind or asset.checkpoint_kind,
            "quick_fingerprint": asset.quick_fingerprint,
            "component_count": len(snapshot_list),
            "component_roles": [item.component_role for item in snapshot_list],
            "components": [
                {
                    "role": item.component_role,
                    "component_sha256": item.component_sha256,
                    "structure_sha256": item.structure_sha256,
                    "tensor_count": item.tensor_count,
                    "total_bytes": item.total_bytes,
                }
                for item in snapshot_list
            ],
        }

    def _store_checkpoint_inspection(self, asset_id: int, path: Path, report) -> None:
        self.registry.store_inspection(
            asset_id,
            {
                "asset_type": "checkpoint",
                "format_type": "safetensors",
                "architecture": report.architecture,
                "checkpoint_kind": report.checkpoint_kind,
                "has_unet": report.has_unet,
                "has_vae": report.has_vae,
                "has_text_encoder": report.has_text_encoder,
                "has_text_encoder_2": report.has_sdxl_text_encoder_2 or report.has_clip_g,
                "key_count": report.total_keys,
                "prefix_summary": {"prefixes": report.key_prefixes},
                "example_keys": report.example_keys,
                "dtype_summary": dict(report.dtype_summary),
                "tensor_shape_summary": dict(report.tensor_shape_summary),
                "metadata": {
                    "inventory_scan": True,
                    "file_name": path.name,
                    "architecture_variant": report.architecture_variant,
                    "denoiser_type": report.denoiser_type,
                    "flow_matching": report.flow_matching,
                },
                "inspector_version": "component_inventory_v1",
            },
        )

    def _store_unknown_inspection(self, asset_id: int, path: Path, report) -> None:
        self.registry.store_inspection(
            asset_id,
            {
                "asset_type": "safetensors_asset",
                "format_type": "safetensors",
                "architecture": report.architecture or "",
                "checkpoint_kind": report.checkpoint_kind or "unknown",
                "has_unet": report.has_unet,
                "has_vae": report.has_vae,
                "has_text_encoder": report.has_text_encoder,
                "has_text_encoder_2": report.has_sdxl_text_encoder_2 or report.has_clip_g,
                "key_count": report.total_keys,
                "prefix_summary": {"prefixes": report.key_prefixes},
                "example_keys": report.example_keys,
                "dtype_summary": dict(report.dtype_summary),
                "tensor_shape_summary": dict(report.tensor_shape_summary),
                "metadata": {
                    "inventory_scan": True,
                    "observation_state": ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED,
                    "observation_reason": f"checkpoint_kind={report.checkpoint_kind or 'unknown'}",
                    "file_name": path.name,
                    "architecture_variant": report.architecture_variant,
                    "architecture_evidence": list(report.architecture_evidence),
                },
                "inspector_version": "component_inventory_v1",
            },
        )

    def _store_standalone_inspection(
        self,
        asset_id: int,
        *,
        asset_kind: str,
        role: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {"inventory_scan": True, "component_role": role}
        payload.update(dict(metadata or {}))
        self.registry.store_inspection(
            asset_id,
            {
                "asset_type": asset_kind,
                "format_type": "safetensors",
                "architecture": "",
                "architecture_state": ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED,
                "checkpoint_kind": "standalone_component",
                "has_unet": False,
                "has_vae": asset_kind == "vae",
                "has_text_encoder": asset_kind == "text_encoder",
                "has_text_encoder_2": False,
                "key_count": None,
                "prefix_summary": {},
                "example_keys": [],
                "dtype_summary": {},
                "tensor_shape_summary": {},
                "metadata": payload,
                "inspector_version": "component_inventory_v1",
            },
        )

    def _source_metadata_for_path(
        self,
        path: str | Path,
        *,
        repository_roots: Iterable[Path] = (),
    ) -> tuple[str, str | None]:
        resolved = Path(path).resolve()
        models_root = self.models_scan_root()
        if self._is_under(resolved, models_root):
            return "project_models", str(models_root)
        for root in repository_roots:
            if self._is_under(resolved, root):
                return "supplied_repository", str(root)
        return "registered_external", None

    def build_match_report(self, *, repository_roots: Iterable[Path] = ()) -> dict[str, Any]:
        snapshots = self.registry.list_component_snapshots(limit=1_000_000)
        assets: dict[int, AssetRecord] = {}
        active: list[ComponentSnapshotRecord] = []
        for snapshot in snapshots:
            asset = assets.get(snapshot.asset_id)
            if asset is None:
                asset = self.registry.get_asset_by_id(snapshot.asset_id)
                if asset is not None:
                    assets[snapshot.asset_id] = asset
            if asset is None or not asset.exists_on_disk or not Path(asset.path).is_file():
                continue
            active.append(snapshot)

        exact_groups: dict[str, list[ComponentSnapshotRecord]] = defaultdict(list)
        structure_groups: dict[str, list[ComponentSnapshotRecord]] = defaultdict(list)
        role_counts: Counter[str] = Counter()
        for snapshot in active:
            exact_groups[snapshot.component_sha256].append(snapshot)
            structure_groups[snapshot.structure_sha256].append(snapshot)
            role_counts[snapshot.component_role] += 1

        exact_matches = []
        for digest, group in exact_groups.items():
            if len({item.asset_id for item in group}) < 2:
                continue
            exact_matches.append(
                self._group_payload(
                    digest, group, assets, kind="payload", repository_roots=repository_roots
                )
            )

        structure_variants = []
        for digest, group in structure_groups.items():
            asset_ids = {item.asset_id for item in group}
            payload_hashes = {item.component_sha256 for item in group}
            if len(asset_ids) < 2 or len(payload_hashes) < 2:
                continue
            structure_variants.append(
                self._group_payload(
                    digest, group, assets, kind="structure", repository_roots=repository_roots
                )
            )

        exact_matches.sort(key=lambda item: (-item["distinct_asset_count"], item["hash"]))
        structure_variants.sort(key=lambda item: (-item["distinct_asset_count"], item["hash"]))

        return {
            "active_snapshot_count": len(active),
            "active_asset_count": len({item.asset_id for item in active}),
            "role_counts": dict(sorted(role_counts.items())),
            "exact_payload_match_group_count": len(exact_matches),
            "structure_variant_group_count": len(structure_variants),
            "exact_payload_matches": exact_matches,
            "same_structure_different_weights": structure_variants,
        }

    def _group_payload(
        self,
        digest: str,
        group: list[ComponentSnapshotRecord],
        assets: Mapping[int, AssetRecord],
        *,
        kind: str,
        repository_roots: Iterable[Path] = (),
    ) -> dict[str, Any]:
        members = []
        for snapshot in sorted(
            group,
            key=lambda item: (
                str(assets[item.asset_id].filename).casefold(),
                item.component_role,
                item.asset_id,
            ),
        ):
            asset = assets[snapshot.asset_id]
            source_scope, source_root = self._source_metadata_for_path(
                asset.path, repository_roots=repository_roots
            )
            members.append(
                {
                    "asset_id": snapshot.asset_id,
                    "source_scope": source_scope,
                    "source_root": source_root,
                    "filename": asset.filename,
                    "path": asset.path,
                    "asset_type": asset.asset_type,
                    "architecture": asset.architecture,
                    "component_role": snapshot.component_role,
                    "component_sha256": snapshot.component_sha256,
                    "structure_sha256": snapshot.structure_sha256,
                    "tensor_count": snapshot.tensor_count,
                    "total_bytes": snapshot.total_bytes,
                }
            )
        return {
            "match_kind": kind,
            "hash": digest,
            "distinct_asset_count": len({item.asset_id for item in group}),
            "distinct_payload_count": len({item.component_sha256 for item in group}),
            "component_roles": sorted({item.component_role for item in group}),
            "members": members,
        }
