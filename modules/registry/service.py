from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from modules.project_context import ProjectContext

from .asset_registry import AssetRegistry
from .component_inventory import ComponentInventoryScanner
from .component_refresh import ComponentRegistryRefresher
from .component_analysis import ComponentAnalysisEngine
from .exact_overlap import ExactOverlapService
from .component_selection import ComponentSelectionService, canonical_model_family
from .family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY
from .architecture_observation import (
    ARCHITECTURE_STATE_INVALID,
    ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED,
    ARCHITECTURE_STATE_RECOGNIZED_UNSUPPORTED,
    normalize_architecture_identifier,
)
from .models import (
    ComponentScanRequest,
    SCAN_SCOPE_EXTERNAL_REPOSITORY_REFRESH,
    SCAN_SCOPE_LIBRARY_REFRESH,
    SCAN_SCOPE_SELECTED_ASSETS,
    SCAN_SCOPE_SINGLE_ASSET,
    SCAN_STRENGTH_STRUCTURAL,
    ANALYSIS_STRENGTH_NONE,
)


REGISTRY_SERVICE_VERSION = "component-registry-service-v7"
DERIVED_METRICS_VERSION = "component-registry-derived-metrics-v2"
EXACT_MATCH_RELATIONSHIP = "exact_component_match"
STRUCTURE_VARIANT_RELATIONSHIP = "same_structure_different_weights"
EXACT_FILE_DUPLICATE_RELATIONSHIP = "exact_file_duplicate_of"
ARCHIVED_LOCATION_RELATIONSHIP = "archived_location_of"


class ComponentRegistryService:
    """Stable service boundary for component-native registry consumers.

    Phase 02 generalizes scanning into a single request contract. The service now
    owns scan orchestration, discovered-family visibility, and derived registry
    metrics/relationships while still reusing AssetRegistry, ComponentRegistryRefresher,
    ComponentInventoryScanner, and ComponentSelectionService as the implementation core.
    """

    def __init__(
        self,
        context: ProjectContext,
        *,
        registry: AssetRegistry | None = None,
        refresher: ComponentRegistryRefresher | None = None,
        inventory_scanner: ComponentInventoryScanner | None = None,
    ) -> None:
        self.context = context
        self.registry = registry or AssetRegistry(str(Path(context.registry_db_path).resolve()))
        self.providers = DEFAULT_FAMILY_PROVIDER_REGISTRY
        self.selection = ComponentSelectionService(context, registry=self.registry)
        base_refresher = refresher or ComponentRegistryRefresher(context, registry=self.registry)
        if not hasattr(base_refresher, "inspector") or not hasattr(base_refresher, "snapshotter"):
            support_refresher = ComponentRegistryRefresher(context, registry=self.registry)
            if not hasattr(base_refresher, "inspector"):
                setattr(base_refresher, "inspector", support_refresher.inspector)
            if not hasattr(base_refresher, "snapshotter"):
                setattr(base_refresher, "snapshotter", support_refresher.snapshotter)
        self.refresher = base_refresher
        self.inventory = inventory_scanner or ComponentInventoryScanner(
            context,
            registry=self.registry,
            inspector=self.refresher.inspector,
            snapshotter=self.refresher.snapshotter,
        )
        self.analysis = ComponentAnalysisEngine(self.registry, providers=self.providers)
        self.exact_overlap = ExactOverlapService(self.registry)

    def provider_contracts(self) -> dict[str, Any]:
        return self.providers.to_dict()

    def configured_library_roots(self) -> dict[str, Any]:
        """Return configured extra model-library roots split by current accessibility."""
        model_library = self.context.config.get("model_library") or {}
        raw_roots = list(model_library.get("additional_scan_roots") or [])
        # Every multi-path project asset directory participates in the unified
        # structural registry refresh, including removable-drive roots.
        for key in (
            "checkpoints_dir", "vae_dir", "lora_dir", "text_encoders_dir",
            "controlnet_dir", "control_lora_dir", "embeddings_dir",
            "upscalers_comfyui_dir",
        ):
            raw_roots.extend(str(path) for path in self.context.roots_for(key))
        accessible: list[str] = []
        unavailable: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw in raw_roots:
            if isinstance(raw, str):
                value = raw
            elif isinstance(raw, Mapping):
                value = str(raw.get("path") or "")
            else:
                continue
            value = str(value or "").strip()
            if not value:
                continue
            try:
                path = self.context.resolve_project_path(value).expanduser().resolve()
            except (OSError, ValueError) as exc:
                unavailable.append({"path": value, "reason": f"resolve_error:{type(exc).__name__}"})
                continue
            token = str(path).casefold()
            if token in seen:
                continue
            seen.add(token)
            try:
                if path.is_dir():
                    accessible.append(str(path))
                else:
                    unavailable.append({"path": str(path), "reason": "not_accessible"})
            except OSError as exc:
                unavailable.append({"path": str(path), "reason": f"access_error:{type(exc).__name__}"})
        return {
            "accessible": accessible,
            "unavailable": unavailable,
            "accessible_count": len(accessible),
            "unavailable_count": len(unavailable),
        }

    def connected_storage_roots(self) -> dict[str, Any]:
        """Enumerate currently mounted roots without reporting absent drives."""
        candidates: list[Path] = []
        if os.name == "nt":
            candidates = [Path(f"{letter}:\\") for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        else:
            candidates = [Path("/")]
            for parent in (Path("/mnt"), Path("/media"), Path("/run/media")):
                try:
                    candidates.extend(path for path in parent.glob("**/*") if path.is_dir())
                except OSError:
                    continue
        roots: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if not resolved.is_dir():
                    continue
            except OSError:
                continue
            token = str(resolved).casefold()
            if token in seen:
                continue
            seen.add(token)
            roots.append({"path": str(resolved), "label": candidate.name or str(candidate)})
        return {"roots": roots, "count": len(roots)}

    def discover_asset_locations(
        self,
        roots: Iterable[str | Path],
        *,
        max_files: int = 10000,
    ) -> dict[str, Any]:
        """Find candidate model directories on connected storage.

        This stage reads directory entries only. Tensor inspection and hashing
        happen during import, keeping drive search quick and cancellable by the
        request boundary.
        """
        extensions = {".safetensors", ".pth", ".pt", ".ckpt", ".bin"}
        excluded = {"$recycle.bin", "system volume information", "windows", "program files", "program files (x86)", "node_modules", ".git"}
        locations: dict[str, dict[str, Any]] = {}
        quarantine: list[dict[str, Any]] = []
        candidate_count = 0
        truncated = False
        for raw in roots:
            root = Path(raw).expanduser().resolve(strict=False)
            if not root.is_dir():
                continue
            try:
                for directory, directory_names, filenames in os.walk(root, topdown=True, onerror=lambda _error: None, followlinks=False):
                    directory_names[:] = [name for name in directory_names if name.casefold() not in excluded and not name.startswith(".")]
                    for filename in filenames:
                        path = Path(directory) / filename
                        if path.suffix.lower() not in extensions:
                            continue
                        candidate_count += 1
                        parent = str(path.parent.resolve(strict=False))
                        row = locations.setdefault(parent, {"path": parent, "file_count": 0, "extensions": {}})
                        row["file_count"] += 1
                        suffix = path.suffix.lower()
                        row["extensions"][suffix] = int(row["extensions"].get(suffix, 0)) + 1
                        if suffix != ".safetensors":
                            try:
                                digest = hashlib.sha256()
                                with path.open("rb") as handle:
                                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                                        digest.update(block)
                                quarantine.append({
                                    "path": str(path.resolve(strict=False)),
                                    "filename": path.name,
                                    "extension": suffix,
                                    "sha256": digest.hexdigest(),
                                    "size_bytes": int(path.stat().st_size),
                                    "state": "awaiting_user_verification",
                                    "reason": "pickle_bearing_format",
                                })
                            except OSError:
                                pass
                        if candidate_count >= max(1, int(max_files)):
                            truncated = True
                            break
                    if truncated:
                        break
                if truncated:
                    break
            except OSError:
                continue
        rows = sorted(locations.values(), key=lambda item: (-int(item["file_count"]), item["path"].casefold()))
        self._save_pickle_quarantine(quarantine)
        return {"locations": rows, "location_count": len(rows), "candidate_file_count": candidate_count, "truncated": truncated, "quarantined_files": quarantine, "quarantined_count": len(quarantine)}

    def _pickle_quarantine_path(self) -> Path:
        return Path(self.context.data_root) / "pickle_asset_quarantine.json"

    def _save_pickle_quarantine(self, discovered: list[dict[str, Any]]) -> None:
        path = self._pickle_quarantine_path()
        previous: dict[str, dict[str, Any]] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            previous = {str(item.get("path") or "").casefold(): item for item in payload.get("files", [])}
        except (OSError, ValueError, TypeError):
            pass
        for item in discovered:
            prior = previous.get(str(item["path"]).casefold())
            if prior and prior.get("sha256") == item.get("sha256") and prior.get("state") == "approved":
                item.update({"state": "approved", "asset_type": prior.get("asset_type", "checkpoint")})
            previous[str(item["path"]).casefold()] = item
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(json.dumps({"version": 1, "files": list(previous.values())}, indent=2), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            return

    def pickle_quarantine(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._pickle_quarantine_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {"version": 1, "files": []}
        files = list(payload.get("files") or [])
        return {"files": files, "count": len(files), "awaiting_count": sum(1 for item in files if item.get("state") != "approved")}

    def approve_pickle_asset(self, path_value: str, expected_sha256: str, asset_type: str) -> dict[str, Any]:
        selected_type = str(asset_type or "").strip().lower()
        if selected_type not in {"checkpoint", "lora", "upscaler", "embedding"}:
            raise ValueError("A quarantined file requires an explicit asset_type.")
        path = Path(path_value).expanduser().resolve(strict=False)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual.casefold() != str(expected_sha256 or "").strip().casefold():
            raise ValueError("The file changed after discovery and must be reviewed again.")
        payload = self.pickle_quarantine()
        matched = False
        for item in payload["files"]:
            if str(item.get("path") or "").casefold() == str(path).casefold():
                item.update({"sha256": actual, "state": "approved", "asset_type": selected_type})
                matched = True
        if not matched:
            raise ValueError("The file is not in the quarantine catalog.")
        self._pickle_quarantine_path().write_text(json.dumps({"version": 1, "files": payload["files"]}, indent=2), encoding="utf-8")
        if selected_type == "upscaler":
            upscaling = self.context.config.setdefault("upscaling", {})
            roots = list(upscaling.get("additional_roots") or [])
            if str(path.parent).casefold() not in {str(item).casefold() for item in roots}:
                roots.append(str(path.parent))
            upscaling["additional_roots"] = roots
            import yaml
            config_path = Path(self.context.config_path)
            user_payload = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
            user_payload.setdefault("upscaling", {})["additional_roots"] = roots
            temporary = config_path.with_name(f".{config_path.name}.tmp")
            temporary.write_text(yaml.safe_dump(user_payload, sort_keys=False), encoding="utf-8", newline="\n")
            temporary.replace(config_path)
        return {"approved": True, "path": str(path), "sha256": actual, "asset_type": selected_type}

    def import_asset_locations(self, roots: Iterable[str | Path], *, force: bool = False) -> dict[str, Any]:
        """Persist selected roots and run evidence-based structural import."""
        model_library = self.context.config.setdefault("model_library", {})
        configured = list(model_library.get("additional_scan_roots") or [])
        known = {
            str(self.context.resolve_project_path(item if isinstance(item, str) else item.get("path", "")).resolve(strict=False)).casefold()
            for item in configured if isinstance(item, (str, Mapping))
        }
        imported: list[str] = []
        for raw in roots:
            path = Path(raw).expanduser().resolve(strict=False)
            if not path.is_dir():
                continue
            token = str(path).casefold()
            if token not in known:
                configured.append({"path": str(path), "mode": "scan_only"})
                known.add(token)
            imported.append(str(path))
        model_library["additional_scan_roots"] = configured

        # Persist only the user-owned config, never the generated system file.
        import yaml
        config_path = Path(self.context.config_path)
        user_payload = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
        user_payload.setdefault("model_library", {})["additional_scan_roots"] = configured
        temporary = config_path.with_name(f".{config_path.name}.tmp")
        temporary.write_text(yaml.safe_dump(user_payload, sort_keys=False), encoding="utf-8", newline="\n")
        temporary.replace(config_path)

        result = self.run_scan({
            "scope": SCAN_SCOPE_EXTERNAL_REPOSITORY_REFRESH,
            "strength": SCAN_STRENGTH_STRUCTURAL,
            "analysis_strength": ANALYSIS_STRENGTH_NONE,
            "force": bool(force),
            "repository_roots": imported,
        }) if imported else {"candidate_count": 0, "results": [], "errors": []}
        result["imported_roots"] = imported
        result["location_catalog"] = self.location_catalog()
        return result

    def refresh_configured_library(
        self,
        *,
        force: bool = False,
        strength: str = SCAN_STRENGTH_STRUCTURAL,
    ) -> dict[str, Any]:
        """Refresh reachable configured model roots while retaining unreachable registry history."""
        roots = self.configured_library_roots()
        payload = self.run_scan(
            {
                "scope": SCAN_SCOPE_LIBRARY_REFRESH,
                "strength": strength,
                "analysis_strength": ANALYSIS_STRENGTH_NONE,
                "force": bool(force),
                "repository_roots": list(roots["accessible"]),
            }
        )
        payload["configured_roots"] = roots
        payload["location_catalog"] = self.location_catalog()
        return payload

    def location_catalog(self, *, accessible_only: bool = False) -> dict[str, Any]:
        """Expose registry locations without erasing unavailable/disconnected history."""
        assets = self.registry.list_assets(limit=1_000_000)
        snapshots_by_asset: dict[int, int] = defaultdict(int)
        for snapshot in self.registry.list_component_snapshots(limit=1_000_000):
            snapshots_by_asset[int(snapshot.asset_id)] += 1

        rows: list[dict[str, Any]] = []
        state_counts: dict[str, int] = defaultdict(int)
        for asset in assets:
            state = str(asset.location_state or "missing")
            state_counts[state] += 1
            accessible = bool(asset.exists_on_disk and state == "available")
            if accessible_only and not accessible:
                continue
            rows.append(
                {
                    "asset_id": int(asset.id),
                    "path": str(asset.path),
                    "filename": str(asset.filename),
                    "asset_type": str(asset.asset_type or "unclassified_asset"),
                    "architecture": str(asset.architecture or ""),
                    "sha256": str(asset.sha256 or ""),
                    "exists_on_disk": bool(asset.exists_on_disk),
                    "accessible": accessible,
                    "location_state": state,
                    "component_snapshot_count": int(snapshots_by_asset.get(int(asset.id), 0)),
                    "last_seen_at": str(asset.last_seen_at or ""),
                }
            )

        rows.sort(
            key=lambda item: (
                0 if item["accessible"] else 1,
                item["asset_type"],
                item["filename"].casefold(),
                item["path"].casefold(),
            )
        )
        return {
            "registered_location_count": len(assets),
            "accessible_location_count": sum(
                1 for item in assets
                if item.exists_on_disk and str(item.location_state or "") == "available"
            ),
            "unavailable_location_count": sum(
                1 for item in assets
                if not item.exists_on_disk or str(item.location_state or "") != "available"
            ),
            "location_state_counts": dict(sorted(state_counts.items())),
            "accessible_only": bool(accessible_only),
            "locations": rows,
        }

    def describe_analysis_layout(self, *, family: str, role: str) -> dict[str, Any] | None:
        provider = self.providers.require(family)
        layout = provider.describe_analysis_layout(role)
        return layout.to_dict() if layout is not None else None

    def list_analysis_manifest_records(
        self,
        *,
        component_sha256: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return self.registry.list_component_analysis_manifests(
            component_sha256=component_sha256,
            limit=limit,
        )

    def list_component_relationship_records(
        self,
        *,
        component_sha256: str | None = None,
        relationship_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return self.registry.list_component_relationships(
            component_sha256=component_sha256,
            relationship_type=relationship_type,
            limit=limit,
        )

    def list_relationship_evidence_records(
        self,
        *,
        component_sha256: str | None = None,
        relationship_type: str | None = None,
        evidence_source: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return self.registry.list_relationship_evidence(
            component_sha256=component_sha256,
            relationship_type=relationship_type,
            evidence_source=evidence_source,
            status=status,
            limit=limit,
        )

    def set_component_policy(self, **payload: Any) -> dict[str, Any]:
        return self.registry.set_component_policy(**payload)

    def clear_component_policy(self, **payload: Any) -> int:
        return self.registry.clear_component_policy(**payload)

    def list_component_policies(
        self,
        *,
        component_sha256: str | None = None,
        base_component_sha256: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        return self.registry.list_component_policies(
            component_sha256=component_sha256,
            base_component_sha256=base_component_sha256,
            limit=limit,
        )

    def record_component_validation(self, **payload: Any) -> dict[str, Any]:
        return self.registry.record_component_validation(**payload)

    def clear_component_validations(self, **payload: Any) -> int:
        return self.registry.clear_component_validations(**payload)

    def list_component_validations(
        self,
        *,
        component_sha256: str | None = None,
        base_component_sha256: str | None = None,
        composition_sha256: str | None = None,
        validation_stage: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        return self.registry.list_component_validations(
            component_sha256=component_sha256,
            base_component_sha256=base_component_sha256,
            composition_sha256=composition_sha256,
            validation_stage=validation_stage,
            limit=limit,
        )

    def registry_browser(
        self,
        *,
        family: str | None = None,
        role: str | None = None,
        accessible_only: bool = False,
        search: str = "",
        limit: int = 500,
    ) -> dict[str, Any]:
        """Return an inspection-oriented registry view without changing identity.

        Paths are source occurrences only. Component rows are keyed by immutable
        component fingerprints and expose policy, validation, analytical evidence,
        and all known locations separately.
        """
        normalized_family = canonical_model_family(family) if family else ""
        normalized_role = str(role or "").strip()
        needle = str(search or "").strip().casefold()
        catalog = self.selection.catalog()
        component_rows: dict[str, dict[str, Any]] = {}
        for family_entry in catalog.get("families", []):
            family_id = str(family_entry.get("family") or "")
            if normalized_family and family_id != normalized_family:
                continue
            for role_entry in family_entry.get("roles", []):
                role_id = str(role_entry.get("role") or "")
                if normalized_role and role_id != normalized_role:
                    continue
                for component in role_entry.get("components", []):
                    digest = str(component.get("component_sha256") or "").strip().lower()
                    if not digest:
                        continue
                    row = component_rows.setdefault(
                        digest,
                        {
                            "component_sha256": digest,
                            "short_hash": digest[:12],
                            "families": [],
                            "roles": [],
                            "component_bytes": int(component.get("component_bytes") or 0),
                            "tensor_count": int(component.get("tensor_count") or 0),
                            "sources": [],
                            "policy": dict(component.get("phase05") or {}),
                        },
                    )
                    if family_id not in row["families"]:
                        row["families"].append(family_id)
                    if role_id not in row["roles"]:
                        row["roles"].append(role_id)
                    known_source_keys = {
                        (int(item.get("asset_id") or 0), str(item.get("component_role") or ""), str(item.get("snapshot_version") or ""))
                        for item in row["sources"]
                    }
                    for source in component.get("sources", []):
                        key = (
                            int(source.get("asset_id") or 0),
                            str(source.get("component_role") or ""),
                            str(source.get("snapshot_version") or ""),
                        )
                        if key not in known_source_keys:
                            row["sources"].append(dict(source))
                            known_source_keys.add(key)

        policies = self.registry.list_component_policies(limit=1_000_000)
        validations = self.registry.list_component_validations(limit=1_000_000)
        relationships = self.registry.list_relationship_evidence(limit=1_000_000)
        policies_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        validations_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        relationship_count: dict[str, int] = defaultdict(int)
        for item in policies:
            policies_by_component[str(item.get("component_sha256") or "").strip().lower()].append(dict(item))
        for item in validations:
            validations_by_component[str(item.get("component_sha256") or "").strip().lower()].append(dict(item))
        for item in relationships:
            seen: set[str] = set()
            for participant in item.get("participants", []):
                digest = str(participant.get("component_sha256") or "").strip().lower()
                if digest and digest not in seen:
                    relationship_count[digest] += 1
                    seen.add(digest)

        components: list[dict[str, Any]] = []
        for digest, row in component_rows.items():
            sources = list(row.get("sources") or [])
            accessible_sources = [
                item for item in sources
                if bool(item.get("exists_on_disk")) and str(item.get("location_state") or "") == "available"
            ]
            if accessible_only and not accessible_sources:
                continue
            if needle:
                haystack = " ".join(
                    [
                        digest,
                        *row.get("families", []),
                        *row.get("roles", []),
                        *(str(item.get("filename") or "") for item in sources),
                        *(str(item.get("path") or "") for item in sources),
                    ]
                ).casefold()
                if needle not in haystack:
                    continue
            row["families"] = sorted(row.get("families", []))
            row["roles"] = sorted(row.get("roles", []))
            row["accessible_source_count"] = len(accessible_sources)
            row["registered_source_count"] = len(sources)
            row["policy_records"] = policies_by_component.get(digest, [])
            row["validation_records"] = validations_by_component.get(digest, [])[:50]
            row["relationship_evidence_count"] = int(relationship_count.get(digest, 0))
            components.append(row)

        components.sort(
            key=lambda item: (
                0 if int(item.get("accessible_source_count") or 0) else 1,
                ",".join(item.get("families", [])),
                ",".join(item.get("roles", [])),
                str(item.get("component_sha256") or ""),
            )
        )
        components = components[: max(1, int(limit))]

        location_catalog = self.location_catalog(accessible_only=accessible_only)
        snapshots_by_asset: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for snapshot in self.registry.list_component_snapshots(limit=1_000_000):
            snapshots_by_asset[int(snapshot.asset_id)].append(
                {
                    "component_role": snapshot.component_role,
                    "component_sha256": snapshot.component_sha256,
                    "structure_sha256": snapshot.structure_sha256,
                    "tensor_count": int(snapshot.tensor_count),
                    "total_bytes": int(snapshot.total_bytes),
                }
            )
        models: list[dict[str, Any]] = []
        for item in location_catalog.get("locations", []):
            if str(item.get("asset_type") or "") != "checkpoint":
                continue
            model = dict(item)
            model["components"] = snapshots_by_asset.get(int(item.get("asset_id") or 0), [])
            if needle:
                haystack = f"{model.get('filename', '')} {model.get('path', '')} {model.get('sha256', '')}".casefold()
                if needle not in haystack:
                    continue
            models.append(model)
        models = models[: max(1, int(limit))]
        return {
            "service_version": REGISTRY_SERVICE_VERSION,
            "filters": {
                "family": normalized_family or None,
                "role": normalized_role or None,
                "accessible_only": bool(accessible_only),
                "search": str(search or ""),
                "limit": max(1, int(limit)),
            },
            "summary": {
                "component_count": len(components),
                "model_location_count": len(models),
                "accessible_location_count": int(location_catalog.get("accessible_location_count") or 0),
                "unavailable_location_count": int(location_catalog.get("unavailable_location_count") or 0),
            },
            "models": models,
            "components": components,
        }

    def refresh_exact_overlap_relationships(
        self,
        *,
        component_sha256s: Iterable[str] | None = None,
        persist: bool = True,
        min_matching_nodes: int = 1,
        include_tensor_only: bool = False,
        min_matching_tensors: int = 1,
    ) -> dict[str, Any]:
        return self.exact_overlap.refresh_relationships(
            component_sha256s=component_sha256s,
            persist=persist,
            min_matching_nodes=min_matching_nodes,
            include_tensor_only=include_tensor_only,
            min_matching_tensors=min_matching_tensors,
        )

    def compare_exact_overlap(
        self,
        left_component_sha256: str,
        right_component_sha256: str,
        *,
        include_tensor_evidence: bool = False,
    ) -> dict[str, Any]:
        return self.exact_overlap.compare_components(
            left_component_sha256,
            right_component_sha256,
            include_tensor_evidence=include_tensor_evidence,
        )

    def rank_exact_overlap_candidates(
        self,
        component_sha256: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.exact_overlap.rank_candidates_by_node_overlap(
            component_sha256,
            limit=limit,
        )

    def _normalize_scan_request(
        self,
        request: ComponentScanRequest | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> ComponentScanRequest:
        if isinstance(request, ComponentScanRequest):
            base = request.to_dict()
        elif request is None:
            base = {}
        else:
            base = dict(request)
        for key, value in overrides.items():
            if value is not None:
                base[key] = value
        return ComponentScanRequest.from_mapping(base)

    def run_scan(
        self,
        request: ComponentScanRequest | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        normalized = self._normalize_scan_request(request, **overrides)
        results: list[dict[str, Any]]
        inventory_payload: dict[str, Any] | None = None

        if normalized.scope in {SCAN_SCOPE_SINGLE_ASSET, SCAN_SCOPE_SELECTED_ASSETS}:
            results = self.refresher.refresh_registered(
                paths=normalized.paths,
                force=normalized.force,
                dry_run=normalized.dry_run,
                source=f"component_registry_service:{normalized.scope}",
                strength=normalized.strength,
            )
            self.registry.mark_missing_assets()
        elif normalized.scope in {SCAN_SCOPE_LIBRARY_REFRESH, SCAN_SCOPE_EXTERNAL_REPOSITORY_REFRESH}:
            inventory_payload = self.inventory.scan(
                force=normalized.force,
                repository_roots=normalized.repository_roots,
                strength=normalized.strength,
            )
            results = list(inventory_payload.get("results") or [])
        else:
            raise ValueError(f"Unsupported component scan scope: {normalized.scope!r}")

        touched_asset_ids = sorted(
            {
                int(item["asset_id"])
                for item in results
                if item.get("asset_id") is not None
            }
        )
        analysis_results: list[dict[str, Any]] = []
        if normalized.analysis_strength != ANALYSIS_STRENGTH_NONE and touched_asset_ids:
            analysis_results = self.analysis.analyze_assets(
                touched_asset_ids,
                analysis_strength=normalized.analysis_strength,
                persist=not normalized.dry_run,
                force=normalized.force,
            )

        overlap_refresh: dict[str, Any] | None = None
        if normalized.analysis_strength == "exact" and analysis_results:
            component_sha256s = sorted({
                str((item.get("manifest") or {}).get("component_sha256") or "").strip().lower()
                for item in analysis_results
                if isinstance(item.get("manifest"), Mapping)
                and str((item.get("manifest") or {}).get("component_sha256") or "").strip()
            })
            if component_sha256s:
                overlap_refresh = self.exact_overlap.refresh_relationships(
                    component_sha256s=component_sha256s,
                    persist=not normalized.dry_run,
                )

        derived = self._refresh_derived_state(
            repository_roots=normalized.repository_roots,
            asset_ids=(None if normalized.scope in {SCAN_SCOPE_LIBRARY_REFRESH, SCAN_SCOPE_EXTERNAL_REPOSITORY_REFRESH} else touched_asset_ids),
        )

        return {
            "service_version": REGISTRY_SERVICE_VERSION,
            "request": normalized.to_dict(),
            "candidate_count": (
                inventory_payload.get("candidate_count")
                if inventory_payload is not None
                else len(results)
            ),
            "result_count": len(results),
            "touched_asset_ids": touched_asset_ids,
            "results": results,
            "inventory": inventory_payload,
            "analysis": {
                "requested_strength": normalized.analysis_strength,
                "result_count": len(analysis_results),
                "results": analysis_results,
            },
            "exact_overlap": overlap_refresh,
            "derived": derived,
            "selector_families": self.list_discovered_families(),
            "diagnostics": self.list_family_discovery().get("diagnostics", []),
            "observations": self.list_family_discovery().get("observations", []),
        }

    def list_family_discovery(self) -> dict[str, Any]:
        catalog = self.selection.catalog()
        catalog_by_family = {item["family"]: item for item in catalog.get("families", [])}
        role_maps = {
            family: {role["role"]: role for role in entry.get("roles", [])}
            for family, entry in catalog_by_family.items()
        }

        provider_ids = {provider.family_id for provider in self.providers.providers()}
        asset_counts: dict[str, int] = defaultdict(int)
        observation_buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"asset_count": 0, "paths": []}
        )

        live_assets = []
        for asset in self.registry.list_assets(limit=1_000_000):
            path = Path(asset.path)
            if not asset.exists_on_disk or not path.is_file() or path.suffix.lower() != ".safetensors":
                continue
            live_assets.append(asset)
            canonical = canonical_model_family(asset.architecture)
            if canonical in provider_ids:
                asset_counts[canonical] += 1
                continue

            state = str(asset.architecture_state or ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED)
            architecture = normalize_architecture_identifier(asset.architecture)
            label = architecture or "unclassified_safetensors"
            bucket = observation_buckets[(state, label)]
            bucket["asset_count"] += 1
            if len(bucket["paths"]) < 10:
                bucket["paths"].append(asset.path)

        supported: list[dict[str, Any]] = []
        for provider in self.providers.providers():
            role_entries = role_maps.get(provider.family_id, {})
            if not asset_counts.get(provider.family_id) and not role_entries:
                continue
            required = [item.canonical_role_id for item in provider.required_roles]
            missing_required = [
                role
                for role in required
                if not any(
                    bool(component.get("selectable_with_digital"))
                    for component in ((role_entries.get(role) or {}).get("components") or [])
                )
            ]
            role_coverage = {
                role_id: int((role_entries.get(role_id) or {}).get("unique_component_count") or 0)
                for role_id in [definition.canonical_role_id for definition in provider.role_definitions()]
            }
            eligible_role_coverage = {
                role_id: sum(
                    1
                    for component in ((role_entries.get(role_id) or {}).get("components") or [])
                    if bool(component.get("selectable_with_digital"))
                )
                for role_id in [definition.canonical_role_id for definition in provider.role_definitions()]
            }
            supported.append(
                {
                    "family": provider.family_id,
                    "label": provider.display_label,
                    "provider_supported": True,
                    "provider_version": provider.version,
                    "asset_count": int(asset_counts.get(provider.family_id, 0)),
                    "base_weight_components": int((role_entries.get(provider.base_weight_role) or {}).get("unique_component_count") or 0),
                    "required_role_coverage": {
                        role: int((role_entries.get(role) or {}).get("unique_component_count") or 0)
                        for role in required
                    },
                    "required_role_eligible_coverage": {
                        role: int(eligible_role_coverage.get(role) or 0)
                        for role in required
                    },
                    "optional_role_coverage": {
                        role_id: count
                        for role_id, count in role_coverage.items()
                        if role_id not in required
                    },
                    "constructible": not missing_required and provider.supports_runtime_composition(),
                    "selectable": not missing_required and provider.supports_runtime_composition(),
                    "missing_required_roles": missing_required,
                    "roles": list(catalog_by_family.get(provider.family_id, {}).get("roles", [])),
                    "runtime_composition_validation_state": provider.to_dict().get("runtime_composition_validation_state"),
                    "base_weight_role": provider.base_weight_role,
                }
            )

        observations: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for (state, key), value in sorted(observation_buckets.items()):
            payload = {
                "family": key if key != "unclassified_safetensors" else None,
                "label": key,
                "provider_supported": False,
                "architecture_state": state,
                "asset_count": int(value["asset_count"]),
                "constructible": False,
                "selectable": False,
                "sample_paths": list(value["paths"]),
            }
            if state == ARCHITECTURE_STATE_INVALID:
                payload["reason"] = "invalid_asset"
                diagnostics.append(payload)
            else:
                payload["reason"] = (
                    "recognized_but_not_yet_supported"
                    if state == ARCHITECTURE_STATE_RECOGNIZED_UNSUPPORTED
                    else "observed_for_future_provider_learning"
                )
                observations.append(payload)

        supported.sort(key=lambda item: item["family"])
        return {
            "version": REGISTRY_SERVICE_VERSION,
            "supported": supported,
            "observations": observations,
            "diagnostics": diagnostics,
            "selector_families": [
                item for item in supported if item["provider_supported"] and item["selectable"]
            ],
        }

    def list_discovered_families(self) -> list[dict[str, Any]]:
        return list(self.list_family_discovery()["selector_families"])

    def list_component_candidates(
        self,
        *,
        family: str,
        role: str,
        base_component_sha256: str | None = None,
        allow_digital_components: bool = True,
    ) -> dict[str, Any]:
        canonical = canonical_model_family(family)
        provider = self.providers.require(canonical)
        definition = provider.role_definition(role)
        if definition is None:
            raise ValueError(f"Role {role!r} is not defined by provider {canonical!r}.")
        family_entry = next((item for item in self.selection.catalog()["families"] if item["family"] == canonical), None)
        role_entry = next((item for item in (family_entry or {}).get("roles", []) if item["role"] == role), None)
        candidates = list((role_entry or {}).get("components", []))
        filtered = [
            item
            for item in candidates
            if self.selection._preferred_source(
                item.get("sources") or [],
                require_checkpoint=bool(getattr(definition, "base_weight_role", False)),
                allow_digital_components=allow_digital_components,
                family=canonical,
            ) is not None
        ]
        return {
            "family": canonical,
            "role": role,
            "base_component_sha256": str(base_component_sha256 or "") or None,
            "provider_version": provider.version,
            "definition": definition.to_dict(),
            "digital_hydration_validation_state": provider.to_dict().get("digital_hydration_validation", {}).get(role),
            "allow_digital_components": bool(allow_digital_components),
            "candidates": candidates,
            "eligible_candidates": filtered,
            "policy_filtering": {
                "source_policy": "allow_physical_and_digital" if allow_digital_components else "physical_and_standalone_only",
                "eligible_count": len(filtered),
                "total_count": len(candidates),
                "base_component_exclusions_applied": False,
            },
        }

    def resolve_selection(
        self,
        family: str,
        selections: Mapping[str, Any] | None,
        *,
        t5_device: Any = "cpu",
        allow_digital_components: Any = True,
    ) -> dict[str, Any]:
        return self.selection.resolve_selection(
            family,
            selections,
            t5_device=t5_device,
            allow_digital_components=allow_digital_components,
        )

    def explain_auto_unresolved(self, *, family: str, role: str, allow_digital_components: bool = True) -> dict[str, Any]:
        payload = self.list_component_candidates(
            family=family,
            role=role,
            allow_digital_components=allow_digital_components,
        )
        definition = payload["definition"]
        candidates = payload["candidates"]
        eligible_candidates = payload["eligible_candidates"]
        if not definition["required"]:
            return {
                "resolved": False,
                "reason": "optional_roles_do_not_auto_enable",
                "candidate_count": len(candidates),
            }
        if len(eligible_candidates) == 1:
            return {
                "resolved": True,
                "reason": "exactly_one_unique_eligible_fingerprint",
                "candidate_count": 1,
                "component_sha256": eligible_candidates[0]["component_sha256"],
            }
        return {
            "resolved": False,
            "reason": (
                "no_candidates"
                if not candidates
                else "no_current_policy_candidates"
                if not eligible_candidates
                else "multiple_candidates_require_explicit_selection"
            ),
            "candidate_count": len(candidates),
            "eligible_candidate_count": len(eligible_candidates),
            "component_sha256": None,
        }

    def explain_component_sources(self, component_sha256: str) -> dict[str, Any]:
        identity = self.registry.get_component_identity(component_sha256)
        sources = self.registry.list_component_sources(component_sha256=component_sha256)
        return {
            "component_sha256": str(component_sha256 or "").strip().lower(),
            "identity": identity.__dict__ if identity is not None else None,
            "source_count": len(sources),
            "sources": [item.__dict__ for item in sources],
        }

    def get_registry_health_metrics(self) -> dict[str, Any]:
        discovery = self.list_family_discovery()
        health = self.registry.registry_health()
        return {
            "service_version": REGISTRY_SERVICE_VERSION,
            "health": health,
            "discovered_family_count": len(discovery["supported"]),
            "selectable_family_count": sum(1 for item in discovery["supported"] if item["selectable"]),
            "families": [
                {
                    "family": item["family"],
                    "selectable": item["selectable"],
                    "missing_required_roles": list(item["missing_required_roles"]),
                }
                for item in discovery["supported"]
            ],
            "diagnostics": discovery["diagnostics"],
            "materialized_metrics": self.registry.list_registry_metrics(),
        }

    def request_targeted_refresh(
        self,
        path: str | Path,
        *,
        force: bool = False,
        dry_run: bool = False,
        strength: str = SCAN_STRENGTH_STRUCTURAL,
    ) -> dict[str, Any]:
        return self.refresher.ensure_path(
            path,
            force=force,
            dry_run=dry_run,
            source="component_registry_service_targeted",
            strength=strength,
        )

    def request_registered_refresh(
        self,
        *,
        force: bool = False,
        dry_run: bool = False,
        strength: str = SCAN_STRENGTH_STRUCTURAL,
    ) -> list[dict[str, Any]]:
        return self.refresher.refresh_registered(
            force=force,
            dry_run=dry_run,
            source="component_registry_service_registered_refresh",
            strength=strength,
        )

    def list_blueprint_records(self, limit: int = 1000) -> list[dict[str, Any]]:
        return self.registry.list_model_blueprints(limit=limit)

    def list_saved_composition_records(self, limit: int = 1000) -> list[dict[str, Any]]:
        return self.registry.list_saved_compositions(limit=limit)

    def list_exact_file_duplicate_groups(self, limit: int = 1000) -> list[dict[str, Any]]:
        return self.registry.list_exact_file_duplicate_groups(limit=limit)

    def list_component_occurrence_groups(self, limit: int = 1000) -> list[dict[str, Any]]:
        return self.registry.list_component_occurrence_groups(limit=limit)

    def _refresh_derived_state(
        self,
        *,
        repository_roots: Iterable[str] = (),
        asset_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        reconciliation = self.registry.reconcile_asset_locations(asset_ids=asset_ids)
        relationship_count = self._rebuild_relationships(asset_ids=asset_ids)
        family_discovery = self.list_family_discovery()
        match_report = self.inventory.build_match_report(
            repository_roots=[Path(item).expanduser().resolve() for item in repository_roots]
        )
        file_duplicates = {
            "group_count": len(self.registry.list_exact_file_duplicate_groups(limit=1_000_000)),
            "groups": self.registry.list_exact_file_duplicate_groups(limit=100),
        }
        component_occurrences = {
            "group_count": len(self.registry.list_component_occurrence_groups(limit=1_000_000)),
            "groups": self.registry.list_component_occurrence_groups(limit=100),
        }
        summary = {
            "service_version": REGISTRY_SERVICE_VERSION,
            "family_discovery": family_discovery,
            "match_report": match_report,
            "reconciliation": reconciliation,
            "exact_file_duplicates": file_duplicates,
            "component_occurrences": component_occurrences,
            "health": self.registry.registry_health(),
        }
        self.registry.replace_registry_metrics(
            {
                "component_family_discovery": family_discovery,
                "component_match_report": match_report,
                "component_reconciliation_report": reconciliation,
                "component_file_duplicate_report": file_duplicates,
                "component_occurrence_report": component_occurrences,
                "component_scan_summary": summary,
            },
            calculation_version=DERIVED_METRICS_VERSION,
        )
        return {
            "relationship_count": relationship_count,
            "metrics_version": DERIVED_METRICS_VERSION,
            "family_discovery": family_discovery,
            "match_report": match_report,
            "reconciliation": reconciliation,
            "exact_file_duplicates": file_duplicates,
            "component_occurrences": component_occurrences,
        }

    def _rebuild_relationships(self, *, asset_ids: Iterable[int] | None = None) -> int:
        scoped_ids = {int(item) for item in asset_ids or []}
        self.registry.clear_asset_relationships(asset_ids=(scoped_ids or None))

        all_assets = {asset.id: asset for asset in self.registry.list_assets(limit=1_000_000)}
        active_assets = {
            asset_id: asset
            for asset_id, asset in all_assets.items()
            if asset.exists_on_disk and Path(asset.path).is_file()
        }
        snapshots = [
            snapshot
            for snapshot in self.registry.list_component_snapshots(limit=1_000_000)
            if snapshot.asset_id in active_assets
        ]
        by_component: dict[str, list[Any]] = defaultdict(list)
        by_structure: dict[str, list[Any]] = defaultdict(list)
        by_file_sha: dict[str, list[Any]] = defaultdict(list)
        for snapshot in snapshots:
            by_component[str(snapshot.component_sha256)].append(snapshot)
            by_structure[str(snapshot.structure_sha256)].append(snapshot)
        for asset in all_assets.values():
            digest = str(asset.sha256 or "").strip().lower()
            if digest:
                by_file_sha[digest].append(asset)

        relationship_payloads: list[tuple[int, int, str, dict[str, Any]]] = []
        seen: set[tuple[int, int, str, str]] = set()

        def add_pair(source_id: int, target_id: int, relationship_type: str, metadata: dict[str, Any], digest: str) -> None:
            if source_id == target_id:
                return
            low, high = sorted((int(source_id), int(target_id)))
            if scoped_ids and low not in scoped_ids and high not in scoped_ids:
                return
            key = (low, high, relationship_type, digest)
            if key in seen:
                return
            seen.add(key)
            relationship_payloads.append((low, high, relationship_type, metadata))

        for digest, group in by_file_sha.items():
            asset_ids_for_group = sorted({item.id for item in group})
            if len(asset_ids_for_group) < 2:
                continue
            state_counts: dict[str, int] = defaultdict(int)
            for asset in group:
                state_counts[str(asset.location_state)] += 1
            for index, source_id in enumerate(asset_ids_for_group):
                for target_id in asset_ids_for_group[index + 1 :]:
                    add_pair(
                        source_id,
                        target_id,
                        EXACT_FILE_DUPLICATE_RELATIONSHIP,
                        {
                            "sha256": digest,
                            "location_count": len(asset_ids_for_group),
                            "location_state_counts": dict(sorted(state_counts.items())),
                        },
                        digest,
                    )
            moved_assets = [asset for asset in group if str(asset.location_state) == "moved_relinked"]
            live_assets = [asset for asset in group if asset.exists_on_disk]
            if len(live_assets) == 1 and moved_assets:
                target_id = live_assets[0].id
                for asset in moved_assets:
                    add_pair(
                        asset.id,
                        target_id,
                        ARCHIVED_LOCATION_RELATIONSHIP,
                        {
                            "sha256": digest,
                            "archived_path": asset.path,
                            "active_path": live_assets[0].path,
                        },
                        f"archived:{digest}:{asset.id}",
                    )

        for digest, group in by_component.items():
            asset_ids_for_group = sorted({item.asset_id for item in group})
            if len(asset_ids_for_group) < 2:
                continue
            roles = sorted({item.component_role for item in group})
            for index, source_id in enumerate(asset_ids_for_group):
                for target_id in asset_ids_for_group[index + 1 :]:
                    add_pair(
                        source_id,
                        target_id,
                        EXACT_MATCH_RELATIONSHIP,
                        {
                            "component_sha256": digest,
                            "roles": roles,
                            "match_asset_count": len(asset_ids_for_group),
                        },
                        digest,
                    )

        for digest, group in by_structure.items():
            asset_ids_for_group = sorted({item.asset_id for item in group})
            payload_hashes = {item.component_sha256 for item in group}
            if len(asset_ids_for_group) < 2 or len(payload_hashes) < 2:
                continue
            roles = sorted({item.component_role for item in group})
            for index, source_id in enumerate(asset_ids_for_group):
                for target_id in asset_ids_for_group[index + 1 :]:
                    add_pair(
                        source_id,
                        target_id,
                        STRUCTURE_VARIANT_RELATIONSHIP,
                        {
                            "structure_sha256": digest,
                            "roles": roles,
                            "match_asset_count": len(asset_ids_for_group),
                            "distinct_payload_count": len(payload_hashes),
                        },
                        digest,
                    )

        for source_id, target_id, relationship_type, metadata in relationship_payloads:
            self.registry.add_relationship(
                source_asset_id=source_id,
                target_asset_id=target_id,
                relationship_type=relationship_type,
                confidence=1.0,
                metadata=metadata,
            )
        return len(relationship_payloads)


__all__ = ["ComponentRegistryService", "REGISTRY_SERVICE_VERSION"]
