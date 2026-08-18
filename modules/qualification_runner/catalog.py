from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

from modules.registry.component_selection import canonical_model_family
from modules.registry.family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY
from image_gen.systems.registry import RuntimeRegistrySystem

from .contracts import BlueprintSnapshot


class QualificationCatalogMixin:
    """Cohesive qualification-runner responsibility mixin used by the public facade."""

    def list_models(self, *, family: str = "") -> list[dict[str, Any]]:
        """List available checkpoint locations with their stored registry identities.

        This intentionally preserves duplicate file locations so the initial model chooser can
        still address a specific filename/path. Family/all-model qualification calls
        :meth:`dedupe_models_by_registry_hash` before generating.
        """
        requested_family = canonical_model_family(family)
        output: list[dict[str, Any]] = []
        for asset in self.registry.list_assets(asset_type="checkpoint", limit=1_000_000):
            asset_family = canonical_model_family(asset.architecture)
            if requested_family and asset_family != requested_family:
                continue
            if not bool(asset.exists_on_disk):
                continue
            if str(asset.location_state or "available") != "available":
                continue
            if not asset_family:
                continue
            snapshots = self.registry.get_component_snapshots(int(asset.id))
            output.append(
                {
                    "asset_id": int(asset.id),
                    "filename": str(asset.filename),
                    "path": str(asset.path),
                    "sha256": str(asset.sha256 or "").strip().lower(),
                    "blake3": str(asset.blake3 or "").strip().lower(),
                    "quick_fingerprint": str(asset.quick_fingerprint or "").strip().lower(),
                    "family": asset_family,
                    "architecture": str(asset.architecture or ""),
                    "component_roles": sorted({str(item.component_role) for item in snapshots}),
                }
            )
        output.sort(key=lambda item: (item["family"], item["filename"].casefold(), item["path"].casefold()))
        return output

    @staticmethod
    def dedupe_models_by_registry_hash(
        models: Iterable[Mapping[str, Any]],
        *,
        preferred_path: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collapse duplicate checkpoint locations using the main registry's stored SHA-256.

        The runner never hashes checkpoint payloads here. Missing strong hashes remain separate
        targets instead of being guessed equivalent from filenames or quick fingerprints.
        If the user selected one location from a duplicate group, that path becomes the
        representative so the output remains organized under the model they deliberately chose.
        """
        preferred = str(Path(preferred_path).expanduser().resolve()) if preferred_path else ""
        hashed_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        unhashed: list[dict[str, Any]] = []
        for raw in models:
            item = dict(raw)
            family = canonical_model_family(item.get("family") or item.get("architecture"))
            digest = str(item.get("sha256") or "").strip().lower()
            if not digest:
                item["dedupe_status"] = "registry_sha256_missing"
                item["duplicate_alias_count"] = 0
                item["duplicate_count"] = 0
                item["registry_location_count"] = 1
                item["duplicate_aliases"] = []
                unhashed.append(item)
                continue
            hashed_groups.setdefault((family, digest), []).append(item)

        unique: list[dict[str, Any]] = []
        group_records: list[dict[str, Any]] = []
        for (family, digest), members in hashed_groups.items():
            members.sort(
                key=lambda item: (
                    str(item.get("filename") or "").casefold(),
                    str(item.get("path") or "").casefold(),
                )
            )
            representative = next(
                (
                    item
                    for item in members
                    if str(Path(str(item.get("path") or "")).expanduser().resolve()) == preferred
                ),
                members[0],
            )
            aliases = [
                {
                    "asset_id": item.get("asset_id"),
                    "filename": item.get("filename"),
                    "path": item.get("path"),
                    "family": item.get("family"),
                }
                for item in members
                if item is not representative
            ]
            selected = dict(representative)
            selected["dedupe_status"] = "registry_sha256_exact"
            selected["duplicate_alias_count"] = len(aliases)
            # Keep the older field for CLI/backward compatibility; it means aliases collapsed.
            selected["duplicate_count"] = len(aliases)
            selected["registry_location_count"] = len(members)
            selected["duplicate_aliases"] = copy.deepcopy(aliases)
            unique.append(selected)
            group_records.append(
                {
                    "family": family,
                    "sha256": digest,
                    "representative_asset_id": selected.get("asset_id"),
                    "representative_filename": selected.get("filename"),
                    "representative_path": selected.get("path"),
                    "registry_location_count": len(members),
                    "duplicate_alias_count": len(aliases),
                    "aliases": copy.deepcopy(aliases),
                }
            )

        unique.extend(unhashed)
        unique.sort(
            key=lambda item: (
                str(item.get("family") or ""),
                str(item.get("filename") or "").casefold(),
                str(item.get("path") or "").casefold(),
            )
        )
        group_records.sort(
            key=lambda item: (
                str(item.get("family") or ""),
                str(item.get("representative_filename") or "").casefold(),
            )
        )
        return unique, group_records

    def component_variant_groups(self, *, family: str, role: str) -> list[dict[str, Any]]:
        """Group original-blueprint component variants by exact component SHA-256.

        A future workspace can render each returned record as a column. The model list under a
        column represents distinct checkpoint *contents* whose original blueprint embeds the
        same exact component bytes. Duplicate checkpoint locations are collapsed first.

        ``identity_status`` proves byte identity. ``runtime_parity_status`` deliberately starts
        untested: image qualification is evidence that extraction/resolution executes equivalently,
        not evidence needed to decide whether identical hashes are identical bytes.
        """
        canonical_family = canonical_model_family(family)
        canonical_role = str(role or "").strip()
        if not canonical_family or not canonical_role:
            return []

        raw_models = self.list_models(family=canonical_family)
        models, _ = self.dedupe_models_by_registry_hash(raw_models)
        grouped: dict[str, dict[str, Any]] = {}
        for model in models:
            snapshots = self.registry.get_component_snapshots(int(model["asset_id"]))
            role_digests = sorted(
                {
                    str(snapshot.component_sha256 or "").strip().lower()
                    for snapshot in snapshots
                    if str(snapshot.component_role or "").strip() == canonical_role
                    and str(snapshot.component_sha256 or "").strip()
                }
            )
            for digest in role_digests:
                group = grouped.setdefault(
                    digest,
                    {
                        "variant_id": f"{canonical_role}-{digest[:12]}",
                        "family": canonical_family,
                        "role": canonical_role,
                        "component_sha256": digest,
                        "identity_status": "exact_component_sha256",
                        "runtime_parity_status": "untested",
                        "models": [],
                    },
                )
                group["models"].append(
                    {
                        "asset_id": model.get("asset_id"),
                        "filename": model.get("filename"),
                        "path": model.get("path"),
                        "model_sha256": model.get("sha256"),
                        "registry_location_count": model.get("registry_location_count", 1),
                        "duplicate_alias_count": model.get("duplicate_alias_count", 0),
                        "duplicate_aliases": copy.deepcopy(model.get("duplicate_aliases") or []),
                    }
                )

        result = list(grouped.values())
        for group in result:
            group["models"].sort(
                key=lambda item: (
                    str(item.get("filename") or "").casefold(),
                    str(item.get("path") or "").casefold(),
                )
            )
            group["model_count"] = len(group["models"])
            group["registry_location_count"] = sum(
                int(item.get("registry_location_count") or 1) for item in group["models"]
            )
        result.sort(
            key=lambda item: (
                -int(item.get("model_count") or 0),
                str(item.get("component_sha256") or ""),
            )
        )
        return result

    def blueprint_for_model(self, model_path: str | Path) -> BlueprintSnapshot:
        path = Path(model_path).expanduser().resolve()
        asset = self.registry.get_asset_by_path(str(path))
        if asset is None:
            raise ValueError(
                f"The model is not registered: {path}. Refresh the asset/component registry before qualification."
            )
        if str(asset.asset_type or "").strip().lower() != "checkpoint":
            raise ValueError(f"Qualification base must be a registered checkpoint: {path}")
        family = canonical_model_family(asset.architecture)
        provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(family)
        if provider is None:
            raise ValueError(f"No component provider is registered for architecture {asset.architecture!r}.")

        snapshots = self.registry.get_component_snapshots(int(asset.id))
        by_role: dict[str, list[Any]] = {}
        for snapshot in snapshots:
            role = str(snapshot.component_role or "").strip()
            if role:
                by_role.setdefault(role, []).append(snapshot)

        components: dict[str, str] = {}
        details: dict[str, dict[str, Any]] = {}
        missing_required: list[str] = []
        for definition in provider.role_definitions():
            role = definition.canonical_role_id
            role_snapshots = sorted(by_role.get(role, []), key=lambda item: int(item.id))
            if not role_snapshots:
                if definition.required:
                    missing_required.append(role)
                continue
            # A checkpoint should have one current component snapshot per canonical role.
            # If historical rows exist, prefer the newest row deterministically.
            snapshot = role_snapshots[-1]
            digest = str(snapshot.component_sha256 or "").strip().lower()
            if not digest:
                if definition.required:
                    missing_required.append(role)
                continue
            components[role] = digest
            details[role] = {
                "component_sha256": digest,
                "structure_sha256": str(snapshot.structure_sha256 or ""),
                "tensor_count": int(snapshot.tensor_count),
                "component_bytes": int(snapshot.total_bytes),
                "snapshot_id": int(snapshot.id),
                "snapshot_version": str(snapshot.snapshot_version or ""),
            }
        if missing_required:
            raise ValueError(
                "The selected checkpoint does not have a complete observed blueprint for required roles: "
                + ", ".join(sorted(missing_required))
            )

        return BlueprintSnapshot(
            asset_id=int(asset.id),
            model_path=str(path),
            model_filename=str(asset.filename),
            model_sha256=str(asset.sha256 or ""),
            family=family,
            family_label=provider.display_label,
            base_weight_role=provider.base_weight_role,
            components=components,
            component_details=details,
        )

    def runtime_choices(self) -> dict[str, list[str]]:
        samplers = sorted(
            {str(item.name) for item in self.runtime_registry.descriptors("sampler")},
            key=str.casefold,
        )
        schedulers = sorted(
            {str(item.name) for item in self.runtime_registry.descriptors("scheduler")},
            key=str.casefold,
        )
        return {"samplers": samplers, "schedulers": schedulers}

    def component_choices(self, blueprint: BlueprintSnapshot) -> dict[str, list[dict[str, Any]]]:
        catalog = self.selection.catalog(
            base_component_sha256=blueprint.components.get(blueprint.base_weight_role, "")
        )
        family_entry = next(
            (item for item in catalog.get("families", []) if item.get("family") == blueprint.family),
            None,
        )
        if family_entry is None:
            return {}
        output: dict[str, list[dict[str, Any]]] = {}
        for role in family_entry.get("roles", []):
            values: list[dict[str, Any]] = []
            for component in role.get("components", []):
                if not bool(component.get("selectable_with_digital")):
                    continue
                values.append(
                    {
                        "value": str(component.get("component_sha256") or ""),
                        "label": str(component.get("selection_label") or component.get("display_name") or ""),
                        "source_status": str(component.get("source_status_label") or ""),
                        "validation": str((component.get("phase05") or {}).get("validation_label") or "Untested"),
                    }
                )
            output[str(role.get("role") or "")] = values
        return output

    def _component_source_payload(
        self,
        *,
        asset_id: int,
        role: str,
        component_sha256: str,
        force_digital_extract: bool = False,
    ) -> dict[str, Any]:
        asset = self.registry.get_asset_by_id(int(asset_id))
        if asset is None:
            raise ValueError(f"Unknown component source asset_id={asset_id}.")
        digest = str(component_sha256 or "").strip().lower()
        snapshot = next(
            (
                item
                for item in self.registry.get_component_snapshots(int(asset_id))
                if str(item.component_role or "").strip() == str(role or "").strip()
                and str(item.component_sha256 or "").strip().lower() == digest
            ),
            None,
        )
        if snapshot is None:
            raise ValueError(
                f"Asset {asset.filename!r} does not contain role {role!r} with component hash {digest}."
            )
        payload = dict(self.selection._source_payload(asset, snapshot))
        if force_digital_extract:
            payload["force_digital_extract"] = True
        return payload
