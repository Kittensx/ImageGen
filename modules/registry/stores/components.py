from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
import json
import sqlite3

from ..analysis_contracts import TensorHashManifest
from ..architecture_observation import normalize_architecture_identifier, normalize_asset_type
from ..component_snapshot import ComponentSnapshot
from ..contracts import (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_MISSING,
    SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT,
    SOURCE_FORM_PHYSICAL_COMPONENT,
    SOURCE_FORM_RECONSTRUCTED_EXPORT,
    SOURCE_FORM_STANDALONE_SHARED,
    SOURCE_FORM_UNKNOWN,
    source_form_for_asset_type,
)
from ..family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY
from ..models import (
    AssetRecord,
    ComponentIdentityRecord,
    ComponentSnapshotRecord,
    ComponentSourceRecord,
)


class ComponentStore:
    """Component snapshot, identity, source, manifest, blueprint, and occurrence persistence operations."""

    def _upsert_normalized_component_row(
        self,
        conn: sqlite3.Connection,
        *,
        asset_id: int,
        asset_type: str,
        architecture: str,
        exists_on_disk: bool,
        snapshot_at: str,
        snapshot_version: str,
        component_role: str,
        source_prefixes_json: str | None,
        tensor_count: int,
        total_bytes: int,
        component_sha256: str,
        structure_sha256: str,
        metadata_json: str | None,
    ) -> None:
        digest = str(component_sha256 or "").strip().lower()
        if not digest:
            return
        metadata = self._json_mapping(metadata_json)
        source_form = source_form_for_asset_type(asset_type, metadata=metadata)
        embedded_state = "embedded" if source_form == SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT else (
            "standalone" if source_form == SOURCE_FORM_STANDALONE_SHARED else "unknown"
        )
        provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(architecture)
        provider_family = provider.family_id if provider is not None else None
        provider_version = provider.version if provider is not None else ""
        availability = AVAILABILITY_AVAILABLE if exists_on_disk else AVAILABILITY_MISSING
        identity_metadata = {
            "identity_basis": metadata.get("identity_basis", "normalized_tensor_schema_plus_payload"),
            "role_is_identity": False,
        }
        conn.execute(
            """
            INSERT INTO component_identities (
                component_sha256, structure_sha256, total_bytes, tensor_count,
                first_seen_at, last_seen_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(component_sha256) DO UPDATE SET
                last_seen_at = excluded.last_seen_at,
                structure_sha256 = COALESCE(component_identities.structure_sha256, excluded.structure_sha256),
                total_bytes = COALESCE(component_identities.total_bytes, excluded.total_bytes),
                tensor_count = COALESCE(component_identities.tensor_count, excluded.tensor_count)
            """,
            (
                digest,
                str(structure_sha256 or "") or None,
                int(total_bytes),
                int(tensor_count),
                snapshot_at,
                snapshot_at,
                json.dumps(identity_metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        try:
            parsed_prefixes = json.loads(source_prefixes_json or "[]") if source_prefixes_json else []
            if not isinstance(parsed_prefixes, list):
                parsed_prefixes = []
        except Exception:
            parsed_prefixes = []
        locator = {
            "source_prefixes": parsed_prefixes,
            "locator_version": "component-source-locator-v1",
        }
        conn.execute(
            """
            INSERT INTO component_sources (
                component_sha256, asset_id, component_role, source_form, embedded_state,
                provider_family, provider_version, availability_state, locator_json,
                scan_timestamp, scanner_version, snapshot_version, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(component_sha256, asset_id, component_role, snapshot_version) DO UPDATE SET
                source_form = excluded.source_form,
                embedded_state = excluded.embedded_state,
                provider_family = excluded.provider_family,
                provider_version = excluded.provider_version,
                availability_state = excluded.availability_state,
                locator_json = excluded.locator_json,
                scan_timestamp = excluded.scan_timestamp,
                scanner_version = excluded.scanner_version,
                metadata_json = excluded.metadata_json
            """,
            (
                digest,
                int(asset_id),
                component_role,
                source_form,
                embedded_state,
                provider_family,
                provider_version,
                availability,
                json.dumps(locator, ensure_ascii=False, sort_keys=True),
                snapshot_at,
                snapshot_version,
                snapshot_version,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
        )

    def store_component_snapshots(
        self,
        asset_id: int,
        snapshots: dict[str, ComponentSnapshot],
        *,
        source_file_sha256: str | None = None,
        source_quick_fingerprint: str | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> list[ComponentSnapshotRecord]:
        """Persist a complete deterministic component snapshot set.

        This public signature is intentionally stable. CNRR-08 incremental
        read-through discovery uses ``merge_component_snapshots`` so existing
        registry callers continue to receive replacement semantics.
        """
        return self._store_component_snapshots_impl(
            asset_id,
            snapshots,
            source_file_sha256=source_file_sha256,
            source_quick_fingerprint=source_quick_fingerprint,
            metadata_extra=metadata_extra,
            replace_snapshot_version=True,
        )

    def merge_component_snapshots(
        self,
        asset_id: int,
        snapshots: dict[str, ComponentSnapshot],
        *,
        source_file_sha256: str | None = None,
        source_quick_fingerprint: str | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> list[ComponentSnapshotRecord]:
        """Transactionally add/update only the supplied component roles.

        Existing current-version roles remain untouched. This is safe only when
        caller freshness checks have already proven those existing rows current.
        """
        return self._store_component_snapshots_impl(
            asset_id,
            snapshots,
            source_file_sha256=source_file_sha256,
            source_quick_fingerprint=source_quick_fingerprint,
            metadata_extra=metadata_extra,
            replace_snapshot_version=False,
        )

    def _store_component_snapshots_impl(
        self,
        asset_id: int,
        snapshots: dict[str, ComponentSnapshot],
        *,
        source_file_sha256: str | None,
        source_quick_fingerprint: str | None,
        metadata_extra: dict[str, Any] | None,
        replace_snapshot_version: bool,
    ) -> list[ComponentSnapshotRecord]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            asset_row = conn.execute(
                "SELECT asset_type, architecture, exists_on_disk FROM assets WHERE id = ?",
                (int(asset_id),),
            ).fetchone()
            if asset_row is None:
                raise ValueError(f"Cannot store component snapshots for unknown asset_id={asset_id}.")
            snapshot_versions = {snapshot.snapshot_version for snapshot in snapshots.values()}
            if replace_snapshot_version:
                for version in snapshot_versions:
                    conn.execute(
                        "DELETE FROM asset_components WHERE asset_id = ? AND snapshot_version = ?",
                        (int(asset_id), version),
                    )
                    conn.execute(
                        "DELETE FROM component_sources WHERE asset_id = ? AND snapshot_version = ?",
                        (int(asset_id), version),
                    )
            for role, snapshot in sorted(snapshots.items()):
                metadata = {
                    "source_file_sha256": str(source_file_sha256 or ""),
                    "source_quick_fingerprint": str(source_quick_fingerprint or ""),
                    "identity_basis": "normalized_tensor_schema_plus_payload",
                    "role_is_identity": False,
                }
                if metadata_extra:
                    metadata.update(dict(metadata_extra))
                conn.execute(
                    """
                    INSERT INTO asset_components (
                        asset_id, snapshot_at, snapshot_version, component_role,
                        source_prefixes_json, tensor_count, total_bytes,
                        component_sha256, structure_sha256, dtype_summary_json,
                        tensor_manifest_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset_id, snapshot_version, component_role) DO UPDATE SET
                        snapshot_at = excluded.snapshot_at,
                        source_prefixes_json = excluded.source_prefixes_json,
                        tensor_count = excluded.tensor_count,
                        total_bytes = excluded.total_bytes,
                        component_sha256 = excluded.component_sha256,
                        structure_sha256 = excluded.structure_sha256,
                        dtype_summary_json = excluded.dtype_summary_json,
                        tensor_manifest_json = excluded.tensor_manifest_json,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        int(asset_id),
                        now,
                        snapshot.snapshot_version,
                        role,
                        json.dumps(list(snapshot.source_prefixes), ensure_ascii=False),
                        int(snapshot.tensor_count),
                        int(snapshot.total_bytes),
                        snapshot.component_sha256,
                        snapshot.structure_sha256,
                        json.dumps(snapshot.dtype_summary, ensure_ascii=False, sort_keys=True),
                        json.dumps(
                            [item.to_dict() for item in snapshot.tensors],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    ),
                )
                self._upsert_normalized_component_row(
                    conn,
                    asset_id=int(asset_id),
                    asset_type=normalize_asset_type(asset_row["asset_type"], format_type=""),
                    architecture=normalize_architecture_identifier(asset_row["architecture"]),
                    exists_on_disk=bool(asset_row["exists_on_disk"]),
                    snapshot_at=now,
                    snapshot_version=snapshot.snapshot_version,
                    component_role=role,
                    source_prefixes_json=json.dumps(list(snapshot.source_prefixes), ensure_ascii=False),
                    tensor_count=int(snapshot.tensor_count),
                    total_bytes=int(snapshot.total_bytes),
                    component_sha256=snapshot.component_sha256,
                    structure_sha256=snapshot.structure_sha256,
                    metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                )
            rows = conn.execute(
                """
                SELECT * FROM asset_components
                WHERE asset_id = ?
                ORDER BY component_role ASC
                """,
                (int(asset_id),),
            ).fetchall()
        return [self._row_to_component_snapshot_record(row) for row in rows]

    def get_component_snapshots(self, asset_id: int) -> list[ComponentSnapshotRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM asset_components
                WHERE asset_id = ?
                ORDER BY component_role ASC
                """,
                (int(asset_id),),
            ).fetchall()
        return [self._row_to_component_snapshot_record(row) for row in rows]

    def find_components_by_sha256(
        self, component_sha256: str, limit: int = 100
    ) -> list[ComponentSnapshotRecord]:
        value = str(component_sha256 or "").strip().lower()
        if not value:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM asset_components
                WHERE component_sha256 = ?
                ORDER BY snapshot_at DESC, id DESC
                LIMIT ?
                """,
                (value, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_component_snapshot_record(row) for row in rows]

    def find_components_by_structure_sha256(
        self, structure_sha256: str, limit: int = 100
    ) -> list[ComponentSnapshotRecord]:
        value = str(structure_sha256 or "").strip().lower()
        if not value:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM asset_components
                WHERE structure_sha256 = ?
                ORDER BY snapshot_at DESC, id DESC
                LIMIT ?
                """,
                (value, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_component_snapshot_record(row) for row in rows]

    def list_component_snapshots(self, limit: int = 1000000) -> list[ComponentSnapshotRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM asset_components
                ORDER BY component_sha256 ASC, asset_id ASC, component_role ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [self._row_to_component_snapshot_record(row) for row in rows]

    def list_component_identities(self, limit: int = 1000000) -> list[ComponentIdentityRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM component_identities
                ORDER BY component_sha256 ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [self._row_to_component_identity_record(row) for row in rows]

    def get_component_identity(self, component_sha256: str) -> ComponentIdentityRecord | None:
        digest = str(component_sha256 or "").strip().lower()
        if not digest:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM component_identities WHERE component_sha256 = ?",
                (digest,),
            ).fetchone()
        return self._row_to_component_identity_record(row) if row else None

    def list_component_sources(
        self,
        *,
        component_sha256: str | None = None,
        asset_id: int | None = None,
        family: str | None = None,
        role: str | None = None,
        available_only: bool = False,
        limit: int = 1000000,
    ) -> list[ComponentSourceRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if component_sha256:
            clauses.append("component_sha256 = ?")
            params.append(str(component_sha256).strip().lower())
        if asset_id is not None:
            clauses.append("asset_id = ?")
            params.append(int(asset_id))
        if family:
            clauses.append("provider_family = ?")
            params.append(str(family))
        if role:
            clauses.append("component_role = ?")
            params.append(str(role))
        if available_only:
            clauses.append("availability_state = ?")
            params.append(AVAILABILITY_AVAILABLE)
        where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM component_sources{where_sql} ORDER BY component_sha256, asset_id, component_role LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_component_source_record(row) for row in rows]

    def list_model_blueprints(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_blueprints ORDER BY updated_at DESC, id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_saved_compositions(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM saved_compositions ORDER BY updated_at DESC, id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_component_analysis_manifest(
        self,
        *,
        component_sha256: str,
        provider_id: str,
        component_role: str,
        layout_version: int,
        algorithm_version: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM component_analysis_manifests
                WHERE component_sha256 = ?
                  AND provider_id = ?
                  AND component_role = ?
                  AND layout_version = ?
                  AND algorithm_version = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (
                    str(component_sha256).strip().lower(),
                    str(provider_id),
                    str(component_role),
                    int(layout_version),
                    str(algorithm_version),
                ),
            ).fetchone()
        return dict(row) if row is not None else None

    def store_component_analysis_manifest(self, manifest: TensorHashManifest) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        payload = manifest.to_dict()
        manifest_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO component_analysis_manifests (
                    component_sha256, provider_id, family_id, component_role,
                    layout_version, algorithm_version, manifest_sha256, manifest_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(component_sha256, provider_id, component_role, layout_version, algorithm_version)
                DO UPDATE SET
                    family_id = excluded.family_id,
                    manifest_sha256 = excluded.manifest_sha256,
                    manifest_json = excluded.manifest_json,
                    updated_at = excluded.updated_at
                """,
                (
                    manifest.component_sha256,
                    manifest.provider_id,
                    manifest.family_id,
                    manifest.component_role,
                    int(manifest.layout_version),
                    manifest.algorithm_version,
                    manifest.analysis_manifest_sha256,
                    manifest_json,
                    now,
                    now,
                ),
            )
        stored = self.get_component_analysis_manifest(
            component_sha256=manifest.component_sha256,
            provider_id=manifest.provider_id,
            component_role=manifest.component_role,
            layout_version=manifest.layout_version,
            algorithm_version=manifest.algorithm_version,
        )
        if stored is None:
            raise RuntimeError("Stored component analysis manifest could not be read back.")
        return stored

    def list_component_analysis_manifests(
        self,
        *,
        component_sha256: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if component_sha256:
            clauses.append("component_sha256 = ?")
            params.append(str(component_sha256).strip().lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM component_analysis_manifests {where} ORDER BY updated_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def list_component_occurrence_groups(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        identities = {item.component_sha256: item for item in self.list_component_identities(limit=1_000_000)}
        sources = self.list_component_sources(limit=1_000_000)
        assets: dict[int, AssetRecord] = {}
        groups: dict[str, list[ComponentSourceRecord]] = defaultdict(list)
        for source in sources:
            groups[source.component_sha256].append(source)

        results: list[dict[str, Any]] = []
        for digest, group in groups.items():
            if len(group) < 2:
                continue
            members: list[dict[str, Any]] = []
            physical_count = 0
            digital_count = 0
            available_count = 0
            roles: set[str] = set()
            for source in sorted(group, key=lambda item: (item.asset_id, item.component_role, item.id)):
                asset = assets.get(source.asset_id)
                if asset is None:
                    asset = self.get_asset_by_id(source.asset_id)
                    if asset is not None:
                        assets[source.asset_id] = asset
                if asset is None:
                    continue
                roles.add(source.component_role)
                effective_source_form = source.source_form
                if effective_source_form == SOURCE_FORM_UNKNOWN and asset.asset_type in {
                    "checkpoint",
                    "lora",
                    "lycoris",
                    "controlnet",
                    "hypernetwork",
                    "adapter",
                }:
                    effective_source_form = SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT
                if effective_source_form in {SOURCE_FORM_PHYSICAL_COMPONENT, SOURCE_FORM_STANDALONE_SHARED, SOURCE_FORM_RECONSTRUCTED_EXPORT}:
                    physical_count += 1
                if effective_source_form == SOURCE_FORM_DIGITAL_CHECKPOINT_COMPONENT:
                    digital_count += 1
                if source.availability_state == AVAILABILITY_AVAILABLE:
                    available_count += 1
                members.append({
                    "source_id": int(source.id),
                    "asset_id": int(source.asset_id),
                    "path": asset.path,
                    "filename": asset.filename,
                    "component_role": source.component_role,
                    "source_form": effective_source_form,
                    "embedded_state": source.embedded_state,
                    "availability_state": source.availability_state,
                    "location_state": asset.location_state,
                    "provider_family": source.provider_family,
                })
            if len(members) < 2:
                continue
            identity = identities.get(digest)
            total_bytes = int(identity.total_bytes or 0) if identity is not None else 0
            results.append({
                "component_sha256": digest,
                "structure_sha256": identity.structure_sha256 if identity is not None else None,
                "total_bytes": total_bytes,
                "source_count": len(members),
                "available_source_count": available_count,
                "physical_source_count": physical_count,
                "digital_source_count": digital_count,
                "distinct_asset_count": len({item["asset_id"] for item in members}),
                "component_roles": sorted(roles),
                "redundant_embedded_bytes": total_bytes * max(0, digital_count - 1),
                "blueprint_reference_count": 0,
                "members": members,
            })
        results.sort(key=lambda item: (-item["source_count"], item["component_sha256"]))
        return results[: max(1, int(limit))]

    def _row_to_component_snapshot_record(self, row: sqlite3.Row) -> ComponentSnapshotRecord:
        return ComponentSnapshotRecord(
            id=row["id"],
            asset_id=row["asset_id"],
            snapshot_at=row["snapshot_at"],
            snapshot_version=row["snapshot_version"],
            component_role=row["component_role"],
            source_prefixes_json=row["source_prefixes_json"],
            tensor_count=row["tensor_count"],
            total_bytes=row["total_bytes"],
            component_sha256=row["component_sha256"],
            structure_sha256=row["structure_sha256"],
            dtype_summary_json=row["dtype_summary_json"],
            tensor_manifest_json=row["tensor_manifest_json"],
            metadata_json=row["metadata_json"],
        )

    def _row_to_component_identity_record(self, row: sqlite3.Row) -> ComponentIdentityRecord:
        return ComponentIdentityRecord(
            component_sha256=row["component_sha256"],
            structure_sha256=row["structure_sha256"],
            total_bytes=row["total_bytes"],
            tensor_count=row["tensor_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            metadata_json=row["metadata_json"],
        )

    def _row_to_component_source_record(self, row: sqlite3.Row) -> ComponentSourceRecord:
        return ComponentSourceRecord(
            id=row["id"],
            component_sha256=row["component_sha256"],
            asset_id=row["asset_id"],
            component_role=row["component_role"],
            source_form=row["source_form"],
            embedded_state=row["embedded_state"],
            provider_family=row["provider_family"],
            provider_version=row["provider_version"],
            availability_state=row["availability_state"],
            locator_json=row["locator_json"],
            scan_timestamp=row["scan_timestamp"],
            scanner_version=row["scanner_version"],
            snapshot_version=row["snapshot_version"],
            metadata_json=row["metadata_json"],
        )
