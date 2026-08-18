from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3

from .architecture_observation import (
    derive_architecture_state,
    normalize_architecture_identifier,
    normalize_asset_type,
)
from .evidence_contracts import (
    RELATIONSHIP_SOURCE_EXACT_ANALYSIS,
    RELATIONSHIP_SOURCE_RECORDED,
    RELATIONSHIP_STATUS_ACTIVE,
    RelationshipParticipant,
    relationship_key,
)
from .family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY
from .models import (
    LOCATION_STATE_ARCHIVED,
    LOCATION_STATE_AVAILABLE,
    LOCATION_STATE_INACCESSIBLE,
    LOCATION_STATE_MISSING,
    LOCATION_STATE_MOVED_RELINKED,
)


class RegistryMigrations:
    """Metadata-only registry schema migrations and backfills."""

    @staticmethod
    def _relationship_backfill_key(
        *,
        relationship_type: str,
        source_component_sha256: str,
        target_component_sha256: str,
    ) -> str:
        participants = [
            RelationshipParticipant(
                component_sha256=str(source_component_sha256),
                participant_role="left",
                position=0,
            ),
            RelationshipParticipant(
                component_sha256=str(target_component_sha256),
                participant_role="right",
                position=1,
            ),
        ]
        return relationship_key(
            relationship_type=str(relationship_type),
            participants=participants,
        )

    def _migrate_relationship_evidence_boundary(self, conn: sqlite3.Connection) -> None:
        """Backfill ML-F03 pairwise evidence into the ML-F04 generic boundary.

        The migration is metadata-only: it reuses existing immutable component
        fingerprints/evidence JSON and never opens model payloads.
        """
        rows = conn.execute(
            "SELECT * FROM component_relationships ORDER BY id ASC"
        ).fetchall()
        for row in rows:
            source = str(row["source_component_sha256"] or "").strip().lower()
            target = str(row["target_component_sha256"] or "").strip().lower()
            relationship_type = str(row["relationship_type"] or "").strip()
            evidence_kind = str(row["evidence_kind"] or "").strip()
            evidence_version = str(row["evidence_version"] or "").strip()
            if not source or not target or not relationship_type or not evidence_kind or not evidence_version:
                continue
            rel_key = self._relationship_backfill_key(
                relationship_type=relationship_type,
                source_component_sha256=source,
                target_component_sha256=target,
            )
            evidence_source = (
                RELATIONSHIP_SOURCE_EXACT_ANALYSIS
                if "exact" in evidence_kind.casefold() or "analysis" in evidence_kind.casefold()
                else RELATIONSHIP_SOURCE_RECORDED
            )
            now = str(row["updated_at"] or row["created_at"] or datetime.now(timezone.utc).isoformat())
            created = str(row["created_at"] or now)
            cursor = conn.execute(
                """
                INSERT INTO component_relationship_evidence (
                    relationship_key, relationship_type, evidence_source,
                    evidence_kind, evidence_version, authoritative, status,
                    evidence_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                ON CONFLICT(relationship_key, evidence_source, evidence_kind, evidence_version)
                DO UPDATE SET evidence_json = excluded.evidence_json, updated_at = excluded.updated_at
                """,
                (
                    rel_key,
                    relationship_type,
                    evidence_source,
                    evidence_kind,
                    evidence_version,
                    RELATIONSHIP_STATUS_ACTIVE,
                    row["evidence_json"],
                    created,
                    now,
                ),
            )
            stored = conn.execute(
                """
                SELECT id FROM component_relationship_evidence
                WHERE relationship_key = ? AND evidence_source = ? AND evidence_kind = ? AND evidence_version = ?
                """,
                (rel_key, evidence_source, evidence_kind, evidence_version),
            ).fetchone()
            if stored is None:
                continue
            evidence_id = int(stored["id"])
            for position, role, digest in ((0, "left", source), (1, "right", target)):
                conn.execute(
                    """
                    INSERT INTO component_relationship_participants (
                        relationship_evidence_id, position, participant_role,
                        component_sha256, metadata_json
                    ) VALUES (?, ?, ?, ?, '{}')
                    ON CONFLICT(relationship_evidence_id, position)
                    DO UPDATE SET participant_role = excluded.participant_role,
                                  component_sha256 = excluded.component_sha256
                    """,
                    (evidence_id, position, role, digest),
                )

    @staticmethod
    def _component_validation_schema_needs_migration(conn: sqlite3.Connection) -> bool:
        required_columns = {
            "provider_version",
            "composition_sha256",
            "validation_stage",
            "validation_result",
            "blocking_state",
            "environment_json",
            "evidence_artifact",
            "error_category",
            "error_message",
            "runtime_version",
            "created_at",
            "updated_at",
        }
        required_indexes = {
            "idx_component_validations_composition",
            "idx_component_validations_stage",
            "idx_component_validations_result",
        }
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(component_validations)").fetchall()
        }
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(component_validations)").fetchall()
        }
        return not required_columns.issubset(columns) or not required_indexes.issubset(indexes)

    def _migrate_component_validation_evidence(self, conn: sqlite3.Connection) -> None:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(component_validations)").fetchall()}
        additions = {
            "provider_version": "TEXT",
            "composition_sha256": "TEXT",
            "validation_stage": "TEXT",
            "validation_result": "TEXT",
            "blocking_state": "TEXT NOT NULL DEFAULT 'advisory'",
            "environment_json": "TEXT",
            "evidence_artifact": "TEXT",
            "error_category": "TEXT",
            "error_message": "TEXT",
            "runtime_version": "TEXT",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        }
        for name, declaration in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE component_validations ADD COLUMN {name} {declaration}")
        conn.execute(
            """
            UPDATE component_validations
            SET validation_stage = COALESCE(NULLIF(validation_stage, ''), evidence_type, 'structural'),
                validation_result = COALESCE(NULLIF(validation_result, ''), validation_state, 'error'),
                blocking_state = COALESCE(NULLIF(blocking_state, ''), 'advisory'),
                created_at = COALESCE(NULLIF(created_at, ''), validated_at),
                updated_at = COALESCE(NULLIF(updated_at, ''), validated_at)
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_component_validations_composition ON component_validations(composition_sha256)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_component_validations_stage ON component_validations(validation_stage)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_component_validations_result ON component_validations(validation_result)")

    def _migrate_asset_location_state(self, conn: sqlite3.Connection) -> None:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(assets)").fetchall()}
        location_state_added = "location_state" not in columns
        if location_state_added:
            conn.execute(
                "ALTER TABLE assets ADD COLUMN location_state TEXT NOT NULL DEFAULT 'available'"
            )
            # SQLite applies the column default to every legacy row. Backfill from
            # the pre-v5 source of truth before treating the new values as valid.
            conn.execute(
                "UPDATE assets SET location_state = CASE WHEN exists_on_disk = 1 THEN ? ELSE ? END",
                (LOCATION_STATE_AVAILABLE, LOCATION_STATE_MISSING),
            )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assets_location_state ON assets(location_state)"
        )
        conn.execute(
            "UPDATE assets SET location_state = CASE WHEN exists_on_disk = 1 THEN ? ELSE ? END "
            "WHERE COALESCE(TRIM(location_state), '') = ''",
            (LOCATION_STATE_AVAILABLE, LOCATION_STATE_MISSING),
        )
        conn.execute(
            "UPDATE assets SET location_state = CASE WHEN exists_on_disk = 1 THEN ? ELSE ? END "
            "WHERE location_state NOT IN (?, ?, ?, ?, ?)",
            (
                LOCATION_STATE_AVAILABLE,
                LOCATION_STATE_MISSING,
                LOCATION_STATE_AVAILABLE,
                LOCATION_STATE_ARCHIVED,
                LOCATION_STATE_MISSING,
                LOCATION_STATE_MOVED_RELINKED,
                LOCATION_STATE_INACCESSIBLE,
            ),
        )

    def _migrate_architecture_observation_state(self, conn: sqlite3.Connection) -> None:
        """Normalize placeholder architecture labels into explicit observation state.

        This migration is metadata-only. It never opens or hashes model payloads.
        Historical values such as ``unknown``/``unknown_model`` are removed from
        canonical architecture/provider-family fields while preserving the raw
        detector value in asset metadata when useful.
        """
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(assets)").fetchall()}
        if "architecture_state" not in columns:
            conn.execute(
                "ALTER TABLE assets ADD COLUMN architecture_state TEXT NOT NULL DEFAULT 'observed_unclassified'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assets_architecture_state ON assets(architecture_state)"
        )

        provider_ids = {provider.family_id for provider in DEFAULT_FAMILY_PROVIDER_REGISTRY.providers()}
        rows = conn.execute(
            "SELECT id, asset_type, format_type, architecture, metadata_json FROM assets"
        ).fetchall()
        for row in rows:
            raw_architecture = str(row["architecture"] or "")
            architecture = normalize_architecture_identifier(raw_architecture)
            provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(architecture) if architecture else None
            state = derive_architecture_state(
                architecture=architecture,
                provider_supported=provider is not None,
                format_type=row["format_type"],
            )
            asset_type = normalize_asset_type(row["asset_type"], format_type=row["format_type"])
            metadata = self._json_mapping(row["metadata_json"])
            if raw_architecture.strip() and architecture != raw_architecture.strip().lower():
                metadata.setdefault("reported_architecture", raw_architecture)
            metadata["architecture_state"] = state
            conn.execute(
                """
                UPDATE assets
                SET asset_type = ?, architecture = ?, architecture_state = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    asset_type,
                    architecture,
                    state,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    int(row["id"]),
                ),
            )

        source_rows = conn.execute(
            "SELECT id, provider_family, provider_version FROM component_sources"
        ).fetchall()
        for row in source_rows:
            family = normalize_architecture_identifier(row["provider_family"])
            if family not in provider_ids:
                conn.execute(
                    "UPDATE component_sources SET provider_family = NULL, provider_version = '' WHERE id = ?",
                    (int(row["id"]),),
                )

    def _migrate_normalized_component_records(self, conn: sqlite3.Connection) -> None:
        """Backfill Phase 01 identity/source rows from existing snapshot evidence.

        Migration is intentionally payload-free: it only transforms hashes and
        metadata already stored in ``asset_components``. Opening an existing
        registry must never trigger a model-file rehash.
        """
        rows = conn.execute(
            """
            SELECT ac.*, a.asset_type, a.architecture, a.exists_on_disk, a.path
            FROM asset_components ac
            JOIN assets a ON a.id = ac.asset_id
            ORDER BY ac.id ASC
            """
        ).fetchall()
        for row in rows:
            self._upsert_normalized_component_row(
                conn,
                asset_id=int(row["asset_id"]),
                asset_type=normalize_asset_type(row["asset_type"], format_type=""),
                architecture=normalize_architecture_identifier(row["architecture"]),
                exists_on_disk=bool(row["exists_on_disk"]),
                snapshot_at=str(row["snapshot_at"] or ""),
                snapshot_version=str(row["snapshot_version"] or ""),
                component_role=str(row["component_role"] or ""),
                source_prefixes_json=row["source_prefixes_json"],
                tensor_count=int(row["tensor_count"] or 0),
                total_bytes=int(row["total_bytes"] or 0),
                component_sha256=str(row["component_sha256"] or ""),
                structure_sha256=str(row["structure_sha256"] or ""),
                metadata_json=row["metadata_json"],
            )
