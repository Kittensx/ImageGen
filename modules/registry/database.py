from __future__ import annotations

from typing import Any, Iterable, Mapping
import json
import sqlite3

from .schema import REGISTRY_SCHEMA_VERSION, SCHEMA_SQL


class RegistryDatabase:
    """Shared SQLite connection, schema initialization, and serialization helpers."""

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            row = conn.execute(
                "SELECT meta_value FROM registry_schema_meta WHERE meta_key = 'schema_version'"
            ).fetchone()
            try:
                current_version = int(row[0]) if row is not None else 0
            except (TypeError, ValueError):
                current_version = 0
            if current_version < 3:
                self._migrate_normalized_component_records(conn)
            if current_version < 4:
                self._migrate_architecture_observation_state(conn)
            if current_version < 5:
                self._migrate_asset_location_state(conn)
            if current_version < 6:
                self._migrate_relationship_evidence_boundary(conn)
            if current_version < 7 or self._component_validation_schema_needs_migration(conn):
                self._migrate_component_validation_evidence(conn)
            current_version = REGISTRY_SCHEMA_VERSION
            conn.execute(
                "INSERT INTO registry_schema_meta(meta_key, meta_value) VALUES('schema_version', ?) "
                "ON CONFLICT(meta_key) DO UPDATE SET meta_value = excluded.meta_value",
                (str(current_version),),
            )

    @staticmethod
    def _json_mapping(value: str | None) -> dict[str, Any]:
        try:
            payload = json.loads(value or "{}")
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    @staticmethod
    def _json_dump(value: Mapping[str, Any]) -> str:
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)

    def _merge_asset_metadata(
        self,
        conn: sqlite3.Connection,
        asset_id: int,
        updates: Mapping[str, Any] | None = None,
        removals: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        row = conn.execute("SELECT metadata_json FROM assets WHERE id = ?", (int(asset_id),)).fetchone()
        metadata = self._json_mapping(row["metadata_json"] if row else None)
        for key in removals or ():
            metadata.pop(str(key), None)
        if updates:
            metadata.update(dict(updates))
        conn.execute(
            "UPDATE assets SET metadata_json = ? WHERE id = ?",
            (self._json_dump(metadata), int(asset_id)),
        )
        return metadata
