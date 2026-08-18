from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional
import json
import sqlite3

from ..architecture_observation import (
    derive_architecture_state,
    normalize_architecture_identifier,
    normalize_asset_type,
)
from ..family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY
from ..models import InspectionRecord, LoadHistoryRecord


class DiagnosticsStore:
    """Inspection, load-history, health, and registry-metric persistence operations."""

    def registry_health(self) -> dict[str, Any]:
        with self._connect() as conn:
            counts = {
                "assets": int(conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]),
                "existing_assets": int(conn.execute("SELECT COUNT(*) FROM assets WHERE exists_on_disk = 1").fetchone()[0]),
                "component_snapshots": int(conn.execute("SELECT COUNT(*) FROM asset_components").fetchone()[0]),
                "unique_components": int(conn.execute("SELECT COUNT(*) FROM component_identities").fetchone()[0]),
                "component_sources": int(conn.execute("SELECT COUNT(*) FROM component_sources").fetchone()[0]),
                "available_component_sources": int(conn.execute("SELECT COUNT(*) FROM component_sources WHERE availability_state = 'available'").fetchone()[0]),
                "blueprints": int(conn.execute("SELECT COUNT(*) FROM model_blueprints").fetchone()[0]),
                "saved_compositions": int(conn.execute("SELECT COUNT(*) FROM saved_compositions").fetchone()[0]),
                "component_policies": int(conn.execute("SELECT COUNT(*) FROM component_policies").fetchone()[0]),
                "component_validations": int(conn.execute("SELECT COUNT(*) FROM component_validations").fetchone()[0]),
                "model_split_eligibility": int(conn.execute("SELECT COUNT(*) FROM model_split_eligibility").fetchone()[0]),
                "component_analysis_manifests": int(conn.execute("SELECT COUNT(*) FROM component_analysis_manifests").fetchone()[0]),
                "component_relationships": int(conn.execute("SELECT COUNT(*) FROM component_relationships").fetchone()[0]),
                "relationship_evidence": int(conn.execute("SELECT COUNT(*) FROM component_relationship_evidence").fetchone()[0]),
                "relationship_participants": int(conn.execute("SELECT COUNT(*) FROM component_relationship_participants").fetchone()[0]),
            }
            schema_row = conn.execute(
                "SELECT meta_value FROM registry_schema_meta WHERE meta_key = 'schema_version'"
            ).fetchone()
        counts["schema_version"] = int(schema_row[0]) if schema_row and str(schema_row[0]).isdigit() else None
        return counts

    def list_registry_metrics(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT metric_key, metric_value_json, calculated_at, calculation_version FROM registry_metrics ORDER BY metric_key"
            ).fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            try:
                value = json.loads(row["metric_value_json"])
            except Exception:
                value = row["metric_value_json"]
            result[str(row["metric_key"])] = {
                "value": value,
                "calculated_at": row["calculated_at"],
                "calculation_version": row["calculation_version"],
            }
        return result

    def store_inspection(self, asset_id: int, report: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()

        metadata = dict(report.get("metadata", {}) or {})
        prefix_summary = report.get("prefix_summary", {})
        example_keys = report.get("example_keys", [])
        dtype_summary = report.get("dtype_summary", {})
        tensor_shape_summary = report.get("tensor_shape_summary", {})

        format_type = str(report.get("format_type") or "other").strip().lower() or "other"
        raw_architecture = str(report.get("architecture") or "").strip()
        architecture = normalize_architecture_identifier(raw_architecture)
        provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(architecture) if architecture else None
        architecture_state = derive_architecture_state(
            architecture=architecture,
            provider_supported=provider is not None,
            format_type=format_type,
            explicit_state=report.get("architecture_state"),
        )
        asset_type = normalize_asset_type(report.get("asset_type"), format_type=format_type)
        if raw_architecture and architecture != raw_architecture.lower():
            metadata.setdefault("reported_architecture", raw_architecture)
        metadata["architecture_state"] = architecture_state

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE assets
                SET asset_type = ?,
                    format_type = ?,
                    architecture = ?,
                    architecture_state = ?,
                    checkpoint_kind = ?,
                    has_unet = ?,
                    has_vae = ?,
                    has_text_encoder = ?,
                    has_text_encoder_2 = ?,
                    key_count = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    asset_type,
                    format_type,
                    architecture,
                    architecture_state,
                    report.get("checkpoint_kind", "unknown"),
                    int(report.get("has_unet", False)),
                    int(report.get("has_vae", False)),
                    int(report.get("has_text_encoder", False)),
                    int(report.get("has_text_encoder_2", False)),
                    report.get("key_count"),
                    json.dumps(metadata, ensure_ascii=False),
                    asset_id,
                ),
            )

            conn.execute(
                """
                UPDATE component_sources
                SET provider_family = ?, provider_version = ?
                WHERE asset_id = ?
                """,
                (
                    provider.family_id if provider is not None else None,
                    provider.version if provider is not None else "",
                    int(asset_id),
                ),
            )

            stored_report = dict(report)
            stored_report["asset_type"] = asset_type
            stored_report["format_type"] = format_type
            stored_report["architecture"] = architecture or None
            stored_report["architecture_state"] = architecture_state
            stored_report["metadata"] = metadata
            if raw_architecture and architecture != raw_architecture.lower():
                stored_report["reported_architecture"] = raw_architecture

            conn.execute(
                """
                INSERT INTO asset_inspections (
                    asset_id,
                    inspected_at,
                    inspector_version,
                    key_count,
                    prefix_summary_json,
                    example_keys_json,
                    dtype_summary_json,
                    tensor_shape_summary_json,
                    result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    now,
                    report.get("inspector_version", "1"),
                    report.get("key_count"),
                    json.dumps(prefix_summary, ensure_ascii=False),
                    json.dumps(example_keys, ensure_ascii=False),
                    json.dumps(dtype_summary, ensure_ascii=False),
                    json.dumps(tensor_shape_summary, ensure_ascii=False),
                    json.dumps(stored_report, ensure_ascii=False),
                ),
            )

    def get_latest_inspection(self, asset_id: int) -> Optional[InspectionRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM asset_inspections
                WHERE asset_id = ?
                ORDER BY inspected_at DESC
                LIMIT 1
                """,
                (asset_id,),
            ).fetchone()

        return self._row_to_inspection_record(row) if row else None

    def log_load_attempt(
        self,
        asset_id: int,
        status: str,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        load_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO load_history (
                    asset_id,
                    loaded_at,
                    status,
                    device,
                    precision,
                    load_time_ms,
                    error_message,
                    context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    device,
                    precision,
                    load_time_ms,
                    error_message,
                    json.dumps(context or {}, ensure_ascii=False),
                ),
            )

    def get_load_history(self, asset_id: int, limit: int = 20) -> list[LoadHistoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM load_history
                WHERE asset_id = ?
                ORDER BY loaded_at DESC
                LIMIT ?
                """,
                (asset_id, limit),
            ).fetchall()

        return [self._row_to_load_history_record(row) for row in rows]

    def upsert_registry_metric(
        self,
        metric_key: str,
        metric_value: Any,
        *,
        calculation_version: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(metric_value, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO registry_metrics (metric_key, metric_value_json, calculated_at, calculation_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(metric_key) DO UPDATE SET
                    metric_value_json = excluded.metric_value_json,
                    calculated_at = excluded.calculated_at,
                    calculation_version = excluded.calculation_version
                """,
                (str(metric_key), payload, now, str(calculation_version)),
            )

    def replace_registry_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        calculation_version: str,
    ) -> None:
        for key, value in dict(metrics).items():
            self.upsert_registry_metric(
                str(key),
                value,
                calculation_version=calculation_version,
            )

    def _row_to_inspection_record(self, row: sqlite3.Row) -> InspectionRecord:
        return InspectionRecord(
            id=row["id"],
            asset_id=row["asset_id"],
            inspected_at=row["inspected_at"],
            inspector_version=row["inspector_version"],
            key_count=row["key_count"],
            prefix_summary_json=row["prefix_summary_json"],
            example_keys_json=row["example_keys_json"],
            dtype_summary_json=row["dtype_summary_json"],
            tensor_shape_summary_json=row["tensor_shape_summary_json"],
            result_json=row["result_json"],
        )

    def _row_to_load_history_record(self, row: sqlite3.Row) -> LoadHistoryRecord:
        return LoadHistoryRecord(
            id=row["id"],
            asset_id=row["asset_id"],
            loaded_at=row["loaded_at"],
            status=row["status"],
            device=row["device"],
            precision=row["precision"],
            load_time_ms=row["load_time_ms"],
            error_message=row["error_message"],
            context_json=row["context_json"],
        )
