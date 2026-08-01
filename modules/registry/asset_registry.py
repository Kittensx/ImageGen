from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json
import os
import sqlite3

from .fingerprint import FileFingerprint, FingerprintResult
from .models import AssetRecord, InspectionRecord, LoadHistoryRecord
from .schema import SCHEMA_SQL


class AssetRegistry:
    """
    Local SQLite registry for model-like assets.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self.fingerprinter = FileFingerprint()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def classify_path(self, path: str, managed_roots: dict[str, str]) -> tuple[str | None, str | None, str]:
        """
        Returns:
            (library_root, managed_category, path_kind)
        """
        resolved = Path(path).resolve()

        for _, root_dir in managed_roots.items():
            root_path = Path(root_dir).resolve()
            try:
                resolved.relative_to(root_path)
                library_root = str(root_path.parent)
                managed_category = root_path.name
                return library_root, managed_category, "managed"
            except ValueError:
                continue

        return None, None, "external"

    def register_file(
        self,
        path: str,
        compute_sha256: bool = False,
        compute_blake3: bool = False,
        library_root: str | None = None,
        managed_category: str | None = None,
        path_kind: str = "external",
    ) -> AssetRecord:
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot register missing file: {file_path}")

        fingerprint = self.fingerprinter.fingerprint_file(
            str(file_path),
            compute_sha256=compute_sha256,
            compute_blake3=compute_blake3,
        )
        return self.upsert_asset_from_fingerprint(
            fingerprint,
            library_root=library_root,
            managed_category=managed_category,
            path_kind=path_kind,
        )

    def upsert_asset_from_fingerprint(
        self,
        fingerprint: FingerprintResult,
        library_root: str | None = None,
        managed_category: str | None = None,
        path_kind: str = "external",
    ) -> AssetRecord:
        now = datetime.utcnow().isoformat()
        file_path = Path(fingerprint.path)
        extension = file_path.suffix.lower()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM assets WHERE path = ?",
                (fingerprint.path,),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE assets
                    SET filename = ?,
                        extension = ?,
                        file_size = ?,
                        modified_time = ?,
                        created_time = ?,
                        last_seen_at = ?,
                        exists_on_disk = 1,
                        quick_fingerprint = ?,
                        sha256 = COALESCE(?, sha256),
                        blake3 = COALESCE(?, blake3),
                        library_root = COALESCE(?, library_root),
                        managed_category = COALESCE(?, managed_category),
                        path_kind = ?
                    WHERE path = ?
                    """,
                    (
                        file_path.name,
                        extension,
                        fingerprint.file_size,
                        fingerprint.modified_time,
                        fingerprint.created_time,
                        now,
                        fingerprint.quick_fingerprint,
                        fingerprint.sha256,
                        fingerprint.blake3,
                        library_root,
                        managed_category,
                        path_kind,
                        fingerprint.path,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO assets (
                        path, filename, extension, file_size, modified_time, created_time,
                        first_seen_at, last_seen_at, exists_on_disk,
                        quick_fingerprint, sha256, blake3,
                        library_root, managed_category, path_kind
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fingerprint.path,
                        file_path.name,
                        extension,
                        fingerprint.file_size,
                        fingerprint.modified_time,
                        fingerprint.created_time,
                        now,
                        now,
                        fingerprint.quick_fingerprint,
                        fingerprint.sha256,
                        fingerprint.blake3,
                        library_root,
                        managed_category,
                        path_kind,
                    ),
                )

            row = conn.execute(
                "SELECT * FROM assets WHERE path = ?",
                (fingerprint.path,),
            ).fetchone()

        return self._row_to_asset_record(row)

    def get_asset_by_path(self, path: str) -> Optional[AssetRecord]:
        file_path = str(Path(path).resolve())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE path = ?",
                (file_path,),
            ).fetchone()
        return self._row_to_asset_record(row) if row else None

    def get_asset_by_id(self, asset_id: int) -> Optional[AssetRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE id = ?",
                (asset_id,),
            ).fetchone()
        return self._row_to_asset_record(row) if row else None

    def get_asset_by_sha256(self, sha256: str) -> Optional[AssetRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        return self._row_to_asset_record(row) if row else None

    def list_assets(
        self,
        asset_type: Optional[str] = None,
        architecture: Optional[str] = None,
        managed_category: Optional[str] = None,
        limit: int = 100,
    ) -> list[AssetRecord]:
        clauses = []
        params: list[Any] = []

        if asset_type:
            clauses.append("asset_type = ?")
            params.append(asset_type)

        if architecture:
            clauses.append("architecture = ?")
            params.append(architecture)

        if managed_category:
            clauses.append("managed_category = ?")
            params.append(managed_category)

        where_sql = ""
        if clauses:
            where_sql = "WHERE " + " AND ".join(clauses)

        sql = f"""
            SELECT * FROM assets
            {where_sql}
            ORDER BY last_seen_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_asset_record(row) for row in rows]

    def mark_missing_assets(self) -> int:
        updated = 0
        with self._connect() as conn:
            rows = conn.execute("SELECT id, path FROM assets").fetchall()

            for row in rows:
                exists = Path(row["path"]).exists()
                conn.execute(
                    "UPDATE assets SET exists_on_disk = ? WHERE id = ?",
                    (1 if exists else 0, row["id"]),
                )
                updated += 1

        return updated

    def store_inspection(self, asset_id: int, report: dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()

        metadata = report.get("metadata", {})
        prefix_summary = report.get("prefix_summary", {})
        example_keys = report.get("example_keys", [])
        dtype_summary = report.get("dtype_summary", {})
        tensor_shape_summary = report.get("tensor_shape_summary", {})

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE assets
                SET asset_type = ?,
                    format_type = ?,
                    architecture = ?,
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
                    report.get("asset_type", "unknown"),
                    report.get("format_type", "other"),
                    report.get("architecture", "unknown"),
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
                    json.dumps(report, ensure_ascii=False),
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
                    datetime.utcnow().isoformat(),
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

    def add_relationship(
        self,
        source_asset_id: int,
        target_asset_id: int,
        relationship_type: str,
        confidence: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO asset_relationships (
                    source_asset_id,
                    target_asset_id,
                    relationship_type,
                    confidence,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source_asset_id,
                    target_asset_id,
                    relationship_type,
                    confidence,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def _row_to_asset_record(self, row: sqlite3.Row) -> AssetRecord:
        return AssetRecord(
            id=row["id"],
            path=row["path"],
            filename=row["filename"],
            extension=row["extension"] or "",
            file_size=row["file_size"] or 0,
            modified_time=row["modified_time"] or 0.0,
            created_time=row["created_time"] or 0.0,
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            exists_on_disk=bool(row["exists_on_disk"]),
            quick_fingerprint=row["quick_fingerprint"],
            sha256=row["sha256"],
            blake3=row["blake3"],
            asset_type=row["asset_type"],
            format_type=row["format_type"],
            architecture=row["architecture"],
            checkpoint_kind=row["checkpoint_kind"],
            has_unet=bool(row["has_unet"]),
            has_vae=bool(row["has_vae"]),
            has_text_encoder=bool(row["has_text_encoder"]),
            has_text_encoder_2=bool(row["has_text_encoder_2"]),
            library_root=row["library_root"],
            managed_category=row["managed_category"],
            path_kind=row["path_kind"],
            key_count=row["key_count"],
            metadata_json=row["metadata_json"],
            notes=row["notes"],
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