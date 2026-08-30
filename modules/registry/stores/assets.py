from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
import os
import sqlite3

from ..architecture_observation import ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED
from ..contracts import AVAILABILITY_AVAILABLE, AVAILABILITY_MISSING
from ..fingerprint import FingerprintResult
from ..models import (
    AssetRecord,
    LOCATION_STATE_ARCHIVED,
    LOCATION_STATE_AVAILABLE,
    LOCATION_STATE_INACCESSIBLE,
    LOCATION_STATE_MISSING,
    LOCATION_STATE_MOVED_RELINKED,
)


class AssetStore:
    """Asset identity, location, lookup, and whole-file duplicate persistence operations."""

    def update_asset_location_state(
        self,
        asset_id: int,
        location_state: str,
        *,
        metadata_updates: Mapping[str, Any] | None = None,
        metadata_removals: Iterable[str] | None = None,
    ) -> None:
        state = str(location_state or LOCATION_STATE_AVAILABLE).strip().lower() or LOCATION_STATE_AVAILABLE
        exists_on_disk = 1 if state in {LOCATION_STATE_AVAILABLE, LOCATION_STATE_ARCHIVED} else 0
        with self._connect() as conn:
            conn.execute(
                "UPDATE assets SET location_state = ?, exists_on_disk = ? WHERE id = ?",
                (state, exists_on_disk, int(asset_id)),
            )
            if metadata_updates or metadata_removals:
                self._merge_asset_metadata(
                    conn,
                    int(asset_id),
                    updates=metadata_updates,
                    removals=metadata_removals,
                )

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
        now = datetime.now(timezone.utc).isoformat()
        file_path = Path(fingerprint.path)
        extension = file_path.suffix.lower()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM assets WHERE path = ?",
                (fingerprint.path,),
            ).fetchone()

            if existing:
                # A strong whole-file hash belongs to the exact file content that
                # produced it. If the cheap fingerprint changes and this refresh did
                # not recompute the strong hash, retaining the previous SHA/BLAKE
                # would falsely bind the new file to the old content identity.
                quick_changed = (
                    bool(existing["quick_fingerprint"])
                    and str(existing["quick_fingerprint"]).lower()
                    != str(fingerprint.quick_fingerprint or "").lower()
                )
                next_sha256 = (
                    fingerprint.sha256
                    if fingerprint.sha256 is not None
                    else (None if quick_changed else existing["sha256"])
                )
                next_blake3 = (
                    fingerprint.blake3
                    if fingerprint.blake3 is not None
                    else (None if quick_changed else existing["blake3"])
                )
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
                        sha256 = ?,
                        blake3 = ?,
                        library_root = COALESCE(?, library_root),
                        managed_category = COALESCE(?, managed_category),
                        path_kind = ?,
                        location_state = ?
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
                        next_sha256,
                        next_blake3,
                        library_root,
                        managed_category,
                        path_kind,
                        LOCATION_STATE_AVAILABLE,
                        fingerprint.path,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO assets (
                        path, filename, extension, file_size, modified_time, created_time,
                        first_seen_at, last_seen_at, exists_on_disk,
                        quick_fingerprint, sha256, blake3,
                        library_root, managed_category, path_kind, location_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
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
                        LOCATION_STATE_AVAILABLE,
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

    def update_asset_sha256(self, asset_id: int, sha256: str) -> None:
        value = str(sha256 or "").strip().lower()
        if not value:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE assets SET sha256 = ? WHERE id = ?",
                (value, int(asset_id)),
            )

    def find_assets_by_filename(self, filename: str, limit: int = 100) -> list[AssetRecord]:
        """Return existing registry rows matching a filename, newest first.

        This is a lookup over the existing asset_registry.db schema; it does not
        create a parallel text-encoder registry or add role-specific tables.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM assets
                WHERE filename = ? COLLATE NOCASE
                ORDER BY exists_on_disk DESC, last_seen_at DESC, id DESC
                LIMIT ?
                """,
                (str(filename), max(1, int(limit))),
            ).fetchall()
        return [self._row_to_asset_record(row) for row in rows]

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

    def _normalized_location_state(self, value: str | None) -> str:
        state = str(value or "").strip().lower()
        if state in {
            LOCATION_STATE_AVAILABLE,
            LOCATION_STATE_ARCHIVED,
            LOCATION_STATE_MISSING,
            LOCATION_STATE_MOVED_RELINKED,
            LOCATION_STATE_INACCESSIBLE,
        }:
            return state
        return LOCATION_STATE_AVAILABLE

    def _derive_location_state(self, current_state: str | None, exists_on_disk: bool) -> str:
        state = self._normalized_location_state(current_state)
        if exists_on_disk:
            if state == LOCATION_STATE_ARCHIVED:
                return LOCATION_STATE_ARCHIVED
            return LOCATION_STATE_AVAILABLE
        if state in {LOCATION_STATE_MOVED_RELINKED, LOCATION_STATE_INACCESSIBLE}:
            return state
        return LOCATION_STATE_MISSING

    @staticmethod
    def _probe_path_location_state(path_value: str, current_state: str | None = None) -> tuple[bool, str]:
        """Probe a registered location without collapsing disconnected roots into missing files.

        A missing file on a reachable filesystem is ``missing``. A path whose root cannot
        currently be reached (for example a disconnected Windows drive or inaccessible
        network share) is ``inaccessible``. Historical ``moved_relinked`` state remains
        sticky while its old path is absent.
        """
        current = str(current_state or "").strip().lower()
        path = Path(str(path_value or ""))
        try:
            path.stat()
            if current == LOCATION_STATE_ARCHIVED:
                return True, LOCATION_STATE_ARCHIVED
            return True, LOCATION_STATE_AVAILABLE
        except PermissionError:
            return False, LOCATION_STATE_INACCESSIBLE
        except FileNotFoundError:
            if current == LOCATION_STATE_MOVED_RELINKED:
                return False, LOCATION_STATE_MOVED_RELINKED
            if os.name == "nt":
                anchor = str(path.anchor or "").strip()
                if anchor:
                    try:
                        if not Path(anchor).exists():
                            return False, LOCATION_STATE_INACCESSIBLE
                    except OSError:
                        return False, LOCATION_STATE_INACCESSIBLE
            return False, LOCATION_STATE_MISSING
        except OSError:
            return False, LOCATION_STATE_INACCESSIBLE

    def mark_missing_assets(self) -> int:
        updated = 0
        with self._connect() as conn:
            rows = conn.execute("SELECT id, path, location_state FROM assets").fetchall()

            for row in rows:
                exists, location_state = self._probe_path_location_state(
                    row["path"], row["location_state"]
                )
                conn.execute(
                    "UPDATE assets SET exists_on_disk = ?, location_state = ? WHERE id = ?",
                    (1 if exists else 0, location_state, row["id"]),
                )
                availability = AVAILABILITY_AVAILABLE if exists else AVAILABILITY_MISSING
                conn.execute(
                    "UPDATE component_sources SET availability_state = ? WHERE asset_id = ?",
                    (availability, row["id"]),
                )
                updated += 1

        return updated

    def reconcile_asset_locations(self, *, asset_ids: Iterable[int] | None = None) -> dict[str, Any]:
        scoped_ids = sorted({int(item) for item in asset_ids or ()})
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, path, filename, file_size, sha256, exists_on_disk, location_state, metadata_json FROM assets"
            ).fetchall()
            by_sha: dict[str, list[sqlite3.Row]] = defaultdict(list)
            for row in rows:
                digest = str(row["sha256"] or "").strip().lower()
                if digest:
                    by_sha[digest].append(row)

            relinked_count = 0
            ambiguous_missing_count = 0
            for row in rows:
                asset_id = int(row["id"])
                if scoped_ids and asset_id not in scoped_ids:
                    continue
                digest = str(row["sha256"] or "").strip().lower()
                if not digest or bool(row["exists_on_disk"]):
                    continue
                candidates = [
                    item for item in by_sha.get(digest, [])
                    if int(item["id"]) != asset_id and bool(item["exists_on_disk"])
                ]
                if len(candidates) == 1:
                    target = candidates[0]
                    relinked_count += 1
                    self._merge_asset_metadata(
                        conn,
                        asset_id,
                        updates={
                            "relinked_to_asset_id": int(target["id"]),
                            "relinked_to_path": target["path"],
                            "relinked_by": "exact_sha256",
                            "relinked_sha256": digest,
                            "relinked_at": datetime.now(timezone.utc).isoformat(),
                        },
                        removals=("relink_candidate_paths",),
                    )
                    conn.execute(
                        "UPDATE assets SET location_state = ?, exists_on_disk = 0 WHERE id = ?",
                        (LOCATION_STATE_MOVED_RELINKED, asset_id),
                    )
                elif len(candidates) > 1:
                    ambiguous_missing_count += 1
                    self._merge_asset_metadata(
                        conn,
                        asset_id,
                        updates={
                            "relink_candidate_paths": [item["path"] for item in candidates],
                            "relinked_by": "ambiguous_exact_sha256",
                        },
                    )

            refreshed_rows = conn.execute(
                "SELECT id, file_size, sha256, exists_on_disk, location_state FROM assets"
            ).fetchall()

        state_counts: Counter[str] = Counter()
        duplicate_group_count = 0
        duplicate_location_count = 0
        recoverable_bytes = 0
        groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in refreshed_rows:
            state_counts[self._normalized_location_state(row["location_state"])] += 1
            digest = str(row["sha256"] or "").strip().lower()
            if digest:
                groups[digest].append(row)
        for group in groups.values():
            if len(group) < 2:
                continue
            duplicate_group_count += 1
            duplicate_location_count += len(group)
            available = [item for item in group if bool(item["exists_on_disk"])]
            if available:
                size = max(int(item["file_size"] or 0) for item in available)
                recoverable_bytes += size * max(0, len(available) - 1)

        return {
            "asset_count": len(refreshed_rows),
            "location_state_counts": dict(sorted(state_counts.items())),
            "relinked_count": relinked_count,
            "ambiguous_missing_count": ambiguous_missing_count,
            "exact_file_duplicate_group_count": duplicate_group_count,
            "exact_file_duplicate_location_count": duplicate_location_count,
            "recoverable_duplicate_bytes": recoverable_bytes,
        }

    def list_exact_file_duplicate_groups(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assets WHERE COALESCE(sha256, '') <> '' ORDER BY sha256, filename, id"
            ).fetchall()

        groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            groups[str(row["sha256"]).strip().lower()].append(row)

        results: list[dict[str, Any]] = []
        for digest, group in groups.items():
            if len(group) < 2:
                continue
            members = []
            available_count = 0
            archive_count = 0
            missing_count = 0
            moved_count = 0
            for row in sorted(group, key=lambda item: (str(item["filename"]).casefold(), int(item["id"]))):
                state = self._normalized_location_state(row["location_state"])
                if state == LOCATION_STATE_AVAILABLE:
                    available_count += 1
                elif state == LOCATION_STATE_ARCHIVED:
                    archive_count += 1
                elif state == LOCATION_STATE_MOVED_RELINKED:
                    moved_count += 1
                else:
                    missing_count += 1
                members.append({
                    "asset_id": int(row["id"]),
                    "path": row["path"],
                    "filename": row["filename"],
                    "file_size": int(row["file_size"] or 0),
                    "exists_on_disk": bool(row["exists_on_disk"]),
                    "location_state": state,
                    "asset_type": row["asset_type"],
                    "architecture": row["architecture"],
                    "managed_category": row["managed_category"],
                    "path_kind": row["path_kind"],
                })
            representative_size = max((item["file_size"] for item in members), default=0)
            results.append({
                "sha256": digest,
                "location_count": len(members),
                "available_location_count": available_count,
                "archived_location_count": archive_count,
                "missing_location_count": missing_count,
                "moved_relinked_location_count": moved_count,
                "total_referenced_bytes": sum(item["file_size"] for item in members),
                "recoverable_duplicate_bytes": representative_size * max(0, available_count - 1),
                "members": members,
            })
        results.sort(key=lambda item: (-item["location_count"], item["sha256"]))
        return results[: max(1, int(limit))]

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
            architecture=row["architecture"] or "",
            architecture_state=row["architecture_state"] or ARCHITECTURE_STATE_OBSERVED_UNCLASSIFIED,
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
            location_state=row["location_state"] or LOCATION_STATE_AVAILABLE,
        )
