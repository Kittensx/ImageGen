from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DownloadJobRecord:
    job_id: str
    provider_id: str
    remote_model_id: str
    remote_version_id: str
    remote_file_id: str
    file_name: str
    staging_directory: str
    status: str
    expected_bytes: int = 0
    received_bytes: int = 0
    expected_sha256: str = ""
    actual_sha256: str = ""
    etag: str = ""
    last_modified: str = ""
    resume_count: int = 0
    resume_note: str = ""
    error_code: str = ""
    error_message: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "remoteVersionId": self.remote_version_id,
            "remoteFileId": self.remote_file_id,
            "fileName": self.file_name,
            "stagingDirectory": self.staging_directory,
            "status": self.status,
            "expectedBytes": self.expected_bytes,
            "receivedBytes": self.received_bytes,
            "expectedSha256": self.expected_sha256 or None,
            "actualSha256": self.actual_sha256 or None,
            "etag": self.etag or None,
            "lastModified": self.last_modified or None,
            "resumeCount": self.resume_count,
            "resumeNote": self.resume_note or None,
            "error": ({"code": self.error_code, "message": self.error_message} if self.error_code else None),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "completedAt": self.completed_at or None,
        }


class DownloadRepository:
    TABLE = "asset_hub_download_jobs"

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.TABLE} (
            job_id TEXT PRIMARY KEY,
            provider_id TEXT NOT NULL,
            remote_model_id TEXT NOT NULL,
            remote_version_id TEXT NOT NULL,
            remote_file_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            staging_directory TEXT NOT NULL,
            status TEXT NOT NULL,
            expected_bytes INTEGER NOT NULL DEFAULT 0,
            received_bytes INTEGER NOT NULL DEFAULT 0,
            expected_sha256 TEXT NOT NULL DEFAULT '',
            actual_sha256 TEXT NOT NULL DEFAULT '',
            etag TEXT NOT NULL DEFAULT '',
            last_modified TEXT NOT NULL DEFAULT '',
            resume_count INTEGER NOT NULL DEFAULT 0,
            resume_note TEXT NOT NULL DEFAULT '',
            error_code TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_asset_hub_download_jobs_status
            ON {self.TABLE}(status, updated_at);
        """
        with self._lock, self._connect() as connection:
            connection.executescript(sql)
            connection.commit()

    @staticmethod
    def _record(row: sqlite3.Row) -> DownloadJobRecord:
        return DownloadJobRecord(**{key: row[key] for key in row.keys()})

    def create(self, record: DownloadJobRecord) -> DownloadJobRecord:
        now = utc_now()
        item = replace(record, created_at=record.created_at or now, updated_at=record.updated_at or now)
        fields = list(item.__dataclass_fields__)
        placeholders = ",".join("?" for _ in fields)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"INSERT INTO {self.TABLE} ({','.join(fields)}) VALUES ({placeholders})",
                [getattr(item, field) for field in fields],
            )
            connection.commit()
        return item

    def get(self, job_id: str) -> DownloadJobRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(f"SELECT * FROM {self.TABLE} WHERE job_id = ?", (str(job_id),)).fetchone()
        return self._record(row) if row is not None else None

    def list(self, *, limit: int = 100) -> list[DownloadJobRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.TABLE} ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [self._record(row) for row in rows]

    def update(self, job_id: str, **changes: Any) -> DownloadJobRecord:
        allowed = set(DownloadJobRecord.__dataclass_fields__) - {"job_id", "created_at"}
        updates = {key: value for key, value in changes.items() if key in allowed}
        updates["updated_at"] = utc_now()
        if not updates:
            record = self.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return record
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [str(job_id)]
        with self._lock, self._connect() as connection:
            cursor = connection.execute(f"UPDATE {self.TABLE} SET {assignments} WHERE job_id = ?", values)
            if cursor.rowcount != 1:
                raise KeyError(job_id)
            connection.commit()
        record = self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    def recover_interrupted(self) -> list[DownloadJobRecord]:
        recovered: list[DownloadJobRecord] = []
        for record in self.list(limit=500):
            if record.status not in {"queued", "resolving", "downloading", "verifying", "cancelling"}:
                continue
            recovered.append(self.update(
                record.job_id,
                status="paused",
                error_code="process_restart",
                error_message="Download was paused because IMAGE_GEN restarted.",
                resume_note="Process restart recovery preserved the staged partial file when present.",
            ))
        return recovered


@dataclass(frozen=True)
class InstallRecord:
    install_id: str
    install_job_id: str
    download_job_id: str
    provider_id: str
    remote_model_id: str
    remote_version_id: str
    remote_file_id: str
    installed_path: str
    verified_sha256: str
    asset_kind: str
    status: str
    source_metadata_json: str = "{}"
    sidecar_path: str = ""
    registry_asset_id: str = ""
    error_code: str = ""
    error_message: str = ""
    installed_at: str = ""
    updated_at: str = ""

    def source_metadata(self) -> dict[str, Any]:
        try:
            value = __import__("json").loads(self.source_metadata_json or "{}")
        except Exception:
            value = {}
        return dict(value) if isinstance(value, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "installId": self.install_id,
            "installJobId": self.install_job_id,
            "downloadJobId": self.download_job_id,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "remoteVersionId": self.remote_version_id,
            "remoteFileId": self.remote_file_id,
            "installedPath": self.installed_path or None,
            "verifiedSha256": self.verified_sha256,
            "assetKind": self.asset_kind,
            "status": self.status,
            "sourceMetadata": self.source_metadata(),
            "sidecarPath": self.sidecar_path or None,
            "registryAssetId": self.registry_asset_id or None,
            "error": ({"code": self.error_code, "message": self.error_message} if self.error_code else None),
            "installedAt": self.installed_at or None,
            "updatedAt": self.updated_at or None,
        }


class InstallRepository:
    TABLE = "asset_hub_installs"

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.TABLE} (
            install_id TEXT PRIMARY KEY,
            install_job_id TEXT NOT NULL,
            download_job_id TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            remote_model_id TEXT NOT NULL,
            remote_version_id TEXT NOT NULL,
            remote_file_id TEXT NOT NULL,
            installed_path TEXT NOT NULL,
            verified_sha256 TEXT NOT NULL,
            asset_kind TEXT NOT NULL,
            status TEXT NOT NULL,
            source_metadata_json TEXT NOT NULL DEFAULT '{{}}',
            sidecar_path TEXT NOT NULL DEFAULT '',
            registry_asset_id TEXT NOT NULL DEFAULT '',
            error_code TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT '',
            installed_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_asset_hub_installs_job ON {self.TABLE}(install_job_id);
        CREATE INDEX IF NOT EXISTS idx_asset_hub_installs_provider ON {self.TABLE}(provider_id, remote_file_id);
        CREATE INDEX IF NOT EXISTS idx_asset_hub_installs_hash ON {self.TABLE}(verified_sha256);
        CREATE INDEX IF NOT EXISTS idx_asset_hub_installs_status ON {self.TABLE}(status, updated_at);
        """
        with self._lock, self._connect() as connection:
            connection.executescript(sql)
            connection.commit()

    @staticmethod
    def _record(row: sqlite3.Row) -> InstallRecord:
        return InstallRecord(**{key: row[key] for key in row.keys()})

    def create(self, record: InstallRecord) -> InstallRecord:
        now = utc_now()
        item = replace(record, updated_at=record.updated_at or now)
        fields = list(item.__dataclass_fields__)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"INSERT INTO {self.TABLE} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                [getattr(item, field) for field in fields],
            )
            connection.commit()
        return item

    def get(self, install_id: str) -> InstallRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(f"SELECT * FROM {self.TABLE} WHERE install_id = ?", (str(install_id),)).fetchone()
        return self._record(row) if row is not None else None

    def get_by_job(self, install_job_id: str) -> InstallRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(f"SELECT * FROM {self.TABLE} WHERE install_job_id = ?", (str(install_job_id),)).fetchone()
        return self._record(row) if row is not None else None

    def list(self, *, status: str = "", limit: int = 500) -> list[InstallRecord]:
        query = f"SELECT * FROM {self.TABLE}"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(str(status))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 2000)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._record(row) for row in rows]

    def update(self, install_id: str, **changes: Any) -> InstallRecord:
        allowed = set(InstallRecord.__dataclass_fields__) - {"install_id"}
        updates = {key: value for key, value in changes.items() if key in allowed}
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE {self.TABLE} SET {assignments} WHERE install_id = ?",
                [*updates.values(), str(install_id)],
            )
            if cursor.rowcount != 1:
                raise KeyError(install_id)
            connection.commit()
        item = self.get(install_id)
        if item is None:
            raise KeyError(install_id)
        return item
