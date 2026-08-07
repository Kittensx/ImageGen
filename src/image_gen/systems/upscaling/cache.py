from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from image_gen.systems.upscaling.contracts import (
    UPSCALER_SCAN_SCHEMA_VERSION,
    UpscalerDescriptor,
)

UPSCALER_CACHE_RELATIVE_PATH = Path("upscalers") / "scan-cache-v1.json"


def canonical_path_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


@dataclass(frozen=True)
class UpscalerCacheRecord:
    schema_version: int
    path: str
    file_size_bytes: int
    modified_time_ns: int
    sha256: str
    loader_backend_version: str
    scan_timestamp_utc: str
    descriptor: UpscalerDescriptor

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "path": self.path,
            "file_size_bytes": int(self.file_size_bytes),
            "modified_time_ns": int(self.modified_time_ns),
            "sha256": self.sha256,
            "loader_backend_version": self.loader_backend_version,
            "scan_timestamp_utc": self.scan_timestamp_utc,
            "descriptor": self.descriptor.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UpscalerCacheRecord":
        payload = dict(value)
        descriptor_payload = payload.get("descriptor")
        if not isinstance(descriptor_payload, Mapping):
            raise ValueError("Upscaler cache descriptor is missing or invalid.")
        return cls(
            schema_version=int(payload.get("schema_version") or 0),
            path=str(payload.get("path") or ""),
            file_size_bytes=int(payload.get("file_size_bytes") or 0),
            modified_time_ns=int(payload.get("modified_time_ns") or 0),
            sha256=str(payload.get("sha256") or ""),
            loader_backend_version=str(payload.get("loader_backend_version") or ""),
            scan_timestamp_utc=str(payload.get("scan_timestamp_utc") or ""),
            descriptor=UpscalerDescriptor.from_mapping(descriptor_payload),
        )

    def is_current(self, path: Path, *, loader_backend_version: str) -> bool:
        try:
            stat = path.stat()
        except OSError:
            return False
        return (
            self.schema_version == UPSCALER_SCAN_SCHEMA_VERSION
            and canonical_path_key(self.path) == canonical_path_key(path)
            and int(self.file_size_bytes) == int(stat.st_size)
            and int(self.modified_time_ns) == int(stat.st_mtime_ns)
            and self.loader_backend_version == str(loader_backend_version)
        )


class UpscalerScanCache:
    def __init__(self, cache_root: str | os.PathLike[str]) -> None:
        root = Path(cache_root).expanduser().resolve()
        self.path = root / UPSCALER_CACHE_RELATIVE_PATH
        self._records: dict[str, UpscalerCacheRecord] | None = None
        self.load_error = ""

    def _load(self) -> dict[str, UpscalerCacheRecord]:
        if self._records is not None:
            return self._records
        records: dict[str, UpscalerCacheRecord] = {}
        if not self.path.is_file():
            self._records = records
            return records
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) != UPSCALER_SCAN_SCHEMA_VERSION:
                self._records = records
                return records
            raw_records = payload.get("records") or {}
            if not isinstance(raw_records, Mapping):
                raise ValueError("Upscaler cache records must be a mapping.")
            for raw_key, raw_value in raw_records.items():
                if not isinstance(raw_value, Mapping):
                    continue
                try:
                    record = UpscalerCacheRecord.from_mapping(raw_value)
                except (TypeError, ValueError):
                    continue
                records[canonical_path_key(raw_key)] = record
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.load_error = f"Unable to read upscaler scan cache: {exc}"
        self._records = records
        return records

    def get(self, path: str | os.PathLike[str]) -> UpscalerCacheRecord | None:
        return self._load().get(canonical_path_key(path))

    def put(self, record: UpscalerCacheRecord) -> None:
        self._load()[canonical_path_key(record.path)] = record

    def remove(self, path: str | os.PathLike[str]) -> None:
        self._load().pop(canonical_path_key(path), None)

    def records(self) -> tuple[UpscalerCacheRecord, ...]:
        return tuple(self._load().values())

    def save(self) -> None:
        records = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": UPSCALER_SCAN_SCHEMA_VERSION,
            "records": {
                key: value.to_dict()
                for key, value in sorted(records.items(), key=lambda item: item[0])
            },
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=self.path.name + ".",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                handle.write(serialized)
                temp_path = Path(handle.name)
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
