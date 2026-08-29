from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx

from image_gen.systems.asset_hub.providers.base import AssetHubError

_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_ALLOWED_DETAIL_FETCH_MODES = {"current_only", "current_and_adjacent"}
_ALLOWED_LIBRARY_GALLERY_MODES = {"hero_only", "selected_version", "all_versions"}
_ALLOWED_RETENTION_MODES = {"session", "days", "until_limit", "permanent"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _is_allowed_image_host(hostname: str | None) -> bool:
    host = _text(hostname).casefold().rstrip(".")
    return (
        host == "civitai.com"
        or host.endswith(".civitai.com")
        or host == "civitai.green"
        or host.endswith(".civitai.green")
    )


@dataclass(frozen=True)
class GalleryCacheSettings:
    detail_fetch_mode: str = "current_only"
    library_gallery_mode: str = "hero_only"
    retention_mode: str = "days"
    retention_days: int = 7
    max_cache_gib: float = 10.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "GalleryCacheSettings":
        root = dict(config or {})
        asset_hub = root.get("asset_hub") if isinstance(root.get("asset_hub"), Mapping) else {}
        raw = asset_hub.get("gallery") if isinstance(asset_hub.get("gallery"), Mapping) else {}
        return cls().updated(raw)

    def updated(self, values: Mapping[str, Any] | None) -> "GalleryCacheSettings":
        raw = dict(values or {})

        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in raw:
                    return raw[name]
            return default

        detail = _text(pick("detail_fetch_mode", "detailFetchMode", default=self.detail_fetch_mode)).casefold()
        if detail not in _ALLOWED_DETAIL_FETCH_MODES:
            detail = self.detail_fetch_mode
        library = _text(pick("library_gallery_mode", "libraryGalleryMode", default=self.library_gallery_mode)).casefold()
        if library not in _ALLOWED_LIBRARY_GALLERY_MODES:
            library = self.library_gallery_mode
        retention = _text(pick("retention_mode", "retentionMode", default=self.retention_mode)).casefold()
        if retention not in _ALLOWED_RETENTION_MODES:
            retention = self.retention_mode
        try:
            retention_days = int(pick("retention_days", "retentionDays", default=self.retention_days))
        except (TypeError, ValueError):
            retention_days = self.retention_days
        retention_days = max(1, min(retention_days, 3650))
        try:
            max_cache_gib = float(pick("max_cache_gib", "maxCacheGiB", default=self.max_cache_gib))
        except (TypeError, ValueError):
            max_cache_gib = self.max_cache_gib
        max_cache_gib = max(0.25, min(max_cache_gib, 4096.0))
        return replace(
            self,
            detail_fetch_mode=detail,
            library_gallery_mode=library,
            retention_mode=retention,
            retention_days=retention_days,
            max_cache_gib=max_cache_gib,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detailFetchMode": self.detail_fetch_mode,
            "libraryGalleryMode": self.library_gallery_mode,
            "retentionMode": self.retention_mode,
            "retentionDays": self.retention_days,
            "maxCacheGiB": self.max_cache_gib,
        }

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "detail_fetch_mode": self.detail_fetch_mode,
            "library_gallery_mode": self.library_gallery_mode,
            "retention_mode": self.retention_mode,
            "retention_days": self.retention_days,
            "max_cache_gib": self.max_cache_gib,
        }


_TOP_LEVEL = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*:\s*(?:#.*)?$")
_TWO_SPACE_KEY = re.compile(r"^  [A-Za-z0-9_][A-Za-z0-9_-]*:\s*(?:#.*)?$")


def _gallery_block(settings: GalleryCacheSettings, *, indent: str = "  ") -> list[str]:
    values = settings.to_config_dict()
    return [
        f"{indent}gallery:",
        f"{indent}  detail_fetch_mode: {values['detail_fetch_mode']}",
        f"{indent}  library_gallery_mode: {values['library_gallery_mode']}",
        f"{indent}  retention_mode: {values['retention_mode']}",
        f"{indent}  retention_days: {values['retention_days']}",
        f"{indent}  max_cache_gib: {values['max_cache_gib']:g}",
    ]


def persist_gallery_settings(config_path: str | Path, settings: GalleryCacheSettings) -> None:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    asset_start = next((i for i, line in enumerate(lines) if line.strip() == "asset_hub:" and not line.startswith((" ", "\t"))), None)
    if asset_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("asset_hub:")
        lines.extend(_gallery_block(settings))
    else:
        asset_end = len(lines)
        for i in range(asset_start + 1, len(lines)):
            line = lines[i]
            if line and not line.startswith((" ", "\t")) and _TOP_LEVEL.match(line):
                asset_end = i
                break
        gallery_start = next(
            (i for i in range(asset_start + 1, asset_end) if lines[i].startswith("  gallery:") and lines[i].strip().startswith("gallery:")),
            None,
        )
        block = _gallery_block(settings)
        if gallery_start is None:
            insert_at = asset_end
            while insert_at > asset_start + 1 and not lines[insert_at - 1].strip():
                insert_at -= 1
            lines[insert_at:insert_at] = block
        else:
            gallery_end = asset_end
            for i in range(gallery_start + 1, asset_end):
                line = lines[i]
                if line.startswith("  ") and not line.startswith("    ") and _TWO_SPACE_KEY.match(line):
                    gallery_end = i
                    break
            lines[gallery_start:gallery_end] = block
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


class AssetGalleryCache:
    """IMAGE_GEN-managed provider gallery cache.

    Search cards never use this cache. P3B only caches a detail image after the
    user asks to view it, while optional multi-image persistence is reserved for
    assets that already resolve as installed in the local library.
    """

    def __init__(
        self,
        cache_root: str | Path,
        database_path: str | Path,
        *,
        settings: GalleryCacheSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.database_path = Path(database_path).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings or GalleryCacheSettings()
        self.transport = transport
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.session_token = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._init_schema()
        self.cleanup(startup=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=15.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS asset_gallery_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    remote_model_id TEXT NOT NULL,
                    remote_version_id TEXT NOT NULL,
                    provider_image_id TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    byte_size INTEGER NOT NULL DEFAULT 0,
                    cache_class TEXT NOT NULL DEFAULT 'detail',
                    protected INTEGER NOT NULL DEFAULT 0,
                    session_token TEXT NOT NULL DEFAULT '',
                    created_at_unix REAL NOT NULL,
                    last_accessed_at_unix REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_asset_gallery_cache_cleanup
                    ON asset_gallery_cache(protected, last_accessed_at_unix ASC);
                CREATE INDEX IF NOT EXISTS idx_asset_gallery_cache_model
                    ON asset_gallery_cache(provider_id, remote_model_id, remote_version_id);
                """
            )

    @staticmethod
    def _cache_key(provider_id: str, url: str) -> str:
        return hashlib.sha256(f"{_text(provider_id).casefold()}|{_text(url)}".encode("utf-8")).hexdigest()

    def update_settings(self, values: Mapping[str, Any] | None) -> GalleryCacheSettings:
        self.settings = self.settings.updated(values)
        self.cleanup()
        return self.settings

    def _delete_rows(self, rows: list[sqlite3.Row]) -> int:
        removed = 0
        if not rows:
            return 0
        with self._lock, self._connect() as connection:
            for row in rows:
                path = Path(str(row["file_path"] or ""))
                try:
                    if path.is_file():
                        path.unlink()
                except OSError:
                    pass
                connection.execute("DELETE FROM asset_gallery_cache WHERE cache_key=?", (str(row["cache_key"]),))
                removed += 1
        return removed

    def cleanup(self, *, startup: bool = False, include_current_session: bool = False) -> dict[str, Any]:
        now = time.time()
        candidates: list[sqlite3.Row] = []
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM asset_gallery_cache ORDER BY last_accessed_at_unix ASC").fetchall()
        for row in rows:
            path = Path(str(row["file_path"] or ""))
            if not path.is_file():
                candidates.append(row)
                continue
            if bool(row["protected"]):
                continue
            mode = self.settings.retention_mode
            if mode == "session":
                if include_current_session or str(row["session_token"] or "") != self.session_token:
                    candidates.append(row)
            elif mode == "days":
                cutoff = now - (self.settings.retention_days * 86400)
                if float(row["last_accessed_at_unix"] or 0.0) < cutoff:
                    candidates.append(row)
        removed = self._delete_rows(candidates)

        # Size limit applies to temporary cache unless the user explicitly chose
        # Permanent. Library-owned/protected gallery media is never evicted here.
        if self.settings.retention_mode != "permanent":
            maximum = int(self.settings.max_cache_gib * 1024 * 1024 * 1024)
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM asset_gallery_cache WHERE protected=0 ORDER BY last_accessed_at_unix ASC"
                ).fetchall()
            total = sum(max(0, int(row["byte_size"] or 0)) for row in rows)
            overflow: list[sqlite3.Row] = []
            for row in rows:
                if total <= maximum:
                    break
                overflow.append(row)
                total -= max(0, int(row["byte_size"] or 0))
            removed += self._delete_rows(overflow)
        return {"removed": removed, **self.status()}

    def clear_temporary(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM asset_gallery_cache WHERE protected=0").fetchall()
        removed = self._delete_rows(list(rows))
        return {"removed": removed, **self.status()}

    def status(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT protected, COUNT(*) AS item_count, COALESCE(SUM(byte_size),0) AS bytes
                FROM asset_gallery_cache GROUP BY protected
                """
            ).fetchall()
        temporary_count = temporary_bytes = library_count = library_bytes = 0
        for row in rows:
            if bool(row["protected"]):
                library_count = int(row["item_count"] or 0)
                library_bytes = int(row["bytes"] or 0)
            else:
                temporary_count = int(row["item_count"] or 0)
                temporary_bytes = int(row["bytes"] or 0)
        return {
            "temporaryImages": temporary_count,
            "temporaryBytes": temporary_bytes,
            "libraryImages": library_count,
            "libraryBytes": library_bytes,
            "totalImages": temporary_count + library_count,
            "totalBytes": temporary_bytes + library_bytes,
            "settings": self.settings.to_dict(),
        }

    def cached_file(self, cache_key: str) -> Path | None:
        key = _text(cache_key)
        if not key:
            return None
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM asset_gallery_cache WHERE cache_key=?", (key,)).fetchone()
            if row is None:
                return None
            path = Path(str(row["file_path"] or ""))
            if not path.is_file():
                connection.execute("DELETE FROM asset_gallery_cache WHERE cache_key=?", (key,))
                return None
            connection.execute("UPDATE asset_gallery_cache SET last_accessed_at_unix=? WHERE cache_key=?", (now, key))
        return path

    async def fetch(
        self,
        *,
        provider_id: str,
        remote_model_id: str,
        remote_version_id: str,
        image_url: str,
        provider_image_id: str = "",
        cache_class: str = "detail",
        protected: bool = False,
    ) -> dict[str, Any]:
        provider = _text(provider_id).casefold()
        if provider != "civitai":
            raise AssetHubError("gallery_provider_unsupported", "Managed gallery caching currently supports CivitAI images only.", status_code=400)
        url = _text(image_url)
        parsed = urlparse(url)
        if parsed.scheme.casefold() != "https" or not _is_allowed_image_host(parsed.hostname):
            raise AssetHubError("gallery_url_blocked", "Refusing to cache an untrusted provider gallery URL.", status_code=400)
        key = self._cache_key(provider, url)
        existing = self.cached_file(key)
        if existing is not None:
            if protected:
                with self._lock, self._connect() as connection:
                    connection.execute(
                        "UPDATE asset_gallery_cache SET protected=1, cache_class='library' WHERE cache_key=?",
                        (key,),
                    )
            return {"cacheKey": key, "byteSize": existing.stat().st_size, "cached": True}

        headers = {
            "Accept": "image/png,image/jpeg,image/webp",
            "Referer": "https://civitai.com/",
            "User-Agent": "IMAGE_GEN-AssetHub-Gallery/1",
        }
        try:
            async with httpx.AsyncClient(
                transport=self.transport,
                timeout=self.timeout_seconds,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = await client.get(url)
        except httpx.TimeoutException as exc:
            raise AssetHubError("gallery_fetch_timeout", "Provider gallery image timed out.", status_code=504) from exc
        except httpx.HTTPError as exc:
            raise AssetHubError("gallery_fetch_failed", "Provider gallery image could not be fetched.", status_code=502) from exc
        if response.status_code >= 400:
            raise AssetHubError("gallery_fetch_failed", f"Provider gallery returned HTTP {response.status_code}.", status_code=502)
        final = urlparse(str(response.url))
        if final.scheme.casefold() != "https" or not _is_allowed_image_host(final.hostname):
            raise AssetHubError("gallery_url_blocked", "Provider gallery redirected to an untrusted host.", status_code=400)
        content_type = _text(response.headers.get("content-type")).split(";", 1)[0].casefold()
        extension = _ALLOWED_CONTENT_TYPES.get(content_type)
        if not extension:
            raise AssetHubError("gallery_content_type", "Provider gallery returned an unsupported image type.", status_code=502)
        content = bytes(response.content)
        if not content or len(content) > _MAX_IMAGE_BYTES:
            raise AssetHubError("gallery_image_size", "Provider gallery image is empty or exceeds the 32 MB safety limit.", status_code=502)

        folder = self.cache_root / provider / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}{extension}"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO asset_gallery_cache(
                    cache_key, provider_id, remote_model_id, remote_version_id,
                    provider_image_id, source_url, file_path, byte_size,
                    cache_class, protected, session_token, created_at_unix, last_accessed_at_unix
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    remote_model_id=excluded.remote_model_id,
                    remote_version_id=excluded.remote_version_id,
                    provider_image_id=excluded.provider_image_id,
                    file_path=excluded.file_path,
                    byte_size=excluded.byte_size,
                    cache_class=CASE WHEN excluded.protected=1 THEN 'library' ELSE asset_gallery_cache.cache_class END,
                    protected=MAX(asset_gallery_cache.protected, excluded.protected),
                    session_token=excluded.session_token,
                    last_accessed_at_unix=excluded.last_accessed_at_unix
                """,
                (
                    key,
                    provider,
                    _text(remote_model_id),
                    _text(remote_version_id),
                    _text(provider_image_id),
                    url,
                    str(path),
                    len(content),
                    "library" if protected else _text(cache_class) or "detail",
                    1 if protected else 0,
                    self.session_token,
                    now,
                    now,
                ),
            )
        self.cleanup()
        return {"cacheKey": key, "byteSize": len(content), "cached": False}
