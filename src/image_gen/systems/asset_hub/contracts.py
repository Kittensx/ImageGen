from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

ASSET_HUB_CONTRACT_VERSION = "image-gen-asset-hub-v1"
ASSET_HUB_SCHEMA_VERSION = 1


def _dict_pairs(value: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(sorted((str(key), item) for key, item in value.items()))


def _pairs_dict(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {str(key): item for key, item in value}


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: str
    display_name: str
    search_enabled: bool = True
    model_details_enabled: bool = True
    version_details_enabled: bool = True
    hash_lookup_enabled: bool = True
    authentication_mode: str = "none"
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "providerId": self.provider_id,
            "displayName": self.display_name,
            "searchEnabled": self.search_enabled,
            "modelDetailsEnabled": self.model_details_enabled,
            "versionDetailsEnabled": self.version_details_enabled,
            "hashLookupEnabled": self.hash_lookup_enabled,
            "authenticationMode": self.authentication_mode,
        }


@dataclass(frozen=True)
class ProviderSearchRequest:
    query: str = ""
    asset_kind: str = "checkpoint"
    base_models: tuple[str, ...] = ()
    creator: str = ""
    sort: str = ""
    period: str = ""
    safe_content: bool = True
    cursor: str = ""
    limit: int = 24
    refresh: bool = False
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "query": self.query,
            "assetKind": self.asset_kind,
            "baseModels": list(self.base_models),
            "creator": self.creator,
            "sort": self.sort,
            "period": self.period,
            "safeContent": self.safe_content,
            "cursor": self.cursor,
            "limit": self.limit,
            "refresh": self.refresh,
        }


@dataclass(frozen=True)
class ProviderPermissionSummary:
    allow_no_credit: bool | None = None
    allow_commercial_use: tuple[str, ...] = ()
    allow_derivatives: bool | None = None
    allow_different_license: bool | None = None
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "allowNoCredit": self.allow_no_credit,
            "allowCommercialUse": list(self.allow_commercial_use),
            "allowDerivatives": self.allow_derivatives,
            "allowDifferentLicense": self.allow_different_license,
        }


@dataclass(frozen=True)
class ProviderScanSummary:
    pickle_scan_result: str = ""
    virus_scan_result: str = ""
    scanned_at: str = ""
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "pickleScanResult": self.pickle_scan_result,
            "virusScanResult": self.virus_scan_result,
            "scannedAt": self.scanned_at,
        }


@dataclass(frozen=True)
class ProviderPreview:
    url: str
    width: int | None = None
    height: int | None = None
    nsfw_level: str = ""
    kind: str = "image"
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "url": self.url,
            "width": self.width,
            "height": self.height,
            "nsfwLevel": self.nsfw_level,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class ProviderSourceIdentity:
    source_provider: str
    source_asset_id: str = ""
    source_version_id: str = ""
    source_file_id: str = ""
    source_url: str = ""
    file_sha256: str = ""
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "source_provider": self.source_provider,
            "source_asset_id": self.source_asset_id,
            "source_version_id": self.source_version_id,
            "source_file_id": self.source_file_id,
            "source_url": self.source_url,
            "file_sha256": self.file_sha256,
        }


@dataclass(frozen=True)
class ProviderFile:
    provider_id: str
    remote_model_id: str
    remote_version_id: str
    remote_file_id: str
    file_name: str
    file_type: str = ""
    format: str = ""
    size_bytes: int = 0
    base_model: str = ""
    architecture: str = ""
    trained_words: tuple[str, ...] = ()
    hashes: tuple[tuple[str, str], ...] = ()
    primary: bool = False
    scan: ProviderScanSummary = field(default_factory=ProviderScanSummary)
    source_page_url: str = ""
    library_status: str = "not_installed"
    local_asset_id: str = ""
    local_asset_type: str = ""
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    @classmethod
    def with_hashes(cls, *, hashes: Mapping[str, Any] | None = None, **kwargs: Any) -> "ProviderFile":
        normalized = {
            str(key).upper(): str(value or "").strip().lower()
            for key, value in dict(hashes or {}).items()
            if str(value or "").strip()
        }
        return cls(hashes=tuple(sorted(normalized.items())), **kwargs)

    def hash_map(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.hashes}

    def source_identity(self) -> ProviderSourceIdentity:
        return ProviderSourceIdentity(
            source_provider=self.provider_id,
            source_asset_id=self.remote_model_id,
            source_version_id=self.remote_version_id,
            source_file_id=self.remote_file_id,
            source_url=self.source_page_url,
            file_sha256=self.hash_map().get("SHA256", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "remoteVersionId": self.remote_version_id,
            "remoteFileId": self.remote_file_id,
            "fileName": self.file_name,
            "fileType": self.file_type,
            "format": self.format,
            "sizeBytes": self.size_bytes,
            "baseModel": self.base_model,
            "architecture": self.architecture,
            "trainedWords": list(self.trained_words),
            "hashes": self.hash_map(),
            "primary": self.primary,
            "scan": self.scan.to_dict(),
            "sourcePageUrl": self.source_page_url,
            "sourceIdentity": self.source_identity().to_dict(),
            "libraryStatus": self.library_status,
            "localAssetId": self.local_asset_id or None,
            "localAssetType": self.local_asset_type or None,
        }


@dataclass(frozen=True)
class ProviderDownloadSource:
    """Ephemeral provider-resolved download target.

    ``url`` is intentionally never serialized by ``to_public_dict``. It may be a
    signed or delivery URL and must not enter persistent job state, API responses,
    logs, or verification reports.
    """

    provider_id: str
    remote_model_id: str
    remote_version_id: str
    remote_file_id: str
    file_name: str
    url: str
    expected_bytes: int = 0
    expected_sha256: str = ""
    auth_hosts: tuple[str, ...] = ()
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "remoteVersionId": self.remote_version_id,
            "remoteFileId": self.remote_file_id,
            "fileName": self.file_name,
            "expectedBytes": self.expected_bytes,
            "expectedSha256": self.expected_sha256 or None,
        }


@dataclass(frozen=True)
class ProviderVersion:
    provider_id: str
    remote_model_id: str
    remote_version_id: str
    name: str
    base_model: str = ""
    architecture: str = ""
    description: str = ""
    trained_words: tuple[str, ...] = ()
    published_at: str = ""
    updated_at: str = ""
    files: tuple[ProviderFile, ...] = ()
    previews: tuple[ProviderPreview, ...] = ()
    stats: tuple[tuple[str, Any], ...] = ()
    library_status: str = "not_installed"
    local_asset_id: str = ""
    local_asset_type: str = ""
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    @classmethod
    def create(cls, *, stats: Mapping[str, Any] | None = None, **kwargs: Any) -> "ProviderVersion":
        return cls(stats=_dict_pairs(stats), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "remoteVersionId": self.remote_version_id,
            "name": self.name,
            "baseModel": self.base_model,
            "architecture": self.architecture,
            "description": self.description,
            "trainedWords": list(self.trained_words),
            "publishedAt": self.published_at,
            "updatedAt": self.updated_at,
            "files": [item.to_dict() for item in self.files],
            "previews": [item.to_dict() for item in self.previews],
            "stats": _pairs_dict(self.stats),
            "libraryStatus": self.library_status,
            "localAssetId": self.local_asset_id or None,
            "localAssetType": self.local_asset_type or None,
        }


@dataclass(frozen=True)
class ProviderModelSummary:
    provider_id: str
    remote_model_id: str
    name: str
    asset_kind: str
    provider_type: str
    creator: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    nsfw: bool = False
    versions: tuple[ProviderVersion, ...] = ()
    library_status: str = "not_installed"
    local_asset_id: str = ""
    local_asset_type: str = ""
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "name": self.name,
            "assetKind": self.asset_kind,
            "providerType": self.provider_type,
            "creator": self.creator,
            "description": self.description,
            "tags": list(self.tags),
            "nsfw": self.nsfw,
            "versions": [item.to_dict() for item in self.versions],
            "libraryStatus": self.library_status,
            "localAssetId": self.local_asset_id or None,
            "localAssetType": self.local_asset_type or None,
        }


@dataclass(frozen=True)
class ProviderModel:
    provider_id: str
    remote_model_id: str
    name: str
    asset_kind: str
    provider_type: str
    creator: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    nsfw: bool = False
    source_page_url: str = ""
    permissions: ProviderPermissionSummary = field(default_factory=ProviderPermissionSummary)
    versions: tuple[ProviderVersion, ...] = ()
    library_status: str = "not_installed"
    local_asset_id: str = ""
    local_asset_type: str = ""
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "providerId": self.provider_id,
            "remoteModelId": self.remote_model_id,
            "name": self.name,
            "assetKind": self.asset_kind,
            "providerType": self.provider_type,
            "creator": self.creator,
            "description": self.description,
            "tags": list(self.tags),
            "nsfw": self.nsfw,
            "sourcePageUrl": self.source_page_url,
            "permissions": self.permissions.to_dict(),
            "versions": [item.to_dict() for item in self.versions],
            "libraryStatus": self.library_status,
            "localAssetId": self.local_asset_id or None,
            "localAssetType": self.local_asset_type or None,
        }


@dataclass(frozen=True)
class ProviderSearchPage:
    provider_id: str
    items: tuple[ProviderModelSummary, ...]
    next_cursor: str = ""
    total_items: int | None = None
    compatibility_scope: tuple[str, ...] = ()
    schema_version: int = ASSET_HUB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "providerId": self.provider_id,
            "items": [item.to_dict() for item in self.items],
            "nextCursor": self.next_cursor or None,
            "totalItems": self.total_items,
            "compatibilityScope": list(self.compatibility_scope),
        }
