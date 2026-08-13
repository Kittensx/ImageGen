from __future__ import annotations

from typing import Protocol

from image_gen.systems.asset_hub.contracts import (
    ProviderDescriptor,
    ProviderDownloadSource,
    ProviderModel,
    ProviderSearchPage,
    ProviderSearchRequest,
    ProviderVersion,
)


class AssetHubError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 502, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.retry_after_seconds = retry_after_seconds

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.retry_after_seconds is not None:
            payload["retryAfterSeconds"] = self.retry_after_seconds
        return payload


class AssetProvider(Protocol):
    provider_id: str

    def descriptor(self) -> ProviderDescriptor: ...

    async def search(self, request: ProviderSearchRequest) -> ProviderSearchPage: ...

    async def get_model(self, remote_model_id: str, *, refresh: bool = False) -> ProviderModel: ...

    async def get_version(self, remote_version_id: str, *, refresh: bool = False) -> ProviderVersion: ...

    async def lookup_hash(self, file_hash: str, *, refresh: bool = False) -> ProviderVersion: ...

    async def validate_secret(self, secret: str) -> bool: ...

    async def resolve_download_source(
        self,
        remote_model_id: str,
        remote_version_id: str,
        remote_file_id: str,
        *,
        secret: str | None = None,
    ) -> ProviderDownloadSource: ...
