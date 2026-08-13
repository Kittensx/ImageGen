from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

from image_gen.systems.asset_hub.contracts import (
    ASSET_HUB_CONTRACT_VERSION,
    ProviderFile,
    ProviderModel,
    ProviderModelSummary,
    ProviderSearchPage,
    ProviderSearchRequest,
    ProviderVersion,
)
from image_gen.systems.asset_hub.policy import ArchitectureCompatibilityPolicy, normalize_asset_kind
from image_gen.systems.asset_hub.providers.base import AssetHubError, AssetProvider


LocalRecordProvider = Callable[[], Iterable[Mapping[str, Any]]]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _norm_id(value: Any) -> str:
    return _norm(value)


def _norm_hash(value: Any) -> str:
    token = _norm(value).lower()
    if len(token) < 12 or any(ch not in "0123456789abcdef" for ch in token):
        return ""
    return token


def _first_sha256(value: Mapping[str, Any]) -> str:
    for key in ("SHA256", "sha256", "Sha256"):
        token = _norm_hash(value.get(key))
        if token:
            return token
    return ""


class LocalPresenceResolver:
    """Provider-neutral installed-state resolver over current local catalog records.

    Records are read afresh for each resolution pass. That means successful local
    removal automatically stops resolving as installed without any provider-side
    state mutation or stale browser filename heuristics.
    """

    def __init__(self, records_provider: LocalRecordProvider | None = None) -> None:
        self._records_provider = records_provider or (lambda: ())

    @staticmethod
    def _record_identity(record: Mapping[str, Any]) -> dict[str, str]:
        metadata = _mapping(record.get("metadata"))
        source = _mapping(metadata.get("_asset_hub_source"))
        if not source:
            source = _mapping(record.get("_asset_hub_source"))

        civitai = _mapping(record.get("civitai_lookup"))
        if not civitai:
            civitai = _mapping(metadata.get("_civitai_lookup"))
        matched_file = _mapping(civitai.get("matched_file"))
        matched_hashes = _mapping(matched_file.get("hashes"))

        provider = _norm(source.get("source_provider") or source.get("provider"))
        model_id = _norm_id(source.get("source_asset_id") or source.get("provider_model_id"))
        version_id = _norm_id(source.get("source_version_id") or source.get("provider_model_version_id"))
        file_id = _norm_id(source.get("source_file_id") or source.get("provider_file_id"))
        sha256 = _norm_hash(source.get("file_sha256") or source.get("sha256"))

        if civitai:
            provider = provider or "civitai"
            model_id = model_id or _norm_id(civitai.get("model_id"))
            version_id = version_id or _norm_id(civitai.get("model_version_id"))
            file_id = file_id or _norm_id(matched_file.get("id"))
            sha256 = (
                sha256
                or _norm_hash(civitai.get("matched_hash"))
                or _first_sha256(matched_hashes)
            )

        sha256 = sha256 or _norm_hash(record.get("sha256"))
        return {
            "provider": provider.casefold(),
            "model_id": model_id,
            "version_id": version_id,
            "file_id": file_id,
            "sha256": sha256,
            "local_asset_id": _norm(record.get("asset_id") or record.get("upscaler_id") or record.get("id")),
            "local_asset_type": normalize_asset_kind(record.get("asset_type") or record.get("kind") or "unknown"),
        }

    def _records(self) -> tuple[dict[str, str], ...]:
        output: list[dict[str, str]] = []
        for raw in self._records_provider():
            if not isinstance(raw, Mapping):
                continue
            if raw.get("exists_on_disk") is False:
                continue
            identity = self._record_identity(raw)
            if identity["provider"] or identity["sha256"]:
                output.append(identity)
        return tuple(output)

    def resolve_file(self, remote: ProviderFile) -> tuple[str, str, str]:
        provider = remote.provider_id.casefold()
        remote_sha256 = remote.hash_map().get("SHA256", "").lower()
        for local in self._records():
            if local["provider"] != provider:
                continue
            if not remote.remote_model_id or local["model_id"] != remote.remote_model_id:
                continue
            if not remote.remote_version_id or local["version_id"] != remote.remote_version_id:
                continue
            # File-level installed state must not broaden a version-only identity
            # into every sibling file. Require the stable provider file ID here;
            # records without it may still resolve through the SHA-256 fallback.
            if not local["file_id"] or not remote.remote_file_id or local["file_id"] != remote.remote_file_id:
                continue
            return "installed", local["local_asset_id"], local["local_asset_type"]

        if remote_sha256:
            for local in self._records():
                if local["sha256"] and local["sha256"] == remote_sha256:
                    return "installed", local["local_asset_id"], local["local_asset_type"]
        return "not_installed", "", ""

    def decorate_file(self, remote: ProviderFile) -> ProviderFile:
        status, local_id, local_type = self.resolve_file(remote)
        return replace(
            remote,
            library_status=status,
            local_asset_id=local_id,
            local_asset_type=local_type,
        )


class AssetHubService:
    def __init__(
        self,
        providers: Iterable[AssetProvider],
        *,
        policy: ArchitectureCompatibilityPolicy | None = None,
        presence: LocalPresenceResolver | None = None,
    ) -> None:
        self.policy = policy or ArchitectureCompatibilityPolicy()
        self.presence = presence or LocalPresenceResolver()
        self.providers = {provider.provider_id: provider for provider in providers}
        if not self.providers:
            raise ValueError("AssetHubService requires at least one provider.")

    def provider_descriptors(self) -> list[dict[str, Any]]:
        return [self.providers[key].descriptor().to_dict() for key in sorted(self.providers)]

    def provider_status(self, provider_id: str) -> dict[str, Any]:
        provider = self._provider(provider_id)
        return {
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "provider": provider.descriptor().to_dict(),
            "mode": "staging_only",
            "downloadsEnabled": True,
            "managedRootWritesEnabled": False,
            "installCommitEnabled": False,
            "compatibility": {
                kind: self.policy.compatibility_payload(kind)
                for kind in ("checkpoint", "lora", "vae", "textual_inversion", "upscaler")
            },
        }

    def _provider(self, provider_id: str) -> AssetProvider:
        selected = str(provider_id or "").strip().casefold()
        provider = self.providers.get(selected)
        if provider is None:
            raise AssetHubError("provider_not_found", f"Unknown asset provider {provider_id!r}.", status_code=404)
        return provider

    def _require_kind(self, asset_kind: str) -> str:
        kind = normalize_asset_kind(asset_kind)
        if not self.policy.is_browsable_kind(kind):
            raise AssetHubError(
                "provider_policy_blocked",
                f"Asset kind {kind!r} is not enabled for normal Asset Hub discovery in this build.",
                status_code=400,
            )
        supported = self.policy.supported_architectures(kind)
        if not supported:
            raise AssetHubError(
                "provider_policy_blocked",
                f"Asset kind {kind!r} has no IMAGE_GEN-supported architecture in this build.",
                status_code=400,
            )
        return kind

    def _enforced_request(self, request: ProviderSearchRequest) -> ProviderSearchRequest:
        kind = self._require_kind(request.asset_kind)
        supported = self.policy.supported_architectures(kind)
        requested = tuple(item for item in request.base_models if str(item or "").strip())
        allowed = self.policy.normalize_requested_architectures(kind, requested)
        if requested and not allowed:
            raise AssetHubError(
                "provider_policy_blocked",
                "The requested architecture is not supported by this IMAGE_GEN build.",
                status_code=400,
            )
        return replace(request, asset_kind=kind, base_models=allowed or supported, limit=max(1, min(int(request.limit or 24), 50)))

    def _decorate_version(self, version: ProviderVersion, *, asset_kind: str) -> ProviderVersion | None:
        if not self.policy.is_compatible(asset_kind, version.base_model):
            return None
        files = tuple(self.presence.decorate_file(item) for item in version.files)
        installed = next((item for item in files if item.library_status == "installed"), None)
        return replace(
            version,
            files=files,
            library_status="installed" if installed else "not_installed",
            local_asset_id=installed.local_asset_id if installed else "",
            local_asset_type=installed.local_asset_type if installed else "",
        )

    def _decorate_summary(
        self,
        model: ProviderModelSummary,
        *,
        requested_kind: str,
        safe_content: bool,
    ) -> ProviderModelSummary | None:
        if model.asset_kind != requested_kind:
            return None
        if safe_content and model.nsfw:
            # The provider request already asks for safe content, but a backend
            # post-filter prevents provider drift from bypassing the preference.
            return None
        versions = tuple(
            item
            for raw in model.versions
            if (item := self._decorate_version(raw, asset_kind=requested_kind)) is not None
        )
        if requested_kind != "upscaler" and not versions:
            return None
        installed = next((item for item in versions if item.library_status == "installed"), None)
        return replace(
            model,
            versions=versions,
            library_status="installed" if installed else "not_installed",
            local_asset_id=installed.local_asset_id if installed else "",
            local_asset_type=installed.local_asset_type if installed else "",
        )

    async def search(self, provider_id: str, request: ProviderSearchRequest) -> ProviderSearchPage:
        provider = self._provider(provider_id)
        enforced = self._enforced_request(request)
        page = await provider.search(enforced)
        items: list[ProviderModelSummary] = []
        for raw in page.items:
            decorated = self._decorate_summary(raw, requested_kind=enforced.asset_kind, safe_content=enforced.safe_content)
            if decorated is not None:
                items.append(decorated)
        return replace(
            page,
            items=tuple(items),
            compatibility_scope=self.policy.supported_architectures(enforced.asset_kind),
        )

    async def get_model(self, provider_id: str, model_id: str, *, refresh: bool = False) -> ProviderModel:
        provider = self._provider(provider_id)
        model = await provider.get_model(model_id, refresh=refresh)
        kind = self._require_kind(model.asset_kind)
        versions = tuple(
            item
            for raw in model.versions
            if (item := self._decorate_version(raw, asset_kind=kind)) is not None
        )
        if kind != "upscaler" and not versions:
            raise AssetHubError(
                "provider_policy_blocked",
                "The requested provider model has no version compatible with this IMAGE_GEN build.",
                status_code=403,
            )
        installed = next((item for item in versions if item.library_status == "installed"), None)
        return replace(
            model,
            versions=versions,
            library_status="installed" if installed else "not_installed",
            local_asset_id=installed.local_asset_id if installed else "",
            local_asset_type=installed.local_asset_type if installed else "",
        )

    async def get_version(self, provider_id: str, version_id: str, *, refresh: bool = False) -> ProviderVersion:
        provider = self._provider(provider_id)
        version = await provider.get_version(version_id, refresh=refresh)
        if not version.remote_model_id:
            raise AssetHubError("provider_bad_response", "Provider version omitted its model identity.", status_code=502)
        model = await provider.get_model(version.remote_model_id, refresh=refresh)
        kind = self._require_kind(model.asset_kind)
        decorated = self._decorate_version(version, asset_kind=kind)
        if decorated is None:
            raise AssetHubError(
                "provider_policy_blocked",
                "The requested provider version is not compatible with this IMAGE_GEN build.",
                status_code=403,
            )
        return decorated

    async def lookup_hash(self, provider_id: str, file_hash: str, *, refresh: bool = False) -> ProviderVersion:
        provider = self._provider(provider_id)
        version = await provider.lookup_hash(file_hash, refresh=refresh)
        if not version.remote_model_id:
            raise AssetHubError("provider_bad_response", "Provider hash lookup omitted its model identity.", status_code=502)
        model = await provider.get_model(version.remote_model_id, refresh=refresh)
        kind = self._require_kind(model.asset_kind)
        decorated = self._decorate_version(version, asset_kind=kind)
        if decorated is None:
            raise AssetHubError(
                "provider_policy_blocked",
                "The hash resolves to an asset architecture not supported by this IMAGE_GEN build.",
                status_code=403,
            )
        return decorated
