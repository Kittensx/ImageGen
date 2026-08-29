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
from image_gen.systems.asset_hub.policy import (
    ArchitectureCompatibilityPolicy,
    normalize_architecture,
    normalize_asset_kind,
)
from image_gen.systems.asset_hub.providers.base import AssetHubError, AssetProvider


LocalRecordProvider = Callable[[], Iterable[Mapping[str, Any]]]
_SUPPORT_FILTERS = {"any", "supported", "unsupported", "unknown"}
_LIBRARY_FILTERS = {"any", "installed", "not_installed"}
_SEARCH_MODES = {"search", "browse"}


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
    def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
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
            sha256 = sha256 or _norm_hash(civitai.get("matched_hash")) or _first_sha256(matched_hashes)

        sha256 = sha256 or _norm_hash(record.get("sha256"))
        asset_hub = _mapping(metadata.get("asset_hub"))
        provider_preview_url = _norm(record.get("provider_preview_url") or civitai.get("image_url"))
        if not provider_preview_url:
            previews = asset_hub.get("preview_images")
            if isinstance(previews, list):
                for item in previews:
                    if not isinstance(item, Mapping):
                        continue
                    candidate = _norm(item.get("url"))
                    if candidate and _norm(item.get("kind") or "image").casefold() == "image":
                        provider_preview_url = candidate
                        break
        return {
            "provider": provider.casefold(),
            "model_id": model_id,
            "version_id": version_id,
            "file_id": file_id,
            "sha256": sha256,
            "local_asset_id": _norm(record.get("asset_id") or record.get("upscaler_id") or record.get("id")),
            "local_asset_type": normalize_asset_kind(record.get("asset_type") or record.get("kind") or "unknown"),
            "local_preview_url": _norm(record.get("local_preview_url") or record.get("preview_url")),
            "local_preview_source": _norm(record.get("local_preview_source")),
            "provider_preview_url": provider_preview_url,
            "local_name": _norm(record.get("display_name") or record.get("name") or asset_hub.get("display_name") or asset_hub.get("title")),
            "local_description": _norm(record.get("description") or asset_hub.get("description_plaintext")),
            "local_tags": tuple(str(item) for item in (record.get("tags") or asset_hub.get("tags") or []) if str(item).strip()),
            "local_creator": _norm(record.get("provider_creator") or record.get("civitai_creator") or asset_hub.get("provider_creator_name")),
            "local_base_model": _norm(record.get("provider_base_model") or record.get("civitai_base_model") or record.get("model_family") or record.get("architecture") or asset_hub.get("model_family") or asset_hub.get("base_model")),
            "local_provider_type": _norm(record.get("civitai_model_type") or asset_hub.get("provider_asset_type")),
            "local_version_name": _norm(record.get("provider_version_name") or record.get("civitai_model_version_name") or asset_hub.get("version_name")),
            "local_file_name": _norm(record.get("filename")),
            "local_size_bytes": str(record.get("size_bytes") or 0),
        }

    def _records(self) -> tuple[dict[str, Any], ...]:
        output: list[dict[str, Any]] = []
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

    def model_overlay(self, provider_id: str, remote_model_id: str) -> dict[str, Any]:
        provider = str(provider_id or "").strip().casefold()
        model_id = str(remote_model_id or "").strip()
        matches = [
            item for item in self._records()
            if item.get("provider") == provider and item.get("model_id") == model_id
        ]
        if not matches:
            return {}
        local_preview = next((item for item in matches if item.get("local_preview_url")), matches[0])
        provider_preview = next((item for item in matches if item.get("provider_preview_url")), matches[0])
        return {
            "library_status": "installed",
            "local_asset_id": str(local_preview.get("local_asset_id") or matches[0].get("local_asset_id") or ""),
            "local_asset_type": str(local_preview.get("local_asset_type") or matches[0].get("local_asset_type") or ""),
            "local_preview_url": str(local_preview.get("local_preview_url") or ""),
            "local_preview_source": str(local_preview.get("local_preview_source") or ""),
            "provider_preview_url": str(provider_preview.get("provider_preview_url") or ""),
        }

    def provider_linked_records(self, provider_id: str = "") -> tuple[dict[str, Any], ...]:
        selected = str(provider_id or "").strip().casefold()
        return tuple(
            item for item in self._records()
            if item.get("provider") and item.get("model_id") and (not selected or item.get("provider") == selected)
        )

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

    @staticmethod
    def _support_filter(value: str) -> str:
        selected = str(value or "supported").strip().casefold().replace("-", "_")
        return selected if selected in _SUPPORT_FILTERS else "supported"

    @staticmethod
    def _library_filter(value: str) -> str:
        selected = str(value or "any").strip().casefold().replace("-", "_")
        aliases = {"in_library": "installed", "already_in_library": "installed", "not_in_library": "not_installed"}
        selected = aliases.get(selected, selected)
        return selected if selected in _LIBRARY_FILTERS else "any"

    @staticmethod
    def _search_mode(value: str) -> str:
        selected = str(value or "search").strip().casefold()
        return selected if selected in _SEARCH_MODES else "search"

    @staticmethod
    def _requested_architectures(values: tuple[str, ...]) -> tuple[str, ...]:
        output: list[str] = []
        for raw in values:
            token = str(raw or "").strip()
            if not token or token.casefold() in {"any", "all"}:
                continue
            normalized = normalize_architecture(token)
            if normalized and normalized not in output:
                output.append(normalized)
        return tuple(output)

    def _enforced_request(self, request: ProviderSearchRequest) -> ProviderSearchRequest:
        support_filter = self._support_filter(request.support_filter)
        library_filter = self._library_filter(request.library_filter)
        search_mode = self._search_mode(request.search_mode)
        raw_kind = str(request.asset_kind or "checkpoint").strip().casefold()
        kind = "any" if raw_kind in {"any", "all", "*"} else normalize_asset_kind(raw_kind)
        requested = self._requested_architectures(tuple(request.base_models))

        # Preserve the original production-discovery contract unless the caller
        # explicitly asks the browser to expose unsupported/unknown resources.
        if support_filter == "supported" and kind != "any":
            kind = self._require_kind(kind)
            supported = self.policy.supported_architectures(kind)
            if requested:
                allowed = tuple(item for item in requested if item in supported)
                if not allowed:
                    raise AssetHubError(
                        "provider_policy_blocked",
                        "The requested architecture is not supported by this IMAGE_GEN build.",
                        status_code=400,
                    )
                requested = allowed
            else:
                requested = supported

        return replace(
            request,
            asset_kind=kind,
            base_models=requested,
            support_filter=support_filter,
            library_filter=library_filter,
            search_mode=search_mode,
            query="" if search_mode == "browse" else str(request.query or "").strip()[:256],
            limit=max(1, min(int(request.limit or 24), 50)),
        )

    def _version_support(self, version: ProviderVersion, *, asset_kind: str) -> tuple[str, str]:
        kind = normalize_asset_kind(asset_kind)
        if kind == "upscaler":
            return "supported", "Upscalers are architecture-independent in the current Asset Hub policy."
        if kind == "unknown":
            return "unknown", "Provider asset type is not recognized by IMAGE_GEN."
        if not self.policy.is_browsable_kind(kind):
            return "unsupported", f"Asset type {kind!r} is discoverable from the provider but is not runnable in this IMAGE_GEN build."
        architecture = normalize_architecture(version.architecture or version.base_model)
        if not architecture:
            return "unknown", "Provider metadata does not identify a known IMAGE_GEN architecture."
        if architecture in self.policy.supported_architectures(kind):
            return "supported", f"Provider architecture {architecture} is supported for {kind}."
        return "unsupported", f"Provider architecture {architecture} is not currently supported for {kind}."

    def _decorate_version(self, version: ProviderVersion, *, asset_kind: str) -> ProviderVersion:
        files = tuple(self.presence.decorate_file(item) for item in version.files)
        installed = next((item for item in files if item.library_status == "installed"), None)
        support_state, support_reason = self._version_support(version, asset_kind=asset_kind)
        return replace(
            version,
            files=files,
            support_state=support_state,
            support_reason=support_reason,
            library_status="installed" if installed else "not_installed",
            local_asset_id=installed.local_asset_id if installed else "",
            local_asset_type=installed.local_asset_type if installed else "",
        )

    @staticmethod
    def _model_support(versions: tuple[ProviderVersion, ...], *, asset_kind: str, policy: ArchitectureCompatibilityPolicy) -> tuple[str, str]:
        states = {item.support_state for item in versions}
        if "supported" in states:
            return "supported", "At least one current provider version has an IMAGE_GEN-supported architecture."
        if "unsupported" in states:
            return "unsupported", "Current provider versions are visible but not runnable under the current IMAGE_GEN capability policy."
        kind = normalize_asset_kind(asset_kind)
        if kind == "upscaler":
            return "supported", "Upscalers are architecture-independent in the current Asset Hub policy."
        if kind != "unknown" and not policy.is_browsable_kind(kind):
            return "unsupported", f"Asset type {kind!r} is not runnable in this IMAGE_GEN build."
        return "unknown", "Compatibility cannot be determined from current provider metadata."

    @staticmethod
    def _matches_support(state: str, support_filter: str) -> bool:
        return support_filter == "any" or state == support_filter

    @staticmethod
    def _matches_library(status: str, library_filter: str) -> bool:
        return library_filter == "any" or status == library_filter

    @staticmethod
    def _search_rank(model: ProviderModelSummary, query: str) -> tuple[int, tuple[str, ...]]:
        phrase = " ".join(str(query or "").casefold().split())
        if not phrase:
            return 0, ()
        words = tuple(dict.fromkeys(part for part in phrase.split() if part))
        name = str(model.name or "").casefold()
        description = str(model.description or "").casefold()
        creator = str(model.creator or "").casefold()
        tags = tuple(str(item or "").casefold() for item in model.tags)
        score = 0
        reasons: list[str] = []

        if name == phrase:
            score += 1200
            reasons.append("exact title")
        elif name.startswith(phrase):
            score += 900
            reasons.append("title prefix")
        elif phrase in name:
            score += 760
            reasons.append("title partial")

        if any(phrase == tag for tag in tags):
            score += 520
            reasons.append("exact tag")
        elif any(phrase in tag for tag in tags):
            score += 420
            reasons.append("tag partial")

        if phrase and phrase in creator:
            score += 260
            reasons.append("creator")
        if phrase and phrase in description:
            score += 180
            reasons.append("description partial")

        for word in words:
            if word in name:
                score += 130
            if any(word in tag for tag in tags):
                score += 80
            if word in description:
                score += 24
        return score, tuple(dict.fromkeys(reasons))

    def _decorate_summary(self, model: ProviderModelSummary, *, request: ProviderSearchRequest) -> ProviderModelSummary | None:
        requested_kind = request.asset_kind
        if requested_kind != "any" and model.asset_kind != requested_kind:
            return None
        if request.safe_content and model.nsfw:
            return None

        requested_architectures = set(request.base_models)
        versions: list[ProviderVersion] = []
        for raw in model.versions:
            architecture = normalize_architecture(raw.architecture or raw.base_model)
            if requested_architectures and architecture not in requested_architectures:
                continue
            versions.append(self._decorate_version(raw, asset_kind=model.asset_kind))
        decorated_versions = tuple(versions)

        # If an architecture filter was requested, a model with no matching version
        # should not leak through even when the provider ignored or broadened the filter.
        if requested_architectures and not decorated_versions:
            return None

        installed = next((item for item in decorated_versions if item.library_status == "installed"), None)
        support_state, support_reason = self._model_support(
            decorated_versions,
            asset_kind=model.asset_kind,
            policy=self.policy,
        )
        overlay = self.presence.model_overlay(model.provider_id, model.remote_model_id)
        library_status = "installed" if installed or overlay else "not_installed"
        if not self._matches_support(support_state, request.support_filter):
            return None
        if not self._matches_library(library_status, request.library_filter):
            return None
        rank, matches = self._search_rank(model, request.query)
        return replace(
            model,
            versions=decorated_versions,
            support_state=support_state,
            support_reason=support_reason,
            search_rank=rank,
            search_matches=matches,
            library_status=library_status,
            local_asset_id=str(overlay.get("local_asset_id") or (installed.local_asset_id if installed else "")),
            local_asset_type=str(overlay.get("local_asset_type") or (installed.local_asset_type if installed else "")),
            provider_preview_url=str(model.provider_preview_url or overlay.get("provider_preview_url") or ""),
            local_preview_url=str(overlay.get("local_preview_url") or ""),
            local_preview_source=str(overlay.get("local_preview_source") or ""),
        )

    async def search(self, provider_id: str, request: ProviderSearchRequest) -> ProviderSearchPage:
        provider = self._provider(provider_id)
        enforced = self._enforced_request(request)
        page = await provider.search(enforced)
        items: list[ProviderModelSummary] = []
        for raw in page.items:
            decorated = self._decorate_summary(raw, request=enforced)
            if decorated is not None:
                items.append(decorated)
        if enforced.query:
            # Stable sorting retains provider ranking among equal local relevance.
            items.sort(key=lambda item: item.search_rank, reverse=True)
        compatibility_scope: tuple[str, ...]
        if enforced.asset_kind == "any":
            combined: list[str] = []
            for kind in ("checkpoint", "lora", "vae", "textual_inversion", "upscaler"):
                for architecture in self.policy.supported_architectures(kind):
                    if architecture not in combined:
                        combined.append(architecture)
            compatibility_scope = tuple(combined)
        else:
            compatibility_scope = self.policy.supported_architectures(enforced.asset_kind)
        return replace(page, items=tuple(items), compatibility_scope=compatibility_scope)

    def _decorate_model_unrestricted(self, model: ProviderModel) -> ProviderModel:
        versions = tuple(self._decorate_version(raw, asset_kind=model.asset_kind) for raw in model.versions)
        installed = next((item for item in versions if item.library_status == "installed"), None)
        support_state, support_reason = self._model_support(versions, asset_kind=model.asset_kind, policy=self.policy)
        overlay = self.presence.model_overlay(model.provider_id, model.remote_model_id)
        return replace(
            model,
            versions=versions,
            support_state=support_state,
            support_reason=support_reason,
            library_status="installed" if installed or overlay else "not_installed",
            local_asset_id=str(overlay.get("local_asset_id") or (installed.local_asset_id if installed else "")),
            local_asset_type=str(overlay.get("local_asset_type") or (installed.local_asset_type if installed else "")),
            provider_preview_url=str(model.provider_preview_url or overlay.get("provider_preview_url") or ""),
            local_preview_url=str(overlay.get("local_preview_url") or ""),
            local_preview_source=str(overlay.get("local_preview_source") or ""),
        )

    async def get_model(
        self,
        provider_id: str,
        model_id: str,
        *,
        refresh: bool = False,
        include_unsupported: bool = False,
    ) -> ProviderModel:
        provider = self._provider(provider_id)
        model = await provider.get_model(model_id, refresh=refresh)
        if include_unsupported:
            return self._decorate_model_unrestricted(model)

        kind = self._require_kind(model.asset_kind)
        versions = tuple(
            item
            for raw in model.versions
            if (item := self._decorate_version(raw, asset_kind=kind)).support_state == "supported"
        )
        if kind != "upscaler" and not versions:
            raise AssetHubError(
                "provider_policy_blocked",
                "The requested provider model has no version compatible with this IMAGE_GEN build.",
                status_code=403,
            )
        installed = next((item for item in versions if item.library_status == "installed"), None)
        support_state, support_reason = self._model_support(versions, asset_kind=kind, policy=self.policy)
        overlay = self.presence.model_overlay(model.provider_id, model.remote_model_id)
        return replace(
            model,
            versions=versions,
            support_state=support_state,
            support_reason=support_reason,
            library_status="installed" if installed or overlay else "not_installed",
            local_asset_id=str(overlay.get("local_asset_id") or (installed.local_asset_id if installed else "")),
            local_asset_type=str(overlay.get("local_asset_type") or (installed.local_asset_type if installed else "")),
            provider_preview_url=str(model.provider_preview_url or overlay.get("provider_preview_url") or ""),
            local_preview_url=str(overlay.get("local_preview_url") or ""),
            local_preview_source=str(overlay.get("local_preview_source") or ""),
        )

    async def get_version(
        self,
        provider_id: str,
        version_id: str,
        *,
        refresh: bool = False,
        include_unsupported: bool = False,
    ) -> ProviderVersion:
        provider = self._provider(provider_id)
        version = await provider.get_version(version_id, refresh=refresh)
        if not version.remote_model_id:
            raise AssetHubError("provider_bad_response", "Provider version omitted its model identity.", status_code=502)
        model = await provider.get_model(version.remote_model_id, refresh=refresh)
        if include_unsupported:
            return self._decorate_version(version, asset_kind=model.asset_kind)
        kind = self._require_kind(model.asset_kind)
        decorated = self._decorate_version(version, asset_kind=kind)
        if decorated.support_state != "supported":
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
        if decorated.support_state != "supported":
            raise AssetHubError(
                "provider_policy_blocked",
                "The hash resolves to an asset architecture not supported by this IMAGE_GEN build.",
                status_code=403,
            )
        return decorated
