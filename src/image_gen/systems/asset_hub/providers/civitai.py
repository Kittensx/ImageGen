from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import httpx

from image_gen.program_metadata import PRODUCT_NAME
from image_gen.systems.asset_hub.contracts import (
    ProviderDescriptor,
    ProviderDownloadSource,
    ProviderFile,
    ProviderModel,
    ProviderModelSummary,
    ProviderPermissionSummary,
    ProviderPreview,
    ProviderScanSummary,
    ProviderSearchPage,
    ProviderSearchRequest,
    ProviderVersion,
)
from image_gen.systems.asset_hub.policy import normalize_architecture, provider_type_to_asset_kind
from image_gen.systems.asset_hub.providers.base import AssetHubError

_API_ROOT = "https://civitai.com/api/v1"
_ALLOWED_HOST = "civitai.com"
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_DEFAULT_TTL_SECONDS = 300
_MAX_CACHE_FILES = 256
_PROVIDER_TYPE_FILTERS = {
    "checkpoint": "Checkpoint",
    "lora": "LORA",
    "vae": "VAE",
    "textual_inversion": "TextualInversion",
    "upscaler": "Upscaler",
}
_PROVIDER_BASE_MODELS = {
    "sd1.x": ("SD 1.4", "SD 1.5"),
    "sd2.x": ("SD 2.0", "SD 2.1"),
    "sdxl": ("SDXL 1.0",),
    "sd3": ("SD 3", "SD 3.5"),
    "flux": ("Flux.1 D", "Flux.1 S"),
}
_ALLOWED_SORTS = {
    "highest rated": "Highest Rated",
    "most downloaded": "Most Downloaded",
    "newest": "Newest",
    "most liked": "Most Liked",
    "most discussed": "Most Discussed",
    "most collected": "Most Collected",
}
_ALLOWED_PERIODS = {
    "alltime": "AllTime",
    "all time": "AllTime",
    "year": "Year",
    "month": "Month",
    "week": "Week",
    "day": "Day",
}
_SECRET_KEYS = {
    "authorization",
    "cookie",
    "cookies",
    "token",
    "api_key",
    "apikey",
    "downloadurl",
    "download_url",
    "signedurl",
    "signed_url",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = str(data or "").strip()
        if text:
            self.parts.append(text)


def _plain_text(value: Any, *, limit: int = 12000) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
        text = " ".join(parser.parts) or raw
    except Exception:
        text = raw
    return " ".join(text.split())[:limit]


def _string_tuple(value: Any, *, limit: int = 128) -> tuple[str, ...]:
    if isinstance(value, str):
        source: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        source = value
    else:
        source = ()
    output: list[str] = []
    seen: set[str] = set()
    for item in source:
        text = str(item or "").strip()
        folded = text.casefold()
        if not text or folded in seen:
            continue
        seen.add(folded)
        output.append(text[:512])
        if len(output) >= limit:
            break
    return tuple(output)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any, *, limit: int = 200) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value[:limit] if isinstance(item, Mapping)]


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sanitize_for_cache(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            token = str(key).replace("-", "").replace("_", "").casefold()
            if token in {item.replace("_", "") for item in _SECRET_KEYS}:
                continue
            output[str(key)] = _sanitize_for_cache(item)
        return output
    if isinstance(value, list):
        return [_sanitize_for_cache(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_cache(item) for item in value]
    return value


class _ProviderCache:
    def __init__(self, root: Path, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS, max_files: int = _MAX_CACHE_FILES) -> None:
        self.root = root.resolve()
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_files = max(16, int(max_files))

    @staticmethod
    def _key(path: str, params: Sequence[tuple[str, str]]) -> str:
        payload = json.dumps([path, sorted((str(k), str(v)) for k, v in params)], separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def path_for(self, path: str, params: Sequence[tuple[str, str]]) -> Path:
        return self.root / f"{self._key(path, params)}.json"

    def load(self, path: str, params: Sequence[tuple[str, str]]) -> dict[str, Any] | None:
        target = self.path_for(path, params)
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def is_fresh(self, record: Mapping[str, Any]) -> bool:
        try:
            fetched_at = float(record.get("fetched_at_unix") or 0.0)
        except (TypeError, ValueError):
            return False
        return self.ttl_seconds > 0 and (time.time() - fetched_at) <= self.ttl_seconds

    def store(
        self,
        path: str,
        params: Sequence[tuple[str, str]],
        payload: Mapping[str, Any],
        *,
        etag: str = "",
        last_modified: str = "",
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "fetched_at_unix": time.time(),
            "etag": str(etag or "")[:512],
            "last_modified": str(last_modified or "")[:512],
            "payload": _sanitize_for_cache(dict(payload)),
        }
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.root),
            prefix=".asset-hub-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(serialized)
            temporary = Path(stream.name)
        temporary.replace(self.path_for(path, params))
        self._prune()

    def touch(self, path: str, params: Sequence[tuple[str, str]], record: Mapping[str, Any]) -> None:
        payload = _mapping(record.get("payload"))
        self.store(
            path,
            params,
            payload,
            etag=str(record.get("etag") or ""),
            last_modified=str(record.get("last_modified") or ""),
        )

    def _prune(self) -> None:
        try:
            files = sorted(
                (item for item in self.root.glob("*.json") if item.is_file()),
                key=lambda item: item.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return
        for item in files[self.max_files :]:
            try:
                item.unlink()
            except OSError:
                pass


class CivitaiProvider:
    provider_id = "civitai"

    def __init__(
        self,
        cache_root: str | os.PathLike[str],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 20.0,
        cache_ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
        secret_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self.cache = _ProviderCache(Path(cache_root), ttl_seconds=cache_ttl_seconds)
        self.transport = transport
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.secret_provider = secret_provider or (lambda: None)

    def descriptor(self) -> ProviderDescriptor:
        return ProviderDescriptor(
            provider_id=self.provider_id,
            display_name="Civitai",
            authentication_mode="optional_bearer_phase_02",
        )

    @staticmethod
    def _validate_path(path: str) -> str:
        selected = str(path or "").strip()
        if not selected.startswith("/") or "://" in selected or ".." in selected:
            raise AssetHubError("provider_policy_blocked", "Provider routes cannot fetch arbitrary upstream URLs.", status_code=400)
        return selected

    async def _request_json(
        self,
        path: str,
        params: Sequence[tuple[str, str]] = (),
        *,
        refresh: bool = False,
        secret_override: str | None = None,
        include_delivery_urls: bool = False,
        cache_response: bool = True,
    ) -> dict[str, Any]:
        path = self._validate_path(path)
        normalized_params = tuple((str(key), str(value)) for key, value in params if str(value).strip())
        cached = None if (refresh or include_delivery_urls) else self.cache.load(path, normalized_params)
        if cached and self.cache.is_fresh(cached):
            return _mapping(cached.get("payload"))

        headers = {
            "Accept": "application/json",
            "User-Agent": f"{PRODUCT_NAME}-AssetHub/2",
            "Referer": "https://civitai.com/",
        }
        secret = str(secret_override if secret_override is not None else (self.secret_provider() or "")).strip()
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        if cached:
            if cached.get("etag"):
                headers["If-None-Match"] = str(cached["etag"])
            if cached.get("last_modified"):
                headers["If-Modified-Since"] = str(cached["last_modified"])

        try:
            async with httpx.AsyncClient(
                base_url=_API_ROOT,
                transport=self.transport,
                timeout=self.timeout_seconds,
                follow_redirects=False,
                headers=headers,
            ) as client:
                response = await client.get(path, params=list(normalized_params))
        except httpx.TimeoutException as exc:
            raise AssetHubError("provider_timeout", "Civitai did not respond before the request timeout.", status_code=504) from exc
        except httpx.HTTPError as exc:
            raise AssetHubError("provider_unavailable", "Civitai could not be reached.", status_code=502) from exc

        if response.status_code == 304 and cached:
            self.cache.touch(path, normalized_params, cached)
            return _mapping(cached.get("payload"))
        if response.status_code in {401, 403}:
            raise AssetHubError("provider_auth_required", "Civitai rejected or requires provider authentication.", status_code=401)
        if response.status_code == 404:
            raise AssetHubError("provider_not_found", "The requested Civitai resource was not found.", status_code=404)
        if response.status_code == 429:
            try:
                retry_after = max(0, int(float(response.headers.get("Retry-After", "0") or 0)))
            except (TypeError, ValueError):
                retry_after = None
            raise AssetHubError(
                "provider_rate_limited",
                "Civitai rate-limited the request.",
                status_code=429,
                retry_after_seconds=retry_after,
            )
        if response.status_code >= 500:
            raise AssetHubError("provider_unavailable", f"Civitai returned HTTP {response.status_code}.", status_code=502)
        if response.status_code >= 400:
            raise AssetHubError("provider_bad_response", f"Civitai returned HTTP {response.status_code}.", status_code=502)
        if len(response.content) > self.max_response_bytes:
            raise AssetHubError("provider_bad_response", "Civitai response exceeded the bounded metadata size limit.", status_code=502)
        try:
            payload = response.json()
        except ValueError as exc:
            raise AssetHubError("provider_bad_response", "Civitai returned malformed JSON.", status_code=502) from exc
        if not isinstance(payload, Mapping):
            raise AssetHubError("provider_bad_response", "Civitai returned an unexpected response shape.", status_code=502)
        raw = dict(payload)
        clean = _sanitize_for_cache(raw)
        if cache_response:
            self.cache.store(
                path,
                normalized_params,
                clean,
                etag=response.headers.get("ETag", ""),
                last_modified=response.headers.get("Last-Modified", ""),
            )
        return raw if include_delivery_urls else dict(clean)

    @staticmethod
    def _preview(raw: Mapping[str, Any]) -> ProviderPreview | None:
        url = str(raw.get("url") or "").strip()
        if not url:
            return None
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            return None
        kind = str(raw.get("type") or "image").strip().casefold()
        if kind not in {"image", "video"}:
            kind = "image"
        return ProviderPreview(
            url=url[:4096],
            width=_safe_int(raw.get("width"), 0) or None,
            height=_safe_int(raw.get("height"), 0) or None,
            nsfw_level=str(raw.get("nsfwLevel") if "nsfwLevel" in raw else raw.get("nsfw") or "")[:64],
            kind=kind,
        )

    @staticmethod
    def _file(raw: Mapping[str, Any], *, model_id: str, version_id: str, base_model: str, trained_words: tuple[str, ...]) -> ProviderFile:
        metadata = _mapping(raw.get("metadata"))
        hashes = _mapping(raw.get("hashes"))
        scan = ProviderScanSummary(
            pickle_scan_result=str(raw.get("pickleScanResult") or "")[:128],
            virus_scan_result=str(raw.get("virusScanResult") or "")[:128],
            scanned_at=str(raw.get("scannedAt") or "")[:128],
        )
        size_kb = raw.get("sizeKB")
        try:
            size_bytes = max(0, int(float(size_kb) * 1024)) if size_kb not in (None, "") else 0
        except (TypeError, ValueError):
            size_bytes = 0
        return ProviderFile.with_hashes(
            provider_id="civitai",
            remote_model_id=model_id,
            remote_version_id=version_id,
            remote_file_id=str(raw.get("id") or ""),
            file_name=str(raw.get("name") or "")[:1024],
            file_type=str(raw.get("type") or "")[:128],
            format=str(metadata.get("format") or "")[:128],
            size_bytes=size_bytes,
            base_model=base_model,
            architecture=normalize_architecture(base_model),
            trained_words=trained_words,
            hashes=hashes,
            primary=bool(raw.get("primary", False)),
            scan=scan,
            source_page_url=f"https://civitai.com/models/{model_id}?modelVersionId={version_id}" if model_id and version_id else "",
        )

    @classmethod
    def _version(cls, raw: Mapping[str, Any], *, model_id: str = "") -> ProviderVersion:
        version_id = str(raw.get("id") or "")
        resolved_model_id = str(raw.get("modelId") or model_id or "")
        base_model = str(raw.get("baseModel") or "").strip()
        trained_words = _string_tuple(raw.get("trainedWords"), limit=64)
        files = tuple(
            cls._file(item, model_id=resolved_model_id, version_id=version_id, base_model=base_model, trained_words=trained_words)
            for item in _mapping_list(raw.get("files"), limit=64)
        )
        previews = tuple(
            preview
            for item in _mapping_list(raw.get("images"), limit=32)
            if (preview := cls._preview(item)) is not None
        )
        return ProviderVersion.create(
            provider_id="civitai",
            remote_model_id=resolved_model_id,
            remote_version_id=version_id,
            name=str(raw.get("name") or "")[:512],
            base_model=base_model[:256],
            architecture=normalize_architecture(base_model),
            description=_plain_text(raw.get("description")),
            trained_words=trained_words,
            published_at=str(raw.get("publishedAt") or "")[:128],
            updated_at=str(raw.get("updatedAt") or "")[:128],
            files=files,
            previews=previews,
            stats=_mapping(raw.get("stats")),
        )

    @classmethod
    def _summary(cls, raw: Mapping[str, Any]) -> ProviderModelSummary:
        model_id = str(raw.get("id") or "")
        provider_type = str(raw.get("type") or "").strip()
        creator = _mapping(raw.get("creator"))
        versions = tuple(cls._version(item, model_id=model_id) for item in _mapping_list(raw.get("modelVersions"), limit=64))
        return ProviderModelSummary(
            provider_id="civitai",
            remote_model_id=model_id,
            name=str(raw.get("name") or "")[:512],
            asset_kind=provider_type_to_asset_kind(provider_type),
            provider_type=provider_type[:128],
            creator=str(creator.get("username") or creator.get("name") or "")[:256],
            description=_plain_text(raw.get("description")),
            tags=_string_tuple(raw.get("tags"), limit=128),
            nsfw=bool(raw.get("nsfw", False)),
            versions=versions,
        )

    @classmethod
    def _model(cls, raw: Mapping[str, Any]) -> ProviderModel:
        summary = cls._summary(raw)
        permissions = ProviderPermissionSummary(
            allow_no_credit=_optional_bool(raw.get("allowNoCredit")),
            allow_commercial_use=_string_tuple(raw.get("allowCommercialUse"), limit=16),
            allow_derivatives=_optional_bool(raw.get("allowDerivatives")),
            allow_different_license=_optional_bool(raw.get("allowDifferentLicense")),
        )
        return ProviderModel(
            provider_id=summary.provider_id,
            remote_model_id=summary.remote_model_id,
            name=summary.name,
            asset_kind=summary.asset_kind,
            provider_type=summary.provider_type,
            creator=summary.creator,
            description=summary.description,
            tags=summary.tags,
            nsfw=summary.nsfw,
            source_page_url=f"https://civitai.com/models/{summary.remote_model_id}" if summary.remote_model_id else "",
            permissions=permissions,
            versions=summary.versions,
        )

    @staticmethod
    def _cursor_page(cursor: str) -> int:
        token = str(cursor or "").strip()
        if not token:
            return 1
        if token.startswith("civitai:page:"):
            token = token.rsplit(":", 1)[-1]
        try:
            page = int(token)
        except ValueError as exc:
            raise AssetHubError("provider_policy_blocked", "Invalid provider continuation cursor.", status_code=400) from exc
        if page < 1 or page > 100000:
            raise AssetHubError("provider_policy_blocked", "Invalid provider continuation cursor.", status_code=400)
        return page

    @staticmethod
    def _next_cursor(metadata: Mapping[str, Any]) -> str:
        next_page = str(metadata.get("nextPage") or "").strip()
        if next_page:
            parsed = urlparse(next_page)
            if parsed.scheme and (parsed.scheme.lower() != "https" or str(parsed.hostname or "").casefold() != _ALLOWED_HOST):
                return ""
            page_values = parse_qs(parsed.query).get("page") or []
            if page_values:
                try:
                    page = int(page_values[0])
                except (TypeError, ValueError):
                    page = 0
                if page > 0:
                    return f"civitai:page:{page}"
        current = _safe_int(metadata.get("currentPage"), 0)
        total = _safe_int(metadata.get("totalPages"), 0)
        if current and total and current < total:
            return f"civitai:page:{current + 1}"
        return ""

    async def search(self, request: ProviderSearchRequest) -> ProviderSearchPage:
        kind = str(request.asset_kind or "checkpoint").strip().casefold()
        provider_type = _PROVIDER_TYPE_FILTERS.get(kind)
        if not provider_type:
            raise AssetHubError("provider_policy_blocked", f"Civitai search does not expose asset kind {kind!r} in Phase 01.", status_code=400)
        params: list[tuple[str, str]] = [
            ("limit", str(max(1, min(int(request.limit or 24), 50)))),
            ("page", str(self._cursor_page(request.cursor))),
            ("types", provider_type),
        ]
        if request.query.strip():
            params.append(("query", request.query.strip()[:256]))
        if request.creator.strip():
            params.append(("username", request.creator.strip()[:256]))
        sort = _ALLOWED_SORTS.get(request.sort.strip().casefold()) if request.sort else None
        period = _ALLOWED_PERIODS.get(request.period.strip().casefold()) if request.period else None
        if sort:
            params.append(("sort", sort))
        if period:
            params.append(("period", period))
        if request.safe_content:
            params.append(("nsfw", "false"))
        for architecture in request.base_models:
            for provider_base in _PROVIDER_BASE_MODELS.get(str(architecture), ()):
                params.append(("baseModels", provider_base))

        payload = await self._request_json("/models", params, refresh=request.refresh)
        items = tuple(self._summary(item) for item in _mapping_list(payload.get("items"), limit=50))
        metadata = _mapping(payload.get("metadata"))
        total_items = _safe_int(metadata.get("totalItems"), -1)
        return ProviderSearchPage(
            provider_id=self.provider_id,
            items=items,
            next_cursor=self._next_cursor(metadata),
            total_items=total_items if total_items >= 0 else None,
        )

    async def get_model(self, remote_model_id: str, *, refresh: bool = False) -> ProviderModel:
        model_id = str(remote_model_id or "").strip()
        if not model_id.isdigit():
            raise AssetHubError("provider_policy_blocked", "Civitai model IDs must be numeric.", status_code=400)
        payload = await self._request_json(f"/models/{model_id}", refresh=refresh)
        return self._model(payload)

    async def get_version(self, remote_version_id: str, *, refresh: bool = False) -> ProviderVersion:
        version_id = str(remote_version_id or "").strip()
        if not version_id.isdigit():
            raise AssetHubError("provider_policy_blocked", "Civitai version IDs must be numeric.", status_code=400)
        payload = await self._request_json(f"/model-versions/{version_id}", refresh=refresh)
        return self._version(payload)

    async def lookup_hash(self, file_hash: str, *, refresh: bool = False) -> ProviderVersion:
        normalized = str(file_hash or "").strip().lower()
        if len(normalized) < 12 or any(character not in "0123456789abcdef" for character in normalized):
            raise AssetHubError("provider_policy_blocked", "Hash lookup requires a hexadecimal file hash.", status_code=400)
        payload = await self._request_json(f"/model-versions/by-hash/{normalized}", refresh=refresh)
        return self._version(payload)

    async def validate_secret(self, secret: str) -> bool:
        token = str(secret or "").strip()
        if not token:
            raise AssetHubError("provider_auth_required", "A Civitai API token is required.", status_code=400)
        await self._request_json(
            "/models",
            (("limit", "1"),),
            refresh=True,
            secret_override=token,
            cache_response=False,
        )
        return True

    async def resolve_download_source(
        self,
        remote_model_id: str,
        remote_version_id: str,
        remote_file_id: str,
        *,
        secret: str | None = None,
    ) -> ProviderDownloadSource:
        model_id = str(remote_model_id or "").strip()
        version_id = str(remote_version_id or "").strip()
        file_id = str(remote_file_id or "").strip()
        if not model_id.isdigit() or not version_id.isdigit() or not file_id.isdigit():
            raise AssetHubError("provider_policy_blocked", "Civitai download identities must be numeric provider IDs.", status_code=400)
        payload = await self._request_json(
            f"/model-versions/{version_id}",
            refresh=True,
            secret_override=secret,
            include_delivery_urls=True,
            cache_response=True,
        )
        resolved_model = str(payload.get("modelId") or model_id).strip()
        if resolved_model and resolved_model != model_id:
            raise AssetHubError("provider_bad_response", "Civitai returned a model identity that does not match the download plan.", status_code=502)
        selected: dict[str, Any] | None = None
        for item in _mapping_list(payload.get("files"), limit=128):
            if str(item.get("id") or "").strip() == file_id:
                selected = item
                break
        if selected is None:
            raise AssetHubError("provider_not_found", "The selected Civitai file no longer exists in this model version.", status_code=404)
        url = str(selected.get("downloadUrl") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme.casefold() != "https" or not parsed.hostname:
            raise AssetHubError("provider_bad_response", "Civitai did not provide a safe HTTPS delivery URL for the selected file.", status_code=502)
        size_bytes = 0
        if selected.get("sizeKB") not in (None, ""):
            try:
                size_bytes = max(0, int(float(selected.get("sizeKB")) * 1024))
            except (TypeError, ValueError):
                size_bytes = 0
        hashes = _mapping(selected.get("hashes"))
        sha256 = str(hashes.get("SHA256") or hashes.get("sha256") or "").strip().lower()
        if sha256 and (len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256)):
            sha256 = ""
        return ProviderDownloadSource(
            provider_id=self.provider_id,
            remote_model_id=model_id,
            remote_version_id=version_id,
            remote_file_id=file_id,
            file_name=str(selected.get("name") or f"civitai-{file_id}.bin")[:1024],
            url=url,
            expected_bytes=size_bytes,
            expected_sha256=sha256,
            auth_hosts=(_ALLOWED_HOST,),
        )

