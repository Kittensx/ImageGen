from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from modules.project_context import ProjectContext


CIVITAI_LORA_METADATA_SCHEMA_VERSION = 2
_DEFAULT_KEY_FILE = Path("secrets") / "civitai_api_key.txt"
_API_ROOT = "https://civitai.com/api/v1"
_USER_AGENT = "IMAGE_GEN-WebUI"
_PLACEHOLDER_KEYS = {
    "",
    "paste_your_civitai_api_key_here",
    "your_api_key",
    "your-civitai-api-key",
}
_MAX_PREVIEW_BYTES = 32 * 1024 * 1024
_IMAGE_CONTENT_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class CivitaiMetadataError(RuntimeError):
    """Base user-facing Civitai metadata lookup error."""


class CivitaiCredentialError(CivitaiMetadataError):
    """Raised when the configured private-key file cannot be used."""


class CivitaiMetadataNotFound(CivitaiMetadataError):
    """Raised when Civitai has no model version matching the supplied hashes."""


class CivitaiRequestError(CivitaiMetadataError):
    """Raised when the Civitai API cannot complete a lookup."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise CivitaiRequestError(
            f"Civitai redirected the metadata request to {newurl}; the private key was not forwarded."
        )


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = str(data or "").strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


def _plain_text(value: Any, *, limit: int = 4000) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
        text = parser.text() or raw
    except Exception:
        text = raw
    normalized = " ".join(text.split())
    return normalized[:limit]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = []
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        folded = text.casefold()
        if not text or folded in seen:
            continue
        seen.add(folded)
        output.append(text)
    return output


def _is_civitai_host(hostname: str | None) -> bool:
    host = str(hostname or "").strip().lower().rstrip(".")
    return host == "civitai.com" or host.endswith(".civitai.com")


def _is_civitai_image_host(hostname: str | None) -> bool:
    host = str(hostname or "").strip().lower().rstrip(".")
    return _is_civitai_host(host) or host == "civitai.green" or host.endswith(".civitai.green")


def _normalize_hashes(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if len(token) < 12 or any(character not in "0123456789abcdef" for character in token):
            continue
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def _configured_key_value(context: ProjectContext) -> str:
    direct = context.config.get("civitai")
    if isinstance(direct, Mapping):
        value = str(direct.get("api_key_file") or "").strip()
        if value:
            return value
    integrations = context.config.get("integrations")
    if isinstance(integrations, Mapping):
        civitai = integrations.get("civitai")
        if isinstance(civitai, Mapping):
            value = str(civitai.get("api_key_file") or "").strip()
            if value:
                return value
    return ""


def resolve_civitai_key_path(context: ProjectContext) -> Path:
    configured = (
        os.environ.get("CIVITAI_API_KEY_FILE", "").strip()
        or _configured_key_value(context)
        or str(_DEFAULT_KEY_FILE)
    )
    path = Path(os.path.expandvars(os.path.expanduser(configured)))
    if not path.is_absolute():
        path = context.project_root / path
    return path.resolve()


def read_civitai_api_key(context: ProjectContext) -> tuple[Path, str]:
    path = resolve_civitai_key_path(context)
    if not path.is_file():
        raise CivitaiCredentialError(
            "Civitai API key file not found: "
            f"{path}. Set CIVITAI_API_KEY_FILE or configure civitai.api_key_file in "
            "user_config/user-config.yml. The file must contain only the key."
        )
    try:
        key = path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise CivitaiCredentialError(f"Could not read the Civitai API key file {path}: {exc}") from exc
    if key.casefold() in _PLACEHOLDER_KEYS or len(key) < 10:
        raise CivitaiCredentialError(f"The Civitai API key file {path} is empty or contains a placeholder.")
    if any(character.isspace() for character in key):
        raise CivitaiCredentialError("The Civitai API key must be a single line without spaces.")
    return path, key


class CivitaiLoraMetadataClient:
    """Authenticated, backend-only Civitai metadata lookup for installed LoRAs."""

    def __init__(
        self,
        context: ProjectContext,
        *,
        request_json: Callable[[str, str], Mapping[str, Any]] | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.context = context
        self._request_json_override = request_json
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_retries = max(0, int(max_retries))
        self._opener = build_opener(_RejectRedirects())

    def key_status(self) -> dict[str, Any]:
        path = resolve_civitai_key_path(self.context)
        return {
            "configured": path.is_file(),
            "key_file": str(path),
        }

    def _request_json(self, url: str, api_key: str) -> dict[str, Any]:
        if self._request_json_override is not None:
            payload = self._request_json_override(url, api_key)
            if not isinstance(payload, Mapping):
                raise CivitaiRequestError("Civitai returned an unexpected non-object response.")
            return dict(payload)

        parsed = urlparse(url)
        if parsed.scheme.lower() != "https" or not _is_civitai_host(parsed.hostname):
            raise CivitaiRequestError(f"Refusing to send the Civitai API key to an untrusted URL: {url}")
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Referer": "https://civitai.com/",
                "User-Agent": _USER_AGENT,
            },
            method="GET",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener.open(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                    encoding = response.headers.get_content_charset() or "utf-8"
                    payload = json.loads(raw.decode(encoding))
                    if not isinstance(payload, Mapping):
                        raise CivitaiRequestError("Civitai returned an unexpected non-object response.")
                    return dict(payload)
            except HTTPError as exc:
                if exc.code == 404:
                    raise CivitaiMetadataNotFound("No Civitai model version matched this LoRA hash.") from exc
                if exc.code in {401, 403}:
                    raise CivitaiCredentialError(
                        "Civitai rejected the configured API key. Verify the key file and account access."
                    ) from exc
                last_error = exc
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                if not retryable or attempt >= self.max_retries:
                    raise CivitaiRequestError(f"Civitai API returned HTTP {exc.code}.") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
                try:
                    delay = min(5.0, max(0.0, float(retry_after))) if retry_after else min(2.0 ** attempt, 5.0)
                except (TypeError, ValueError):
                    delay = min(2.0 ** attempt, 5.0)
                time.sleep(delay)
            except (URLError, TimeoutError, json.JSONDecodeError, CivitaiRequestError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0 ** attempt, 5.0))
        raise CivitaiRequestError(f"Could not read Civitai metadata: {last_error}")

    def download_preview_image(self, image_url: str) -> tuple[str, bytes]:
        """Download a trusted Civitai preview without forwarding the API key."""

        url = str(image_url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https" or not _is_civitai_image_host(parsed.hostname):
            raise CivitaiRequestError(f"Refusing to download an untrusted Civitai preview URL: {url}")
        request = Request(
            url,
            headers={
                "Accept": "image/png,image/jpeg,image/webp",
                "Referer": "https://civitai.com/",
                "User-Agent": _USER_AGENT,
            },
            method="GET",
        )
        try:
            with build_opener().open(request, timeout=self.timeout_seconds) as response:
                final_url = str(response.geturl() or url)
                final_parsed = urlparse(final_url)
                if final_parsed.scheme.lower() != "https" or not _is_civitai_image_host(final_parsed.hostname):
                    raise CivitaiRequestError(
                        f"Civitai preview redirected to an untrusted host: {final_url}"
                    )
                content_type = str(response.headers.get_content_type() or "").lower()
                if content_type not in _IMAGE_CONTENT_EXTENSIONS:
                    raise CivitaiRequestError(
                        f"Civitai preview returned unsupported content type {content_type or 'unknown'}."
                    )
                content_length = response.headers.get("Content-Length")
                try:
                    declared_size = int(content_length) if content_length else 0
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > _MAX_PREVIEW_BYTES:
                    raise CivitaiRequestError("Civitai preview exceeds the 32 MB safety limit.")
                content = response.read(_MAX_PREVIEW_BYTES + 1)
        except HTTPError as exc:
            raise CivitaiRequestError(f"Civitai preview returned HTTP {exc.code}.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise CivitaiRequestError(f"Could not download the Civitai preview: {exc}") from exc
        if len(content) > _MAX_PREVIEW_BYTES:
            raise CivitaiRequestError("Civitai preview exceeds the 32 MB safety limit.")
        if not content:
            raise CivitaiRequestError("Civitai preview download was empty.")
        suffix = _IMAGE_CONTENT_EXTENSIONS[content_type]
        return f"civitai-preview{suffix}", content

    @staticmethod
    def _model_page_url(model_id: Any, version_id: Any) -> str:
        try:
            model = int(model_id)
        except (TypeError, ValueError):
            return ""
        if model <= 0:
            return ""
        try:
            version = int(version_id)
        except (TypeError, ValueError):
            version = 0
        suffix = f"?modelVersionId={version}" if version > 0 else ""
        return f"https://civitai.com/models/{model}{suffix}"

    @staticmethod
    def _normalize_result(
        version: Mapping[str, Any],
        model: Mapping[str, Any],
        *,
        matched_hash: str,
    ) -> dict[str, Any]:
        version_payload = dict(version)
        model_payload = dict(model)
        embedded_model = version_payload.get("model")
        embedded_model = dict(embedded_model) if isinstance(embedded_model, Mapping) else {}
        creator = model_payload.get("creator")
        creator = dict(creator) if isinstance(creator, Mapping) else {}
        model_id = version_payload.get("modelId") or model_payload.get("id")
        version_id = version_payload.get("id")
        trained_words = _string_list(version_payload.get("trainedWords"))
        tags = _string_list(model_payload.get("tags"))
        version_description = _plain_text(version_payload.get("description"))
        model_description = _plain_text(model_payload.get("description"))
        source_url = CivitaiLoraMetadataClient._model_page_url(model_id, version_id)
        model_name = str(
            model_payload.get("name")
            or embedded_model.get("name")
            or version_payload.get("modelName")
            or ""
        ).strip()
        model_type = str(model_payload.get("type") or embedded_model.get("type") or "").strip()
        images = version_payload.get("images")
        images = list(images) if isinstance(images, (list, tuple)) else []
        normalized_images: list[dict[str, Any]] = []
        for raw_image in images:
            if not isinstance(raw_image, Mapping):
                continue
            image_url = str(raw_image.get("url") or "").strip()
            parsed_image = urlparse(image_url)
            image_type = str(raw_image.get("type") or "image").strip().lower()
            if (
                not image_url
                or parsed_image.scheme.lower() != "https"
                or not _is_civitai_image_host(parsed_image.hostname)
                or image_type not in {"", "image"}
            ):
                continue
            normalized_images.append({
                "url": image_url,
                "width": raw_image.get("width"),
                "height": raw_image.get("height"),
                "nsfw": raw_image.get("nsfw") if "nsfw" in raw_image else raw_image.get("nsfwLevel"),
                "type": image_type or "image",
            })
        preferred_image = next(
            (item for item in normalized_images if str(item.get("nsfw") or "").strip().lower() in {"", "false", "none", "0"}),
            normalized_images[0] if normalized_images else {},
        )
        return {
            "schema_version": CIVITAI_LORA_METADATA_SCHEMA_VERSION,
            "status": "matched",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "matched_hash": matched_hash,
            "model_id": model_id,
            "model_version_id": version_id,
            "model_name": model_name,
            "model_version_name": str(version_payload.get("name") or "").strip(),
            "model_type": model_type,
            "creator": str(creator.get("username") or "").strip(),
            "base_model": str(version_payload.get("baseModel") or "").strip(),
            "trained_words": trained_words,
            "activation_text": ", ".join(trained_words),
            "source_url": source_url,
            "description": version_description or model_description,
            "version_description": version_description,
            "model_description": model_description,
            "tags": tags,
            "images": normalized_images,
            "image_url": str(preferred_image.get("url") or ""),
            "image_width": preferred_image.get("width"),
            "image_height": preferred_image.get("height"),
            "image_nsfw": preferred_image.get("nsfw"),
            "manual_activation_text_search_required": not bool(trained_words),
        }

    def lookup_by_hashes(self, hashes: Iterable[Any]) -> dict[str, Any]:
        candidates = _normalize_hashes(hashes)
        if not candidates:
            raise CivitaiMetadataError(
                "This LoRA has no usable SHA-256 or Civitai-compatible hash. Run the LoRA scan first."
            )
        _, api_key = read_civitai_api_key(self.context)
        last_not_found: CivitaiMetadataNotFound | None = None
        for candidate in candidates:
            url = f"{_API_ROOT}/model-versions/by-hash/{quote(candidate, safe='')}"
            try:
                version = self._request_json(url, api_key)
            except CivitaiMetadataNotFound as exc:
                last_not_found = exc
                continue
            model: dict[str, Any] = {}
            model_id = version.get("modelId")
            try:
                parsed_model_id = int(model_id)
            except (TypeError, ValueError):
                parsed_model_id = 0
            if parsed_model_id > 0:
                try:
                    model = self._request_json(f"{_API_ROOT}/models/{parsed_model_id}", api_key)
                except CivitaiMetadataNotFound:
                    model = {}
                except CivitaiRequestError:
                    model = {}
            return self._normalize_result(version, model, matched_hash=candidate)
        raise last_not_found or CivitaiMetadataNotFound(
            "No Civitai model version matched the LoRA hashes calculated by ImageGen."
        )


__all__ = [
    "CIVITAI_LORA_METADATA_SCHEMA_VERSION",
    "CivitaiCredentialError",
    "CivitaiLoraMetadataClient",
    "CivitaiMetadataError",
    "CivitaiMetadataNotFound",
    "CivitaiRequestError",
    "read_civitai_api_key",
    "resolve_civitai_key_path",
]
