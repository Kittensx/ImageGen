from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from image_gen.program_metadata import PRODUCT_NAME
from image_gen.webui.asset_metadata import (
    load_asset_metadata,
    replace_asset_preview,
    resolve_preview_path,
    save_asset_sidecar_fields,
    synchronize_asset_companions,
)
from modules.project_context import ProjectContext


CIVITAI_ASSET_METADATA_SCHEMA_VERSION = 3
_DEFAULT_KEY_FILE = Path("secrets") / "civitai_api_key.txt"
_API_ROOT = "https://civitai.com/api/v1"
_USER_AGENT = f"{PRODUCT_NAME}-WebUI"
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


def _plain_text(value: Any, *, limit: int = 12000) -> str:
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


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        while True:
            chunk = stream.read(max(64 * 1024, int(chunk_size)))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
    try:
        key = _validate_civitai_api_key_value(key)
    except CivitaiCredentialError as exc:
        raise CivitaiCredentialError(f"The CivitAI API key file {path} is invalid: {exc}") from exc
    return path, key




def sync_civitai_api_key_to_secret_store(context: ProjectContext, secret_store: Any) -> dict[str, Any]:
    """Load the existing project CivitAI key into an in-memory provider secret store.

    This intentionally reuses ``secrets/civitai_api_key.txt`` (or the configured
    override) and never creates a second credential file. The Asset Hub secret
    store receives only a session copy so download/runtime code can use the same
    credential source as the existing CivitAI metadata integration.
    """

    path, key = read_civitai_api_key(context)
    secret_store.set("civitai", key, persistent=False)
    try:
        display_path = path.relative_to(context.project_root.resolve()).as_posix()
    except ValueError:
        display_path = str(path)
    return {
        "configured": True,
        "source": "project_file",
        "key_file": display_path,
    }


def _civitai_auth_request_path(context: ProjectContext) -> Path:
    return (Path(context.cache_root) / "asset-hub" / "civitai-auth-request.json").resolve()


def request_civitai_authentication(
    context: ProjectContext,
    *,
    reason: str,
    source: str = "",
    fixture_id: str = "",
) -> dict[str, Any]:
    """Publish a secret-free authentication handoff for the local WebUI."""

    path = _civitai_auth_request_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    request_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    payload = {
        "pending": True,
        "request_id": request_id,
        "provider": "civitai",
        "reason": str(reason or "CivitAI authentication is required.").strip(),
        "source": str(source or "").strip(),
        "fixture_id": str(fixture_id or "").strip(),
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "credential_path": civitai_api_key_status(context).get("key_file", "secrets/civitai_api_key.txt"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return payload


def civitai_authentication_request_status(context: ProjectContext) -> dict[str, Any]:
    path = _civitai_auth_request_path(context)
    if not path.is_file():
        return {"pending": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"pending": False}
    if not isinstance(payload, Mapping) or payload.get("pending") is not True:
        return {"pending": False}
    allowed = {
        "pending",
        "request_id",
        "provider",
        "reason",
        "source",
        "fixture_id",
        "requested_at",
        "credential_path",
    }
    return {key: payload.get(key) for key in allowed if key in payload}


def clear_civitai_authentication_request(context: ProjectContext, *, request_id: str = "") -> None:
    path = _civitai_auth_request_path(context)
    if not path.is_file():
        return
    if request_id:
        current = civitai_authentication_request_status(context)
        if str(current.get("request_id") or "") != str(request_id):
            return
    try:
        path.unlink()
    except FileNotFoundError:
        pass

def _validate_civitai_api_key_value(value: Any) -> str:
    key = str(value or "").strip()
    if key.casefold() in _PLACEHOLDER_KEYS or len(key) < 10:
        raise CivitaiCredentialError("The CivitAI API key is empty or contains a placeholder.")
    if any(character.isspace() for character in key):
        raise CivitaiCredentialError("The CivitAI API key must be a single line without spaces.")
    return key


def _ui_managed_key_path(context: ProjectContext) -> tuple[Path, bool]:
    path = resolve_civitai_key_path(context)
    try:
        path.relative_to(context.project_root.resolve())
        return path, True
    except ValueError:
        return path, False


def civitai_api_key_status(context: ProjectContext) -> dict[str, Any]:
    path, managed = _ui_managed_key_path(context)
    configured = path.is_file()
    usable = False
    message = "CivitAI is not connected."
    if configured:
        try:
            read_civitai_api_key(context)
            usable = True
            message = "CivitAI API key is configured locally."
        except CivitaiCredentialError:
            message = "The configured CivitAI API key file is empty, unreadable, or invalid."
    if managed:
        try:
            display_path = path.relative_to(context.project_root.resolve()).as_posix()
        except ValueError:
            display_path = str(path)
    else:
        display_path = "Externally managed credential file"
    return {
        "configured": configured,
        "usable": usable,
        "managed_by_ui": managed,
        "key_file": display_path,
        "message": message,
    }


def write_civitai_api_key(context: ProjectContext, value: Any) -> dict[str, Any]:
    key = _validate_civitai_api_key_value(value)
    path, managed = _ui_managed_key_path(context)
    if not managed:
        raise CivitaiCredentialError(
            f"The configured CivitAI API key file is outside the {PRODUCT_NAME} project. "
            "For safety, update that externally managed credential file manually."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    default_secrets_root = (context.project_root / _DEFAULT_KEY_FILE.parent).resolve()
    if path.parent == default_secrets_root:
        ignore_file = path.parent / ".gitignore"
        if not ignore_file.exists():
            try:
                ignore_file.write_text("*\n!.gitignore\n", encoding="utf-8")
            except OSError:
                pass
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(key + "\n", encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
    return civitai_api_key_status(context)


def delete_civitai_api_key(context: ProjectContext) -> dict[str, Any]:
    path, managed = _ui_managed_key_path(context)
    if not managed:
        raise CivitaiCredentialError(
            f"The configured CivitAI API key file is outside the {PRODUCT_NAME} project and cannot be removed from the WebUI."
        )
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise CivitaiCredentialError(f"Could not remove the CivitAI API key file {path}: {exc}") from exc
    return civitai_api_key_status(context)

def _base_model_family(value: Any) -> str:
    token = str(value or "").strip().casefold().replace("_", " ").replace("-", " ")
    collapsed = " ".join(token.split())
    if not collapsed:
        return ""
    if "sdxl" in collapsed or "stable diffusion xl" in collapsed or collapsed.startswith("xl"):
        return "sdxl"
    if "sd 3" in collapsed or "sd3" in collapsed or "stable diffusion 3" in collapsed:
        return "sd3"
    if "flux" in collapsed:
        return "flux"
    if "sd 2" in collapsed or "sd2" in collapsed or "stable diffusion 2" in collapsed:
        return "sd2.x"
    if "sd 1" in collapsed or "sd1" in collapsed or "stable diffusion 1" in collapsed:
        return "sd1.x"
    return str(value or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


class CivitaiAssetMetadataService:
    """Backend-only Civitai metadata lookup and sidecar enrichment for any local asset.

    The service is intentionally asset-type agnostic. Callers provide a local file,
    an asset type, and any already-known hashes. The service always retains the
    complete Civitai model/model-version payload in the sidecar while also exposing
    normalized fields for IMAGE_GEN cards and editors.
    """

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
        return civitai_api_key_status(self.context)

    def test_connection(self) -> dict[str, Any]:
        _, api_key = read_civitai_api_key(self.context)
        payload = self._request_json(f"{_API_ROOT}/models?limit=1", api_key)
        return {
            "connected": isinstance(payload, Mapping),
            "message": "CivitAI connection verified.",
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
                    raise CivitaiMetadataNotFound("No Civitai model version matched this asset hash.") from exc
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
                    raise CivitaiRequestError(f"Civitai preview redirected to an untrusted host: {final_url}")
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
        return f"civitai-preview{_IMAGE_CONTENT_EXTENSIONS[content_type]}", content

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
    def _matched_file(files: list[dict[str, Any]], matched_hash: str) -> dict[str, Any]:
        needle = str(matched_hash or "").strip().casefold()
        for item in files:
            hashes = _mapping(item.get("hashes"))
            if any(str(value or "").strip().casefold() == needle for value in hashes.values()):
                return item
        # The by-hash endpoint already resolved the version. Some historical API
        # responses omit the queried digest from the nested file object, so keep
        # the primary file (or first file) as a useful fallback rather than
        # discarding its metadata.
        return next((item for item in files if bool(item.get("primary"))), files[0] if files else {})

    @classmethod
    def _normalize_result(
        cls,
        version: Mapping[str, Any],
        model: Mapping[str, Any],
        *,
        matched_hash: str,
        requested_asset_type: str = "",
    ) -> dict[str, Any]:
        version_payload = dict(version)
        model_payload = dict(model)
        embedded_model = _mapping(version_payload.get("model"))
        creator = _mapping(model_payload.get("creator") or embedded_model.get("creator"))
        model_id = version_payload.get("modelId") or model_payload.get("id") or embedded_model.get("id")
        version_id = version_payload.get("id")
        trained_words = _string_list(version_payload.get("trainedWords"))
        tags = _string_list(model_payload.get("tags") or embedded_model.get("tags"))
        version_description = _plain_text(version_payload.get("description"))
        model_description = _plain_text(model_payload.get("description") or embedded_model.get("description"))
        source_url = cls._model_page_url(model_id, version_id)
        model_name = str(
            model_payload.get("name")
            or embedded_model.get("name")
            or version_payload.get("modelName")
            or ""
        ).strip()
        model_type = str(model_payload.get("type") or embedded_model.get("type") or "").strip()
        files = _list_of_mappings(version_payload.get("files"))
        matched_file = cls._matched_file(files, matched_hash)

        images = _list_of_mappings(version_payload.get("images"))
        normalized_images: list[dict[str, Any]] = []
        for raw_image in images:
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
            normalized_images.append(dict(raw_image))
        preferred_image = next(
            (
                item for item in normalized_images
                if str(item.get("nsfw") if "nsfw" in item else item.get("nsfwLevel") or "").strip().lower()
                in {"", "false", "none", "0"}
            ),
            normalized_images[0] if normalized_images else {},
        )
        stats = _mapping(model_payload.get("stats"))
        version_stats = _mapping(version_payload.get("stats"))
        result = {
            "schema_version": CIVITAI_ASSET_METADATA_SCHEMA_VERSION,
            "status": "matched",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "requested_asset_type": str(requested_asset_type or "").strip().lower(),
            "matched_hash": matched_hash,
            "model_id": model_id,
            "model_version_id": version_id,
            "model_name": model_name,
            "model_version_name": str(version_payload.get("name") or "").strip(),
            "model_type": model_type,
            "creator": str(creator.get("username") or creator.get("name") or "").strip(),
            "creator_data": creator,
            "base_model": str(version_payload.get("baseModel") or "").strip(),
            "base_model_type": str(version_payload.get("baseModelType") or "").strip(),
            "trained_words": trained_words,
            "activation_text": ", ".join(trained_words),
            "source_url": source_url,
            "description": version_description or model_description,
            "version_description": version_description,
            "model_description": model_description,
            "tags": tags,
            "stats": stats,
            "version_stats": version_stats,
            "published_at": str(version_payload.get("publishedAt") or model_payload.get("publishedAt") or "").strip(),
            "updated_at": str(version_payload.get("updatedAt") or model_payload.get("updatedAt") or "").strip(),
            "availability": version_payload.get("availability") or model_payload.get("availability"),
            "nsfw": model_payload.get("nsfw"),
            "poi": model_payload.get("poi"),
            "minor": model_payload.get("minor"),
            "allow_no_credit": model_payload.get("allowNoCredit"),
            "allow_commercial_use": model_payload.get("allowCommercialUse"),
            "allow_derivatives": model_payload.get("allowDerivatives"),
            "allow_different_license": model_payload.get("allowDifferentLicense"),
            "files": files,
            "matched_file": matched_file,
            "images": normalized_images,
            "image_url": str(preferred_image.get("url") or ""),
            "image_width": preferred_image.get("width"),
            "image_height": preferred_image.get("height"),
            "image_nsfw": preferred_image.get("nsfw") if "nsfw" in preferred_image else preferred_image.get("nsfwLevel"),
            "manual_activation_text_search_required": bool(
                str(requested_asset_type or "").strip().lower() in {"lora", "loras", "textual_inversion", "textual-inversion", "embedding"}
                and not trained_words
            ),
            # Preserve the complete remote payload so IMAGE_GEN does not discard
            # fields that the current UI does not yet render.
            "raw": {
                "model_version": version_payload,
                "model": model_payload,
            },
        }
        return result

    def lookup_by_hashes(self, hashes: Iterable[Any], *, asset_type: str = "") -> dict[str, Any]:
        candidates = _normalize_hashes(hashes)
        if not candidates:
            raise CivitaiMetadataError(
                f"This asset has no usable SHA-256 or Civitai-compatible hash. Scan the asset first or allow {PRODUCT_NAME} to hash it."
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
            return self._normalize_result(
                version,
                model,
                matched_hash=candidate,
                requested_asset_type=asset_type,
            )
        raise last_not_found or CivitaiMetadataNotFound(
            f"No Civitai model version matched the hashes calculated by {PRODUCT_NAME}."
        )

    @staticmethod
    def _candidate_updates(
        result: Mapping[str, Any],
        metadata: Mapping[str, Any],
        *,
        overwrite: bool,
    ) -> dict[str, Any]:
        current = dict(metadata)
        desired: dict[str, Any] = {
            # CivitAI model/version names are provider provenance. They must not
            # replace the local model identity or create a user nickname.
            "source_url": str(result.get("source_url") or "").strip(),
            "description": str(result.get("description") or "").strip(),
            "tags": list(result.get("tags") or []),
            "model_family": _base_model_family(result.get("base_model")),
            "activation_text": str(result.get("activation_text") or "").strip(),
        }
        updates: dict[str, Any] = {}
        for field, value in desired.items():
            meaningful = bool(value) if not isinstance(value, list) else bool(value)
            current_value = current.get(field)
            current_meaningful = bool(current_value) if not isinstance(current_value, list) else bool(current_value)
            if meaningful and (overwrite or not current_meaningful):
                updates[field] = value
        return updates

    def enrich_local_asset(
        self,
        asset_path: str | os.PathLike[str],
        *,
        asset_type: str,
        hashes: Iterable[Any] = (),
        overwrite: bool = False,
        download_preview: bool = True,
    ) -> dict[str, Any]:
        path = Path(asset_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Asset file no longer exists: {path}")
        candidates = _normalize_hashes(hashes)
        sha256 = sha256_file(path)
        if sha256 not in candidates:
            candidates.insert(0, sha256)
        try:
            result = self.lookup_by_hashes(candidates, asset_type=asset_type)
        except TypeError as exc:
            # Backward compatibility for extensions/tests that monkeypatch the
            # former LoRA-only lookup signature as lookup_by_hashes(hashes).
            if "asset_type" not in str(exc):
                raise
            result = self.lookup_by_hashes(candidates)
        metadata = synchronize_asset_companions(path, load_asset_metadata(path))
        updates = self._candidate_updates(result, metadata, overwrite=overwrite)

        local_preview = resolve_preview_path(path, metadata)
        preview_downloaded = False
        preview_download_error = ""
        image_url = str(result.get("image_url") or "").strip()
        if download_preview and local_preview is None and image_url:
            try:
                filename, content = self.download_preview_image(image_url)
                local_preview, metadata = replace_asset_preview(path, filename=filename, content=content)
                metadata = save_asset_sidecar_fields(path, {
                    "_preview_provenance": {
                        "source": "civitai_cache",
                        "provider": "civitai",
                        "source_url": image_url,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                })
                metadata = synchronize_asset_companions(path, metadata)
            except (CivitaiMetadataError, OSError, ValueError) as exc:
                preview_download_error = str(exc)
            else:
                preview_downloaded = True

        applied_fields = sorted(updates)
        lookup_payload = {
            **dict(result),
            "local_asset_path": str(path),
            "local_asset_type": str(asset_type or "").strip().lower(),
            "applied_fields": applied_fields,
            "preview_image_downloaded": preview_downloaded,
            "preview_image_path": str(local_preview) if local_preview else "",
            "preview_image_download_error": preview_download_error,
        }
        for field in ("activation_text", "source_url", "model_family", "description", "tags"):
            lookup_payload[f"{field}_applied"] = field in updates

        saved = save_asset_sidecar_fields(path, {**updates, "_civitai_lookup": lookup_payload})
        return {
            "asset_type": str(asset_type or "").strip().lower(),
            "path": str(path),
            "sha256": sha256,
            "metadata": saved,
            "civitai_lookup": lookup_payload,
        }


__all__ = [
    "CIVITAI_ASSET_METADATA_SCHEMA_VERSION",
    "CivitaiAssetMetadataService",
    "CivitaiCredentialError",
    "CivitaiMetadataError",
    "CivitaiMetadataNotFound",
    "CivitaiRequestError",
    "civitai_api_key_status",
    "civitai_authentication_request_status",
    "clear_civitai_authentication_request",
    "delete_civitai_api_key",
    "read_civitai_api_key",
    "request_civitai_authentication",
    "resolve_civitai_key_path",
    "sha256_file",
    "sync_civitai_api_key_to_secret_store",
    "write_civitai_api_key",
]
