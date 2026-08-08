from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_COMPANION_METADATA_SUFFIXES = (
    ".civitai.info",
    ".metadata.json",
    ".json",
    ".info",
    ".txt",
    ".yaml",
    ".yml",
)
ASSET_DISCOVERY_SCHEMA_VERSION = 1
_EDITABLE_FIELDS = {
    "display_name",
    "nickname",
    "description",
    "source_url",
    "tags",
    "category",
    "favorite",
    "notes",
    "model_family",
    "architecture",
    "prediction_type",
    "conditioning_dimension",
    "activation_text",
    "preferred_weight",
    "preview_image",
}


def sidecar_path(asset_path: str | os.PathLike[str]) -> Path:
    path = Path(asset_path).expanduser().resolve()
    return path.with_name(f"{path.stem}.imagegen.json")


def load_asset_metadata(asset_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = sidecar_path(asset_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = []
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item or "").strip()
        folded = token.casefold()
        if not token or folded in seen:
            continue
        seen.add(folded)
        output.append(token)
    return output[:64]


def normalize_asset_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(value or {})
    output: dict[str, Any] = {}
    for key in _EDITABLE_FIELDS:
        if key not in source:
            continue
        raw = source.get(key)
        if key == "tags":
            output[key] = _normalize_tags(raw)
        elif key == "favorite":
            output[key] = bool(raw)
        elif key == "conditioning_dimension":
            try:
                output[key] = int(raw) if raw not in (None, "") else None
            except (TypeError, ValueError):
                output[key] = None
        elif key == "preferred_weight":
            try:
                output[key] = max(-4.0, min(4.0, float(raw)))
            except (TypeError, ValueError):
                output[key] = 1.0
        else:
            output[key] = str(raw or "").strip()
    return output


def _write_sidecar_payload(target: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(serialized)
        temporary = Path(stream.name)
    temporary.replace(target)
    return dict(payload)


def save_asset_metadata(
    asset_path: str | os.PathLike[str],
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    target = sidecar_path(asset_path)
    current = load_asset_metadata(asset_path)
    current.update(normalize_asset_metadata(value))
    return _write_sidecar_payload(target, current)


def save_asset_sidecar_fields(
    asset_path: str | os.PathLike[str],
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return load_asset_metadata(asset_path)
    target = sidecar_path(asset_path)
    current = load_asset_metadata(asset_path)
    current.update(dict(value))
    return _write_sidecar_payload(target, current)


def resolve_preview_path(
    asset_path: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    asset = Path(asset_path).expanduser().resolve()
    source = dict(metadata or {})
    configured = str(source.get("preview_image") or "").strip()
    candidates: list[Path] = []
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = asset.parent / configured_path
        candidates.append(configured_path)
    for suffix in _IMAGE_EXTENSIONS:
        candidates.append(asset.with_name(f"{asset.stem}.preview{suffix}"))
    for suffix in _IMAGE_EXTENSIONS:
        candidates.append(asset.with_suffix(suffix))
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        token = os.path.normcase(str(resolved))
        if token in seen:
            continue
        seen.add(token)
        if resolved.is_file() and resolved.suffix.lower() in _IMAGE_EXTENSIONS:
            return resolved
    return None


def _portable_companion_path(asset: Path, candidate: Path) -> str:
    try:
        return str(candidate.resolve().relative_to(asset.parent.resolve()))
    except (OSError, ValueError):
        try:
            return str(candidate.resolve())
        except OSError:
            return str(candidate)


def discover_asset_companions(
    asset_path: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover deterministic same-name preview and metadata companions.

    The scan deliberately avoids assigning arbitrary images from a shared model
    directory. Only files derived from the asset stem are eligible, which keeps
    multi-LoRA folders deterministic and safe to rescan at startup.
    """

    asset = Path(asset_path).expanduser().resolve()
    source = dict(metadata or {})
    preview = resolve_preview_path(asset, source)
    own_sidecar = sidecar_path(asset)
    metadata_files: list[str] = []
    seen: set[str] = set()
    for suffix in _COMPANION_METADATA_SUFFIXES:
        candidate = asset.with_name(f"{asset.stem}{suffix}")
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        token = os.path.normcase(str(resolved))
        if token in seen or resolved == own_sidecar or not resolved.is_file():
            continue
        seen.add(token)
        metadata_files.append(_portable_companion_path(asset, resolved))
    return {
        "schema_version": ASSET_DISCOVERY_SCHEMA_VERSION,
        "preview_image": _portable_companion_path(asset, preview) if preview else "",
        "metadata_files": metadata_files,
    }


def synchronize_asset_companions(
    asset_path: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist discovered preview/metadata locations without overwriting users.

    A valid explicit ``preview_image`` remains authoritative. When it is blank
    or stale and a deterministic same-stem preview exists, the relative path is
    written to the normal editable field so subsequent catalog loads do not
    depend on rediscovery or an Edit Metadata click.
    """

    asset = Path(asset_path).expanduser().resolve()
    current = dict(metadata) if isinstance(metadata, Mapping) else load_asset_metadata(asset)
    discovery = discover_asset_companions(asset, current)
    updates: dict[str, Any] = {}

    configured = str(current.get("preview_image") or "").strip()
    configured_valid = False
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = asset.parent / configured_path
        try:
            configured_valid = configured_path.resolve().is_file()
        except OSError:
            configured_valid = False
    discovered_preview = str(discovery.get("preview_image") or "").strip()
    if discovered_preview and not configured_valid:
        updates["preview_image"] = discovered_preview

    prior_discovery = current.get("_asset_discovery")
    if not isinstance(prior_discovery, Mapping) or dict(prior_discovery) != discovery:
        updates["_asset_discovery"] = discovery

    if updates:
        return save_asset_sidecar_fields(asset, updates)
    return current


def preview_file_payload(preview: Path | None) -> dict[str, Any]:
    if preview is None:
        return {
            "preview_path": "",
            "has_preview": False,
            "preview_modified_ns": 0,
            "preview_size_bytes": 0,
            "preview_revision": "",
        }
    try:
        resolved = preview.resolve()
        stat = resolved.stat()
    except OSError:
        return {
            "preview_path": "",
            "has_preview": False,
            "preview_modified_ns": 0,
            "preview_size_bytes": 0,
            "preview_revision": "",
        }
    modified_ns = int(stat.st_mtime_ns)
    size_bytes = int(stat.st_size)
    return {
        "preview_path": str(resolved),
        "has_preview": True,
        "preview_modified_ns": modified_ns,
        "preview_size_bytes": size_bytes,
        "preview_revision": f"{modified_ns:x}-{size_bytes:x}",
    }


def replace_asset_preview(
    asset_path: str | os.PathLike[str],
    *,
    filename: str,
    content: bytes,
) -> tuple[Path, dict[str, Any]]:
    asset = Path(asset_path).expanduser().resolve()
    suffix = Path(filename or "preview.png").suffix.lower()
    if suffix not in _IMAGE_EXTENSIONS:
        raise ValueError("Preview images must be PNG, JPG, JPEG, or WebP files.")
    if not content:
        raise ValueError("The uploaded preview image is empty.")
    target = asset.with_name(f"{asset.stem}.preview{suffix}")
    target.write_bytes(content)
    metadata = save_asset_metadata(asset, {"preview_image": target.name})
    return target, metadata


__all__ = [
    "ASSET_DISCOVERY_SCHEMA_VERSION",
    "discover_asset_companions",
    "load_asset_metadata",
    "normalize_asset_metadata",
    "preview_file_payload",
    "replace_asset_preview",
    "resolve_preview_path",
    "save_asset_metadata",
    "save_asset_sidecar_fields",
    "sidecar_path",
    "synchronize_asset_companions",
]
