from __future__ import annotations

import html
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from image_gen.systems.asset_hub.contracts import ProviderFile, ProviderModel, ProviderVersion

PROVENANCE_SCHEMA_VERSION = 1

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_plaintext(value: Any, *, limit: int = 12000) -> str:
    text = html.unescape(str(value or ""))
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:limit]


def build_provenance(
    *,
    model: ProviderModel,
    version: ProviderVersion,
    provider_file: ProviderFile,
    installed_path: Path,
    verified_sha256: str,
    asset_kind: str,
    classification: Mapping[str, Any],
    download_job_id: str,
    install_id: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    creator = str(model.creator or "").strip()
    description = str(model.description or version.description or "").strip()[:50000]
    safe_description = sanitize_plaintext(description)
    source = dict(source_metadata or {})
    install_snapshot = source.get("provider_install_snapshot")
    if not isinstance(install_snapshot, Mapping):
        install_snapshot = {
            "provider": provider_file.provider_id,
            "provider_model_id": provider_file.remote_model_id,
            "provider_model_version_id": provider_file.remote_version_id,
            "provider_file_id": provider_file.remote_file_id,
            "title": str(model.name or installed_path.stem),
            "author": str(model.creator or ""),
            "description": sanitize_plaintext(model.description or version.description or ""),
            "tags": list(model.tags),
            "trained_words": list(version.trained_words or provider_file.trained_words),
            "base_model": str(version.base_model or provider_file.base_model or ""),
            "version_name": str(version.name or ""),
            "source_page_url": str(model.source_page_url or provider_file.source_page_url or ""),
            "captured_at_utc": now,
        }
    return {
        "metadata_schema_version": PROVENANCE_SCHEMA_VERSION,
        "asset_id": str(source.get("asset_id") or uuid.uuid4()),
        "install_id": str(install_id or ""),
        "asset_kind": asset_kind,
        "display_name": str(model.name or installed_path.stem),
        "provider": provider_file.provider_id,
        "provider_asset_type": model.provider_type,
        "provider_model_id": provider_file.remote_model_id,
        "provider_model_version_id": provider_file.remote_version_id,
        "provider_file_id": provider_file.remote_file_id,
        "provider_creator_name": creator,
        "provider_creator_username": creator,
        "source_page_url": str(model.source_page_url or provider_file.source_page_url or ""),
        "source_platform_name": "Civitai" if provider_file.provider_id == "civitai" else provider_file.provider_id,
        "source_platform_domain": "civitai.com" if provider_file.provider_id == "civitai" else "",
        # Phase 02 security boundary: signed/direct delivery URLs are deliberately not persisted.
        "source_download_url": "",
        "source_download_url_persisted": False,
        "downloaded_at_utc": str(source.get("downloaded_at_utc") or now),
        "download_job_id": download_job_id,
        "downloaded_via": str(source.get("downloaded_via") or "asset_hub_ui"),
        "author_name": creator,
        "author_username": creator,
        "title": str(model.name or installed_path.stem),
        "short_description": safe_description[:320],
        "description_markdown": description,
        "description_plaintext": safe_description,
        "tags": list(model.tags),
        "trained_words": list(version.trained_words or provider_file.trained_words),
        "trigger_words": list(version.trained_words or provider_file.trained_words),
        "base_model": str(version.base_model or provider_file.base_model or ""),
        "model_family": str(version.architecture or provider_file.architecture or ""),
        "version_name": str(version.name or ""),
        "version_description": str(version.description or "")[:50000],
        "nsfw_level": "provider_reported" if model.nsfw else "",
        "filename": installed_path.name,
        "original_filename": provider_file.file_name,
        "file_size_bytes": installed_path.stat().st_size if installed_path.exists() else int(provider_file.size_bytes or 0),
        "sha256": verified_sha256,
        "hashes": provider_file.hash_map(),
        "local_path": str(installed_path),
        "installed_path": str(installed_path),
        "installed_at_utc": now,
        "quarantined": False,
        "install_status": "installed",
        "imagegen_classification": dict(classification),
        "imagegen_loader_family": str(classification.get("loader_family") or classification.get("architecture") or ""),
        "imagegen_scan_version": str(classification.get("scan_version") or classification.get("classifier_version") or "asset-hub-phase03-v1"),
        "imagegen_compatibility": classification.get("compatibility") or classification.get("architecture") or "",
        "imagegen_notes": str(classification.get("notes") or ""),
        "preview_images": [item.to_dict() for item in version.previews[:16]],
        "info_card_summary": safe_description[:240],
        "metadata_created_at_utc": now,
        "metadata_updated_at_utc": now,
        "last_verified_at_utc": now,
        "provider_sync_status": "synced",
        "provider_sync_error": "",
        "provider_install_snapshot": dict(install_snapshot),
        "provider_scan_results": provider_file.scan.to_dict(),
        "expected_hashes": provider_file.hash_map(),
        "verified_sha256": verified_sha256,
        "classification_result": dict(classification),
    }
