from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from safetensors import safe_open

from image_gen.systems.registry import RuntimeRegistrySystem
from image_gen.runtime.lora_inspector import (
    LORA_SCAN_CACHE_SCHEMA_VERSION,
    canonical_model_family,
    compute_lora_compatibility_hash,
    inspect_lora_file,
    lora_scan_cache_is_current,
)
from image_gen.runtime.adapters.compatibility import AdapterCompatibilityService
from image_gen.runtime.adapters.contracts import AdapterInspectionRecord
from image_gen.webui.civitai_asset_metadata import (
    CivitaiAssetMetadataService,
    CivitaiCredentialError,
    CivitaiMetadataError,
    read_civitai_api_key,
)
from image_gen.webui.asset_metadata import (
    load_asset_metadata,
    preview_file_payload,
    replace_asset_preview,
    resolve_preview_path,
    save_asset_metadata,
    save_asset_sidecar_fields,
    sidecar_path,
    synchronize_asset_companions,
)
from image_gen.webui.image_refs import encode_external_image_ref, is_within_root
from image_gen.webui.output_details import load_image_file_details, load_output_details
from image_gen.webui.schema_utils import normalize_config_schema
from modules.checkpoint_inspector import CheckpointInspector, detect_model_name
from modules.project_context import ProjectContext
from modules.txt2img.model_selector import MODEL_EXTENSIONS

from .contracts import (
    ASSET_CATALOG_CONTRACT_VERSION,
    _ASSET_PLURAL_KEYS,
    _ASSET_TYPES,
    _IMAGE_EXTENSIONS,
    _LORA_EXTENSIONS,
    _TEXTUAL_INVERSION_EXTENSIONS,
)


class CivitaiCatalogMixin:
    def _known_civitai_hashes(self, asset_type: str, record: dict[str, Any]) -> list[str]:
        values: list[str] = []
        if asset_type == "checkpoint":
            details = self.checkpoint_details(str(record.get("asset_id") or ""))
            values.append(str(details.get("sha256") or ""))
        elif asset_type == "lora":
            technical, _ = self._inspect_lora_record(record)
            values.extend([
                str(technical.get("sha256") or ""),
                str(technical.get("a1111_hash") or ""),
            ])
        else:
            values.append(str(record.get("sha256") or ""))
        return [value for value in values if value]

    @staticmethod
    def _civitai_enrichment_is_complete(
        path: Path,
        metadata: dict[str, Any],
        lookup: dict[str, Any],
    ) -> bool:
        if str(lookup.get("status") or "").strip().lower() != "matched":
            return False

        # A matched sidecar can predate preview downloading, or a previous
        # download can have failed. Treat those records as incomplete so the
        # normal "missing" refresh can repair the browser card without forcing
        # the user to refresh each asset individually.
        if resolve_preview_path(path, metadata) is not None:
            return True
        if str(lookup.get("preview_image_download_error") or "").strip():
            return False
        if str(lookup.get("preview_image_path") or "").strip():
            return False

        remote_preview = str(lookup.get("image_url") or "").strip()
        if not remote_preview:
            images = lookup.get("images")
            if isinstance(images, list):
                remote_preview = next((
                    str(image.get("url") or "").strip()
                    for image in images
                    if isinstance(image, dict) and str(image.get("url") or "").strip()
                ), "")
        return not bool(remote_preview)

    def enrich_asset_from_civitai(
        self,
        asset_type: str,
        asset_id: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if asset_type not in _ASSET_TYPES:
            raise KeyError(asset_type)
        record = self._asset_record(asset_type, asset_id)
        path = Path(record["path"]).resolve()
        enrichment = self._civitai_client.enrich_local_asset(
            path,
            asset_type=asset_type,
            hashes=self._known_civitai_hashes(asset_type, record),
            overwrite=overwrite,
        )
        refreshed = self._catalog_entry(record, asset_type=asset_type)
        self._replace_catalog_record(asset_type, refreshed)
        self._bump_catalog_revision(asset_type)
        details = self._details_for_asset_type(asset_type, asset_id)
        details["civitai_lookup"] = dict(enrichment.get("civitai_lookup") or {})
        return details

    def enrich_assets_from_civitai(self, asset_type: str, *, mode: str = "missing") -> dict[str, Any]:
        if asset_type not in _ASSET_TYPES:
            raise KeyError(asset_type)
        normalized_mode = str(mode or "missing").strip().lower()
        if normalized_mode not in {"missing", "all"}:
            raise ValueError(f"Unsupported CivitAI metadata mode: {mode}")

        # Validate once so bulk requests fail clearly rather than repeating the
        # same credential error for every local file.
        read_civitai_api_key(self.context)

        matched = 0
        activation_text_found = 0
        manual_search_required = 0
        previews_downloaded = 0
        preview_download_errors = 0
        skipped = 0
        errors: list[dict[str, str]] = []
        records = list(getattr(self, self._collection_attribute(asset_type)))
        for record in records:
            path = Path(str(record.get("path") or "")).expanduser().resolve()
            if not path.is_file():
                continue
            metadata = synchronize_asset_companions(path, load_asset_metadata(path))
            lookup = metadata.get("_civitai_lookup")
            lookup = dict(lookup) if isinstance(lookup, dict) else {}
            if normalized_mode == "missing" and self._civitai_enrichment_is_complete(path, metadata, lookup):
                skipped += 1
                continue
            try:
                details = self.enrich_asset_from_civitai(
                    asset_type,
                    str(record.get("asset_id") or ""),
                    overwrite=False,
                )
                civitai = details.get("civitai_lookup")
                civitai = dict(civitai) if isinstance(civitai, dict) else {}
                matched += 1
                if civitai.get("activation_text"):
                    activation_text_found += 1
                if civitai.get("manual_activation_text_search_required"):
                    manual_search_required += 1
                if civitai.get("preview_image_downloaded"):
                    previews_downloaded += 1
                if civitai.get("preview_image_download_error"):
                    preview_download_errors += 1
            except CivitaiCredentialError:
                raise
            except (CivitaiMetadataError, OSError, ValueError) as exc:
                errors.append({
                    "asset_id": str(record.get("asset_id") or ""),
                    "filename": str(record.get("filename") or path.name),
                    "error": str(exc),
                })

        return {
            "catalog": self.catalog_status(asset_type)["catalogs"][asset_type],
            _ASSET_PLURAL_KEYS[asset_type]: self.asset_list(asset_type),
            "civitai": {
                "asset_type": asset_type,
                "mode": normalized_mode,
                "matched": matched,
                "activation_text_found": activation_text_found,
                "manual_search_required": manual_search_required,
                "previews_downloaded": previews_downloaded,
                "preview_download_errors": preview_download_errors,
                "skipped": skipped,
                "errors": errors,
            },
        }

    def enrich_lora_from_civitai(self, asset_id: str, *, overwrite: bool = False) -> dict[str, Any]:
        return self.enrich_asset_from_civitai("lora", asset_id, overwrite=overwrite)

    def enrich_loras_from_civitai(self, *, mode: str = "missing") -> dict[str, Any]:
        return self.enrich_assets_from_civitai("lora", mode=mode)
