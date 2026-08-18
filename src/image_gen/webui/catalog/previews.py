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


class PreviewCatalogMixin:
    @staticmethod
    def _canonical_model_family(value: Any) -> str:
        return canonical_model_family(value)

    def asset_preview_candidates(self, asset_type: str, asset_id: str, *, limit: int = 48) -> list[dict[str, Any]]:
        record = self._asset_record(asset_type, asset_id)
        metadata = load_asset_metadata(record["path"])
        preview_items = self.recent_outputs(limit=max(int(limit or 48), 1), include_subfolders=True, require_metadata_for_external=False)
        target_name = str(record.get("name") or Path(record["path"]).stem or "").casefold()
        target_hash = str(record.get("sha256") or "").casefold()
        target_activation = str(record.get("activation_text") or metadata.get("activation_text") or "").strip().casefold()
        target_family = self._canonical_model_family(record.get("model_family") or metadata.get("model_family") or metadata.get("base_model"))
        ranked: list[dict[str, Any]] = []
        for item in preview_items:
            score = 0
            reasons: list[str] = []
            output_family = self._canonical_model_family(item.get("model_name") or item.get("model_path") or "")
            if asset_type == "checkpoint":
                requested_path = str(item.get("model_path") or "").casefold()
                requested_name = str(item.get("model_name") or Path(str(item.get("model_path") or "")).stem).casefold()
                if target_hash and target_hash == str(item.get("model_hash") or "").casefold():
                    score += 100
                    reasons.append("Exact checkpoint hash match")
                if requested_path and requested_path == str(record.get("path") or "").casefold():
                    score += 90
                    reasons.append("Exact checkpoint path match")
                elif requested_name and requested_name == target_name:
                    score += 70
                    reasons.append("Exact checkpoint name match")
            elif asset_type == "lora":
                loras = item.get("loras") or []
                if isinstance(loras, list):
                    labels = [str(value).casefold() for value in loras]
                    if target_name and target_name in labels:
                        score += 70
                        reasons.append("Exact LoRA name match")
                try:
                    details = load_output_details(self.context, item["output_id"])
                    image_loras = list(details.image.get("loras") or [])
                except Exception:
                    image_loras = []
                for detail in image_loras:
                    if not isinstance(detail, dict):
                        continue
                    extra = dict(detail.get("extra") or {})
                    if target_hash and target_hash == str(detail.get("resolved_hash") or detail.get("requested_hash") or "").casefold():
                        score += 100
                        reasons.append("Exact LoRA hash match")
                    resolved_path = str(detail.get("resolved_path") or detail.get("requested_path") or "").casefold()
                    if resolved_path and resolved_path == str(record.get("path") or "").casefold():
                        score += 90
                        reasons.append("Exact LoRA path match")
                    name = str(detail.get("requested_display_name") or detail.get("resolved_display_name") or "").casefold()
                    if name and name == target_name:
                        score += 80
                        reasons.append("Exact LoRA name match")
                    activation = str(extra.get("activation_text") or "").strip().casefold()
                    if target_activation and activation and activation == target_activation:
                        score += 25
                        reasons.append("Matching activation text")
            if target_family and output_family and output_family == target_family:
                score += 20
                reasons.append("Matching model family")
            ranked.append({**item, "match_score": score, "match_reasons": reasons})
        ranked.sort(key=lambda item: (int(item.get("match_score") or 0), int(item.get("modified_ns") or 0)), reverse=True)
        return ranked[: max(int(limit or 48), 1)]

    def replace_asset_preview_from_output(self, asset_type: str, asset_id: str, output_id: str) -> dict[str, Any]:
        self._asset_record(asset_type, asset_id)
        try:
            details = load_output_details(self.context, output_id)
            image_path = Path(details.image_path).resolve()
        except Exception as exc:
            raise ValueError(f"Recent output could not be loaded: {exc}") from exc
        if not image_path.is_file():
            raise ValueError("The chosen recent output image no longer exists on disk.")
        return self.replace_asset_preview(asset_type, asset_id, filename=image_path.name, content=image_path.read_bytes())

    def asset_preview_path(self, asset_type: str, asset_id: str) -> Path:
        record = self._asset_record(asset_type, asset_id)
        metadata = load_asset_metadata(record["path"])
        if asset_type == "lora":
            metadata = synchronize_asset_companions(record["path"], metadata)
        preview = resolve_preview_path(record["path"], metadata)
        if preview is None:
            raise FileNotFoundError("No preview image is configured for this asset.")
        return preview

    def replace_asset_preview(
        self,
        asset_type: str,
        asset_id: str,
        *,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        record = self._asset_record(asset_type, asset_id)
        replace_asset_preview(record["path"], filename=filename, content=content)
        refreshed = self._catalog_entry(record, asset_type=asset_type)
        self._replace_catalog_record(asset_type, refreshed)
        self._bump_catalog_revision(asset_type)
        if asset_type == "checkpoint":
            return self.checkpoint_details(asset_id)
        if asset_type == "lora":
            return self.lora_details(asset_id)
        if asset_type == "vae":
            return self.vae_details(asset_id)
        return self._catalog_entry_payload(asset_type, refreshed)
