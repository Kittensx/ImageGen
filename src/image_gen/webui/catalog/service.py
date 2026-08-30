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

from .assets import AssetCatalogMixin
from .civitai import CivitaiCatalogMixin
from .lora import LoRACatalogMixin
from .outputs import OutputCatalogMixin
from .previews import PreviewCatalogMixin


class WebUICatalog(
    AssetCatalogMixin,
    LoRACatalogMixin,
    CivitaiCatalogMixin,
    PreviewCatalogMixin,
    OutputCatalogMixin,
):
    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self._registry = RuntimeRegistrySystem(project_context=context)
        self._models: list[dict[str, Any]] = []
        self._vaes: list[dict[str, Any]] = []
        self._loras: list[dict[str, Any]] = []
        self._textual_inversions: list[dict[str, Any]] = []
        self._asset_indexes: dict[str, dict[str, dict[str, Any]]] = {
            "checkpoint": {},
            "lora": {},
            "vae": {},
            "textual_inversion": {},
        }
        self._catalog_lock = threading.RLock()
        self._catalog_revisions: dict[str, int] = {asset_type: 0 for asset_type in _ASSET_TYPES}
        self._catalog_refreshed_at: dict[str, str] = {asset_type: "" for asset_type in _ASSET_TYPES}
        self._checkpoint_detail_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._lora_detail_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._checkpoint_inspector = CheckpointInspector()
        self._adapter_compatibility = AdapterCompatibilityService()
        self._civitai_client = CivitaiAssetMetadataService(context)
        # Compatibility alias for older extensions/tests; the implementation is generic.
        self._civitai_lora_client = self._civitai_client
        self._output_summary_cache: dict[tuple[str, int, int, int], dict[str, Any]] = {}
        self._lora_root_report: dict[str, Any] = {"roots": [], "diagnostics": [], "summary": {}}
        self.refresh_models()

    def reload_plugins(self) -> None:
        self._registry = RuntimeRegistrySystem(project_context=self.context)

    def invalidate_output_cache(self) -> None:
        self._output_summary_cache.clear()

    def _descriptor_payload(self, kind: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in self._registry.descriptors(kind):
            payload = item.to_dict()
            payload["config_schema"] = normalize_config_schema(payload.get("config_schema") or {}, kind=kind)
            output.append(payload)
        return output

    def plugins(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "samplers": self._descriptor_payload("sampler"),
            "schedulers": self._descriptor_payload("scheduler"),
        }

    def validate_pair(self, sampler: str, scheduler: str):
        """Forward sampler/scheduler compatibility to the canonical runtime registry."""

        return self._registry.validate_pair(sampler, scheduler)

__all__ = ["ASSET_CATALOG_CONTRACT_VERSION", "WebUICatalog"]
