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


class LoRACatalogMixin:
    @staticmethod
    def _lora_model_family_from_metadata(metadata: dict[str, Any], keys: list[str]) -> str:
        values = [
            metadata.get("modelspec.architecture"),
            metadata.get("modelspec.description"),
            metadata.get("ss_base_model_version"),
            metadata.get("ss_sd_model_name"),
            metadata.get("ss_v2"),
        ]
        joined = " ".join(str(value or "") for value in values).strip().lower()
        if "sdxl" in joined or "stable-diffusion-xl" in joined or any("lora_te2" in key or "text_encoder_2" in key for key in keys):
            return "sdxl"
        if "sd2" in joined or "2.0" in joined or "2.1" in joined or str(metadata.get("ss_v2") or "").strip().lower() in {"true", "1", "yes"}:
            return "sd2.x"
        if "sd1" in joined or "1.4" in joined or "1.5" in joined:
            return "sd1.x"
        return ""

    @staticmethod
    def _lora_tensor_format(keys: list[str]) -> str:
        lowered = [key.lower() for key in keys]
        if any("hada_w1_a" in key or "hada_w2_a" in key or "lokr_" in key for key in lowered):
            return "LyCORIS"
        if any(key.startswith("lora_unet_") or key.startswith("lora_te_") or key.startswith("lora_te1_") for key in lowered):
            return "Kohya"
        if any(".lora_a.weight" in key or ".lora_b.weight" in key for key in lowered):
            return "Diffusers PEFT"
        if any("lora_down.weight" in key or "lora_up.weight" in key for key in lowered):
            return "LoRA up/down"
        return "Unknown"

    @staticmethod
    def _lora_scan_signature(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }

    @staticmethod
    def _normalize_lora_scan_cache(value: Any) -> dict[str, Any]:
        cache = dict(value) if isinstance(value, dict) else {}
        signature = cache.get("file_signature") if isinstance(cache.get("file_signature"), dict) else {}
        try:
            size_bytes = int(signature.get("size_bytes", 0) or 0)
        except (TypeError, ValueError):
            size_bytes = 0
        try:
            modified_ns = int(signature.get("modified_ns", 0) or 0)
        except (TypeError, ValueError):
            modified_ns = 0
        return {
            "schema_version": int(cache.get("schema_version") or 0),
            "scan_status": str(cache.get("scan_status") or "unknown"),
            "scanned_at": str(cache.get("scanned_at") or ""),
            "file_signature": {
                "path": str(signature.get("path") or ""),
                "size_bytes": size_bytes,
                "modified_ns": modified_ns,
            },
            "sha256": str(cache.get("sha256") or ""),
            "a1111_hash": str(cache.get("a1111_hash") or ""),
            "a1111_short_hash": str(cache.get("a1111_short_hash") or ""),
            "a1111_hash_source": str(cache.get("a1111_hash_source") or ""),
            "a1111_hash_error": str(cache.get("a1111_hash_error") or ""),
            "network_type": str(cache.get("network_type") or "Unknown"),
            "tensor_key_format": str(cache.get("tensor_key_format") or "Unknown"),
            "tensor_key_count": int(cache.get("tensor_key_count") or 0),
            "detected_model_family": str(cache.get("detected_model_family") or ""),
            "activation_text": str(cache.get("activation_text") or ""),
            "activation_text_source": str(cache.get("activation_text_source") or ""),
            "network_dimension": cache.get("network_dimension"),
            "network_alpha": cache.get("network_alpha"),
            "adapter_format": str(cache.get("adapter_format") or ""),
            "adapter_extensions": [str(item) for item in (cache.get("adapter_extensions") or []) if str(item)],
            "target_scopes": [str(item) for item in (cache.get("target_scopes") or []) if str(item)],
            "target_counts": dict(cache.get("target_counts") or {}),
            "runtime_support_state": str(cache.get("runtime_support_state") or ""),
            "runtime_loadable": bool(cache.get("runtime_loadable", False)),
            "support_reason": str(cache.get("support_reason") or ""),
            "loader_id": str(cache.get("loader_id") or ""),
            "adapter_inspection": dict(cache.get("adapter_inspection") or {}),
            "inspection_error": str(cache.get("inspection_error") or ""),
        }

    def _lora_scan_cache_payload(self, path: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        data = metadata if isinstance(metadata, dict) else load_asset_metadata(path)
        cache = self._normalize_lora_scan_cache(data.get("_lora_scan_cache"))
        valid = lora_scan_cache_is_current(
            path,
            cache,
            require_compatibility_hash=True,
        )
        signature = dict(cache.get("file_signature") or {})
        current_signature = self._lora_scan_signature(path)
        legacy_signature_current = bool(
            cache.get("scan_status")
            and int(signature.get("size_bytes") or 0) == int(current_signature.get("size_bytes") or 0)
            and int(signature.get("modified_ns") or 0) == int(current_signature.get("modified_ns") or 0)
        )
        if not valid and not legacy_signature_current:
            return {
                "detected_model_family": "",
                "network_type": "Unknown",
                "tensor_key_format": "Unknown",
                "tensor_key_count": 0,
                "adapter_format": "",
                "adapter_extensions": [],
                "target_scopes": [],
                "target_counts": {},
                "runtime_support_state": "",
                "runtime_loadable": False,
                "support_reason": "",
                "loader_id": "",
                "adapter_inspection": {},
                "activation_text": str(data.get("activation_text") or ""),
                "activation_text_source": "sidecar" if str(data.get("activation_text") or "").strip() else "",
                "scan_status": "unscanned",
                "scanned_at": "",
                "inspection_error": "",
                "scan_cached": False,
            }

        adapter_inspection = dict(cache.get("adapter_inspection") or {})
        adapter_format = str(cache.get("adapter_format") or adapter_inspection.get("adapter_format") or "")
        if not adapter_format:
            legacy_format = str(cache.get("tensor_key_format") or "").strip().lower()
            adapter_format = {
                "kohya": "standard_kohya_lora",
                "diffusers peft": "standard_diffusers_peft_lora",
                "lora up/down": "standard_lora_up_down",
            }.get(legacy_format, "")
        migration_pending = (
            not bool(adapter_inspection)
            or int(cache.get("schema_version") or 0) < LORA_SCAN_CACHE_SCHEMA_VERSION
        )
        support_state = str(cache.get("runtime_support_state") or "")
        runtime_loadable = bool(cache.get("runtime_loadable", False))
        support_reason = str(cache.get("support_reason") or "")
        loader_id = str(cache.get("loader_id") or "")
        if adapter_format and not support_state:
            migrated_record = AdapterInspectionRecord.from_mapping({
                "source_path": str(path),
                "file_signature": cache.get("file_signature") or {},
                "model_family": cache.get("detected_model_family") or "",
                "adapter_format": adapter_format,
                "network_type": cache.get("network_type") or "Unknown",
                "tensor_key_count": cache.get("tensor_key_count") or 0,
                "target_scopes": cache.get("target_scopes") or [],
                "source_rank": cache.get("network_dimension"),
                "source_alpha": cache.get("network_alpha"),
                "inspection_warnings": ["Legacy LoRA scan cache is awaiting bounded LORA-01 target-scope refresh."],
            })
            decision = self._adapter_compatibility.evaluate(migrated_record, active_checkpoint_family="")
            support_state = decision.overall_support_state
            runtime_loadable = decision.runtime_loadable
            support_reason = decision.blocking_reason
            loader_id = decision.loader_id

        return {
            "detected_model_family": str(cache.get("detected_model_family") or ""),
            "network_type": str(cache.get("network_type") or "Unknown"),
            "tensor_key_format": str(cache.get("tensor_key_format") or "Unknown"),
            "tensor_key_count": int(cache.get("tensor_key_count") or 0),
            "network_dimension": cache.get("network_dimension"),
            "network_alpha": cache.get("network_alpha"),
            "adapter_format": adapter_format,
            "adapter_extensions": list(cache.get("adapter_extensions") or adapter_inspection.get("adapter_extensions") or []),
            "target_scopes": list(cache.get("target_scopes") or adapter_inspection.get("target_scopes") or []),
            "target_counts": dict(cache.get("target_counts") or adapter_inspection.get("target_counts") or {}),
            "runtime_support_state": support_state,
            "runtime_loadable": runtime_loadable,
            "support_reason": support_reason,
            "loader_id": loader_id,
            "adapter_inspection": adapter_inspection,
            "activation_text": str(cache.get("activation_text") or ""),
            "activation_text_source": str(cache.get("activation_text_source") or ""),
            "sha256": str(cache.get("sha256") or ""),
            "a1111_hash": str(cache.get("a1111_hash") or ""),
            "a1111_short_hash": str(cache.get("a1111_short_hash") or ""),
            "a1111_hash_source": str(cache.get("a1111_hash_source") or ""),
            "a1111_hash_error": str(cache.get("a1111_hash_error") or ""),
            "inspection_error": str(cache.get("inspection_error") or ""),
            "scan_status": str(cache.get("scan_status") or "cached"),
            "scanned_at": str(cache.get("scanned_at") or ""),
            "scan_cached": bool(valid and not migration_pending),
            "scan_migration_pending": migration_pending,
        }


    @staticmethod
    def _restricted_lora_technical(path: Path, *, extension: str, a1111_hash_error: str = "") -> dict[str, Any]:
        message = "Technical tensor inspection is intentionally restricted for pickle-bearing legacy adapter formats (.pt/.ckpt/.bin/.pth)."
        adapter_format = "inspection_restricted"
        inspection_record = AdapterInspectionRecord(
            source_path=str(path),
            file_signature={
                "path": str(path),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "modified_ns": int(path.stat().st_mtime_ns) if path.exists() else 0,
            },
            adapter_format=adapter_format,
            adapter_format_evidence=(f"restricted_extension:{extension}",),
            inspection_errors=(message,),
        )
        return {
            "sha256": "",
            "a1111_hash": "",
            "a1111_short_hash": "",
            "a1111_hash_source": "",
            "a1111_hash_error": a1111_hash_error,
            "network_type": "Unknown",
            "tensor_key_format": "Restricted",
            "tensor_key_count": 0,
            "safetensors_metadata": {},
            "detected_model_family": "",
            "activation_text": "",
            "activation_text_source": "",
            "network_dimension": None,
            "network_alpha": None,
            "adapter_format": adapter_format,
            "adapter_extensions": [],
            "target_scopes": [],
            "target_counts": {},
            "runtime_support_state": "restricted",
            "runtime_loadable": False,
            "support_reason": message,
            "loader_id": "",
            "adapter_inspection": inspection_record.to_dict(),
            "inspection_error": message,
        }

    def _inspect_lora_record(self, record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        path = Path(record["path"]).resolve()
        metadata = load_asset_metadata(path)
        cached_payload = self._lora_scan_cache_payload(path, metadata=metadata)
        if cached_payload.get("scan_cached"):
            return cached_payload, False

        scan_signature = self._lora_scan_signature(path)
        technical = {
            "sha256": "",
            "a1111_hash": "",
            "a1111_short_hash": "",
            "a1111_hash_source": "",
            "a1111_hash_error": "",
            "network_type": "Unknown",
            "tensor_key_format": "Unknown",
            "tensor_key_count": 0,
            "safetensors_metadata": {},
            "detected_model_family": "",
            "activation_text": "",
            "activation_text_source": "",
            "network_dimension": None,
            "network_alpha": None,
            "adapter_format": "",
            "adapter_extensions": [],
            "target_scopes": [],
            "target_counts": {},
            "runtime_support_state": "",
            "runtime_loadable": False,
            "support_reason": "",
            "loader_id": "",
            "adapter_inspection": {},
            "inspection_error": "",
        }
        scan_status = "scanned"
        if path.suffix.lower() == ".safetensors":
            try:
                report = self._checkpoint_inspector.inspect(str(path))
                lora_analysis = inspect_lora_file(path, sidecar_metadata=metadata)
                metadata_map = dict(lora_analysis.get("safetensors_metadata") or report.safetensors_metadata or {})
                network_module = str(metadata_map.get("ss_network_module") or metadata_map.get("ss_network_type") or "").strip()
                network_type = str(lora_analysis.get("network_type") or network_module or ("LoRA" if report.checkpoint_kind == "lora" else report.checkpoint_kind or "Unknown"))
                inspection_record = AdapterInspectionRecord.from_mapping({
                    **dict(lora_analysis.get("adapter_inspection") or {}),
                    "sha256": report.sha256,
                })
                support = self._adapter_compatibility.evaluate(inspection_record, active_checkpoint_family="")
                technical = {
                    "sha256": report.sha256,
                    "a1111_hash": str(lora_analysis.get("a1111_hash") or ""),
                    "a1111_short_hash": str(lora_analysis.get("a1111_short_hash") or ""),
                    "a1111_hash_source": str(lora_analysis.get("a1111_hash_source") or ""),
                    "a1111_hash_error": str(lora_analysis.get("a1111_hash_error") or ""),
                    "network_type": network_type,
                    "tensor_key_format": str(lora_analysis.get("tensor_key_format") or "Unknown"),
                    "tensor_key_count": int(lora_analysis.get("tensor_key_count") or report.total_keys or 0),
                    "safetensors_metadata": metadata_map,
                    "detected_model_family": inspection_record.model_family,
                    "adapter_format": inspection_record.adapter_format,
                    "adapter_extensions": list(inspection_record.adapter_extensions),
                    "target_scopes": list(inspection_record.target_scopes),
                    "target_counts": dict(inspection_record.target_counts),
                    "runtime_support_state": support.overall_support_state,
                    "runtime_loadable": support.runtime_loadable,
                    "support_reason": support.blocking_reason,
                    "support_warnings": list(support.warnings),
                    "loader_id": support.loader_id,
                    "adapter_inspection": inspection_record.to_dict(),
                    "activation_text": str(lora_analysis.get("activation_text") or metadata.get("activation_text") or ""),
                    "activation_text_source": str(lora_analysis.get("activation_text_source") or ""),
                    "network_dimension": metadata_map.get("ss_network_dim"),
                    "network_alpha": metadata_map.get("ss_network_alpha"),
                    "inspection_error": str(lora_analysis.get("inspection_error") or ""),
                }
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                technical = {
                    **technical,
                    "adapter_format": "invalid",
                    "runtime_support_state": "invalid",
                    "runtime_loadable": False,
                    "support_reason": "Adapter file could not be inspected as a valid Safetensors adapter.",
                    "inspection_error": message,
                }
                scan_status = "error"
        else:
            scan_status = "restricted"
            a1111_hash_error = ""
            try:
                compatibility_hash = compute_lora_compatibility_hash(path)
            except Exception as exc:
                compatibility_hash = {
                    "a1111_hash": "",
                    "a1111_short_hash": "",
                    "a1111_hash_source": "",
                }
                a1111_hash_error = f"{type(exc).__name__}: {exc}"
            technical = {
                **self._restricted_lora_technical(path, extension=path.suffix.lower(), a1111_hash_error=a1111_hash_error),
                **compatibility_hash,
            }

        persisted_cache = {
            "schema_version": LORA_SCAN_CACHE_SCHEMA_VERSION,
            "scan_status": scan_status,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "file_signature": scan_signature,
            "sha256": technical.get("sha256") or "",
            "a1111_hash": technical.get("a1111_hash") or "",
            "a1111_short_hash": technical.get("a1111_short_hash") or "",
            "a1111_hash_source": technical.get("a1111_hash_source") or "",
            "a1111_hash_error": technical.get("a1111_hash_error") or "",
            "network_type": technical.get("network_type") or "Unknown",
            "tensor_key_format": technical.get("tensor_key_format") or "Unknown",
            "tensor_key_count": int(technical.get("tensor_key_count") or 0),
            "detected_model_family": technical.get("detected_model_family") or "",
            "activation_text": technical.get("activation_text") or "",
            "activation_text_source": technical.get("activation_text_source") or "",
            "network_dimension": technical.get("network_dimension"),
            "network_alpha": technical.get("network_alpha"),
            "adapter_format": technical.get("adapter_format") or "",
            "adapter_extensions": list(technical.get("adapter_extensions") or []),
            "target_scopes": list(technical.get("target_scopes") or []),
            "target_counts": dict(technical.get("target_counts") or {}),
            "runtime_support_state": technical.get("runtime_support_state") or "",
            "runtime_loadable": bool(technical.get("runtime_loadable", False)),
            "support_reason": technical.get("support_reason") or "",
            "loader_id": technical.get("loader_id") or "",
            "adapter_inspection": dict(technical.get("adapter_inspection") or {}),
            "inspection_error": technical.get("inspection_error") or "",
        }
        save_asset_sidecar_fields(path, {"_lora_scan_cache": persisted_cache})
        cache_key = (str(path), int(scan_signature["size_bytes"]), int(scan_signature["modified_ns"]))
        self._lora_detail_cache = {
            cache_key: dict(technical),
            **{key: value for key, value in self._lora_detail_cache.items() if key != cache_key},
        }
        payload = {
            **technical,
            "scan_status": scan_status,
            "scanned_at": persisted_cache["scanned_at"],
            "scan_cached": True,
        }
        return payload, True

    def scan_loras(self, *, mode: str = "missing") -> dict[str, Any]:
        normalized_mode = str(mode or "missing").strip().lower()
        if normalized_mode not in {"missing", "all"}:
            raise ValueError(f"Unsupported LoRA scan mode: {mode}")
        scanned = 0
        refreshed = 0
        errors = 0
        unsupported = 0
        restricted = 0
        for record in list(self._loras):
            path = Path(str(record.get("path") or "")).expanduser().resolve()
            if not path.is_file():
                continue
            metadata = load_asset_metadata(path)
            cached = self._lora_scan_cache_payload(path, metadata=metadata)
            needs_scan = normalized_mode == "all" or not cached.get("scan_cached")
            if not needs_scan:
                continue
            payload, changed = self._inspect_lora_record(record)
            merged = self._merge_technical_record("lora", record["asset_id"], {
                **payload,
                "model_family": str(payload.get("detected_model_family") or record.get("model_family") or ""),
                "detected_model_family": str(payload.get("detected_model_family") or ""),
                "activation_text": str(record.get("activation_text") or payload.get("activation_text") or ""),
            })
            if changed:
                scanned += 1
            refreshed += 1
            if str(merged.get("scan_status") or "") == "error":
                errors += 1
            status_token = str(merged.get("scan_status") or "")
            if status_token == "unsupported":
                unsupported += 1
            if status_token == "restricted":
                restricted += 1
        if refreshed:
            self._bump_catalog_revision("lora")
        return {
            "catalog": self.catalog_status("lora")["catalogs"]["lora"],
            "loras": self.asset_list("lora"),
            "scan": {
                "mode": normalized_mode,
                "scanned": scanned,
                "refreshed": refreshed,
                "errors": errors,
                "unsupported": unsupported,
                "restricted": restricted,
            },
        }

    def lora_details(self, asset_id: str, *, inspect_technical: bool = True) -> dict[str, Any]:
        record = self._asset_record("lora", asset_id)
        path = Path(record["path"]).resolve()
        metadata = synchronize_asset_companions(path, load_asset_metadata(path))
        if inspect_technical:
            technical, _ = self._inspect_lora_record(record)
        else:
            technical = self._lora_scan_cache_payload(path, metadata=metadata)
        preview = resolve_preview_path(path, metadata)
        preview_payload = preview_file_payload(preview)
        preview_revision = str(preview_payload.get("preview_revision") or "")
        resolved_family = str(technical.get("detected_model_family") or metadata.get("model_family") or record.get("model_family") or "")
        merged = self._merge_technical_record("lora", asset_id, {
            **technical,
            "model_family": resolved_family,
            "detected_model_family": str(technical.get("detected_model_family") or ""),
            "activation_text": str(record.get("activation_text") or metadata.get("activation_text") or technical.get("activation_text") or ""),
        })
        return self._catalog_entry_payload("lora", {
            **merged,
            "metadata": metadata,
            **preview_payload,
            "preview_url": f"/api/assets/loras/{asset_id}/preview?v={quote(preview_revision, safe='')}" if preview_payload.get("has_preview") else "",
        })
