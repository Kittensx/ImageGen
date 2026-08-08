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

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt"}
_TEXTUAL_INVERSION_EXTENSIONS = {".safetensors", ".pt", ".bin"}
ASSET_CATALOG_CONTRACT_VERSION = "image-gen-asset-catalog-v1"
_ASSET_TYPES = ("checkpoint", "lora", "vae", "textual_inversion")
_ASSET_PLURAL_KEYS = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "vae": "vaes",
    "textual_inversion": "textual_inversions",
}


class WebUICatalog:
    """Manual-refresh catalogs for plugins, models, VAEs, and recent outputs."""

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
        self._civitai_client = CivitaiAssetMetadataService(context)
        # Compatibility alias for older extensions/tests; the implementation is generic.
        self._civitai_lora_client = self._civitai_client
        self._output_summary_cache: dict[tuple[str, int], dict[str, Any]] = {}
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

    def _additional_roots(self) -> list[Path]:
        model_library = self.context.config.get("model_library") or {}
        raw_roots = model_library.get("additional_scan_roots") or []
        output: list[Path] = []
        for item in raw_roots:
            if isinstance(item, str):
                output.append(self.context.resolve_project_path(item))
            elif isinstance(item, dict) and item.get("path"):
                output.append(self.context.resolve_project_path(str(item["path"])))
        return output

    @staticmethod
    def _scan_files(roots: Iterable[Path], extensions: set[str]) -> list[dict[str, Any]]:
        candidates: list[tuple[Path, int, int]] = []
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in extensions:
                    continue
                try:
                    resolved = path.resolve()
                    stat = resolved.stat()
                except OSError:
                    continue
                token = str(resolved).casefold()
                if token in seen:
                    continue
                seen.add(token)
                candidates.append((resolved, stat.st_size, stat.st_mtime_ns))

        output: list[dict[str, Any]] = []
        for path, size, mtime_ns in candidates:
            try:
                stable = path.stat()
            except OSError:
                continue
            if stable.st_size != size or stable.st_mtime_ns != mtime_ns:
                continue
            output.append(
                {
                    "name": path.stem,
                    "path": str(path),
                    "extension": path.suffix.lower(),
                    "size_mb": round(size / (1024 * 1024), 2),
                    "modified_ns": mtime_ns,
                }
            )
        return sorted(output, key=lambda item: (item["name"].casefold(), item["path"].casefold()))

    @staticmethod
    def _embedded_safetensors_name(path: Path, asset_type: str) -> str:
        """Read a conservative display title from the safetensors header only.

        This never hashes the file or materializes tensors. Checkpoint/VAE names
        use ModelSpec title/name fields only. LoRAs/textual inversions may also
        expose ``ss_output_name`` as an informational display title, while their
        technical identity remains the filename stem.
        """
        if path.suffix.lower() != ".safetensors":
            return ""
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
        except Exception:
            return ""
        if asset_type in {"checkpoint", "vae"}:
            value, source = detect_model_name(metadata, "")
            return value if source and source != "filename" else ""
        for key in ("modelspec.title", "modelspec.name", "ss_output_name"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _asset_id(path: str | os.PathLike[str], asset_type: str) -> str:
        resolved = os.path.normcase(str(Path(path).expanduser().resolve()))
        return hashlib.sha256(f"{asset_type}|{resolved}".encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _iso_modified(modified_ns: int) -> str:
        try:
            return datetime.fromtimestamp(int(modified_ns) / 1_000_000_000, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, TypeError, ValueError):
            return ""

    def _catalog_entry(self, item: dict[str, Any], *, asset_type: str) -> dict[str, Any]:
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        metadata = synchronize_asset_companions(path, load_asset_metadata(path))
        preview = resolve_preview_path(path, metadata)
        preview_payload = preview_file_payload(preview)
        preview_revision = str(preview_payload.get("preview_revision") or "")
        preview_url = ""
        if preview_payload.get("has_preview"):
            plural = {
                "checkpoint": "checkpoints",
                "lora": "loras",
                "vae": "vaes",
                "textual_inversion": "textual-inversions",
            }.get(asset_type, f"{asset_type}s")
            preview_url = f"/api/assets/{plural}/{self._asset_id(path, asset_type)}/preview?v={quote(preview_revision, safe='')}"
        canonical_name = str(item.get("name") or path.stem).strip() or path.stem
        embedded_name = str(item.get("embedded_name") or "").strip()
        lookup = metadata.get("_civitai_lookup")
        lookup = dict(lookup) if isinstance(lookup, dict) else {}
        nickname_explicit = "nickname" in metadata
        nickname = str(metadata.get("nickname") or "").strip()
        if not nickname_explicit and not nickname:
            legacy_display_name = str(metadata.get("display_name") or "").strip()
            civitai_names = {
                str(lookup.get("model_version_name") or "").strip().casefold(),
                str(lookup.get("model_name") or "").strip().casefold(),
            }
            civitai_names.discard("")
            civitai_applied = bool(lookup.get("display_name_applied")) or "display_name" in set(lookup.get("applied_fields") or [])
            # Older CivitAI enrichment wrote provider model/version names into
            # display_name. Ignore that historical value when it still matches
            # CivitAI provenance. Genuine user-created legacy display names are
            # preserved as nicknames for backward compatibility.
            if legacy_display_name and not (civitai_applied and legacy_display_name.casefold() in civitai_names):
                nickname = legacy_display_name
        display_name = nickname or embedded_name or canonical_name
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
        try:
            preferred_weight = float(metadata.get("preferred_weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            preferred_weight = 1.0
        entry = {
            **item,
            "asset_id": self._asset_id(path, asset_type),
            "asset_type": asset_type,
            # ``name`` is the stable local technical identity. Display labels may
            # use a user nickname or an embedded ModelSpec title, but neither is
            # allowed to change the selected file path or LoRA prompt token.
            "name": canonical_name,
            "nickname": nickname,
            "embedded_name": embedded_name,
            "display_name": display_name,
            "filename": path.name,
            "size_bytes": int(path.stat().st_size) if path.is_file() else int(float(item.get("size_mb") or 0) * 1024 * 1024),
            "modified_iso": self._iso_modified(int(item.get("modified_ns") or 0)),
            "metadata_path": str(sidecar_path(path)),
            **preview_payload,
            "preview_url": preview_url,
            "source_url": str(metadata.get("source_url") or ""),
            "description": str(metadata.get("description") or ""),
            "notes": str(metadata.get("notes") or ""),
            "tags": [str(value) for value in tags if str(value).strip()],
            "category": str(metadata.get("category") or ""),
            "favorite": bool(metadata.get("favorite", False)),
            "model_family": str(metadata.get("model_family") or ""),
            "architecture": str(metadata.get("architecture") or ""),
            "prediction_type": str(metadata.get("prediction_type") or ""),
            "conditioning_dimension": metadata.get("conditioning_dimension"),
            "activation_text": str(metadata.get("activation_text") or ""),
            "preferred_weight": max(-4.0, min(4.0, preferred_weight)),
        }
        entry.update({
            "civitai_lookup": lookup,
            "civitai_model_id": lookup.get("model_id"),
            "civitai_model_version_id": lookup.get("model_version_id"),
            "civitai_model_name": str(lookup.get("model_name") or ""),
            "civitai_model_version_name": str(lookup.get("model_version_name") or ""),
            "civitai_model_type": str(lookup.get("model_type") or ""),
            "civitai_creator": str(lookup.get("creator") or ""),
            "civitai_base_model": str(lookup.get("base_model") or ""),
            "civitai_stats": dict(lookup.get("stats") or {}) if isinstance(lookup.get("stats"), dict) else {},
        })
        if asset_type == "lora":
            entry.update(self._lora_scan_cache_payload(path, metadata=metadata))
        self._asset_indexes.setdefault(asset_type, {})[entry["asset_id"]] = entry
        return entry

    def _catalog_entries(self, items: list[dict[str, Any]], *, asset_type: str) -> list[dict[str, Any]]:
        self._asset_indexes[asset_type] = {}
        return [self._catalog_entry(item, asset_type=asset_type) for item in items]

    @staticmethod
    def _catalog_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _collection_attribute(asset_type: str) -> str:
        return {
            "checkpoint": "_models",
            "lora": "_loras",
            "vae": "_vaes",
            "textual_inversion": "_textual_inversions",
        }[asset_type]

    def _asset_scan_config(self, asset_type: str) -> tuple[list[Path], set[str]]:
        if asset_type == "checkpoint":
            return [self.context.checkpoints_dir, *self._additional_roots()], set(MODEL_EXTENSIONS)
        if asset_type == "lora":
            return [self.context.lora_dir], set(_LORA_EXTENSIONS)
        if asset_type == "vae":
            return [self.context.vae_dir], set(MODEL_EXTENSIONS)
        if asset_type == "textual_inversion":
            return [self.context.embeddings_dir], set(_TEXTUAL_INVERSION_EXTENSIONS)
        raise KeyError(asset_type)

    def _bump_catalog_revision(self, asset_type: str) -> None:
        with self._catalog_lock:
            self._catalog_revisions[asset_type] = int(self._catalog_revisions.get(asset_type, 0)) + 1
            self._catalog_refreshed_at[asset_type] = self._catalog_timestamp()

    def _catalog_entry_payload(self, asset_type: str, item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item)
        payload["catalog_contract_version"] = ASSET_CATALOG_CONTRACT_VERSION
        payload["catalog_revision"] = int(self._catalog_revisions.get(asset_type, 0))
        payload["catalog_refreshed_at"] = str(self._catalog_refreshed_at.get(asset_type, ""))
        if asset_type == "lora":
            payload["compatible_model_family"] = str(
                payload.get("model_family") or payload.get("detected_model_family") or ""
            )
        return payload

    def catalog_status(self, asset_type: str | None = None) -> dict[str, Any]:
        requested = [asset_type] if asset_type else list(_ASSET_TYPES)
        catalogs: dict[str, dict[str, Any]] = {}
        with self._catalog_lock:
            for kind in requested:
                if kind not in _ASSET_TYPES:
                    raise KeyError(kind)
                records = list(getattr(self, self._collection_attribute(kind)))
                catalogs[kind] = {
                    "asset_type": kind,
                    "plural_key": _ASSET_PLURAL_KEYS[kind],
                    "revision": int(self._catalog_revisions.get(kind, 0)),
                    "refreshed_at": str(self._catalog_refreshed_at.get(kind, "")),
                    "count": len(records),
                }
        return {
            "contract_version": ASSET_CATALOG_CONTRACT_VERSION,
            "catalogs": catalogs,
        }

    def refresh_asset_type(self, asset_type: str) -> dict[str, Any]:
        roots, extensions = self._asset_scan_config(asset_type)
        items = self._scan_files(roots, extensions)
        for item in items:
            path = Path(str(item.get("path") or "")).expanduser().resolve()
            item["embedded_name"] = self._embedded_safetensors_name(path, asset_type)
        with self._catalog_lock:
            self._bump_catalog_revision(asset_type)
            records = self._catalog_entries(items, asset_type=asset_type)
            setattr(self, self._collection_attribute(asset_type), records)
        return self.asset_payload(asset_type)

    def asset_payload(self, asset_type: str) -> dict[str, Any]:
        if asset_type not in _ASSET_TYPES:
            raise KeyError(asset_type)
        return {
            "catalog": self.catalog_status(asset_type)["catalogs"][asset_type],
            _ASSET_PLURAL_KEYS[asset_type]: self.asset_list(asset_type),
        }

    def catalog_payload(self) -> dict[str, Any]:
        return {
            **self.catalog_status(),
            "assets": {
                kind: self.asset_list(kind)
                for kind in _ASSET_TYPES
            },
        }

    def refresh_models(self) -> dict[str, Any]:
        for asset_type in _ASSET_TYPES:
            self.refresh_asset_type(asset_type)
        return self.model_payload()

    def model_payload(self) -> dict[str, Any]:
        return {
            "models": self.asset_list("checkpoint"),
            "vaes": self.asset_list("vae"),
            "loras": self.asset_list("lora"),
            "textual_inversions": self.asset_list("textual_inversion"),
            "asset_catalog": self.catalog_status(),
        }

    def asset_record(self, asset_type: str, asset_id: str) -> dict[str, Any]:
        return self._catalog_entry_payload(asset_type, self._asset_record(asset_type, asset_id))

    def _asset_record(self, asset_type: str, asset_id: str) -> dict[str, Any]:
        record = self._asset_indexes.get(asset_type, {}).get(str(asset_id or ""))
        if record is None:
            raise KeyError(f"Unknown {asset_type} asset: {asset_id}")
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Asset file no longer exists: {path}")
        return dict(record)

    def checkpoint_details(self, asset_id: str) -> dict[str, Any]:
        record = self._asset_record("checkpoint", asset_id)
        path = Path(record["path"]).resolve()
        stat = path.stat()
        cache_key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        technical = self._checkpoint_detail_cache.get(cache_key)
        if technical is None:
            technical = {}
            if path.suffix.lower() == ".safetensors":
                try:
                    report = self._checkpoint_inspector.inspect(str(path))
                    technical = {
                        "sha256": report.sha256,
                        "architecture": report.architecture,
                        "prediction_type": report.prediction_type,
                        "conditioning_dimension": report.model_dimension,
                        "architecture_summary": report.architecture_summary,
                        "checkpoint_kind": report.checkpoint_kind,
                        "tensor_key_count": report.total_keys,
                        "safetensors_metadata": dict(report.safetensors_metadata),
                    }
                except Exception as exc:
                    technical = {"inspection_error": f"{type(exc).__name__}: {exc}"}
            self._checkpoint_detail_cache = {cache_key: dict(technical)}
        metadata = load_asset_metadata(path)
        preview = resolve_preview_path(path, metadata)
        preview_payload = preview_file_payload(preview)
        preview_revision = str(preview_payload.get("preview_revision") or "")
        merged = self._merge_technical_record("checkpoint", asset_id, technical)
        return self._catalog_entry_payload("checkpoint", {
            **merged,
            "metadata": metadata,
            **preview_payload,
            "preview_url": f"/api/assets/checkpoints/{asset_id}/preview?v={quote(preview_revision, safe='')}" if preview_payload.get("has_preview") else "",
        })

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
        if not valid:
            return {
                "detected_model_family": "",
                "network_type": "Unknown",
                "tensor_key_format": "Unknown",
                "tensor_key_count": 0,
                "activation_text": str(data.get("activation_text") or ""),
                "activation_text_source": "sidecar" if str(data.get("activation_text") or "").strip() else "",
                "scan_status": "unscanned",
                "scanned_at": "",
                "inspection_error": "",
                "scan_cached": False,
            }
        return {
            "detected_model_family": str(cache.get("detected_model_family") or ""),
            "network_type": str(cache.get("network_type") or "Unknown"),
            "tensor_key_format": str(cache.get("tensor_key_format") or "Unknown"),
            "tensor_key_count": int(cache.get("tensor_key_count") or 0),
            "network_dimension": cache.get("network_dimension"),
            "network_alpha": cache.get("network_alpha"),
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
            "scan_cached": True,
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
            "inspection_error": "",
        }
        scan_status = "scanned"
        if path.suffix.lower() == ".safetensors":
            try:
                report = self._checkpoint_inspector.inspect(str(path))
                lora_analysis = inspect_lora_file(path, sidecar_metadata=metadata)
                metadata_map = dict(lora_analysis.get("safetensors_metadata") or report.safetensors_metadata or {})
                network_module = str(metadata_map.get("ss_network_module") or metadata_map.get("ss_network_type") or "").strip()
                network_type = network_module or ("LoRA" if report.checkpoint_kind == "lora" else report.checkpoint_kind or "Unknown")
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
                    "detected_model_family": str(lora_analysis.get("detected_model_family") or ""),
                    "activation_text": str(lora_analysis.get("activation_text") or metadata.get("activation_text") or ""),
                    "activation_text_source": str(lora_analysis.get("activation_text_source") or ""),
                    "network_dimension": metadata_map.get("ss_network_dim"),
                    "network_alpha": metadata_map.get("ss_network_alpha"),
                    "inspection_error": str(lora_analysis.get("inspection_error") or ""),
                }
            except Exception as exc:
                technical = {**technical, "inspection_error": f"{type(exc).__name__}: {exc}"}
                scan_status = "error"
        else:
            scan_status = "unsupported"
            try:
                compatibility_hash = compute_lora_compatibility_hash(path)
            except Exception as exc:
                compatibility_hash = {
                    "a1111_hash": "",
                    "a1111_short_hash": "",
                    "a1111_hash_source": "",
                }
                technical["a1111_hash_error"] = f"{type(exc).__name__}: {exc}"
            technical = {
                **technical,
                **compatibility_hash,
                "inspection_error": "Technical tensor inspection is currently available only for .safetensors LoRA files.",
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
                "model_family": str(record.get("model_family") or payload.get("detected_model_family") or ""),
                "detected_model_family": str(payload.get("detected_model_family") or ""),
                "activation_text": str(record.get("activation_text") or payload.get("activation_text") or ""),
            })
            if changed:
                scanned += 1
            refreshed += 1
            if str(merged.get("scan_status") or "") == "error":
                errors += 1
            if str(merged.get("scan_status") or "") == "unsupported":
                unsupported += 1
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
            },
        }

    def lora_details(self, asset_id: str) -> dict[str, Any]:
        record = self._asset_record("lora", asset_id)
        path = Path(record["path"]).resolve()
        technical, _ = self._inspect_lora_record(record)
        metadata = synchronize_asset_companions(path, load_asset_metadata(path))
        preview = resolve_preview_path(path, metadata)
        preview_payload = preview_file_payload(preview)
        preview_revision = str(preview_payload.get("preview_revision") or "")
        resolved_family = str(metadata.get("model_family") or record.get("model_family") or technical.get("detected_model_family") or "")
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

    def vae_details(self, asset_id: str) -> dict[str, Any]:
        record = self._asset_record("vae", asset_id)
        path = Path(record["path"]).resolve()
        metadata = synchronize_asset_companions(path, load_asset_metadata(path))
        preview = resolve_preview_path(path, metadata)
        preview_payload = preview_file_payload(preview)
        preview_revision = str(preview_payload.get("preview_revision") or "")
        refreshed = self._catalog_entry(record, asset_type="vae")
        self._replace_catalog_record("vae", refreshed)
        return self._catalog_entry_payload("vae", {
            **refreshed,
            "metadata": metadata,
            **preview_payload,
            "preview_url": f"/api/assets/vaes/{asset_id}/preview?v={quote(preview_revision, safe='')}" if preview_payload.get("has_preview") else "",
        })

    def _details_for_asset_type(self, asset_type: str, asset_id: str) -> dict[str, Any]:
        if asset_type == "checkpoint":
            return self.checkpoint_details(asset_id)
        if asset_type == "lora":
            return self.lora_details(asset_id)
        if asset_type == "vae":
            return self.vae_details(asset_id)
        return self.asset_record(asset_type, asset_id)

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
            if normalized_mode == "missing" and str(lookup.get("status") or "").strip().lower() == "matched":
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

    # Backwards-compatible LoRA methods now delegate to the generic service.
    def enrich_lora_from_civitai(self, asset_id: str, *, overwrite: bool = False) -> dict[str, Any]:
        return self.enrich_asset_from_civitai("lora", asset_id, overwrite=overwrite)

    def enrich_loras_from_civitai(self, *, mode: str = "missing") -> dict[str, Any]:
        return self.enrich_assets_from_civitai("lora", mode=mode)

    def asset_list(self, asset_type: str) -> list[dict[str, Any]]:
        mapping = {
            "checkpoint": self._models,
            "lora": self._loras,
            "vae": self._vaes,
            "textual_inversion": self._textual_inversions,
        }
        if asset_type not in mapping:
            raise KeyError(asset_type)
        return [self._catalog_entry_payload(asset_type, item) for item in mapping[asset_type]]



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

    def _replace_catalog_record(self, asset_type: str, refreshed: dict[str, Any]) -> None:
        attribute = self._collection_attribute(asset_type)
        records = list(getattr(self, attribute))
        records = [refreshed if item.get("asset_id") == refreshed.get("asset_id") else item for item in records]
        setattr(self, attribute, records)
        self._asset_indexes.setdefault(asset_type, {})[str(refreshed.get("asset_id") or "")] = refreshed

    def _merge_technical_record(self, asset_type: str, asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = self._asset_record(asset_type, asset_id)
        merged = {**record, **payload}
        self._replace_catalog_record(asset_type, merged)
        return merged

    def update_asset_metadata(self, asset_type: str, asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = self._asset_record(asset_type, asset_id)
        save_asset_metadata(record["path"], payload)
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

    def delete_asset(self, asset_type: str, asset_id: str) -> dict[str, Any]:
        record = self._asset_record(asset_type, asset_id)
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        metadata = load_asset_metadata(path)
        preview = resolve_preview_path(path, metadata)
        sidecar = sidecar_path(path)
        deleted: list[str] = []
        for candidate in (preview, sidecar, path):
            if candidate is None:
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.is_file():
                resolved.unlink()
                deleted.append(str(resolved))
        if asset_type == "checkpoint":
            self._models = [item for item in self._models if item.get("asset_id") != asset_id]
        elif asset_type == "lora":
            self._loras = [item for item in self._loras if item.get("asset_id") != asset_id]
        elif asset_type == "vae":
            self._vaes = [item for item in self._vaes if item.get("asset_id") != asset_id]
        elif asset_type == "textual_inversion":
            self._textual_inversions = [item for item in self._textual_inversions if item.get("asset_id") != asset_id]
        self._asset_indexes.get(asset_type, {}).pop(asset_id, None)
        self._bump_catalog_revision(asset_type)
        return {
            "deleted": True,
            "asset_id": asset_id,
            "paths": deleted,
            "catalog": self.catalog_status(asset_type)["catalogs"][asset_type],
        }

    def resolve_output_root(self, raw_path: str | os.PathLike[str]) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return self.context.resolve_project_path(path)

    def configured_output_roots(self, extra_paths: Iterable[str] | None = None) -> list[Path]:
        seen: set[str] = set()
        output: list[Path] = []
        for raw in [self.context.txt2img_output_root, *(extra_paths or [])]:
            try:
                root = self.resolve_output_root(raw)
            except OSError:
                continue
            token = str(root).casefold()
            if token in seen:
                continue
            seen.add(token)
            output.append(root)
        return output

    @staticmethod
    def _mtime_cutoff(hours: int | None) -> datetime | None:
        if hours is None or int(hours) <= 0:
            return None
        return datetime.now(timezone.utc) - timedelta(hours=int(hours))

    @staticmethod
    def _iter_image_files(root: Path, *, include_subfolders: bool) -> Iterable[Path]:
        iterator = root.rglob("*") if include_subfolders else root.glob("*")
        for path in iterator:
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
                yield path

    @staticmethod
    def _asset_label(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("display_name") or value.get("name") or value.get("path") or "")
        if isinstance(value, str):
            return value
        return ""

    @staticmethod
    def _infer_generation_mode(replay: dict[str, Any]) -> str:
        if replay.get("init_image") or replay.get("source_image") or replay.get("img2img"):
            return "img2img"
        if replay.get("input_image") or replay.get("inpaint"):
            return "img2img"
        if replay:
            return "txt2img"
        return "unknown"

    @staticmethod
    def _infer_hires(replay: dict[str, Any], manifest: dict[str, Any]) -> bool | None:
        for key in ("enable_hr", "hires_fix", "hires_enabled"):
            if key in replay:
                return bool(replay.get(key))
        required = manifest.get("required_for_rerun") if isinstance(manifest, dict) else {}
        if isinstance(required, dict):
            for key in ("enable_hr", "hires_fix", "hires_enabled"):
                if key in required:
                    return bool(required.get(key))
        return None

    def output_summary_from_path(self, image_path: Path) -> dict[str, Any] | None:
        try:
            resolved = image_path.resolve()
        except OSError:
            return None

        try:
            modified_ns = resolved.stat().st_mtime_ns
        except OSError:
            modified_ns = 0

        cache_key = (str(resolved).casefold(), int(modified_ns))
        cached = self._output_summary_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        root = self.context.txt2img_output_root.resolve()
        is_managed = is_within_root(resolved, root)
        relative = resolved.name
        source_root = ""
        output_id = relative
        details_url = ""
        if is_managed:
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError:
                relative = resolved.name
            output_id = relative
            url = f"/outputs/{quote(relative, safe='/')}"
            details_url = f"/api/outputs/{quote(output_id, safe='/')}/details"
            source_root = str(root)
        else:
            output_id = encode_external_image_ref(resolved)
            url = f"/api/image-files/{quote(output_id, safe='')}"
            details_url = f"/api/image-files/{quote(output_id, safe='')}/details"
            source_root = str(resolved.parent)

        try:
            details = load_image_file_details(
                self.context,
                resolved,
                display_name=relative,
            )
        except Exception:
            details = None

        replay = dict(details.replay) if details is not None else {}
        image = dict(details.image) if details is not None else {}
        metadata_source = str(getattr(details, "metadata_source", "partial_summary") or "partial_summary")
        manifest = dict(getattr(details, "manifest", {}) or {})
        model = image.get("model") or {}
        vae = image.get("vae") or {}
        loras = image.get("loras") or []
        payload = {
            "output_id": output_id,
            "name": resolved.name,
            "relative_name": relative,
            "url": url,
            "details_url": details_url,
            "prompt": replay.get("positive_prompt") or "",
            "negative_prompt": replay.get("negative_prompt") or "",
            "seed": replay.get("seed"),
            "width": replay.get("width") or image.get("width"),
            "height": replay.get("height") or image.get("height"),
            "steps": replay.get("steps"),
            "cfg_scale": replay.get("cfg_scale"),
            "sampler_name": replay.get("sampler_name") or "",
            "scheduler_name": replay.get("scheduler_name") or "",
            "model_path": replay.get("model_path") or "",
            "model_name": str(model.get("display_name") or Path(str(replay.get("model_path") or "")).name or ""),
            "model_hash": str(model.get("hash") or ""),
            "vae_path": replay.get("vae_path") or "",
            "vae_name": str(vae.get("display_name") or Path(str(replay.get("vae_path") or "")).name or ""),
            "loras": [self._asset_label(item) for item in loras if self._asset_label(item)],
            "timestamp": image.get("timestamp"),
            "modified_ns": modified_ns,
            "metadata_source": metadata_source,
            "source_kind": "output_root" if is_managed else "external_image",
            "source_root": source_root,
            "absolute_path": str(resolved),
            "generation_mode": self._infer_generation_mode(replay),
            "hires": self._infer_hires(replay, manifest),
        }
        self._output_summary_cache[cache_key] = dict(payload)
        if len(self._output_summary_cache) > 4096:
            self._output_summary_cache = dict(list(self._output_summary_cache.items())[-2048:])
        return payload

    def recent_outputs(
        self,
        limit: int | None = None,
        *,
        hours: int | None = None,
        include_subfolders: bool = True,
        extra_paths: Iterable[str] | None = None,
        require_metadata_for_external: bool = True,
    ) -> list[dict[str, Any]]:
        roots = self.configured_output_roots(extra_paths)
        cutoff = self._mtime_cutoff(hours)
        candidates: list[tuple[int, Path, bool]] = []
        seen: set[str] = set()
        managed_root = self.context.txt2img_output_root.resolve()

        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for path in self._iter_image_files(root, include_subfolders=include_subfolders):
                try:
                    resolved = path.resolve()
                    stat = resolved.stat()
                except OSError:
                    continue
                token = str(resolved).casefold()
                if token in seen:
                    continue
                if cutoff is not None:
                    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    if modified < cutoff:
                        continue
                seen.add(token)
                candidates.append((stat.st_mtime_ns, resolved, is_within_root(resolved, managed_root)))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if limit is not None:
            candidates = candidates[: max(1, int(limit))]

        output: list[dict[str, Any]] = []
        for _, image_path, is_managed in candidates:
            summary = self.output_summary_from_path(image_path)
            if summary is None:
                continue
            if not is_managed and require_metadata_for_external and summary.get("metadata_source") == "partial_summary":
                continue
            output.append(summary)
        return output


__all__ = ["ASSET_CATALOG_CONTRACT_VERSION", "WebUICatalog"]
