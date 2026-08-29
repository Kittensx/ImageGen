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
from image_gen.runtime.lora_inventory import discover_lora_library_roots, scan_lora_library_files, summarize_lora_scan_roots

from .contracts import (
    ASSET_CATALOG_CONTRACT_VERSION,
    _ASSET_PLURAL_KEYS,
    _ASSET_TYPES,
    _IMAGE_EXTENSIONS,
    _LORA_EXTENSIONS,
    _TEXTUAL_INVERSION_EXTENSIONS,
)


class AssetCatalogMixin:
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
        asset_hub_metadata = metadata.get("asset_hub")
        asset_hub_metadata = dict(asset_hub_metadata) if isinstance(asset_hub_metadata, dict) else {}
        asset_hub_provider = str(asset_hub_metadata.get("provider") or "").strip().casefold()
        preview_provenance = metadata.get("_preview_provenance")
        preview_provenance = dict(preview_provenance) if isinstance(preview_provenance, dict) else {}
        provider_preview_url = str(lookup.get("image_url") or "").strip()
        if not provider_preview_url:
            provider_previews = asset_hub_metadata.get("preview_images")
            if isinstance(provider_previews, list):
                for provider_preview in provider_previews:
                    if not isinstance(provider_preview, dict):
                        continue
                    candidate = str(provider_preview.get("url") or "").strip()
                    if candidate and str(provider_preview.get("kind") or "image").casefold() == "image":
                        provider_preview_url = candidate
                        break
        local_preview_source = str(preview_provenance.get("source") or "").strip().casefold()
        if preview is not None and not local_preview_source:
            lookup_preview_path = str(lookup.get("preview_image_path") or "").strip()
            if bool(lookup.get("preview_image_downloaded")) and lookup_preview_path:
                try:
                    same_preview = Path(lookup_preview_path).expanduser().resolve() == preview.resolve()
                except OSError:
                    same_preview = False
                local_preview_source = "civitai_cache" if same_preview else "local"
            else:
                local_preview_source = "local"
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
        if not tags and isinstance(asset_hub_metadata.get("tags"), list):
            tags = list(asset_hub_metadata.get("tags") or [])
        provider_trigger_words = asset_hub_metadata.get("trigger_words") or asset_hub_metadata.get("trained_words") or []
        if isinstance(provider_trigger_words, str):
            provider_trigger_words = [provider_trigger_words]
        elif not isinstance(provider_trigger_words, list):
            provider_trigger_words = list(provider_trigger_words) if isinstance(provider_trigger_words, tuple) else []
        provider_classification = asset_hub_metadata.get("classification_result")
        provider_classification = dict(provider_classification) if isinstance(provider_classification, dict) else {}
        provider_model_family = str(asset_hub_metadata.get("model_family") or provider_classification.get("architecture") or "").strip()
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
            "local_preview_url": preview_url,
            "local_preview_source": local_preview_source,
            "provider_preview_url": provider_preview_url,
            "source_url": str(metadata.get("source_url") or asset_hub_metadata.get("source_page_url") or ""),
            "description": str(metadata.get("description") or asset_hub_metadata.get("description_plaintext") or ""),
            "notes": str(metadata.get("notes") or ""),
            "tags": [str(value) for value in tags if str(value).strip()],
            "category": str(metadata.get("category") or ""),
            "favorite": bool(metadata.get("favorite", False)),
            "model_family": str(metadata.get("model_family") or provider_model_family or ""),
            "architecture": str(metadata.get("architecture") or provider_classification.get("architecture") or provider_model_family or ""),
            "prediction_type": str(metadata.get("prediction_type") or ""),
            "conditioning_dimension": metadata.get("conditioning_dimension"),
            "activation_text": str(metadata.get("activation_text") or (provider_trigger_words[0] if provider_trigger_words else "")),
            "provider_metadata": asset_hub_metadata,
            "provider_id": str(asset_hub_metadata.get("provider") or ""),
            "provider_model_id": str(asset_hub_metadata.get("provider_model_id") or ""),
            "provider_model_version_id": str(asset_hub_metadata.get("provider_model_version_id") or ""),
            "provider_file_id": str(asset_hub_metadata.get("provider_file_id") or ""),
            "provider_creator": str(asset_hub_metadata.get("provider_creator_name") or asset_hub_metadata.get("author_name") or ""),
            "provider_base_model": str(asset_hub_metadata.get("base_model") or ""),
            "provider_version_name": str(asset_hub_metadata.get("version_name") or ""),
            "preferred_weight": max(-4.0, min(4.0, preferred_weight)),
        }
        if not lookup and asset_hub_provider == "civitai":
            lookup = {
                "status": "matched",
                "model_id": asset_hub_metadata.get("provider_model_id"),
                "model_version_id": asset_hub_metadata.get("provider_model_version_id"),
                "model_name": str(asset_hub_metadata.get("display_name") or asset_hub_metadata.get("title") or ""),
                "model_version_name": str(asset_hub_metadata.get("version_name") or ""),
                "model_type": str(asset_hub_metadata.get("provider_asset_type") or ""),
                "creator": str(asset_hub_metadata.get("provider_creator_name") or ""),
                "base_model": str(asset_hub_metadata.get("base_model") or ""),
                "source_url": str(asset_hub_metadata.get("source_page_url") or ""),
                "image_url": provider_preview_url,
                "matched_file": {
                    "id": asset_hub_metadata.get("provider_file_id"),
                    "name": str(asset_hub_metadata.get("original_filename") or asset_hub_metadata.get("filename") or ""),
                    "hashes": dict(asset_hub_metadata.get("hashes") or {}) if isinstance(asset_hub_metadata.get("hashes"), dict) else {},
                },
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
            root_report = discover_lora_library_roots(self.context)
            roots = [Path(str(item.get("path") or "")).expanduser().resolve(strict=False) for item in root_report.get("roots") or []]
            return roots, set(_LORA_EXTENSIONS)
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
                payload = {
                    "asset_type": kind,
                    "plural_key": _ASSET_PLURAL_KEYS[kind],
                    "revision": int(self._catalog_revisions.get(kind, 0)),
                    "refreshed_at": str(self._catalog_refreshed_at.get(kind, "")),
                    "count": len(records),
                }
                if kind == "lora":
                    payload["scan_roots"] = dict(getattr(self, "_lora_root_report", {"roots": [], "diagnostics": [], "summary": {}}))
                catalogs[kind] = payload
        return {
            "contract_version": ASSET_CATALOG_CONTRACT_VERSION,
            "catalogs": catalogs,
        }

    def refresh_asset_type(self, asset_type: str) -> dict[str, Any]:
        if asset_type == "lora":
            root_report = discover_lora_library_roots(self.context)
            items = scan_lora_library_files(root_report.get("roots") or [])
            for item in items:
                path = Path(str(item.get("path") or "")).expanduser().resolve()
                item["embedded_name"] = self._embedded_safetensors_name(path, asset_type)
            scan_root_summary = summarize_lora_scan_roots(root_report.get("roots") or [], items)
            self._lora_root_report = {
                **scan_root_summary,
                "diagnostics": list(root_report.get("diagnostics") or []),
            }
            with self._catalog_lock:
                self._bump_catalog_revision(asset_type)
                records = self._catalog_entries(items, asset_type=asset_type)
                setattr(self, self._collection_attribute(asset_type), records)
            return self.asset_payload(asset_type)

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

    def checkpoint_details(self, asset_id: str, *, inspect_technical: bool = True) -> dict[str, Any]:
        record = self._asset_record("checkpoint", asset_id)
        path = Path(record["path"]).resolve()
        stat = path.stat()
        cache_key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        technical = self._checkpoint_detail_cache.get(cache_key)
        if technical is None:
            technical = {}
            if inspect_technical and path.suffix.lower() == ".safetensors":
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
            return self.checkpoint_details(asset_id, inspect_technical=False)
        if asset_type == "lora":
            return self.lora_details(asset_id, inspect_technical=False)
        if asset_type == "vae":
            return self.vae_details(asset_id)
        return self.asset_record(asset_type, asset_id)

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
            return self.checkpoint_details(asset_id, inspect_technical=False)
        if asset_type == "lora":
            return self.lora_details(asset_id, inspect_technical=False)
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
