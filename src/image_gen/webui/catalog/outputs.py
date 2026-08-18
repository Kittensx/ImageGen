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


class OutputCatalogMixin:
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
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            try:
                relative_parts = path.relative_to(root).parts
            except ValueError:
                relative_parts = path.parts
            # Output saving uses dot-prefixed image-suffix temp files such as
            # .image.<uuid>.tmp.png before os.replace(). They are implementation
            # details and can change size while a browser is reading them.
            if any(part.startswith(".") for part in relative_parts):
                continue
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
