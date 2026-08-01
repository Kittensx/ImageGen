from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from image_gen.systems.registry import RuntimeRegistrySystem
from image_gen.webui.image_refs import encode_external_image_ref, is_within_root
from image_gen.webui.output_details import load_image_file_details
from image_gen.webui.schema_utils import normalize_config_schema
from modules.project_context import ProjectContext
from modules.txt2img.model_selector import MODEL_EXTENSIONS

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class WebUICatalog:
    """Manual-refresh catalogs for plugins, models, VAEs, and recent outputs."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self._registry = RuntimeRegistrySystem(project_context=context)
        self._models: list[dict[str, Any]] = []
        self._vaes: list[dict[str, Any]] = []
        self._output_summary_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self.refresh_models()

    def reload_plugins(self) -> None:
        self._registry = RuntimeRegistrySystem(project_context=self.context)

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

    def refresh_models(self) -> dict[str, list[dict[str, Any]]]:
        checkpoint_roots = [self.context.checkpoints_dir, *self._additional_roots()]
        self._models = self._scan_files(checkpoint_roots, MODEL_EXTENSIONS)
        self._vaes = self._scan_files([self.context.vae_dir], MODEL_EXTENSIONS)
        return self.model_payload()

    def model_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {"models": list(self._models), "vaes": list(self._vaes)}

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
