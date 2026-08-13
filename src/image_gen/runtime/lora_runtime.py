
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from modules.asset_discovery import resolve_nested_asset
from modules.project_context import ProjectContext
from image_gen.runtime.lora_inspector import (
    cached_or_compute_lora_compatibility_hash,
    canonical_model_family,
    inspect_lora_file,
)
from image_gen.runtime.adapters.compatibility import AdapterCompatibilityService
from image_gen.runtime.adapters.contracts import AdapterInspectionRecord, AdapterRuntimePlan
from image_gen.runtime.adapters.registry import (
    AdapterLoaderRegistry,
    default_adapter_loader_registry,
)
from image_gen.contracts import (
    PROMPT_ASSET_CONTRACT_VERSION,
    GenerationRequest,
    PromptAssetSelection,
    canonical_prompt_asset_source,
    normalize_prompt_asset_list,
)

_LORA_TOKEN_RE = re.compile(r"<lora:([^:>]+?)(?::([-+]?\d*\.?\d+))?>", re.IGNORECASE)


def _canonical_family(value: Any) -> str:
    return canonical_model_family(value)


def _path_token(value: str | os.PathLike[str]) -> str:
    try:
        resolved = Path(value).expanduser().resolve()
    except OSError:
        resolved = Path(value).expanduser()
    return os.path.normcase(str(resolved))


def _coerce_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"", "none"}:
        return default
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    return default


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ResolvedLoRAAsset:
    requested_name: str
    requested_path: str
    resolved_path: str
    asset_id: str = ""
    catalog_asset_id: str = ""
    requested_hash: str = ""
    weight: float = 1.0
    enabled: bool = True
    polarity: str = "positive"
    activation_text: str = ""
    model_family: str = ""
    source_url: str = ""
    source: str = "visual_selection"
    original_source: str = ""
    order: int = 0
    file_hash: str = ""
    a1111_hash: str = ""
    a1111_short_hash: str = ""
    a1111_hash_source: str = ""
    adapter_name: str = ""
    metadata: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        metadata = dict(self.metadata or {})
        if self.a1111_hash:
            metadata["a1111_hash"] = self.a1111_hash
            metadata["a1111_short_hash"] = self.a1111_short_hash or self.a1111_hash[:12]
            metadata["a1111_hash_source"] = self.a1111_hash_source
        inspection = dict(metadata.get("_adapter_inspection") or {})
        runtime_plan = dict(metadata.get("_adapter_runtime_plan") or {})
        compatibility = dict(runtime_plan.get("compatibility") or {})
        return {
            "asset_type": "lora",
            "asset_id": self.asset_id,
            "catalog_asset_id": self.catalog_asset_id or self.asset_id,
            "requested_name": self.requested_name,
            "name": self.requested_name or Path(self.resolved_path).stem,
            "requested_path": self.requested_path,
            "path": self.resolved_path,
            "resolved_path": self.resolved_path,
            "requested_hash": self.requested_hash,
            "resolved_hash": self.file_hash,
            "file_hash": self.file_hash,
            "a1111_hash": self.a1111_hash,
            "a1111_short_hash": self.a1111_short_hash,
            "a1111_hash_source": self.a1111_hash_source,
            "weight": self.weight,
            "enabled": self.enabled,
            "polarity": self.polarity,
            "activation_text": self.activation_text,
            "model_family": self.model_family,
            "source_url": self.source_url,
            "source": self.source,
            "original_source": self.original_source,
            "origin": self.source,
            "order": self.order,
            "adapter_name": self.adapter_name,
            "adapter_format": str(inspection.get("adapter_format") or ""),
            "adapter_extensions": list(inspection.get("adapter_extensions") or []),
            "detected_model_family": str(inspection.get("model_family") or ""),
            "target_scopes": list(inspection.get("target_scopes") or []),
            "runtime_support_state": str(compatibility.get("overall_support_state") or ""),
            "runtime_loadable": bool(compatibility.get("runtime_loadable", False)),
            "support_reason": str(compatibility.get("blocking_reason") or ""),
            "loader_id": str(runtime_plan.get("loader_id") or ""),
            "adapter_inspection": inspection,
            "adapter_runtime_plan": runtime_plan,
            "metadata": metadata,
        }


class LoRAResolver:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self._catalog_cache: dict[str, Path] | None = None
        self._hash_cache: dict[tuple[str, int, int], str] = {}
        self._compatibility_hash_cache: dict[tuple[str, int, int], dict[str, str]] = {}
        self._metadata_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def _scan_catalog(self) -> dict[str, Path]:
        if self._catalog_cache is not None:
            return dict(self._catalog_cache)
        catalog: dict[str, Path] = {}
        root = getattr(self.context, "lora_dir", None)
        if root is not None:
            try:
                root_path = Path(root).expanduser().resolve()
                if root_path.exists():
                    for path in root_path.rglob("*"):
                        if not path.is_file() or path.suffix.lower() not in {".safetensors", ".pt", ".ckpt", ".bin"}:
                            continue
                        keys = {
                            path.name.casefold(),
                            path.stem.casefold(),
                            str(path).casefold(),
                        }
                        try:
                            relative = path.relative_to(root_path).as_posix().casefold()
                            keys.add(relative)
                        except ValueError:
                            pass
                        for key in keys:
                            catalog.setdefault(key, path)
            except OSError:
                pass
        self._catalog_cache = dict(catalog)
        return dict(catalog)

    def resolve(self, requested_name: str = "", requested_path: str = "") -> Path:
        candidates: list[Path] = []
        for raw in (requested_path, requested_name):
            text = str(raw or "").strip()
            if not text:
                continue
            direct = Path(text).expanduser()
            if direct.is_file():
                candidates.append(direct.resolve())
                continue
            try:
                project = self.context.resolve_project_path(text)
                if project.is_file():
                    candidates.append(project.resolve())
                    continue
            except Exception:
                pass
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        lora_root = getattr(self.context, "lora_dir", None)
        if lora_root is not None:
            for raw in (requested_path, requested_name):
                text = str(raw or "").strip()
                if not text:
                    continue
                nested = resolve_nested_asset(
                    lora_root,
                    text,
                    extensions={".safetensors", ".pt", ".ckpt", ".bin"},
                )
                if nested is not None:
                    return nested

        catalog = self._scan_catalog()
        for raw in (requested_path, requested_name):
            text = str(raw or "").strip()
            if not text:
                continue
            key_candidates = [text.casefold(), Path(text).name.casefold(), Path(text).stem.casefold()]
            normalized = text.replace("\\", "/").lstrip("/")
            key_candidates.append(normalized.casefold())
            for key in key_candidates:
                resolved = catalog.get(key)
                if resolved is not None and resolved.is_file():
                    return resolved
        raise ValueError(f"LoRA could not be resolved from installed assets: {requested_name or requested_path!r}.")

    def file_hash(self, path: Path) -> str:
        try:
            stat = path.stat()
        except OSError:
            return ""
        cache_key = (_path_token(path), int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._hash_cache.get(cache_key)
        if cached is not None:
            return cached
        digest = _hash_file(path)
        self._hash_cache = {cache_key: digest, **self._hash_cache}
        if len(self._hash_cache) > 128:
            self._hash_cache = dict(list(self._hash_cache.items())[:128])
        return digest

    def compatibility_hash(
        self,
        path: Path,
        *,
        sidecar_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        try:
            stat = path.stat()
        except OSError:
            return {
                "a1111_hash": "",
                "a1111_short_hash": "",
                "a1111_hash_source": "",
            }
        cache_key = (_path_token(path), int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._compatibility_hash_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        try:
            payload = cached_or_compute_lora_compatibility_hash(
                path,
                sidecar_metadata=sidecar_metadata,
            )
        except Exception:
            payload = {
                "a1111_hash": "",
                "a1111_short_hash": "",
                "a1111_hash_source": "",
            }
        self._compatibility_hash_cache = {
            cache_key: dict(payload),
            **self._compatibility_hash_cache,
        }
        if len(self._compatibility_hash_cache) > 128:
            self._compatibility_hash_cache = dict(
                list(self._compatibility_hash_cache.items())[:128]
            )
        return dict(payload)


    def metadata(self, path: Path) -> dict[str, Any]:
        sidecar = path.with_name(f"{path.stem}.imagegen.json")
        if not sidecar.is_file():
            return {}
        try:
            modified_ns = int(sidecar.stat().st_mtime_ns)
        except OSError:
            return {}
        cache_key = (_path_token(sidecar), modified_ns)
        cached = self._metadata_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        self._metadata_cache = {cache_key: dict(payload), **self._metadata_cache}
        if len(self._metadata_cache) > 128:
            self._metadata_cache = dict(list(self._metadata_cache.items())[:128])
        return dict(payload)


class LoRARuntimeManager:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.resolver = LoRAResolver(context)
        self.loader_registry: AdapterLoaderRegistry = default_adapter_loader_registry()
        self.compatibility_service = AdapterCompatibilityService(self.loader_registry)
        self._bound_component_id: int | None = None
        self._loaded_adapters: dict[str, dict[str, Any]] = {}
        self._active_signature: tuple[tuple[str, float], ...] = ()
        self._active_stack: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._bound_component_id = None
        self._loaded_adapters.clear()
        self._active_signature = ()
        self._active_stack = []

    def prepare_request(
        self,
        request: GenerationRequest,
        extras: dict[str, Any] | None,
        *,
        checkpoint_family: str = "",
    ) -> list[dict[str, Any]]:
        extras = extras if isinstance(extras, dict) else {}
        checkpoint_family = _canonical_family(checkpoint_family)
        inline_records: list[dict[str, Any]] = []
        for field_name in ("positive_prompt", "negative_prompt", "hires_positive_prompt", "hires_negative_prompt"):
            value = getattr(request, field_name, "")
            cleaned, extracted = self._strip_inline_loras(value)
            setattr(request, field_name, cleaned)
            for item in extracted:
                item["source"] = "inline_syntax"
                item["field"] = field_name
                item["polarity"] = "negative" if "negative" in field_name else "positive"
                inline_records.append(item)

        structured = self._normalize_structured_loras(
            getattr(request, "loras", None) or extras.get("loras"),
            extras.get("lora_paths"),
        )
        merged = self._merge_requested_stack(structured, inline_records)

        resolved_stack: list[dict[str, Any]] = []
        contract_stack: list[dict[str, Any]] = []
        preflight_items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for order, item in enumerate(merged):
            if not _coerce_bool(item.get("enabled", True), True):
                disabled = PromptAssetSelection.from_value(
                    {**item, "enabled": False, "order": order},
                    asset_type="lora",
                    order=order,
                )
                contract_stack.append(disabled.to_serializable_dict())
                continue
            requested_name = str(item.get("name") or item.get("requested_name") or "").strip()
            requested_path = str(item.get("path") or item.get("requested_path") or "").strip()
            resolved_path = self.resolver.resolve(requested_name, requested_path)
            sidecar_metadata = self.resolver.metadata(resolved_path)
            inspection_metadata = {**sidecar_metadata, **dict(item.get("metadata") or {})}
            requested_family = (
                item.get("model_family")
                or item.get("base_model")
                or item.get("sd_version")
            )
            if requested_family and not inspection_metadata.get("model_family"):
                inspection_metadata["model_family"] = requested_family
            inspection_payload = inspect_lora_file(
                resolved_path,
                sidecar_metadata=inspection_metadata,
                include_compatibility_hash=False,
            )
            inspection = AdapterInspectionRecord.from_mapping(inspection_payload.get("adapter_inspection"))
            compatibility = self.compatibility_service.evaluate(
                inspection,
                active_checkpoint_family=checkpoint_family,
            )
            requested_label = requested_name or resolved_path.stem
            preflight_item = {
                "requested_name": requested_label,
                "resolved_path": str(resolved_path),
                "inspection": inspection.to_dict(),
                "compatibility": compatibility.to_dict(),
            }
            preflight_items.append(preflight_item)
            if not compatibility.runtime_loadable:
                extras["adapter_preflight"] = {
                    "adapter_count": len(merged),
                    "checkpoint_family": checkpoint_family,
                    "runtime_loadable": False,
                    "blocked_adapter": requested_label,
                    "blocking_reason": compatibility.blocking_reason,
                    "items": list(preflight_items),
                }
                raise ValueError(f"LoRA '{requested_label}' cannot be loaded: {compatibility.blocking_reason}")

            # Expensive identity hashing happens only after format/family/target preflight
            # proves that this adapter has a valid runtime path.
            file_hash = self.resolver.file_hash(resolved_path)
            inspection = AdapterInspectionRecord.from_mapping({**inspection.to_dict(), "sha256": file_hash})
            compatibility_hash = self.resolver.compatibility_hash(
                resolved_path,
                sidecar_metadata=sidecar_metadata,
            )
            adapter_token = file_hash[:12] if file_hash else f"{abs(hash(str(resolved_path))) & 0xffffffff:08x}"
            source = canonical_prompt_asset_source(
                item.get("source")
                or item.get("selection_source")
                or item.get("selection_origin")
                or item.get("source_scope")
                or item.get("origin")
                or "visual_selection"
            )
            runtime_plan = AdapterRuntimePlan(
                adapter_identity=file_hash or str(resolved_path),
                asset_id=str(item.get("asset_id") or item.get("catalog_asset_id") or "").strip(),
                requested_name=requested_name or resolved_path.stem,
                resolved_path=str(resolved_path),
                file_hash=file_hash,
                inspection_contract_version=inspection.contract_version,
                adapter_format=inspection.adapter_format,
                model_family=inspection.model_family,
                active_checkpoint_family=checkpoint_family,
                compatibility=compatibility.to_dict(),
                loader_id=compatibility.loader_id,
                requested_weight=_coerce_float(item.get("weight"), 1.0),
                effective_weight=_coerce_float(item.get("weight"), 1.0),
                weight_semantics="user multiplier after loader-native rank/alpha normalization",
                expected_component_targets=inspection.target_scopes,
                blocking_reason=compatibility.blocking_reason,
                warnings=compatibility.warnings,
            )
            technical_metadata = {
                **sidecar_metadata,
                **dict(item.get("metadata") or {}),
                "adapter_format": inspection.adapter_format,
                "adapter_extensions": list(inspection.adapter_extensions),
                "target_scopes": list(inspection.target_scopes),
                "runtime_support_state": compatibility.overall_support_state,
                "runtime_loadable": compatibility.runtime_loadable,
                "support_reason": compatibility.blocking_reason,
                "loader_id": compatibility.loader_id,
                "_adapter_inspection": inspection.to_dict(),
                "_adapter_runtime_plan": runtime_plan.to_dict(),
            }
            entry = ResolvedLoRAAsset(
                requested_name=requested_name or resolved_path.stem,
                requested_path=requested_path,
                resolved_path=str(resolved_path),
                asset_id=str(item.get("asset_id") or "").strip(),
                catalog_asset_id=str(item.get("catalog_asset_id") or item.get("asset_id") or "").strip(),
                requested_hash=str(item.get("requested_hash") or item.get("file_hash") or "").strip(),
                weight=_coerce_float(item.get("weight"), 1.0),
                enabled=_coerce_bool(item.get("enabled", True), True),
                polarity=str(item.get("polarity") or "positive").strip().lower(),
                activation_text=str(item.get("activation_text") or inspection_payload.get("activation_text") or sidecar_metadata.get("activation_text") or "").strip(),
                model_family=inspection.model_family,
                source_url=str(item.get("source_url") or sidecar_metadata.get("source_url") or "").strip(),
                source=source,
                original_source=canonical_prompt_asset_source(item.get("original_source"), default="") if item.get("original_source") else "",
                order=order,
                file_hash=file_hash,
                a1111_hash=str(compatibility_hash.get("a1111_hash") or ""),
                a1111_short_hash=str(compatibility_hash.get("a1111_short_hash") or ""),
                a1111_hash_source=str(compatibility_hash.get("a1111_hash_source") or ""),
                adapter_name=f"lora_{adapter_token}",
                metadata=technical_metadata,
            )
            dedupe_key = (entry.file_hash or _path_token(entry.resolved_path)).casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            resolved_payload = entry.to_payload()
            resolved_stack.append(resolved_payload)
            contract_stack.append(resolved_payload)

        activation_report = self._apply_activation_texts(request, resolved_stack)
        if activation_report:
            extras["lora_activation_text"] = activation_report

        request.prompt_asset_contract_version = PROMPT_ASSET_CONTRACT_VERSION
        request.loras = normalize_prompt_asset_list(
            contract_stack,
            asset_type="lora",
        )
        extras["loras"] = [asset.to_serializable_dict() for asset in request.loras]
        extras["lora_paths"] = [
            str(item.get("resolved_path") or item.get("path") or "")
            for item in resolved_stack
            if item.get("resolved_path") or item.get("path")
        ]
        extras["resolved_lora_stack"] = [dict(item) for item in resolved_stack]
        extras["adapter_runtime_plans"] = [
            dict(item.get("adapter_runtime_plan") or {})
            for item in resolved_stack
            if isinstance(item.get("adapter_runtime_plan"), Mapping)
        ]
        extras["adapter_preflight"] = {
            "adapter_count": len(resolved_stack),
            "checkpoint_family": checkpoint_family,
            "runtime_loadable": True,
            "items": list(preflight_items),
            "plans": list(extras["adapter_runtime_plans"]),
        }
        extras["prompt_asset_contract_version"] = PROMPT_ASSET_CONTRACT_VERSION
        return [dict(item) for item in resolved_stack]

    def apply(
        self,
        *,
        components: Any,
        stack: Iterable[Mapping[str, Any]] | None,
        extras: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        extras = extras if isinstance(extras, dict) else {}
        component_id = id(components)
        if self._bound_component_id != component_id:
            self.reset()
            self._bound_component_id = component_id

        normalized_stack = [dict(item) for item in (stack or [])]
        signature = tuple(
            (
                str(item.get("file_hash") or item.get("resolved_path") or ""),
                round(_coerce_float(item.get("weight"), 1.0), 6),
                str((item.get("adapter_runtime_plan") or {}).get("loader_id") or item.get("loader_id") or ""),
                str((item.get("adapter_runtime_plan") or {}).get("active_checkpoint_family") or ""),
            )
            for item in normalized_stack
        )
        if not normalized_stack:
            self._deactivate_components(components)
            self._active_signature = ()
            self._active_stack = []
            extras["lora_runtime"] = {
                "active": [],
                "reused": False,
                "loaded_adapter_count": len(self._loaded_adapters),
                "adapter_reports": [],
            }
            return []
        if signature == self._active_signature:
            reused_stack = self._decorate_stack_with_runtime_details(normalized_stack)
            extras["lora_runtime"] = {
                "active": [dict(item) for item in reused_stack],
                "reused": True,
                "loaded_adapter_count": len(self._loaded_adapters),
                "adapter_reports": [
                    dict(item.get("runtime_load_report") or {})
                    for item in reused_stack
                    if isinstance(item.get("runtime_load_report"), Mapping)
                ],
            }
            return [dict(item) for item in reused_stack]

        self._deactivate_components(components)
        runtime_stack: list[dict[str, Any]] = []
        for item in normalized_stack:
            adapter_name = str(item.get("adapter_name") or "").strip()
            loaded_entry = dict(item)
            if adapter_name:
                loader_id, implementation = self._loader_implementation_for_entry(item)
                if adapter_name not in self._loaded_adapters:
                    self._loaded_adapters[adapter_name] = implementation.load(components=components, entry=dict(item))
                cached = dict(self._loaded_adapters.get(adapter_name) or {})
                if cached:
                    loaded_entry.update({
                        key: value
                        for key, value in cached.items()
                        if key in {"runtime_applied", "runtime_load_report"}
                    })
                loaded_entry["loader_id"] = loader_id
            runtime_stack.append(loaded_entry)

        runtime_stack = self._decorate_stack_with_runtime_details(runtime_stack)
        self._activate_stack(components, runtime_stack)
        self._active_signature = signature
        self._active_stack = [dict(item) for item in runtime_stack]
        extras["lora_runtime"] = {
            "active": [dict(item) for item in runtime_stack],
            "reused": False,
            "loaded_adapter_count": len(self._loaded_adapters),
            "adapter_reports": [
                dict(item.get("runtime_load_report") or {})
                for item in runtime_stack
                if isinstance(item.get("runtime_load_report"), Mapping)
            ],
        }
        return [dict(item) for item in runtime_stack]

    @staticmethod
    def _append_prompt_text(prompt: Any, addition: str) -> tuple[str, bool]:
        base = str(prompt or "").strip()
        extra = str(addition or "").strip(" ,")
        if not extra:
            return base, False
        if extra.casefold() in base.casefold():
            return base, False
        if not base:
            return extra, True
        return f"{base.rstrip(' ,')}, {extra}", True

    def _apply_activation_texts(
        self,
        request: GenerationRequest,
        stack: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        for item in stack:
            activation_text = str(item.get("activation_text") or "").strip()
            if not activation_text:
                continue
            polarity = str(item.get("polarity") or "positive").strip().lower()
            if polarity == "negative":
                request.negative_prompt, base_applied = self._append_prompt_text(
                    request.negative_prompt,
                    activation_text,
                )
                hires_negative_source = (
                    request.hires_negative_prompt
                    if str(request.hires_negative_prompt or "").strip()
                    else request.negative_prompt
                )
                request.hires_negative_prompt, hires_applied = self._append_prompt_text(
                    hires_negative_source,
                    activation_text,
                )
                targets = ["negative_prompt", "hires_negative_prompt"]
            else:
                request.positive_prompt, base_applied = self._append_prompt_text(
                    request.positive_prompt,
                    activation_text,
                )
                hires_positive_source = (
                    request.hires_positive_prompt
                    if str(request.hires_positive_prompt or "").strip()
                    else request.positive_prompt
                )
                request.hires_positive_prompt, hires_applied = self._append_prompt_text(
                    hires_positive_source,
                    activation_text,
                )
                targets = ["positive_prompt", "hires_positive_prompt"]
            report.append(
                {
                    "adapter_name": str(item.get("adapter_name") or ""),
                    "name": str(item.get("name") or item.get("requested_name") or ""),
                    "activation_text": activation_text,
                    "polarity": polarity if polarity in {"positive", "negative"} else "positive",
                    "targets": targets,
                    "base_applied": bool(base_applied),
                    "hires_applied": bool(hires_applied),
                }
            )
        return report

    def _strip_inline_loras(self, text: Any) -> tuple[str, list[dict[str, Any]]]:
        source = str(text or "")
        extracted: list[dict[str, Any]] = []

        def _replace(match: re.Match[str]) -> str:
            name = str(match.group(1) or "").strip()
            weight_raw = match.group(2)
            extracted.append({
                "name": name,
                "path": "",
                "weight": _coerce_float(weight_raw, 1.0),
                "activation_text": "",
                "enabled": True,
            })
            return " "

        cleaned = _LORA_TOKEN_RE.sub(_replace, source)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
        return cleaned, extracted

    def _normalize_structured_loras(self, raw: Any, raw_paths: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, PromptAssetSelection):
                    records.append(item.to_serializable_dict())
                elif isinstance(item, Mapping):
                    records.append(dict(item))
                elif isinstance(item, str) and item.strip():
                    records.append({"name": Path(item).stem, "path": item, "weight": 1.0, "source": "api_request"})
        if not records and isinstance(raw_paths, list):
            for item in raw_paths:
                if isinstance(item, str) and item.strip():
                    records.append({"name": Path(item).stem, "path": item, "weight": 1.0, "source": "imported"})
        return records

    def _merge_requested_stack(self, structured: list[dict[str, Any]], inline_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = [dict(item) for item in structured]
        index: dict[str, int] = {}

        def aliases(item: Mapping[str, Any]) -> list[str]:
            values = [
                item.get("resolved_hash"),
                item.get("requested_hash"),
                item.get("file_hash"),
                item.get("resolved_path"),
                item.get("path"),
                item.get("requested_path"),
                item.get("catalog_asset_id"),
                item.get("asset_id"),
                item.get("requested_name"),
                item.get("name"),
            ]
            output: list[str] = []
            for value in values:
                token = str(value or "").strip().casefold()
                if token and token not in output:
                    output.append(token)
                if token and ("/" in token or "\\" in token):
                    stem = Path(token).stem.casefold()
                    if stem and stem not in output:
                        output.append(stem)
            return output

        for position, item in enumerate(merged):
            for key in aliases(item):
                index[key] = position
        for item in inline_records:
            match = next((index[key] for key in aliases(item) if key in index), None)
            if match is not None:
                existing = merged[match]
                existing["original_source"] = existing.get("original_source") or existing.get("source") or ""
                existing.update({key_name: value for key_name, value in item.items() if value not in (None, "")})
                existing["source"] = "inline_syntax"
                for key in aliases(existing):
                    index[key] = match
            else:
                merged.append(dict(item))
                position = len(merged) - 1
                for key in aliases(item):
                    index[key] = position
        return merged

    def _decorate_stack_with_runtime_details(self, stack: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in stack:
            payload = dict(item)
            adapter_name = str(payload.get("adapter_name") or "").strip()
            cached = dict(self._loaded_adapters.get(adapter_name) or {}) if adapter_name else {}
            if cached.get("runtime_applied") is not None:
                payload["runtime_applied"] = bool(cached.get("runtime_applied"))
            if isinstance(cached.get("runtime_load_report"), Mapping):
                report = dict(cached["runtime_load_report"])
                current_weight = _coerce_float(payload.get("weight"), 1.0)
                report["requested_weight"] = current_weight
                report["effective_user_multiplier"] = current_weight
                report["final_effective_scale"] = current_weight
                payload["runtime_load_report"] = report
            output.append(payload)
        return output

    def _loader_implementation_for_entry(self, entry: Mapping[str, Any]) -> tuple[str, Any]:
        runtime_plan = dict(entry.get("adapter_runtime_plan") or {})
        if not runtime_plan:
            runtime_plan = dict((entry.get("metadata") or {}).get("_adapter_runtime_plan") or {})
        loader_id = str(runtime_plan.get("loader_id") or entry.get("loader_id") or "").strip()
        capability = self.loader_registry.capability(loader_id)
        if capability is None:
            raise ValueError(
                f"Adapter '{entry.get('requested_name') or entry.get('name') or entry.get('adapter_name') or 'adapter'}' "
                "has no registered runtime loader capability."
            )
        implementation = self.loader_registry.implementation(loader_id)
        if implementation is None:
            raise ValueError(
                f"Adapter loader '{loader_id}' is registered as a capability but has no runtime implementation."
            )
        return loader_id, implementation

    def _deactivate_components(self, components: Any) -> None:
        for implementation in self.loader_registry.implementations():
            deactivate = getattr(implementation, "deactivate", None)
            if callable(deactivate):
                deactivate(components=components)

    def _activate_stack(self, components: Any, stack: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in stack:
            loader_id, _implementation = self._loader_implementation_for_entry(item)
            grouped.setdefault(loader_id, []).append(item)
        activated = False
        for loader_id, loader_stack in grouped.items():
            implementation = self.loader_registry.implementation(loader_id)
            if implementation is None:
                raise ValueError(f"Adapter loader '{loader_id}' has no runtime implementation.")
            activate = getattr(implementation, "activate", None)
            if not callable(activate):
                raise ValueError(f"Adapter loader '{loader_id}' does not implement stack activation.")
            activate(components=components, stack=loader_stack)
            activated = True
        if stack and not activated:
            raise ValueError("LoRA adapters were loaded but no registered adapter loader activated the runtime stack.")

