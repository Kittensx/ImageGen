
from __future__ import annotations

import hashlib
import json
import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from modules.project_context import ProjectContext
from image_gen.runtime.lora_inspector import canonical_model_family
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
    adapter_name: str = ""
    metadata: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
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
            "metadata": dict(self.metadata or {}),
        }


class LoRAResolver:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self._catalog_cache: dict[str, Path] | None = None
        self._hash_cache: dict[tuple[str, int, int], str] = {}
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
            file_hash = self.resolver.file_hash(resolved_path)
            sidecar_metadata = self.resolver.metadata(resolved_path)
            adapter_token = file_hash[:12] if file_hash else f"{abs(hash(str(resolved_path))) & 0xffffffff:08x}"
            source = canonical_prompt_asset_source(
                item.get("source")
                or item.get("selection_source")
                or item.get("selection_origin")
                or item.get("source_scope")
                or item.get("origin")
                or "visual_selection"
            )
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
                activation_text=str(item.get("activation_text") or sidecar_metadata.get("activation_text") or "").strip(),
                model_family=_canonical_family(
                    item.get("model_family")
                    or item.get("base_model")
                    or item.get("sd_version")
                    or sidecar_metadata.get("model_family")
                    or sidecar_metadata.get("base_model")
                ),
                source_url=str(item.get("source_url") or sidecar_metadata.get("source_url") or "").strip(),
                source=source,
                original_source=canonical_prompt_asset_source(item.get("original_source"), default="") if item.get("original_source") else "",
                order=order,
                file_hash=file_hash,
                adapter_name=f"lora_{adapter_token}",
                metadata={**sidecar_metadata, **dict(item.get("metadata") or {})},
            )
            if entry.model_family and checkpoint_family and entry.model_family != checkpoint_family:
                raise ValueError(
                    f"LoRA '{entry.requested_name}' is tagged for model family '{entry.model_family}', "
                    f"but the active checkpoint family is '{checkpoint_family}'."
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
        signature = tuple((str(item.get("file_hash") or item.get("resolved_path") or ""), round(_coerce_float(item.get("weight"), 1.0), 6)) for item in normalized_stack)
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
                if adapter_name not in self._loaded_adapters:
                    self._loaded_adapters[adapter_name] = self._load_adapter_into_components(components, dict(item))
                cached = dict(self._loaded_adapters.get(adapter_name) or {})
                if cached:
                    loaded_entry.update({
                        key: value
                        for key, value in cached.items()
                        if key in {"runtime_applied", "runtime_load_report"}
                    })
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
                payload["runtime_load_report"] = dict(cached["runtime_load_report"])
            output.append(payload)
        return output

    def _module_supports_load(self, module: Any) -> bool:
        return hasattr(module, "load_lora_adapter") or hasattr(module, "load_attn_procs")

    def _stable_diffusion_lora_loader_mixin(self) -> Any:
        try:
            from diffusers.loaders import StableDiffusionLoraLoaderMixin

            return StableDiffusionLoraLoaderMixin
        except Exception:
            try:
                from diffusers.loaders import LoraLoaderMixin

                return LoraLoaderMixin
            except Exception as exc:
                raise ValueError(
                    "Unable to import diffusers LoRA loader support. Expected a diffusers build "
                    "with StableDiffusionLoraLoaderMixin or LoraLoaderMixin."
                ) from exc

    def _call_with_supported_kwargs(self, target: Any, kwargs: dict[str, Any]) -> Any:
        filtered = {key: value for key, value in kwargs.items() if value is not None}
        try:
            signature = inspect.signature(target)
            accepts_var_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if not accepts_var_kwargs:
                filtered = {
                    key: value
                    for key, value in filtered.items()
                    if key in signature.parameters
                }
        except (TypeError, ValueError):
            pass
        return target(**filtered)

    @staticmethod
    def _looks_like_lora_parameter_key(key: Any) -> bool:
        token = str(key or "").lower()
        return any(
            marker in token
            for marker in (
                ".lora_a.",
                ".lora_b.",
                ".lora_down.",
                ".lora_up.",
                ".lora_magnitude_vector",
                "lora_down.weight",
                "lora_up.weight",
                "lora_a.weight",
                "lora_b.weight",
            )
        )

    @staticmethod
    def _has_component_prefix(key: Any) -> bool:
        token = str(key or "")
        return token.startswith(("unet.", "text_encoder.", "text_encoder_2."))

    def _normalize_lora_component_prefixes(
        self,
        state_dict: Mapping[str, Any],
        network_alphas: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        """Normalize pipeline and component-native Diffusers LoRA state dicts.

        Kohya conversion normally produces ``unet.`` / ``text_encoder.`` keys.
        Diffusers component exports can instead contain keys indexed directly
        into a UNet. The pipeline mixin always supplies ``prefix='unet'`` to the
        component loader, so direct keys must be wrapped with ``unet.`` first.
        """

        normalized_state = dict(state_dict)
        normalized_alphas = dict(network_alphas)
        if any(self._has_component_prefix(key) for key in normalized_state):
            return normalized_state, normalized_alphas, "pipeline_prefixed"

        direct_lora_keys = [
            key for key in normalized_state if self._looks_like_lora_parameter_key(key)
        ]
        if direct_lora_keys:
            normalized_state = {
                f"unet.{key}": value
                for key, value in normalized_state.items()
            }
            normalized_alphas = {
                f"unet.{key}": value
                for key, value in normalized_alphas.items()
            }
            return normalized_state, normalized_alphas, "component_native_unet_prefixed"

        return normalized_state, normalized_alphas, "unrecognized"

    def _load_lora_state(
        self,
        *,
        path: str,
        components: Any,
        mixin: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        kwargs = {
            "pretrained_model_name_or_path_or_dict": path,
            "unet_config": getattr(getattr(components, "unet", None), "config", None),
            "return_lora_metadata": True,
        }
        try:
            loaded = self._call_with_supported_kwargs(mixin.lora_state_dict, kwargs)
        except TypeError:
            kwargs.pop("return_lora_metadata", None)
            loaded = self._call_with_supported_kwargs(mixin.lora_state_dict, kwargs)
        if not isinstance(loaded, tuple):
            raise ValueError("diffusers LoRA state conversion returned an unexpected payload.")
        if len(loaded) == 3:
            state_dict, network_alphas, metadata = loaded
        elif len(loaded) == 2:
            state_dict, network_alphas = loaded
            metadata = {}
        else:
            raise ValueError("diffusers LoRA state conversion returned an unexpected tuple shape.")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("diffusers LoRA state conversion did not produce a usable state dict.")
        network_alphas = dict(network_alphas or {}) if isinstance(network_alphas, Mapping) else {}
        metadata = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
        state_dict, network_alphas, prefix_mode = self._normalize_lora_component_prefixes(
            dict(state_dict),
            network_alphas,
        )
        return state_dict, network_alphas, metadata, prefix_mode

    @staticmethod
    def _count_prefixed_keys(state_dict: Mapping[str, Any], prefixes: tuple[str, ...]) -> int:
        total = 0
        for key in state_dict.keys():
            token = str(key or "")
            if any(token.startswith(prefix) for prefix in prefixes):
                total += 1
        return total

    def _build_runtime_load_report(
        self,
        *,
        entry: Mapping[str, Any],
        state_dict: Mapping[str, Any],
        network_alphas: Mapping[str, Any],
        metadata: Mapping[str, Any],
        prefix_mode: str,
    ) -> dict[str, Any]:
        entry_metadata = dict(entry.get("metadata") or {})
        scan_cache = dict(entry_metadata.get("_lora_scan_cache") or {})
        source_tensor_format = str(
            scan_cache.get("tensor_key_format")
            or entry_metadata.get("tensor_key_format")
            or ""
        )
        source_network_type = str(
            scan_cache.get("network_type")
            or entry_metadata.get("network_type")
            or ""
        )
        return {
            "adapter_name": str(entry.get("adapter_name") or ""),
            "resolved_path": str(entry.get("resolved_path") or entry.get("path") or ""),
            "loader_path": "diffusers_pipeline_mixin",
            "key_prefix_mode": prefix_mode,
            "source_tensor_format": source_tensor_format,
            "source_network_type": source_network_type,
            "converted_key_count": len(state_dict),
            "network_alpha_count": len(network_alphas),
            "metadata_key_count": len(metadata),
            "unet_candidate_keys": self._count_prefixed_keys(state_dict, ("unet.",)),
            "text_encoder_candidate_keys": self._count_prefixed_keys(state_dict, ("text_encoder.",)),
            "text_encoder_2_candidate_keys": self._count_prefixed_keys(state_dict, ("text_encoder_2.",)),
            "converted_key_examples": [str(key) for key in list(state_dict.keys())[:8]],
            "unet_loaded": False,
            "text_encoder_loaded": False,
            "verified": False,
        }

    @staticmethod
    def _verify_adapter_presence(module: Any, adapter_name: str) -> bool:
        if not adapter_name:
            return False
        peft_config = getattr(module, "peft_config", None)
        if isinstance(peft_config, Mapping) and adapter_name in peft_config:
            return True
        active = getattr(module, "active_adapters", None)
        if callable(active):
            try:
                current = active()
            except Exception:
                current = []
        else:
            current = active
        if isinstance(current, str):
            return current == adapter_name
        if isinstance(current, (list, tuple, set)):
            return adapter_name in current
        return False

    def _load_adapter_into_components(self, components: Any, entry: dict[str, Any]) -> dict[str, Any]:
        path = str(entry.get("resolved_path") or entry.get("path") or "").strip()
        adapter_name = str(entry.get("adapter_name") or "").strip()
        if not path:
            raise ValueError("LoRA runtime entry is missing a resolved adapter path.")
        if not adapter_name:
            raise ValueError("LoRA runtime entry is missing an adapter name.")
        if getattr(components, "unet", None) is None:
            raise ValueError("The active runtime components do not expose a UNet for LoRA loading.")
        if getattr(components, "text_encoder", None) is None:
            raise ValueError("The active runtime components do not expose a text encoder for LoRA loading.")

        mixin = self._stable_diffusion_lora_loader_mixin()
        state_dict, network_alphas, metadata, prefix_mode = self._load_lora_state(
            path=path,
            components=components,
            mixin=mixin,
        )
        report = self._build_runtime_load_report(
            entry=entry,
            state_dict=state_dict,
            network_alphas=network_alphas,
            metadata=metadata,
            prefix_mode=prefix_mode,
        )
        if report["unet_candidate_keys"] <= 0:
            examples = ", ".join(report.get("converted_key_examples") or []) or "none"
            raise ValueError(
                "Resolved LoRA did not produce any UNet adapter weights after conversion. "
                f"key_prefix_mode={prefix_mode}; converted_key_examples={examples}."
            )

        # Kohya conversion supplies network alphas. Newer serialized PEFT LoRAs
        # may instead supply adapter metadata. Diffusers rejects both at once,
        # so prefer the explicit alpha map when it exists.
        metadata_for_load = metadata if metadata and not network_alphas else None
        if prefix_mode == "component_native_unet_prefixed":
            # Component-native exports do not carry pipeline-prefixed metadata.
            # Let Diffusers derive the PEFT config from the normalized weights.
            metadata_for_load = None

        self._call_with_supported_kwargs(
            mixin.load_lora_into_unet,
            {
                "state_dict": state_dict,
                "network_alphas": network_alphas,
                "unet": getattr(components, "unet", None),
                "adapter_name": adapter_name,
                "metadata": metadata_for_load,
            },
        )
        report["unet_loaded"] = self._verify_adapter_presence(getattr(components, "unet", None), adapter_name)

        text_encoder_expected = report["text_encoder_candidate_keys"] > 0
        if text_encoder_expected:
            self._call_with_supported_kwargs(
                mixin.load_lora_into_text_encoder,
                {
                    "state_dict": state_dict,
                    "network_alphas": network_alphas,
                    "text_encoder": getattr(components, "text_encoder", None),
                    "prefix": "text_encoder",
                    "adapter_name": adapter_name,
                    "metadata": metadata_for_load,
                },
            )
            report["text_encoder_loaded"] = self._verify_adapter_presence(getattr(components, "text_encoder", None), adapter_name)
        elif report["text_encoder_2_candidate_keys"] > 0:
            report["text_encoder_loaded"] = False
            report["warning"] = (
                "Converted LoRA includes text_encoder_2 weights, but the active runtime exposes only a single text encoder."
            )
        else:
            report["text_encoder_loaded"] = True

        report["verified"] = bool(report["unet_loaded"] and report["text_encoder_loaded"])
        if not report["verified"]:
            raise ValueError(
                "LoRA conversion completed but runtime verification failed. "
                f"UNet loaded={report['unet_loaded']}, text_encoder loaded={report['text_encoder_loaded']}."
            )
        return {
            **dict(entry),
            "runtime_applied": True,
            "runtime_load_report": report,
        }

    def _deactivate_components(self, components: Any) -> None:
        for module_name in ("unet", "text_encoder"):
            module = getattr(components, module_name, None)
            if module is None:
                continue
            if hasattr(module, "set_adapters"):
                try:
                    module.set_adapters([], adapter_weights=[])
                    continue
                except TypeError:
                    try:
                        module.set_adapters([])
                        continue
                    except Exception:
                        pass
                except Exception:
                    pass
            disable = getattr(module, "disable_adapters", None)
            if callable(disable):
                try:
                    disable()
                except Exception:
                    pass

    def _activate_stack(self, components: Any, stack: list[dict[str, Any]]) -> None:
        names = [str(item.get("adapter_name") or "").strip() for item in stack if str(item.get("adapter_name") or "").strip()]
        weights = [_coerce_float(item.get("weight"), 1.0) for item in stack if str(item.get("adapter_name") or "").strip()]
        if not names:
            return
        activated_any = False
        for module_name in ("unet", "text_encoder"):
            module = getattr(components, module_name, None)
            if module is None:
                continue
            setter = getattr(module, "set_adapters", None)
            if callable(setter):
                try:
                    setter(names, adapter_weights=weights)
                    activated_any = True
                    continue
                except TypeError:
                    try:
                        setter(names, weights)
                        activated_any = True
                        continue
                    except Exception:
                        pass
                except Exception:
                    pass
            enable = getattr(module, "enable_adapters", None)
            if callable(enable):
                try:
                    enable()
                    activated_any = True
                except Exception:
                    pass
        if not activated_any:
            raise ValueError("LoRA adapters were loaded but the runtime could not activate the requested adapter stack.")
