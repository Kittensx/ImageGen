from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from modules.project_context import ProjectContext


_LORA_LIBRARY_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".pth", ".bin"}
_SPECIALIZED_ROOT_KEYS: tuple[tuple[str, str], ...] = (
    ("a1111_lora_roots", "a1111"),
    ("automatic1111_lora_roots", "a1111"),
    ("comfyui_lora_roots", "comfyui"),
    ("shared_lora_roots", "shared"),
    ("external_lora_roots", "external"),
)


@dataclass(frozen=True)
class LoRALibraryRoot:
    root_id: str
    path: Path
    root_mode: str
    source_hint: str
    source_config_key: str
    exists: bool
    asset_types: tuple[str, ...]
    is_managed: bool
    is_external: bool
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "path": str(self.path),
            "root_mode": self.root_mode,
            "source_hint": self.source_hint,
            "source_config_key": self.source_config_key,
            "exists": self.exists,
            "asset_types": list(self.asset_types),
            "is_managed": self.is_managed,
            "is_external": self.is_external,
            "label": self.label,
        }


def _stable_root_id(*parts: str) -> str:
    import hashlib

    token = "|".join(str(part or "").strip().lower() for part in parts)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _normalize_asset_types(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray, str, Mapping)):
        values = list(raw)
    else:
        values = []
    normalized: list[str] = []
    for value in values:
        token = str(value or "").strip().lower().replace("-", "_")
        if not token:
            continue
        if token in {"lora", "loras", "adapter", "adapters"}:
            normalized.append("lora")
        else:
            normalized.append(token)
    seen: set[str] = set()
    ordered: list[str] = []
    for token in normalized:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered)


def _entry_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("path") or "")
    return ""


def _entry_enabled(value: Any) -> bool:
    if isinstance(value, Mapping) and value.get("enabled") is False:
        return False
    return True


def _entry_mode(value: Any, *, default_mode: str) -> str:
    if not isinstance(value, Mapping):
        return default_mode
    mode = str(value.get("mode") or default_mode).strip().lower()
    if mode in {"managed", "scan_only", "external", "specialized"}:
        return mode
    return default_mode


def _entry_source_hint(value: Any, *, default_hint: str) -> str:
    if not isinstance(value, Mapping):
        return default_hint
    hint = str(value.get("source_hint") or default_hint).strip().lower()
    return hint or default_hint


def _entry_label(value: Any, *, fallback: str) -> str:
    if isinstance(value, Mapping):
        label = str(value.get("label") or value.get("name") or "").strip()
        if label:
            return label
    return fallback


def _entry_asset_types(value: Any, *, default_types: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return default_types
    parsed = _normalize_asset_types(value.get("asset_types") or value.get("asset_type"))
    return parsed or default_types


def _applies_to_lora(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    asset_types = _entry_asset_types(value, default_types=("lora",))
    return "lora" in asset_types or not asset_types


def _root_record(
    *,
    context: ProjectContext,
    path_text: str,
    root_mode: str,
    source_hint: str,
    source_config_key: str,
    asset_types: tuple[str, ...],
    label: str,
) -> LoRALibraryRoot:
    candidate = context.resolve_project_path(path_text)
    resolved = candidate.expanduser().resolve(strict=False)
    normalized_mode = root_mode if root_mode in {"managed", "scan_only", "external", "specialized"} else "scan_only"
    effective_hint = source_hint or ("managed" if normalized_mode == "managed" else "configured")
    return LoRALibraryRoot(
        root_id=_stable_root_id(normalized_mode, effective_hint, source_config_key, str(resolved)),
        path=resolved,
        root_mode=normalized_mode,
        source_hint=effective_hint,
        source_config_key=source_config_key,
        exists=resolved.exists() and resolved.is_dir(),
        asset_types=asset_types,
        is_managed=normalized_mode == "managed",
        is_external=normalized_mode != "managed",
        label=label or resolved.name or str(resolved),
    )


def discover_lora_library_roots(context: ProjectContext) -> dict[str, Any]:
    model_library = context.config.get("model_library") or {}
    configured: list[LoRALibraryRoot] = []
    diagnostics: list[dict[str, Any]] = []

    managed_config = model_library.get("managed_roots") if isinstance(model_library.get("managed_roots"), Mapping) else {}
    managed_path = str((managed_config or {}).get("Lora") or (managed_config or {}).get("lora") or context.lora_dir)
    configured.append(
        _root_record(
            context=context,
            path_text=managed_path,
            root_mode="managed",
            source_hint="image_gen",
            source_config_key="managed_roots.Lora",
            asset_types=("lora",),
            label="IMAGE_GEN LoRA Library",
        )
    )

    def _consume_entries(raw_entries: Any, *, source_config_key: str, default_hint: str, default_mode: str = "scan_only") -> None:
        if isinstance(raw_entries, (str, Mapping)):
            entries = [raw_entries]
        elif isinstance(raw_entries, Iterable) and not isinstance(raw_entries, (bytes, bytearray, str, Mapping)):
            entries = list(raw_entries)
        else:
            entries = []
        for index, entry in enumerate(entries):
            if not _entry_enabled(entry):
                continue
            if not _applies_to_lora(entry):
                continue
            path_text = _entry_path(entry)
            if not path_text.strip():
                continue
            hint = _entry_source_hint(entry, default_hint=default_hint)
            mode = _entry_mode(entry, default_mode=default_mode)
            label = _entry_label(entry, fallback=f"{source_config_key}[{index}]")
            configured.append(
                _root_record(
                    context=context,
                    path_text=path_text,
                    root_mode=mode,
                    source_hint=hint,
                    source_config_key=source_config_key,
                    asset_types=_entry_asset_types(entry, default_types=("lora",)),
                    label=label,
                )
            )

    _consume_entries(model_library.get("lora_scan_roots"), source_config_key="lora_scan_roots", default_hint="configured")
    _consume_entries(model_library.get("additional_scan_roots"), source_config_key="additional_scan_roots", default_hint="configured")
    for key, hint in _SPECIALIZED_ROOT_KEYS:
        _consume_entries(model_library.get(key), source_config_key=key, default_hint=hint, default_mode="specialized")

    unique: list[LoRALibraryRoot] = []
    seen_paths: dict[str, LoRALibraryRoot] = {}
    for root in configured:
        token = str(root.path).casefold()
        previous = seen_paths.get(token)
        if previous is not None:
            diagnostics.append({
                "severity": "warning",
                "code": "duplicate_root",
                "message": f"Duplicate LoRA root ignored: {root.path}",
                "path": str(root.path),
                "root_id": root.root_id,
                "replaced_by": previous.root_id,
            })
            continue
        seen_paths[token] = root
        unique.append(root)

    for root in unique:
        if not root.exists:
            diagnostics.append({
                "severity": "warning",
                "code": "missing_root",
                "message": f"Configured LoRA root does not exist on disk: {root.path}",
                "path": str(root.path),
                "root_id": root.root_id,
            })

    for index, left in enumerate(unique):
        for right in unique[index + 1:]:
            left_path = left.path
            right_path = right.path
            if left_path == right_path:
                continue
            try:
                if right_path.is_relative_to(left_path):
                    diagnostics.append({
                        "severity": "info",
                        "code": "nested_root",
                        "message": f"LoRA root {right_path} is nested inside {left_path}.",
                        "path": str(right_path),
                        "root_id": right.root_id,
                        "parent_root_id": left.root_id,
                    })
                elif left_path.is_relative_to(right_path):
                    diagnostics.append({
                        "severity": "info",
                        "code": "nested_root",
                        "message": f"LoRA root {left_path} is nested inside {right_path}.",
                        "path": str(left_path),
                        "root_id": left.root_id,
                        "parent_root_id": right.root_id,
                    })
            except Exception:
                continue

    roots_payload = [root.to_dict() for root in unique]
    summary = {
        "count": len(unique),
        "managed_count": sum(1 for root in unique if root.is_managed),
        "external_count": sum(1 for root in unique if root.is_external),
        "missing_count": sum(1 for root in unique if not root.exists),
    }
    return {
        "roots": roots_payload,
        "diagnostics": diagnostics,
        "summary": summary,
    }


def scan_lora_library_files(roots: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        root_path = Path(str(root.get("path") or "")).expanduser().resolve(strict=False)
        if not root_path.exists() or not root_path.is_dir():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _LORA_LIBRARY_EXTENSIONS:
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
            try:
                relative_path = str(resolved.relative_to(root_path)).replace("\\", "/")
            except ValueError:
                relative_path = resolved.name
            candidates.append(
                {
                    "name": resolved.stem,
                    "path": str(resolved),
                    "extension": resolved.suffix.lower(),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified_ns": int(stat.st_mtime_ns),
                    "source_root_id": str(root.get("root_id") or ""),
                    "source_root_path": str(root_path),
                    "source_root_mode": str(root.get("root_mode") or "scan_only"),
                    "source_root_source_hint": str(root.get("source_hint") or ""),
                    "source_root_label": str(root.get("label") or root_path.name or str(root_path)),
                    "source_config_key": str(root.get("source_config_key") or ""),
                    "is_managed_library_asset": bool(root.get("is_managed", False)),
                    "is_external_library_asset": bool(root.get("is_external", False)),
                    "root_id": str(root.get("root_id") or ""),
                    "root_mode": str(root.get("root_mode") or "scan_only"),
                    "source_hint": str(root.get("source_hint") or ""),
                    "is_managed": bool(root.get("is_managed", False)),
                    "is_external": bool(root.get("is_external", False)),
                    "managed_vs_external": "managed" if bool(root.get("is_managed", False)) else "external",
                    "discovered_via": str(root.get("source_config_key") or root.get("source_hint") or "configured_root"),
                    "relative_path": relative_path,
                }
            )
    return sorted(candidates, key=lambda item: (str(item.get("name") or "").casefold(), str(item.get("path") or "").casefold()))


def summarize_lora_scan_roots(roots: Iterable[Mapping[str, Any]], files: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    payload_roots = [dict(root) for root in roots]
    counts = Counter(str(item.get("source_root_id") or "") for item in files)
    for root in payload_roots:
        root["discovered_count"] = int(counts.get(str(root.get("root_id") or ""), 0))
    summary = {
        "count": len(payload_roots),
        "managed_count": sum(1 for root in payload_roots if root.get("is_managed")),
        "external_count": sum(1 for root in payload_roots if root.get("is_external")),
        "missing_count": sum(1 for root in payload_roots if not root.get("exists")),
        "discovered_asset_count": sum(int(root.get("discovered_count") or 0) for root in payload_roots),
    }
    return {"roots": payload_roots, "summary": summary}


__all__ = [
    "discover_lora_library_roots",
    "scan_lora_library_files",
    "summarize_lora_scan_roots",
]
