from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

DEFAULT_ASSET_CONTRACT_VERSION = "image-gen-default-assets-v1"
SUPPORTED_ASSET_TYPES = {"lora", "textual_inversion"}
SUPPORTED_POLARITIES = {"positive", "negative"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_path(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        return os.path.normcase(str(Path(text).expanduser().resolve()))
    except OSError:
        return os.path.normcase(text.replace("\\", "/"))


def normalize_model_family(value: Any) -> str:
    token = _text(value).lower().replace("stable diffusion", "sd")
    token = token.replace("_", " ").replace("-", " ")
    token = " ".join(token.split())
    aliases = {
        "sd 1": "sd1.x",
        "sd 1.x": "sd1.x",
        "sd1": "sd1.x",
        "sd1.x": "sd1.x",
        "1.5": "sd1.x",
        "sd 1.5": "sd1.x",
        "sd15": "sd1.x",
        "sd 2": "sd2.x",
        "sd 2.x": "sd2.x",
        "sd2": "sd2.x",
        "sd2.x": "sd2.x",
        "2.0": "sd2.x",
        "2.1": "sd2.x",
        "sd 2.0": "sd2.x",
        "sd 2.1": "sd2.x",
        "sd20": "sd2.x",
        "sd21": "sd2.x",
        "sdxl": "sdxl",
        "sd xl": "sdxl",
        "any": "any",
        "all": "any",
        "unknown": "",
    }
    return aliases.get(token, token.replace(" ", ""))


def model_identity(model: Mapping[str, Any] | None) -> dict[str, str]:
    source = dict(model or {})
    path = _text(source.get("resolved_path") or source.get("model_path") or source.get("path"))
    name = _text(source.get("model_name") or source.get("name") or (Path(path).stem if path else ""))
    family = normalize_model_family(
        source.get("architecture")
        or (source.get("architecture_contract") or {}).get("family")
        or source.get("model_family")
    )
    normalized_path = _normalize_path(path)
    key_source = normalized_path or name.casefold()
    key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:20] if key_source else ""
    return {
        "model_key": key,
        "model_path": path,
        "model_name": name,
        "model_family": family,
    }


def _asset_identity(asset: Mapping[str, Any]) -> str:
    asset_type = _text(asset.get("asset_type") or "lora").lower()
    polarity = _text(asset.get("polarity") or "positive").lower()
    path = _normalize_path(asset.get("path"))
    name = _text(asset.get("name")).casefold()
    activation = _text(asset.get("activation_text")).casefold()
    basis = path or name or activation
    return f"{asset_type}|{polarity}|{basis}"


def normalize_asset(value: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(value or {})
    asset_type = _text(source.get("asset_type") or "lora").lower()
    if asset_type not in SUPPORTED_ASSET_TYPES:
        asset_type = "lora"
    polarity = _text(source.get("polarity") or "positive").lower()
    if polarity not in SUPPORTED_POLARITIES:
        polarity = "positive"
    name = _text(source.get("name") or source.get("display_name") or source.get("filename"))
    path = _text(source.get("path") or source.get("resolved_path"))
    if not name and path:
        name = Path(path).stem
    try:
        weight = float(source.get("weight", 1.0) if source.get("weight") is not None else 1.0)
    except (TypeError, ValueError):
        weight = 1.0
    weight = max(-4.0, min(4.0, weight))
    normalized = {
        "asset_id": _text(source.get("asset_id")),
        "asset_type": asset_type,
        "polarity": polarity,
        "name": name or "Unnamed asset",
        "path": path,
        "weight": weight,
        "enabled": bool(source.get("enabled", True)),
        "activation_text": _text(source.get("activation_text")),
        "model_family": normalize_model_family(source.get("model_family")),
        "source_url": _text(source.get("source_url")),
        "preview_path": _text(source.get("preview_path")),
        "catalog_asset_id": _text(source.get("catalog_asset_id")),
        "notes": _text(source.get("notes")),
    }
    if not normalized["asset_id"]:
        normalized["asset_id"] = hashlib.sha256(_asset_identity(normalized).encode("utf-8")).hexdigest()[:16]
    return normalized


def normalize_profile(value: Mapping[str, Any] | None, *, fallback_id: str, model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = dict(value or {})
    identity = model_identity(model or source)
    assets = [normalize_asset(item) for item in source.get("assets", []) if isinstance(item, Mapping)]
    return {
        "profile_id": _text(source.get("profile_id") or fallback_id),
        "display_name": _text(source.get("display_name") or (identity["model_name"] if identity["model_name"] else "Global defaults")),
        "model_key": _text(source.get("model_key") or identity["model_key"]),
        "model_path": _text(source.get("model_path") or identity["model_path"]),
        "model_name": _text(source.get("model_name") or identity["model_name"]),
        "model_family": normalize_model_family(source.get("model_family") or identity["model_family"]),
        "assets": assets,
    }


def default_document() -> dict[str, Any]:
    return {
        "contract_version": DEFAULT_ASSET_CONTRACT_VERSION,
        "apply_saved_defaults": False,
        "auto_apply_on_model_load": True,
        "global_profile": normalize_profile({}, fallback_id="global"),
        "model_profiles": {},
    }


def normalize_document(value: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(value or {})
    output = default_document()
    output["apply_saved_defaults"] = bool(source.get("apply_saved_defaults", False))
    output["auto_apply_on_model_load"] = bool(source.get("auto_apply_on_model_load", True))
    output["global_profile"] = normalize_profile(source.get("global_profile"), fallback_id="global")
    raw_models = source.get("model_profiles") if isinstance(source.get("model_profiles"), Mapping) else {}
    models: dict[str, dict[str, Any]] = {}
    for raw_key, raw_profile in raw_models.items():
        if not isinstance(raw_profile, Mapping):
            continue
        profile = normalize_profile(raw_profile, fallback_id=_text(raw_key) or "model")
        key = profile["model_key"] or _text(raw_key)
        if key:
            profile["profile_id"] = profile["profile_id"] or key
            models[key] = profile
    output["model_profiles"] = models
    return output


def _compatible(asset_family: str, model_family: str) -> tuple[bool, str]:
    asset_family = normalize_model_family(asset_family)
    model_family = normalize_model_family(model_family)
    if not asset_family or asset_family == "any":
        return True, "Asset does not restrict the model family."
    if not model_family:
        return True, "Model family is not known yet; compatibility is pending."
    if asset_family == model_family:
        return True, f"Compatible with {model_family}."
    return False, f"Asset targets {asset_family}, but the active checkpoint is {model_family}."


def resolve_default_assets(document: Mapping[str, Any] | None, model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_document(document)
    identity = model_identity(model)
    global_profile = normalized["global_profile"]
    selected_profile: dict[str, Any] | None = None
    if identity["model_key"]:
        selected_profile = normalized["model_profiles"].get(identity["model_key"])
    if selected_profile is None and identity["model_name"]:
        for profile in normalized["model_profiles"].values():
            if _text(profile.get("model_name")).casefold() == identity["model_name"].casefold():
                selected_profile = profile
                break

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for scope, profile in (("global", global_profile), ("model", selected_profile)):
        if not profile:
            continue
        for item in profile.get("assets", []):
            asset = normalize_asset(item)
            key = _asset_identity(asset)
            if key not in merged:
                order.append(key)
            merged[key] = {**asset, "source_scope": scope, "source_profile_id": profile.get("profile_id", scope)}

    enabled: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    incompatible: list[dict[str, Any]] = []
    for key in order:
        asset = merged[key]
        if not asset.get("enabled", True):
            disabled.append(asset)
            continue
        compatible, message = _compatible(asset.get("model_family", ""), identity["model_family"])
        record = {**asset, "compatible": compatible, "compatibility_message": message}
        if compatible:
            enabled.append(record)
        else:
            incompatible.append(record)

    counts = {
        "total": len(enabled),
        "loras": sum(1 for item in enabled if item["asset_type"] == "lora"),
        "textual_inversions": sum(1 for item in enabled if item["asset_type"] == "textual_inversion"),
        "positive": sum(1 for item in enabled if item["polarity"] == "positive"),
        "negative": sum(1 for item in enabled if item["polarity"] == "negative"),
        "incompatible": len(incompatible),
        "disabled": len(disabled),
    }
    return {
        "contract_version": DEFAULT_ASSET_CONTRACT_VERSION,
        "profiles": deepcopy(normalized),
        "active_model": identity,
        "global_profile": deepcopy(global_profile),
        "model_profile": deepcopy(selected_profile) if selected_profile else None,
        "effective_assets": enabled,
        "incompatible_assets": incompatible,
        "disabled_assets": disabled,
        "counts": counts,
        "apply_saved_defaults": normalized["apply_saved_defaults"],
        "auto_apply_on_model_load": normalized["auto_apply_on_model_load"],
    }


__all__ = [
    "DEFAULT_ASSET_CONTRACT_VERSION",
    "default_document",
    "model_identity",
    "normalize_asset",
    "normalize_document",
    "normalize_model_family",
    "normalize_profile",
    "resolve_default_assets",
]
