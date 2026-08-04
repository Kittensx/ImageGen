from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from image_gen.contracts import PromptAssetSelection, normalize_prompt_asset_list

_INLINE_LORA_PATTERN = re.compile(r"<lora:([^:>]+?)(?::([-+]?\d*\.?\d+))?>", re.IGNORECASE)


def _coerce_weight(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def extract_inline_loras_from_prompts(*texts: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for text in texts:
        source = str(text or "")
        if not source:
            continue
        for match in _INLINE_LORA_PATTERN.finditer(source):
            name = str(match.group(1) or "").strip()
            if not name:
                continue
            output.append({
                "asset_type": "lora",
                "name": name,
                "path": "",
                "weight": _coerce_weight(match.group(2), 1.0),
                "enabled": True,
                "polarity": "positive",
                "activation_text": "",
                "source": "inline_syntax",
                "original_source": "",
            })
    return output


def _asset_aliases(item: Mapping[str, Any]) -> list[str]:
    values = [
        item.get("resolved_hash"),
        item.get("requested_hash"),
        item.get("file_hash"),
        item.get("resolved_path"),
        item.get("path"),
        item.get("requested_path"),
        item.get("catalog_asset_id"),
        item.get("asset_id"),
        item.get("name"),
        item.get("requested_name"),
    ]
    output: list[str] = []
    for value in values:
        token = str(value or "").strip().casefold()
        if token and token not in output:
            output.append(token)
        if token and ('/' in token or '\\' in token):
            stem = Path(token).stem.casefold()
            if stem and stem not in output:
                output.append(stem)
    return output


def merge_replay_loras(
    recorded: Iterable[Any] | None,
    inline_records: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized = normalize_prompt_asset_list(
        list(recorded or []),
        asset_type="lora",
        default_source="replay",
    )
    merged: list[dict[str, Any]] = []
    index: dict[str, int] = {}

    for asset in normalized:
        original_source = asset.original_source or asset.source
        asset.source = "replay"
        asset.original_source = "" if original_source == "replay" else original_source
        payload = asset.to_serializable_dict()
        merged.append(payload)
        for key in _asset_aliases(payload):
            index[key] = len(merged) - 1

    for item in inline_records or []:
        inline_asset = PromptAssetSelection.from_value(
            dict(item),
            asset_type="lora",
            default_source="inline_syntax",
        )
        payload = inline_asset.to_serializable_dict()
        payload["source"] = "inline_syntax"
        payload.setdefault("enabled", True)
        payload.setdefault("polarity", "positive")
        match = next((index[key] for key in _asset_aliases(payload) if key in index), None)
        if match is not None:
            existing = dict(merged[match])
            original_source = (
                existing.get("original_source")
                or existing.get("source")
                or payload.get("original_source")
                or ""
            )
            existing.update({
                key: value for key, value in payload.items()
                if value not in (None, "")
            })
            existing["source"] = "inline_syntax"
            existing["original_source"] = "" if original_source == "inline_syntax" else original_source
            merged[match] = existing
            for key in _asset_aliases(existing):
                index[key] = match
        else:
            payload["source"] = "inline_syntax"
            payload["original_source"] = payload.get("original_source") or ""
            merged.append(payload)
            position = len(merged) - 1
            for key in _asset_aliases(payload):
                index[key] = position

    return merged
