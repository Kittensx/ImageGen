from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from image_gen.webui.asset_metadata import load_asset_metadata, save_asset_sidecar_fields, sidecar_path


def _initial_card_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    triggers = source.get("trigger_words") or source.get("trained_words") or []
    if isinstance(triggers, str):
        trigger_values = [triggers]
    elif isinstance(triggers, (list, tuple)):
        trigger_values = [str(value).strip() for value in triggers if str(value).strip()]
    else:
        trigger_values = []
    classification = source.get("classification_result")
    classification = dict(classification) if isinstance(classification, Mapping) else {}
    family = str(source.get("model_family") or classification.get("architecture") or "").strip()
    values = {
        "source_url": str(source.get("source_page_url") or "").strip(),
        "description": str(source.get("description_plaintext") or source.get("short_description") or "").strip(),
        "tags": [str(value).strip() for value in (source.get("tags") or []) if str(value).strip()],
        "model_family": family,
        "architecture": str(classification.get("architecture") or family).strip(),
        "activation_text": trigger_values[0] if trigger_values else "",
    }
    return {key: value for key, value in values.items() if value not in ("", [], None)}


def write_provenance_sidecar(
    asset_path: str | Path,
    metadata: Mapping[str, Any],
    *,
    hydrate_card_fields: bool = False,
) -> dict[str, Any]:
    source = dict(metadata)
    identity = {
        "source_provider": str(source.get("provider") or ""),
        "source_asset_id": str(source.get("provider_model_id") or ""),
        "source_version_id": str(source.get("provider_model_version_id") or ""),
        "source_file_id": str(source.get("provider_file_id") or ""),
        "source_url": str(source.get("source_page_url") or ""),
        "file_sha256": str(source.get("sha256") or source.get("verified_sha256") or ""),
    }
    payload: dict[str, Any] = {
        "_asset_hub_source": identity,
        "asset_hub": source,
    }
    if hydrate_card_fields:
        current = load_asset_metadata(asset_path)
        # Provider metadata seeds a newly downloaded card, but never overwrites
        # user-owned editable fields on later metadata refreshes.
        for key, value in _initial_card_fields(source).items():
            if key not in current:
                payload[key] = value
    return save_asset_sidecar_fields(asset_path, payload)


def read_provenance_sidecar(asset_path: str | Path) -> dict[str, Any]:
    payload = load_asset_metadata(asset_path)
    value = payload.get("asset_hub")
    return dict(value) if isinstance(value, Mapping) else {}


def provenance_sidecar_path(asset_path: str | Path) -> Path:
    return sidecar_path(asset_path)
