from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from image_gen.webui.asset_metadata import load_asset_metadata, save_asset_sidecar_fields, sidecar_path


def write_provenance_sidecar(asset_path: str | Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(metadata)
    identity = {
        "source_provider": str(source.get("provider") or ""),
        "source_asset_id": str(source.get("provider_model_id") or ""),
        "source_version_id": str(source.get("provider_model_version_id") or ""),
        "source_file_id": str(source.get("provider_file_id") or ""),
        "source_url": str(source.get("source_page_url") or ""),
        "file_sha256": str(source.get("sha256") or source.get("verified_sha256") or ""),
    }
    return save_asset_sidecar_fields(asset_path, {
        "_asset_hub_source": identity,
        "asset_hub": source,
    })


def read_provenance_sidecar(asset_path: str | Path) -> dict[str, Any]:
    payload = load_asset_metadata(asset_path)
    value = payload.get("asset_hub")
    return dict(value) if isinstance(value, Mapping) else {}


def provenance_sidecar_path(asset_path: str | Path) -> Path:
    return sidecar_path(asset_path)
