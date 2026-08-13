from __future__ import annotations

from typing import Any, Mapping


def build_card_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(metadata or {})
    description = str(source.get("short_description") or source.get("description_plaintext") or "").strip()
    return {
        "title": str(source.get("title") or source.get("display_name") or source.get("filename") or "Asset"),
        "author": str(source.get("author_name") or source.get("provider_creator_name") or ""),
        "description": description[:320],
        "source": str(source.get("source_platform_name") or source.get("provider") or ""),
        "assetKind": str(source.get("asset_kind") or "other"),
        "baseModel": str(source.get("base_model") or ""),
        "versionName": str(source.get("version_name") or ""),
        "installedAt": str(source.get("installed_at_utc") or ""),
    }
