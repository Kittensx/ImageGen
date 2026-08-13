from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

FAVORITES_SCHEMA_VERSION = 1


class UpscalerFavoriteStore:
    def __init__(self, cache_root: str | Path) -> None:
        self.path = Path(cache_root).resolve() / "upscalers" / "favorites-v1.json"
        self._lock = threading.RLock()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": FAVORITES_SCHEMA_VERSION, "favorite_upscaler_ids": [], "updated_at_utc": ""}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        values = payload.get("favorite_upscaler_ids") if isinstance(payload, Mapping) else []
        if not isinstance(values, list):
            values = []
        unique = []
        seen = set()
        for value in values:
            token = str(value or "").strip()
            if token and token not in seen:
                seen.add(token)
                unique.append(token)
        return {
            "schema_version": FAVORITES_SCHEMA_VERSION,
            "favorite_upscaler_ids": unique,
            "updated_at_utc": str(payload.get("updated_at_utc") or "") if isinstance(payload, Mapping) else "",
        }

    def _save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, prefix=".favorites-", suffix=".tmp", delete=False) as stream:
            stream.write(content)
            temp = Path(stream.name)
        temp.replace(self.path)
        return dict(payload)

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return self._load()

    def set_favorites(self, values: Iterable[str]) -> dict[str, Any]:
        unique = []
        seen = set()
        for value in values:
            token = str(value or "").strip()
            if token and token not in seen:
                seen.add(token)
                unique.append(token)
        payload = {
            "schema_version": FAVORITES_SCHEMA_VERSION,
            "favorite_upscaler_ids": unique[:512],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            return self._save(payload)


def compatible_upscaler_payload(catalog_payload: Mapping[str, Any], favorites: Mapping[str, Any], *, model_architecture: str = "") -> dict[str, Any]:
    all_items = [dict(item) for item in (catalog_payload.get("neural") or []) if isinstance(item, Mapping)]
    # The qualified Phase 14N families are architecture-independent with respect to
    # the diffusion checkpoint. Compatibility is still decided here, server-side,
    # so later model-specific constraints can be added without changing clients.
    compatible = [item for item in all_items if bool(item.get("selectable"))]
    favorite_ids = {str(item) for item in (favorites.get("favorite_upscaler_ids") or [])}
    compatible_ids = {str(item.get("upscaler_id") or item.get("id") or "") for item in compatible}
    favorite_compatible = [item for item in compatible if str(item.get("upscaler_id") or item.get("id") or "") in favorite_ids]
    hidden = sorted(item for item in favorite_ids if item and item not in compatible_ids)
    return {
        "modelArchitecture": str(model_architecture or ""),
        "allInstalled": all_items,
        "compatible": compatible,
        "favoriteCompatible": favorite_compatible,
        "incompatibleFavoriteCount": len(hidden),
        "hiddenFavoriteIds": hidden,
    }
