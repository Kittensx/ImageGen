from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from image_gen.systems.asset_hub.card_summary import build_card_summary

INDEX_SCHEMA_VERSION = 1


class InstalledAssetMetadataIndex:
    def __init__(self, cache_root: str | os.PathLike[str]) -> None:
        self.path = Path(cache_root).resolve() / "asset-hub" / "installed-assets-v1.json"
        self._lock = threading.RLock()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schemaVersion": INDEX_SCHEMA_VERSION, "assets": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schemaVersion": INDEX_SCHEMA_VERSION, "assets": {}}
        assets = payload.get("assets") if isinstance(payload, Mapping) else {}
        return {"schemaVersion": INDEX_SCHEMA_VERSION, "assets": dict(assets) if isinstance(assets, Mapping) else {}}

    def _save(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, prefix=".installed-assets-", suffix=".tmp", delete=False) as stream:
            stream.write(serialized)
            temp = Path(stream.name)
        temp.replace(self.path)

    def upsert(self, install_id: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = self._load()
            record = dict(metadata)
            record["card"] = build_card_summary(record)
            payload["assets"][str(install_id)] = record
            self._save(payload)
            return record

    def remove(self, install_id: str) -> None:
        with self._lock:
            payload = self._load()
            payload["assets"].pop(str(install_id), None)
            self._save(payload)

    def get(self, install_id: str) -> dict[str, Any] | None:
        record = self._load()["assets"].get(str(install_id))
        return dict(record) if isinstance(record, Mapping) else None

    def list(self) -> list[dict[str, Any]]:
        values = self._load()["assets"].values()
        return [dict(item) for item in values if isinstance(item, Mapping)]
