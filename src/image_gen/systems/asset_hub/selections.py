from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from image_gen.systems.asset_hub.providers.base import AssetHubError

_PURPOSE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


def _text(value: Any, *, limit: int = 1024) -> str:
    return str(value or "").strip()[:limit]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AssetHubSelection:
    purpose: str
    provider_id: str
    remote_model_id: str
    remote_version_id: str
    remote_file_id: str
    asset_kind: str = ""
    architecture: str = ""
    base_model: str = ""
    model_name: str = ""
    version_name: str = ""
    file_name: str = ""
    sha256: str = ""
    source_url: str = ""
    selected_at: str = ""
    schema_version: int = 1

    @classmethod
    def from_mapping(cls, purpose: str, value: Mapping[str, Any]) -> "AssetHubSelection":
        token = str(purpose or "").strip().casefold()
        if not _PURPOSE_RE.fullmatch(token):
            raise AssetHubError("selection_purpose_invalid", "Asset selection purpose is invalid.", status_code=400)
        provider_id = _text(value.get("providerId") or value.get("provider_id"), limit=64).casefold()
        model_id = _text(value.get("remoteModelId") or value.get("remote_model_id"), limit=128)
        version_id = _text(value.get("remoteVersionId") or value.get("remote_version_id"), limit=128)
        file_id = _text(value.get("remoteFileId") or value.get("remote_file_id"), limit=128)
        if not provider_id or not model_id or not version_id or not file_id:
            raise AssetHubError(
                "selection_identity_incomplete",
                "Provider, model, version, and file identities are required for an asset selection.",
                status_code=400,
            )
        sha256 = _text(value.get("sha256"), limit=64).casefold()
        if sha256 and (len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256)):
            raise AssetHubError("selection_hash_invalid", "Selection SHA-256 must be 64 hexadecimal characters.", status_code=400)
        return cls(
            purpose=token,
            provider_id=provider_id,
            remote_model_id=model_id,
            remote_version_id=version_id,
            remote_file_id=file_id,
            asset_kind=_text(value.get("assetKind") or value.get("asset_kind"), limit=64).casefold(),
            architecture=_text(value.get("architecture"), limit=128).casefold(),
            base_model=_text(value.get("baseModel") or value.get("base_model"), limit=256),
            model_name=_text(value.get("modelName") or value.get("model_name"), limit=512),
            version_name=_text(value.get("versionName") or value.get("version_name"), limit=512),
            file_name=Path(_text(value.get("fileName") or value.get("file_name"), limit=1024)).name,
            sha256=sha256,
            source_url=_text(value.get("sourceUrl") or value.get("source_url"), limit=2048),
            selected_at=_text(value.get("selectedAt") or value.get("selected_at"), limit=128) or _utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "schemaVersion": payload.pop("schema_version"),
            "purpose": payload.pop("purpose"),
            "providerId": payload.pop("provider_id"),
            "remoteModelId": payload.pop("remote_model_id"),
            "remoteVersionId": payload.pop("remote_version_id"),
            "remoteFileId": payload.pop("remote_file_id"),
            "assetKind": payload.pop("asset_kind"),
            "architecture": payload.pop("architecture"),
            "baseModel": payload.pop("base_model"),
            "modelName": payload.pop("model_name"),
            "versionName": payload.pop("version_name"),
            "fileName": payload.pop("file_name"),
            "sha256": payload.pop("sha256"),
            "sourceUrl": payload.pop("source_url"),
            "selectedAt": payload.pop("selected_at"),
        }


class AssetHubSelectionStore:
    """Small provider-neutral store for user-selected remote asset identities.

    The store persists only public provider identity/metadata. It never stores
    credentials, signed delivery URLs, cookies, or arbitrary provider payloads.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()

    def _path(self, purpose: str) -> Path:
        token = str(purpose or "").strip().casefold()
        if not _PURPOSE_RE.fullmatch(token):
            raise AssetHubError("selection_purpose_invalid", "Asset selection purpose is invalid.", status_code=400)
        return self.root / f"{token}.json"

    def get(self, purpose: str) -> AssetHubSelection | None:
        path = self._path(purpose)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetHubError("selection_store_corrupt", f"Unable to read asset selection: {exc}", status_code=500) from exc
        if not isinstance(payload, Mapping):
            raise AssetHubError("selection_store_corrupt", "Stored asset selection is invalid.", status_code=500)
        return AssetHubSelection.from_mapping(purpose, payload)

    def set(self, purpose: str, value: Mapping[str, Any]) -> AssetHubSelection:
        selection = AssetHubSelection.from_mapping(purpose, value)
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(purpose)
        serialized = json.dumps(selection.to_dict(), indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.root),
            prefix=f".{target.stem}-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(serialized)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(target)
        return selection

    def delete(self, purpose: str) -> bool:
        path = self._path(purpose)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False


__all__ = ["AssetHubSelection", "AssetHubSelectionStore"]
