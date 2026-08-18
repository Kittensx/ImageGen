from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from PIL import Image


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _slug(value: Any) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or ""))
    return "-".join(part for part in text.split("-") if part) or "case"

def _safe_folder_name(value: Any) -> str:
    raw = Path(str(value or "model")).stem
    cleaned = []
    last_was_sep = False
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
            last_was_sep = False
        else:
            if not last_was_sep:
                cleaned.append("_")
                last_was_sep = True
    name = "".join(cleaned).strip("_")
    return name or "model"

def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _pixel_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        with Image.open(path) as image:
            normalized = image.convert("RGBA")
            digest = hashlib.sha256()
            digest.update(str(normalized.size).encode("ascii"))
            digest.update(normalized.tobytes())
            return digest.hexdigest()
    except Exception:
        return ""

def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return dict(payload)

def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_json_safe(dict(payload)), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
