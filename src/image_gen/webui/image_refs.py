from __future__ import annotations

import base64
from pathlib import Path

_EXTERNAL_PREFIX = "external::"


def is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def encode_external_image_ref(path: Path) -> str:
    raw = str(Path(path).expanduser().resolve()).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{_EXTERNAL_PREFIX}{token}"



def decode_external_image_ref(reference: str) -> Path | None:
    if not isinstance(reference, str) or not reference.startswith(_EXTERNAL_PREFIX):
        return None
    token = reference[len(_EXTERNAL_PREFIX):].strip()
    if not token:
        return None
    padded = token + ("=" * (-len(token) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    try:
        return Path(decoded).expanduser().resolve()
    except OSError:
        return None
