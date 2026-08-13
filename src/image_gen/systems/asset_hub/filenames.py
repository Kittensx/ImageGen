from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(value: str, *, fallback: str = "asset.bin", max_length: int = 180) -> str:
    """Return a Windows-safe basename while preserving a useful extension.

    Provider filenames are untrusted. This function deliberately returns only a
    basename; path separators, device names, trailing dots/spaces, and control
    characters are removed or replaced.
    """

    raw = Path(str(value or "").replace("\\", "/")).name.strip()
    cleaned = _INVALID.sub("_", raw).rstrip(" .")
    if cleaned in {"", ".", ".."}:
        cleaned = fallback

    stem = Path(cleaned).stem.rstrip(" .") or "asset"
    suffix = Path(cleaned).suffix
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"

    limit = max(32, int(max_length))
    suffix = suffix[: min(len(suffix), 32)]
    allowed_stem = max(1, limit - len(suffix))
    stem = stem[:allowed_stem].rstrip(" .") or "asset"
    cleaned = f"{stem}{suffix}".rstrip(" .")
    return cleaned or fallback
