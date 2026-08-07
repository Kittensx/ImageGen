from __future__ import annotations

import re
from typing import Any

from image_gen.systems.upscaling.contracts import UpscalerDiscoveryResult

MAX_UPSCALER_ERROR_CHARS = 512
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def bounded_error_text(value: Any, *, limit: int = MAX_UPSCALER_ERROR_CHARS) -> str:
    text = _ANSI_ESCAPE.sub("", str(value or ""))
    text = _CONTROL_CHARS.sub(" ", text)
    text = " ".join(text.split())
    safe_limit = max(32, int(limit))
    if len(text) <= safe_limit:
        return text
    return text[: max(0, safe_limit - 3)].rstrip() + "..."


def summarize_discovery(result: UpscalerDiscoveryResult) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    cache_statuses: dict[str, int] = {}
    architectures: dict[str, int] = {}
    for descriptor in result.neural_descriptors:
        statuses[descriptor.load_status] = statuses.get(descriptor.load_status, 0) + 1
        cache_statuses[descriptor.scan_cache_status] = (
            cache_statuses.get(descriptor.scan_cache_status, 0) + 1
        )
        architectures[descriptor.architecture] = architectures.get(descriptor.architecture, 0) + 1
    return {
        "mode": result.mode,
        "root_count": len(result.roots),
        "file_count": len(result.neural_descriptors),
        "supported_count": len(result.supported_neural),
        "unavailable_count": len(result.unavailable_neural),
        "status_counts": dict(sorted(statuses.items())),
        "cache_status_counts": dict(sorted(cache_statuses.items())),
        "architecture_counts": dict(sorted(architectures.items())),
        "diagnostic_count": len(result.diagnostics),
        "cache_path": result.cache_path,
    }
