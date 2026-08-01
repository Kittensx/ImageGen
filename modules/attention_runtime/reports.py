from __future__ import annotations

import hashlib
import json
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe representation for diagnostics."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def normalized_json(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(normalized_json(value).encode("utf-8")).hexdigest()


def shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except Exception:
        return None


def tensor_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    shape = shape_list(value)
    if shape is None:
        return {
            "python_type": f"{type(value).__module__}.{type(value).__name__}",
        }
    return {
        "shape": shape,
        "dtype": str(getattr(value, "dtype", "unknown")).replace("torch.", ""),
        "device": str(getattr(value, "device", "unknown")),
        "python_type": f"{type(value).__module__}.{type(value).__name__}",
    }


def module_device_dtype(module: Any) -> dict[str, str]:
    try:
        parameter = next(module.parameters())
    except Exception:
        return {
            "device": str(getattr(module, "device", "unknown")),
            "dtype": str(getattr(module, "dtype", "unknown")).replace("torch.", ""),
        }
    return {
        "device": str(parameter.device),
        "dtype": str(parameter.dtype).replace("torch.", ""),
    }
