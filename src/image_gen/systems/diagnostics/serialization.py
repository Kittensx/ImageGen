from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch


_SECRET_TOKENS = ("password", "passwd", "secret", "token", "api_key", "apikey", "private_key")


def is_secret_key(key: str) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(token in normalized for token in _SECRET_TOKENS)


def json_safe(value: Any, *, redact_secrets: bool = False) -> Any:
    """Convert runtime values to deterministic JSON-safe diagnostics metadata.

    Tensors are summarized and never serialized with their full contents.
    Objects that expose an explicit serialization method are allowed to control
    their representation before generic dataclass handling is attempted. This
    matters for frozen registry descriptors whose fields contain mappingproxy
    objects that cannot be deep-copied by ``dataclasses.asdict``.
    """

    return _json_safe(value, redact_secrets=redact_secrets, seen=set())


def _json_safe(value: Any, *, redact_secrets: bool, seen: set[int]) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value, redact_secrets=redact_secrets, seen=seen)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if torch.is_tensor(value):
        return tensor_summary(value)

    object_id = id(value)
    if object_id in seen:
        return {
            "runtime_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "cycle": True,
        }

    if hasattr(value, "to_serializable_dict") and callable(value.to_serializable_dict):
        seen.add(object_id)
        try:
            payload = value.to_serializable_dict()
            return _json_safe(payload, redact_secrets=redact_secrets, seen=seen)
        finally:
            seen.discard(object_id)

    if hasattr(value, "to_dict") and callable(value.to_dict):
        seen.add(object_id)
        try:
            payload = value.to_dict()
            return _json_safe(payload, redact_secrets=redact_secrets, seen=seen)
        finally:
            seen.discard(object_id)

    if is_dataclass(value):
        seen.add(object_id)
        try:
            payload = {
                item.name: getattr(value, item.name)
                for item in fields(value)
            }
            return _json_safe(payload, redact_secrets=redact_secrets, seen=seen)
        finally:
            seen.discard(object_id)

    if isinstance(value, Mapping):
        seen.add(object_id)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if redact_secrets and is_secret_key(key_text):
                    result[key_text] = "<redacted>"
                else:
                    result[key_text] = _json_safe(
                        item,
                        redact_secrets=redact_secrets,
                        seen=seen,
                    )
            return result
        finally:
            seen.discard(object_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(object_id)
        try:
            return [
                _json_safe(item, redact_secrets=redact_secrets, seen=seen)
                for item in value
            ]
        finally:
            seen.discard(object_id)

    return {"runtime_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def tensor_summary(value: torch.Tensor | None, *, include_statistics: bool = False) -> dict[str, Any] | None:
    if value is None:
        return None
    summary: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "numel": int(value.numel()),
    }
    if include_statistics and value.numel():
        detached = value.detach()
        finite = torch.isfinite(detached)
        summary["finite"] = bool(finite.all().item())
        summary["finite_count"] = int(finite.sum().item())
        if finite.any():
            data = detached[finite].to(dtype=torch.float32, device="cpu")
            summary.update(
                {
                    "min": float(data.min().item()),
                    "max": float(data.max().item()),
                    "mean": float(data.mean().item()),
                    "std": float(data.std(unbiased=False).item()),
                    "norm": float(torch.linalg.vector_norm(data).item()),
                }
            )
    return summary
