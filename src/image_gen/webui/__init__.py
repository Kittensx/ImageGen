"""Local, source-readable IMAGE_GEN WebUI.

``create_app`` is loaded lazily so ``image_gen.webui.server`` can establish
process-start CUDA allocator settings before the WebUI job modules import
Torch.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from .app import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name != "create_app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("image_gen.webui.app"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"create_app"})
