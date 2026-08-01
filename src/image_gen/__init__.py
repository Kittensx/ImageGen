"""Canonical IMAGE_GEN Python package.

The public contract symbols are loaded lazily so process-start configuration
modules can be imported before Torch. CUDA allocator environment variables must
be established before importing ``image_gen.contracts``, which carries Torch
runtime types.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from image_gen.contracts import (
        ConditioningOutput,
        GenerationRequest,
        GenerationResult,
        PipelineComponents,
        SamplerOutput,
        SchedulerOutput,
    )

_CONTRACT_EXPORTS = {
    "ConditioningOutput",
    "GenerationRequest",
    "GenerationResult",
    "PipelineComponents",
    "SamplerOutput",
    "SchedulerOutput",
}

__all__ = sorted(_CONTRACT_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _CONTRACT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    contracts = import_module("image_gen.contracts")
    value = getattr(contracts, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _CONTRACT_EXPORTS)
