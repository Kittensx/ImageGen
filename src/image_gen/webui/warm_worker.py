"""Compatibility aliases for the removed Phase 13F warm-worker feature.

The WebUI now uses the canonical resident model runtime directly.
"""
from image_gen.webui.model_runtime import ModelRuntimeUnavailable, ResidentModelRuntimeClient

WarmWorkerUnavailable = ModelRuntimeUnavailable
PersistentWarmWorkerClient = ResidentModelRuntimeClient

__all__ = [
    "ModelRuntimeUnavailable",
    "ResidentModelRuntimeClient",
    "WarmWorkerUnavailable",
    "PersistentWarmWorkerClient",
]
