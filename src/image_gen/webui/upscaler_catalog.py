from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Mapping

from image_gen.systems.upscaling.contracts import UpscalerDescriptor, UpscalerDiscoveryResult
from image_gen.systems.upscaling.discovery import discover_upscalers
from image_gen.systems.upscaling.registry import UpscalerModelRegistry
from modules.project_context import ProjectContext

UPSCALER_CATALOG_CONTRACT_VERSION = "image-gen-upscaler-catalog-v1"


class WebUIUpscalerCatalog:
    """Cached discovery handoff for WebUI selection, diagnostics, and replay.

    Discovery and registry construction happen before the commit lock is held.
    A refresh therefore publishes exactly one complete catalog revision and never
    mutates the generation-critical runtime registry used by an active job.
    """

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self._lock = threading.RLock()
        self._revision = 0
        self._refreshed_at = ""
        self._result: UpscalerDiscoveryResult | None = None
        self._registry: UpscalerModelRegistry | None = None
        self.refresh(mode="unidentified")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def refresh(
        self,
        *,
        mode: str = "all",
        selected_file: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = str(mode or "all").strip().casefold()
        result = discover_upscalers(
            self.context,
            mode=normalized_mode,
            selected_file=selected_file,
        )
        registry = UpscalerModelRegistry.from_discovery(result)
        with self._lock:
            self._result = result
            self._registry = registry
            self._revision += 1
            self._refreshed_at = self._timestamp()
            return self._payload_locked()

    def _payload_locked(self) -> dict[str, Any]:
        if self._result is None or self._registry is None:
            raise RuntimeError("The upscaler catalog has not been initialized.")
        discovery = self._result.to_dict()
        runtime = self._registry.snapshot()
        neural = list(runtime.get("neural") or [])
        supported = [item for item in neural if bool(item.get("selectable"))]
        unavailable = [item for item in neural if not bool(item.get("selectable"))]
        return {
            "contract_version": UPSCALER_CATALOG_CONTRACT_VERSION,
            "catalog_revision": int(self._revision),
            "catalog_refreshed_at": self._refreshed_at,
            "startup_scan_mode": "unidentified",
            "manual_refresh_modes": ["all", "selected"],
            "mode": discovery.get("mode"),
            "roots": discovery.get("roots") or [],
            "cache_path": discovery.get("cache_path") or "",
            "built_in_latent": [],
            "interpolation_baselines": [],
            "neural": neural,
            "supported_neural": supported,
            "unavailable_neural": unavailable,
            "diagnostics": discovery.get("diagnostics") or [],
            "supported_neural_count": len(supported),
            "unavailable_neural_count": len(unavailable),
            "discovery_support_is_runtime_qualification": False,
            "target_gpu_qualification_phase": "14N-8",
        }

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return self._payload_locked()

    def descriptor(self, upscaler_id: str) -> UpscalerDescriptor | None:
        selected = str(upscaler_id or "").strip()
        with self._lock:
            return self._result.descriptor_by_id(selected) if self._result is not None else None

    def validate_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = dict(payload or {})
        if not bool(request.get("hires_enabled", False)):
            return request

        strategy = str(request.get("hires_strategy") or "pixel_neural").strip().casefold()
        selected_id = str(
            request.get("hires_upscaler_id")
            or request.get("hires_upscaler")
            or ""
        ).strip()

        if strategy != "pixel_neural":
            raise ValueError(
                f"Neural upscaler ID {selected_id!r} requires hires_strategy='pixel_neural'."
            )
        descriptor = self.descriptor(selected_id)
        if descriptor is None:
            raise ValueError(
                f"Pixel-neural hires could not resolve stable upscaler ID {selected_id!r}. "
                "Refresh the upscaler catalog; replay never falls back to another model."
            )
        if not descriptor.selectable:
            reason = descriptor.bounded_error or descriptor.load_status
            raise ValueError(
                f"Pixel-neural upscaler {selected_id!r} is not selectable: {reason}."
            )
        request["hires_strategy"] = "pixel_neural"
        request["hires_upscaler"] = descriptor.upscaler_id
        request["hires_upscaler_id"] = descriptor.upscaler_id
        return request
