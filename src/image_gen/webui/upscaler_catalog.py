from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Mapping

from image_gen.systems.upscaling.contracts import UpscalerDescriptor, UpscalerDiscoveryResult
from image_gen.systems.upscaling.discovery import discover_upscalers
from image_gen.systems.upscaling.registry import UpscalerModelRegistry
from image_gen.webui.asset_metadata import load_asset_metadata, synchronize_asset_companions
from image_gen.webui.civitai_asset_metadata import (
    CivitaiAssetMetadataService,
    CivitaiCredentialError,
    CivitaiMetadataError,
    read_civitai_api_key,
)
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
        self._civitai_client = CivitaiAssetMetadataService(context)
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
        neural = [self._decorate_descriptor(item) for item in list(runtime.get("neural") or [])]
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

    @staticmethod
    def _metadata_lookup(metadata: Mapping[str, Any]) -> dict[str, Any]:
        lookup = metadata.get("_civitai_lookup")
        return dict(lookup) if isinstance(lookup, Mapping) else {}

    def _decorate_descriptor(self, value: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(value)
        path = str(item.get("path") or "").strip()
        if not path:
            return item
        metadata = synchronize_asset_companions(path, load_asset_metadata(path))
        lookup = self._metadata_lookup(metadata)
        item.update({
            "metadata": metadata,
            "source_url": str(metadata.get("source_url") or lookup.get("source_url") or ""),
            "description": str(metadata.get("description") or lookup.get("description") or ""),
            "tags": list(metadata.get("tags") or lookup.get("tags") or []),
            "civitai_lookup": lookup,
            "civitai_model_id": lookup.get("model_id"),
            "civitai_model_version_id": lookup.get("model_version_id"),
            "civitai_model_name": str(lookup.get("model_name") or ""),
            "civitai_model_version_name": str(lookup.get("model_version_name") or ""),
            "civitai_creator": str(lookup.get("creator") or ""),
        })
        return item

    def enrich_from_civitai(self, upscaler_id: str, *, overwrite: bool = False) -> dict[str, Any]:
        descriptor = self.descriptor(upscaler_id)
        if descriptor is None:
            raise KeyError(f"Unknown upscaler asset: {upscaler_id}")
        enrichment = self._civitai_client.enrich_local_asset(
            descriptor.path,
            asset_type="upscaler",
            hashes=(descriptor.sha256,),
            overwrite=overwrite,
        )
        asset = self._decorate_descriptor(descriptor.to_dict())
        asset["civitai_lookup"] = dict(enrichment.get("civitai_lookup") or {})
        return {
            "asset_type": "upscaler",
            "asset": asset,
            "catalog": self.payload(),
        }

    def enrich_all_from_civitai(self, *, mode: str = "missing") -> dict[str, Any]:
        normalized_mode = str(mode or "missing").strip().casefold()
        if normalized_mode not in {"missing", "all"}:
            raise ValueError(f"Unsupported CivitAI metadata mode: {mode}")
        read_civitai_api_key(self.context)
        with self._lock:
            descriptors = list(self._result.neural_descriptors) if self._result is not None else []
        matched = 0
        skipped = 0
        previews_downloaded = 0
        errors: list[dict[str, str]] = []
        for descriptor in descriptors:
            metadata = synchronize_asset_companions(descriptor.path, load_asset_metadata(descriptor.path))
            lookup = self._metadata_lookup(metadata)
            if normalized_mode == "missing" and str(lookup.get("status") or "").strip().casefold() == "matched":
                skipped += 1
                continue
            try:
                result = self.enrich_from_civitai(descriptor.upscaler_id, overwrite=False)
                lookup = dict(result.get("asset", {}).get("civitai_lookup") or {})
                matched += 1
                if lookup.get("preview_image_downloaded"):
                    previews_downloaded += 1
            except CivitaiCredentialError:
                raise
            except (CivitaiMetadataError, OSError, ValueError) as exc:
                errors.append({
                    "asset_id": descriptor.upscaler_id,
                    "filename": descriptor.file_name,
                    "error": str(exc),
                })
        return {
            "asset_type": "upscaler",
            "catalog": self.payload(),
            "civitai": {
                "asset_type": "upscaler",
                "mode": normalized_mode,
                "matched": matched,
                "skipped": skipped,
                "previews_downloaded": previews_downloaded,
                "errors": errors,
            },
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
