from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from image_gen.webui.model_selection import ModelSelectionUnavailableError
from image_gen.webui.model_runtime import ModelRuntimeCommandSuperseded
from image_gen.webui.routes.payloads import ModelActivationPayload


def build_models_router(*, context, catalog, component_registry, component_selection, jobs, model_selection, generation_capabilities, _default_asset_payload, _webui_failure) -> APIRouter:
    router = APIRouter()

    @router.get("/api/models/active")
    async def active_model() -> dict[str, Any]:
        return {
            "active_model": model_selection.current_payload(),
            "generation_capabilities": generation_capabilities.resolve_active(),
            "model_runtime": jobs.model_runtime_status(),
        }


    @router.post("/api/generation/capabilities")
    async def generation_capability_contract(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        # GFP-02 authority boundary: Advanced Models are resolved through the
        # component registry inside GenerationCapabilityService. Browser family
        # labels are never promoted directly into a capability architecture.
        return generation_capabilities.resolve_request(dict(payload or {}))


    @router.get("/api/models/runtime-status")
    async def model_runtime_status() -> dict[str, Any]:
        return jobs.model_runtime_status()


    @router.get("/api/models/components")
    async def advanced_model_components(base_component_sha256: str = "") -> dict[str, Any]:
        """Return fingerprint-deduplicated component choices for Advanced Models."""
        try:
            return component_selection.catalog(
                base_component_sha256=(base_component_sha256 or None)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "advanced_model_component_catalog",
                exc,
                request_path="/api/models/components",
                status_code=400,
            ) from exc


    @router.get("/api/models/components/browser")
    async def component_registry_browser(
        family: str = "",
        role: str = "",
        accessible_only: bool = False,
        q: str = "",
        limit: int = 500,
    ) -> dict[str, Any]:
        try:
            return component_registry.registry_browser(
                family=(family or None),
                role=(role or None),
                accessible_only=accessible_only,
                search=q,
                limit=max(1, min(int(limit), 2000)),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "component_registry_browser",
                exc,
                request_path="/api/models/components/browser",
                status_code=400,
            ) from exc


    @router.get("/api/models/components/{component_sha256}/evidence")
    async def component_registry_evidence(component_sha256: str) -> dict[str, Any]:
        try:
            return {
                "component_sha256": component_sha256,
                "policies": component_registry.list_component_policies(
                    component_sha256=component_sha256,
                    limit=1000,
                ),
                "validations": component_registry.list_component_validations(
                    component_sha256=component_sha256,
                    limit=1000,
                ),
                "relationships": component_registry.list_relationship_evidence_records(
                    component_sha256=component_sha256,
                    limit=1000,
                ),
                "sources": component_registry.explain_component_sources(component_sha256),
            }
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "component_registry_evidence",
                exc,
                request_path=f"/api/models/components/{component_sha256}/evidence",
                status_code=400,
            ) from exc


    @router.post("/api/models/components/policy")
    async def set_component_policy(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            record = component_registry.set_component_policy(
                component_sha256=str(payload.get("component_sha256") or ""),
                policy_scope=str(payload.get("policy_scope") or "global"),
                base_component_sha256=payload.get("base_component_sha256"),
                component_role=payload.get("component_role"),
                policy_source=str(payload.get("policy_source") or "user"),
                reason=str(payload.get("reason") or ""),
                metadata=dict(payload.get("metadata") or {}),
            )
            return {"record": record, "component_catalog": component_selection.catalog()}
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "component_policy_set",
                exc,
                payload=payload,
                request_path="/api/models/components/policy",
                status_code=400,
            ) from exc


    @router.delete("/api/models/components/policy")
    async def clear_component_policy(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            removed = component_registry.clear_component_policy(
                component_sha256=str(payload.get("component_sha256") or ""),
                policy_scope=str(payload.get("policy_scope") or "global"),
                base_component_sha256=payload.get("base_component_sha256"),
                component_role=payload.get("component_role"),
            )
            return {"removed": removed, "component_catalog": component_selection.catalog()}
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "component_policy_clear",
                exc,
                payload=payload,
                request_path="/api/models/components/policy",
                status_code=400,
            ) from exc


    @router.post("/api/models/components/validation")
    async def record_component_validation(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            record = component_registry.record_component_validation(
                component_sha256=str(payload.get("component_sha256") or ""),
                validation_stage=str(payload.get("validation_stage") or ""),
                validation_result=str(payload.get("validation_result") or ""),
                family_id=str(payload.get("family_id") or ""),
                provider_version=str(payload.get("provider_version") or ""),
                base_component_sha256=payload.get("base_component_sha256"),
                composition_sha256=payload.get("composition_sha256"),
                component_role=str(payload.get("component_role") or ""),
                blocking_state=str(payload.get("blocking_state") or "advisory"),
                evidence_type=str(payload.get("evidence_type") or "runtime_validation"),
                evidence=dict(payload.get("evidence") or {}),
                environment=dict(payload.get("environment") or {}),
                evidence_artifact=str(payload.get("evidence_artifact") or ""),
                error_category=str(payload.get("error_category") or ""),
                error_message=str(payload.get("error_message") or ""),
                runtime_version=str(payload.get("runtime_version") or ""),
            )
            return {"record": record}
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "component_validation_record",
                exc,
                payload=payload,
                request_path="/api/models/components/validation",
                status_code=400,
            ) from exc


    @router.delete("/api/models/components/validation")
    async def clear_component_validation(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            removed = component_registry.clear_component_validations(
                component_sha256=payload.get("component_sha256"),
                base_component_sha256=payload.get("base_component_sha256"),
                composition_sha256=payload.get("composition_sha256"),
            )
            return {"removed": removed}
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "component_validation_clear",
                exc,
                payload=payload,
                request_path="/api/models/components/validation",
                status_code=400,
            ) from exc


    @router.get("/api/models/components/registry-status")
    async def advanced_model_registry_status(accessible_only: bool = False) -> dict[str, Any]:
        try:
            return {
                "configured_roots": component_registry.configured_library_roots(),
                "location_catalog": component_registry.location_catalog(accessible_only=accessible_only),
            }
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "advanced_model_registry_status",
                exc,
                request_path="/api/models/components/registry-status",
                status_code=400,
            ) from exc


    @router.post("/api/models/components/refresh")
    async def refresh_advanced_model_registry(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = dict(payload or {})
        try:
            result = await asyncio.to_thread(
                component_registry.refresh_configured_library,
                force=bool(request.get("force", False)),
                strength=str(request.get("strength") or "structural"),
            )
            # Keep the traditional checkpoint/VAE selectors synchronized with files
            # discovered during the same user-requested model-library refresh.
            await asyncio.to_thread(catalog.refresh_models)
            result["component_catalog"] = component_selection.catalog()
            return result
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "advanced_model_registry_refresh",
                exc,
                request_path="/api/models/components/refresh",
                status_code=400,
            ) from exc


    @router.post("/api/models/unload")
    async def unload_model() -> dict[str, Any]:
        status = jobs.model_runtime_status()
        if status.get("current_job_id"):
            raise HTTPException(status_code=409, detail="Cancel or finish the active generation before unloading the checkpoint.")
        result = await jobs.unload_model()
        model_selection.deactivate()
        return {
            "unloaded": True,
            "result": result,
            "active_model": None,
            "generation_capabilities": generation_capabilities.resolve_for_model({}, request={}),
            "model_runtime": jobs.model_runtime_status(),
            "default_assets": _default_asset_payload(None),
        }


    async def _activate_model_impl(payload: ModelActivationPayload) -> dict[str, Any]:
        runtime_before = jobs.model_runtime_status()
        current_job_id = str(runtime_before.get("current_job_id") or "").strip()
        current_model_path = str(runtime_before.get("current_model_path") or "").strip()
        requested_model_path = str(payload.model_path or "").strip()

        def _model_path_identity(value: str) -> str:
            if not value:
                return ""
            try:
                return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))
            except OSError:
                return os.path.normcase(value)

        if current_job_id and _model_path_identity(current_model_path) != _model_path_identity(requested_model_path):
            raise HTTPException(
                status_code=409,
                detail=(
                    "A generation is currently using the resident checkpoint. "
                    "Wait for the active job to finish or cancel it before changing models."
                ),
            )
        # Checkpoint dropdown changes are user intent, not a FIFO load queue. If a
        # previous activation is still hydrating a different checkpoint, supersede
        # it so the newest selection does not wait behind an obsolete multi-GB load.
        if not current_job_id:
            await jobs.supersede_model_activation(requested_model_path)
        try:
            selected = model_selection.activate(payload.model_path)
        except ModelSelectionUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise _webui_failure(
                "model_activation",
                exc,
                payload=payload.model_dump(),
                request_path="/api/models/activate",
                status_code=400,
            ) from exc
        try:
            activation = await jobs.activate_model(
                selected.resolved_path,
                selection=selected.to_dict(),
            )
        except ModelRuntimeCommandSuperseded as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "model_gpu_activation",
                exc,
                payload=payload.model_dump(),
                request_path="/api/models/activate",
                status_code=400,
                extra={
                    "model_runtime_before": runtime_before,
                    "model_runtime_after": jobs.model_runtime_status(),
                },
            ) from exc
        return {
            "active_model": selected.to_dict(),
            "generation_capabilities": generation_capabilities.resolve_for_model(selected.to_dict(), request={"model_path": selected.resolved_path}),
            "model_runtime": jobs.model_runtime_status(),
            "default_assets": _default_asset_payload(selected.to_dict()),
            "activation": activation,
        }


    @router.post("/api/models/activate")
    async def activate_model(payload: ModelActivationPayload) -> dict[str, Any]:
        return await _activate_model_impl(payload)


    @router.post("/api/model/activate")
    async def activate_model_alias(payload: ModelActivationPayload) -> dict[str, Any]:
        return await _activate_model_impl(payload)


    @router.post("/api/activate-model")
    async def activate_model_legacy_alias(payload: ModelActivationPayload) -> dict[str, Any]:
        return await _activate_model_impl(payload)


    @router.get("/api/model-activation/debug")
    async def model_activation_debug() -> dict[str, Any]:
        return {
            "project_root": str(context.project_root),
            "config_path": str(context.config_path),
            "checkpoints_dir": str(context.checkpoints_dir),
            "default_model_path": str(context.default_model_path) if context.default_model_path else None,
            "active_model": model_selection.current_payload(),
            "checkpoints_dir_exists": context.checkpoints_dir.is_dir(),
        }


    return router
