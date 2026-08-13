from __future__ import annotations

import json
from typing import Any, Mapping

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse, StreamingResponse

from image_gen.systems.asset_hub import (
    ASSET_HUB_CONTRACT_VERSION,
    AssetHubDownloadManager,
    AssetHubInstaller,
    UpscalerFavoriteStore,
    compatible_upscaler_payload,
    AssetHubError,
    AssetHubSecretStore,
    AssetHubService,
    ProviderSearchRequest,
)


def _error_response(exc: AssetHubError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "error": exc.to_dict(),
        },
    )


def _body_mapping(payload: Any) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_asset_hub_router(
    service: AssetHubService,
    *,
    secrets: AssetHubSecretStore | None = None,
    downloads: AssetHubDownloadManager | None = None,
    installer: AssetHubInstaller | None = None,
    upscaler_catalog: Any | None = None,
    upscaler_favorites: UpscalerFavoriteStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/asset-hub", tags=["asset-hub"])

    @router.get("/providers")
    async def providers() -> dict[str, Any]:
        return {
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "providers": service.provider_descriptors(),
        }

    @router.get("/providers/{provider_id}/status")
    async def provider_status(provider_id: str) -> Any:
        try:
            payload = service.provider_status(provider_id)
            if secrets is not None:
                payload["secret"] = secrets.status(provider_id).to_dict()
            return payload
        except (AssetHubError, ValueError) as exc:
            if isinstance(exc, AssetHubError):
                return _error_response(exc)
            return _error_response(AssetHubError("provider_policy_blocked", str(exc), status_code=400))

    @router.put("/providers/{provider_id}/secret")
    async def set_provider_secret(provider_id: str, payload: Any = Body(default={})) -> Any:
        if secrets is None:
            return _error_response(AssetHubError("secret_store_unavailable", "Provider credential storage is unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            if str(provider_id or "").strip().casefold() not in service.providers:
                raise AssetHubError("provider_not_found", "Unknown asset provider.", status_code=404)
            secrets.set(provider_id, str(body.get("secret") or ""), persistent=bool(body.get("persistent", False)))
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "secret": secrets.status(provider_id).to_dict(),
            }
        except AssetHubError as exc:
            return _error_response(exc)
        except (ValueError, RuntimeError, OSError) as exc:
            return _error_response(AssetHubError("secret_store_error", str(exc), status_code=400))

    @router.delete("/providers/{provider_id}/secret")
    async def delete_provider_secret(provider_id: str) -> Any:
        if secrets is None:
            return _error_response(AssetHubError("secret_store_unavailable", "Provider credential storage is unavailable.", status_code=503))
        try:
            if str(provider_id or "").strip().casefold() not in service.providers:
                raise AssetHubError("provider_not_found", "Unknown asset provider.", status_code=404)
            secrets.delete(provider_id)
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "secret": secrets.status(provider_id).to_dict(),
            }
        except AssetHubError as exc:
            return _error_response(exc)
        except (ValueError, RuntimeError, OSError) as exc:
            return _error_response(AssetHubError("secret_store_error", str(exc), status_code=400))

    @router.post("/providers/{provider_id}/secret/validate")
    async def validate_provider_secret(provider_id: str, payload: Any = Body(default={})) -> Any:
        if secrets is None:
            return _error_response(AssetHubError("secret_store_unavailable", "Provider credential storage is unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            provider = service.providers.get(str(provider_id or "").strip().casefold())
            if provider is None:
                raise AssetHubError("provider_not_found", "Unknown asset provider.", status_code=404)
            candidate = str(body.get("secret") or "").strip() or (secrets.get(provider_id) or "")
            if not candidate:
                raise AssetHubError("provider_auth_required", "No provider token is configured.", status_code=400)
            valid = await provider.validate_secret(candidate)
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "valid": bool(valid),
                "secret": secrets.status(provider_id).to_dict(),
            }
        except AssetHubError as exc:
            return _error_response(exc)
        except (ValueError, RuntimeError, OSError) as exc:
            return _error_response(AssetHubError("secret_store_error", str(exc), status_code=400))

    @router.get("/search")
    async def search(
        provider: str = Query(default="civitai", max_length=64),
        query: str = Query(default="", max_length=256),
        asset_kind: str = Query(default="checkpoint", alias="type", max_length=64),
        base_model: list[str] = Query(default=[]),
        creator: str = Query(default="", max_length=256),
        sort: str = Query(default="", max_length=64),
        period: str = Query(default="", max_length=64),
        safe_content: bool = Query(default=True),
        cursor: str = Query(default="", max_length=128),
        limit: int = Query(default=24, ge=1, le=50),
        refresh: bool = Query(default=False),
    ) -> Any:
        try:
            request = ProviderSearchRequest(
                query=query,
                asset_kind=asset_kind,
                base_models=tuple(base_model),
                creator=creator,
                sort=sort,
                period=period,
                safe_content=safe_content,
                cursor=cursor,
                limit=limit,
                refresh=refresh,
            )
            page = await service.search(provider, request)
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "readOnly": True,
                "page": page.to_dict(),
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/models/{provider_id}/{model_id}")
    async def model(provider_id: str, model_id: str, refresh: bool = Query(default=False)) -> Any:
        try:
            result = await service.get_model(provider_id, model_id, refresh=refresh)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "model": result.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/versions/{provider_id}/{version_id}")
    async def version(provider_id: str, version_id: str, refresh: bool = Query(default=False)) -> Any:
        try:
            result = await service.get_version(provider_id, version_id, refresh=refresh)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "version": result.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/hash/{provider_id}/{file_hash}")
    async def hash_lookup(provider_id: str, file_hash: str, refresh: bool = Query(default=False)) -> Any:
        try:
            result = await service.lookup_hash(provider_id, file_hash, refresh=refresh)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "version": result.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/download-plans")
    async def create_download_plan(payload: Any = Body(default={})) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            plan = await downloads.create_plan(
                provider_id=str(body.get("providerId") or ""),
                remote_model_id=str(body.get("remoteModelId") or ""),
                remote_version_id=str(body.get("remoteVersionId") or ""),
                remote_file_id=str(body.get("remoteFileId") or ""),
            )
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "plan": plan.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/download-jobs")
    async def create_download_job(payload: Any = Body(default={})) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        body = _body_mapping(payload)
        if any(key in body for key in ("url", "downloadUrl", "download_url")):
            return _error_response(AssetHubError("download_plan_invalid", "Arbitrary download URLs are not accepted.", status_code=400))
        try:
            job = await downloads.enqueue(str(body.get("planId") or ""))
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": job.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/download-jobs")
    async def list_download_jobs(limit: int = Query(default=100, ge=1, le=500)) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        return {
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "jobs": [item.to_dict() for item in downloads.list_jobs(limit=limit)],
        }

    @router.get("/download-jobs/{job_id}")
    async def get_download_job(job_id: str) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": downloads.get_job(job_id).to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/download-jobs/{job_id}/cancel")
    async def cancel_download_job(job_id: str) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            job = await downloads.cancel(job_id)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": job.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/download-jobs/{job_id}/resume")
    async def resume_download_job(job_id: str) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            job = await downloads.resume(job_id)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": job.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/install-plans")
    async def create_install_plan(payload: Any = Body(default={})) -> Any:
        if installer is None:
            return _error_response(AssetHubError("install_runtime_unavailable", "Asset Hub installation is unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            plan = await installer.create_plan(
                str(body.get("downloadJobId") or ""),
                conflict_policy=str(body.get("conflictPolicy") or "hash_suffix"),
                archive_member=str(body.get("archiveMember") or ""),
            )
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "plan": plan.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/install-jobs")
    async def create_install_job(payload: Any = Body(default={})) -> Any:
        if installer is None:
            return _error_response(AssetHubError("install_runtime_unavailable", "Asset Hub installation is unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            record = await installer.install(str(body.get("planId") or ""), confirmed=bool(body.get("confirmed", False)))
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": record.to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/install-jobs/{job_id}")
    async def get_install_job(job_id: str) -> Any:
        if installer is None:
            return _error_response(AssetHubError("install_runtime_unavailable", "Asset Hub installation is unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": installer.get_install_job(job_id).to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/installed")
    async def list_installed() -> Any:
        if installer is None:
            return _error_response(AssetHubError("install_runtime_unavailable", "Asset Hub installation is unavailable.", status_code=503))
        return {
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "installed": [item.to_dict() for item in installer.list_installed()],
        }

    @router.delete("/installed/{install_id}")
    async def uninstall_asset(install_id: str) -> Any:
        if installer is None:
            return _error_response(AssetHubError("install_runtime_unavailable", "Asset Hub installation is unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "install": installer.uninstall(install_id).to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/installed/{install_id}/refresh")
    async def refresh_installed_metadata(install_id: str) -> Any:
        if installer is None:
            return _error_response(AssetHubError("install_runtime_unavailable", "Asset Hub installation is unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "install": (await installer.refresh_metadata(install_id)).to_dict()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/upscalers/compatibility")
    async def upscaler_compatibility(model_architecture: str = Query(default="", max_length=64)) -> Any:
        if upscaler_catalog is None or upscaler_favorites is None:
            return _error_response(AssetHubError("upscaler_catalog_unavailable", "Upscaler compatibility catalog is unavailable.", status_code=503))
        return {
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "upscalers": compatible_upscaler_payload(
                upscaler_catalog.payload(),
                upscaler_favorites.payload(),
                model_architecture=model_architecture,
            ),
        }

    @router.get("/upscalers/favorites")
    async def get_upscaler_favorites() -> Any:
        if upscaler_favorites is None:
            return _error_response(AssetHubError("upscaler_catalog_unavailable", "Upscaler favorites are unavailable.", status_code=503))
        return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "favorites": upscaler_favorites.payload()}

    @router.put("/upscalers/favorites")
    async def set_upscaler_favorites(payload: Any = Body(default={})) -> Any:
        if upscaler_favorites is None:
            return _error_response(AssetHubError("upscaler_catalog_unavailable", "Upscaler favorites are unavailable.", status_code=503))
        body = _body_mapping(payload)
        values = body.get("favoriteUpscalerIds") or []
        if not isinstance(values, list):
            return _error_response(AssetHubError("upscaler_favorites_invalid", "favoriteUpscalerIds must be a list.", status_code=400))
        return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "favorites": upscaler_favorites.set_favorites(values)}

    @router.get("/download-jobs/{job_id}/events")
    async def download_job_events(job_id: str) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            queue = await downloads.subscribe(job_id)
        except AssetHubError as exc:
            return _error_response(exc)

        async def stream():
            try:
                while True:
                    payload = await queue.get()
                    if payload is None:
                        break
                    event = str(payload.get("event") or "message")
                    yield f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
            finally:
                downloads.unsubscribe(job_id, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
