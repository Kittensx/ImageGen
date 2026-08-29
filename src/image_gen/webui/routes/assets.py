from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from image_gen.program_metadata import PRODUCT_NAME
from image_gen.webui.civitai_asset_metadata import (
    CivitaiCredentialError,
    CivitaiMetadataNotFound,
    CivitaiRequestError,
    civitai_api_key_status,
    civitai_authentication_request_status,
    delete_civitai_api_key,
    write_civitai_api_key,
    sync_civitai_api_key_to_secret_store,
)
from image_gen.webui.routes.payloads import CivitaiCredentialPayload


def build_assets_router(
    *,
    context,
    catalog,
    upscaler_catalog,
    civitai_connection,
    asset_hub_secrets=None,
    _lora_auto_scan_enabled,
    _preview_media_type,
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/models/refresh")
    async def refresh_models() -> dict[str, Any]:
        return catalog.refresh_models()


    @router.get("/api/assets/catalog")
    async def asset_catalog_status() -> dict[str, Any]:
        return catalog.catalog_payload()


    @router.post("/api/assets/refresh")
    async def refresh_asset_catalog(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        requested = str((payload or {}).get("asset_type") or "").strip().lower()
        aliases = {
            "checkpoints": "checkpoint",
            "loras": "lora",
            "textual-inversions": "textual_inversion",
            "textual_inversions": "textual_inversion",
        }
        requested = aliases.get(requested, requested)
        try:
            if requested:
                return await asyncio.to_thread(catalog.refresh_asset_type, requested)
            await asyncio.to_thread(catalog.refresh_models)
            return catalog.catalog_payload()
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Unsupported asset type: {requested}") from exc


    def _normalize_civitai_asset_type(value: str) -> str:
        token = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "checkpoint": "checkpoint",
            "checkpoints": "checkpoint",
            "model": "checkpoint",
            "models": "checkpoint",
            "lora": "lora",
            "loras": "lora",
            "vae": "vae",
            "vaes": "vae",
            "textual_inversion": "textual_inversion",
            "textual_inversions": "textual_inversion",
            "embedding": "textual_inversion",
            "embeddings": "textual_inversion",
            "upscaler": "upscaler",
            "upscalers": "upscaler",
        }
        normalized = aliases.get(token, token)
        if normalized not in {"checkpoint", "lora", "vae", "textual_inversion", "upscaler"}:
            raise ValueError(f"Unsupported CivitAI asset type: {value}")
        return normalized


    def _civitai_credential_required(exc: CivitaiCredentialError) -> HTTPException:
        return HTTPException(
            status_code=400,
            detail={
                "code": "civitai_credentials_required",
                "message": f"Connect CivitAI in {PRODUCT_NAME} Settings to use this action.",
            },
        )


    @router.get("/api/integrations/civitai")
    async def civitai_connection_status() -> dict[str, Any]:
        return civitai_api_key_status(context)


    @router.get("/api/integrations/civitai/auth-request")
    async def civitai_auth_request_status() -> dict[str, Any]:
        return civitai_authentication_request_status(context)


    @router.put("/api/integrations/civitai/credential")
    async def save_civitai_credential(payload: CivitaiCredentialPayload) -> dict[str, Any]:
        try:
            status = await asyncio.to_thread(write_civitai_api_key, context, payload.api_key)
            if asset_hub_secrets is not None:
                await asyncio.to_thread(sync_civitai_api_key_to_secret_store, context, asset_hub_secrets)
            return status
        except CivitaiCredentialError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.delete("/api/integrations/civitai/credential")
    async def remove_civitai_credential() -> dict[str, Any]:
        try:
            status = await asyncio.to_thread(delete_civitai_api_key, context)
            if asset_hub_secrets is not None:
                asset_hub_secrets.delete("civitai")
            return status
        except CivitaiCredentialError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/integrations/civitai/test")
    async def test_civitai_connection() -> dict[str, Any]:
        try:
            return await asyncio.to_thread(civitai_connection.test_connection)
        except CivitaiCredentialError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "civitai_credentials_invalid",
                    "message": "CivitAI rejected the configured API key. Replace it and try again.",
                },
            ) from exc
        except CivitaiRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


    @router.post("/api/civitai/assets/{asset_type}/metadata")
    async def enrich_assets_from_civitai(
        asset_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            normalized = _normalize_civitai_asset_type(asset_type)
            mode = str((payload or {}).get("mode") or "missing").strip().casefold()
            if normalized == "upscaler":
                return await asyncio.to_thread(upscaler_catalog.enrich_all_from_civitai, mode=mode)
            return await asyncio.to_thread(catalog.enrich_assets_from_civitai, normalized, mode=mode)
        except CivitaiCredentialError as exc:
            raise _civitai_credential_required(exc) from exc
        except CivitaiRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/civitai/assets/{asset_type}/{asset_id}/metadata")
    async def enrich_asset_from_civitai(
        asset_type: str,
        asset_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        overwrite = bool((payload or {}).get("overwrite", False))
        try:
            normalized = _normalize_civitai_asset_type(asset_type)
            if normalized == "upscaler":
                return await asyncio.to_thread(
                    upscaler_catalog.enrich_from_civitai,
                    asset_id,
                    overwrite=overwrite,
                )
            return await asyncio.to_thread(
                catalog.enrich_asset_from_civitai,
                normalized,
                asset_id,
                overwrite=overwrite,
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CivitaiMetadataNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CivitaiCredentialError as exc:
            raise _civitai_credential_required(exc) from exc
        except CivitaiRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/checkpoints")
    async def checkpoint_assets() -> dict[str, Any]:
        return catalog.asset_payload("checkpoint")


    @router.post("/api/assets/checkpoints/refresh")
    async def refresh_checkpoint_assets() -> dict[str, Any]:
        return await asyncio.to_thread(catalog.refresh_asset_type, "checkpoint")


    @router.get("/api/assets/checkpoints/{asset_id}")
    async def checkpoint_asset_details(asset_id: str, inspect: bool = True) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(catalog.checkpoint_details, asset_id, inspect_technical=inspect)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.patch("/api/assets/checkpoints/{asset_id}")
    async def update_checkpoint_asset(asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return catalog.update_asset_metadata("checkpoint", asset_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/checkpoints/{asset_id}/preview")
    async def checkpoint_asset_preview(asset_id: str) -> FileResponse:
        try:
            path = catalog.asset_preview_path("checkpoint", asset_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type=_preview_media_type(path))


    @router.post("/api/assets/checkpoints/{asset_id}/preview")
    async def replace_checkpoint_asset_preview(asset_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            return catalog.replace_asset_preview(
                "checkpoint",
                asset_id,
                filename=file.filename or "preview.png",
                content=await file.read(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/checkpoints/{asset_id}/preview/recent-outputs")
    async def checkpoint_asset_preview_candidates(asset_id: str, limit: int = 48) -> list[dict[str, Any]]:
        try:
            return catalog.asset_preview_candidates("checkpoint", asset_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.post("/api/assets/checkpoints/{asset_id}/preview/from-output")
    async def checkpoint_asset_preview_from_output(asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return catalog.replace_asset_preview_from_output("checkpoint", asset_id, str(payload.get("output_id") or ""))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/assets/checkpoints/{asset_id}/open-folder")
    async def open_checkpoint_asset_folder(asset_id: str) -> dict[str, Any]:
        try:
            record = catalog.asset_record("checkpoint", asset_id)
            target = Path(str(record.get("path") or "")).resolve().parent
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)], start_new_session=True)
            else:
                subprocess.Popen(["xdg-open", str(target)], start_new_session=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to open the checkpoint folder: {exc}") from exc
        return {"opened": True, "path": str(target)}


    @router.get("/api/assets/loras")
    async def lora_assets() -> dict[str, Any]:
        return catalog.asset_payload("lora")


    @router.post("/api/assets/loras/refresh")
    async def refresh_lora_assets() -> dict[str, Any]:
        payload = await asyncio.to_thread(catalog.refresh_asset_type, "lora")
        if _lora_auto_scan_enabled():
            return await asyncio.to_thread(catalog.scan_loras, mode="missing")
        return payload


    @router.post("/api/assets/loras/scan")
    async def scan_lora_assets(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        mode = str((payload or {}).get("mode") or "missing").strip().lower()
        try:
            return await asyncio.to_thread(catalog.scan_loras, mode=mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/assets/loras/civitai-metadata")
    async def enrich_lora_assets_from_civitai(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        mode = str((payload or {}).get("mode") or "missing").strip().lower()
        try:
            return await asyncio.to_thread(catalog.enrich_loras_from_civitai, mode=mode)
        except CivitaiCredentialError as exc:
            raise _civitai_credential_required(exc) from exc
        except CivitaiRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/assets/loras/{asset_id}/civitai-metadata")
    async def enrich_lora_asset_from_civitai(
        asset_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        overwrite = bool((payload or {}).get("overwrite", False))
        try:
            return await asyncio.to_thread(
                catalog.enrich_lora_from_civitai,
                asset_id,
                overwrite=overwrite,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CivitaiMetadataNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CivitaiCredentialError as exc:
            raise _civitai_credential_required(exc) from exc
        except CivitaiRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/loras/{asset_id}")
    async def lora_asset_details(asset_id: str, inspect: bool = True) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(catalog.lora_details, asset_id, inspect_technical=inspect)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.patch("/api/assets/loras/{asset_id}")
    async def update_lora_asset(asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return catalog.update_asset_metadata("lora", asset_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/loras/{asset_id}/preview")
    async def lora_asset_preview(asset_id: str) -> FileResponse:
        try:
            path = catalog.asset_preview_path("lora", asset_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type=_preview_media_type(path))


    @router.post("/api/assets/loras/{asset_id}/preview")
    async def replace_lora_asset_preview(asset_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            return catalog.replace_asset_preview(
                "lora",
                asset_id,
                filename=file.filename or "preview.png",
                content=await file.read(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/loras/{asset_id}/preview/recent-outputs")
    async def lora_asset_preview_candidates(asset_id: str, limit: int = 48) -> list[dict[str, Any]]:
        try:
            return catalog.asset_preview_candidates("lora", asset_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.post("/api/assets/loras/{asset_id}/preview/from-output")
    async def lora_asset_preview_from_output(asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return catalog.replace_asset_preview_from_output("lora", asset_id, str(payload.get("output_id") or ""))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/assets/loras/{asset_id}/open-folder")
    async def open_lora_asset_folder(asset_id: str) -> dict[str, Any]:
        try:
            record = catalog.asset_record("lora", asset_id)
            target = Path(str(record.get("path") or "")).resolve().parent
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)], start_new_session=True)
            else:
                subprocess.Popen(["xdg-open", str(target)], start_new_session=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to open the LoRA folder: {exc}") from exc
        return {"opened": True, "path": str(target)}


    @router.delete("/api/assets/loras/{asset_id}")
    async def delete_lora_asset(asset_id: str) -> dict[str, Any]:
        try:
            return catalog.delete_asset("lora", asset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/vaes")
    async def vae_assets() -> dict[str, Any]:
        return catalog.asset_payload("vae")


    @router.post("/api/assets/vaes/refresh")
    async def refresh_vae_assets() -> dict[str, Any]:
        return await asyncio.to_thread(catalog.refresh_asset_type, "vae")


    @router.get("/api/assets/vaes/{asset_id}")
    async def vae_asset_details(asset_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(catalog.vae_details, asset_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.patch("/api/assets/vaes/{asset_id}")
    async def update_vae_asset(asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return catalog.update_asset_metadata("vae", asset_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/vaes/{asset_id}/preview")
    async def vae_asset_preview(asset_id: str) -> FileResponse:
        try:
            path = catalog.asset_preview_path("vae", asset_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type=_preview_media_type(path))


    @router.post("/api/assets/vaes/{asset_id}/preview")
    async def replace_vae_asset_preview(asset_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            return catalog.replace_asset_preview(
                "vae",
                asset_id,
                filename=file.filename or "preview.png",
                content=await file.read(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/textual-inversions")
    async def textual_inversion_assets() -> dict[str, Any]:
        return catalog.asset_payload("textual_inversion")


    @router.post("/api/assets/textual-inversions/refresh")
    async def refresh_textual_inversion_assets() -> dict[str, Any]:
        return await asyncio.to_thread(catalog.refresh_asset_type, "textual_inversion")


    @router.get("/api/assets/textual-inversions/{asset_id}")
    async def textual_inversion_asset_details(asset_id: str) -> dict[str, Any]:
        try:
            return catalog.asset_record("textual_inversion", asset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.patch("/api/assets/textual-inversions/{asset_id}")
    async def update_textual_inversion_asset(asset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return catalog.update_asset_metadata("textual_inversion", asset_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/assets/textual-inversions/{asset_id}/preview")
    async def textual_inversion_asset_preview(asset_id: str) -> FileResponse:
        try:
            path = catalog.asset_preview_path("textual_inversion", asset_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type=_preview_media_type(path))


    @router.post("/api/assets/textual-inversions/{asset_id}/preview")
    async def replace_textual_inversion_asset_preview(asset_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            return catalog.replace_asset_preview(
                "textual_inversion",
                asset_id,
                filename=file.filename or "preview.png",
                content=await file.read(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/assets/textual-inversions/{asset_id}/open-folder")
    async def open_textual_inversion_asset_folder(asset_id: str) -> dict[str, Any]:
        try:
            record = catalog.asset_record("textual_inversion", asset_id)
            target = Path(str(record.get("path") or "")).resolve().parent
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)], start_new_session=True)
            else:
                subprocess.Popen(["xdg-open", str(target)], start_new_session=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to open the textual-inversion folder: {exc}") from exc
        return {"opened": True, "path": str(target)}


    return router
