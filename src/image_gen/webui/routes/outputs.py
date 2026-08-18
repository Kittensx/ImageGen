from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from image_gen.webui.image_refs import is_within_root
from image_gen.webui.output_details import load_image_file_details, load_output_details
from image_gen.webui.routes.payloads import OutputFolderPayload


def build_outputs_router(*, context, catalog, store, prompt_configuration, upscaler_catalog, _preview_media_type, _recent_output_browser_settings, _resolve_external_image_ref_or_404, _visible_recent_outputs) -> APIRouter:
    router = APIRouter()

    @router.post("/api/workspace/reload")
    async def reload_workspace() -> dict[str, Any]:
        catalog.reload_plugins()
        catalog.refresh_models()
        return {
            "plugins": catalog.plugins(),
            "upscalers": upscaler_catalog.payload(),
            **prompt_configuration.bootstrap_payload(),
            "models": catalog.model_payload(),
            "recent_outputs": _visible_recent_outputs(),
        }


    @router.get("/api/recent-outputs")
    async def recent_outputs(
        limit: int | None = None,
        hours: int | None = None,
        include_subfolders: bool | None = None,
        source_paths: str = "",
        require_metadata_for_external: bool | None = None,
    ) -> list[dict[str, Any]]:
        extra_paths = [item for item in source_paths.split("|") if item.strip()] if source_paths else None
        return _visible_recent_outputs(
            limit=limit,
            hours=hours,
            include_subfolders=include_subfolders,
            source_paths=extra_paths,
            require_metadata_for_external=require_metadata_for_external,
        )


    @router.post("/api/recent-outputs/reload")
    async def reload_recent_outputs() -> JSONResponse:
        # A manual folder reload is an explicit full disk reconciliation. Reset
        # the non-destructive clear marker, invalidate metadata summaries, and
        # scan all configured folders regardless of the previous time window.
        store.restore_recent_outputs_visibility()
        store.save_application_settings({
            "recent_outputs_browser": {"time_window": "all"}
        })
        catalog.invalidate_output_cache()
        browser = _recent_output_browser_settings()
        items = catalog.recent_outputs(
            limit=None,
            hours=None,
            include_subfolders=browser["include_subfolders"],
            extra_paths=browser["source_paths"],
            require_metadata_for_external=browser["require_metadata_for_external"],
        )
        return JSONResponse({
            "recent_outputs": items,
            "count": len(items),
            "time_window": "all",
            "full_rescan": True,
        })


    @router.post("/api/recent-outputs/clear")
    async def clear_recent_outputs() -> dict[str, Any]:
        visible = _visible_recent_outputs(limit=None)
        newest_modified_ns = max(
            (int(item.get("modified_ns", 0) or 0) for item in visible),
            default=0,
        )
        visibility = store.clear_recent_outputs_through(newest_modified_ns)
        return {
            "cleared_count": len(visible),
            "files_deleted": 0,
            "cleared_through_modified_ns": visibility["cleared_through_modified_ns"],
            "recent_outputs": _visible_recent_outputs(),
        }


    @router.post("/api/outputs/open-folder")
    async def open_output_folder(payload: OutputFolderPayload) -> dict[str, Any]:
        raw_path = str(payload.path or "").strip()
        target = (
            catalog.resolve_output_root(raw_path)
            if raw_path
            else context.txt2img_output_root.resolve()
        )
        allowed_roots = catalog.configured_output_roots(
            _recent_output_browser_settings().get("source_paths") or []
        )
        if not any(target == root or is_within_root(target, root) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="The requested folder is outside configured output locations.")
        target.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)], start_new_session=True)
            else:
                subprocess.Popen(["xdg-open", str(target)], start_new_session=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to open the output folder: {exc}") from exc
        return {"opened": True, "path": str(target)}


    @router.get("/api/image-files/{image_ref:path}")
    async def recent_output_image(image_ref: str) -> FileResponse:
        path = _resolve_external_image_ref_or_404(image_ref)
        return FileResponse(path, media_type=_preview_media_type(path))


    @router.get("/api/image-files/{image_ref:path}/details")
    async def recent_output_image_details(image_ref: str) -> dict[str, Any]:
        path = _resolve_external_image_ref_or_404(image_ref)
        return load_image_file_details(context, path, display_name=path.name).to_dict()


    @router.get("/api/outputs/{output_id:path}/details")
    async def output_details(output_id: str) -> dict[str, Any]:
        try:
            return load_output_details(context, output_id).to_dict()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/outputs/inspect-upload")
    async def inspect_uploaded_output(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "upload.png").suffix or ".png"
        temp_root = context.data_root / "webui" / "temp" / "uploaded-output-details"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_path = temp_root / f"upload-{abs(hash((file.filename or '', suffix, id(file))))}{suffix}"
        data = await file.read()
        temp_path.write_bytes(data)
        try:
            return load_image_file_details(context, temp_path, display_name=file.filename or temp_path.name).to_dict()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            try:
                temp_path.unlink(missing_ok=True)
                for sidecar in (temp_path.with_suffix(".json"), temp_path.with_suffix(".txt")):
                    sidecar.unlink(missing_ok=True)
            except OSError:
                pass


    return router
