from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, HTTPException, UploadFile

from image_gen.contracts import PROMPT_ASSET_CONTRACT_VERSION
from image_gen.program_metadata import build_program_metadata
from image_gen.runtime.hires_sizing import resolve_hires_dimensions
from image_gen.systems.asset_hub import ASSET_HUB_CONTRACT_VERSION
from image_gen.systems.outpainting import plan_outpaint_canvas
from image_gen.webui.catalog import ASSET_CATALOG_CONTRACT_VERSION
from image_gen.webui.help_center import HELP_CENTER_CONTRACT_VERSION
from image_gen.webui.generation_capabilities import (
    GENERATION_CAPABILITY_CONTRACT_VERSION,
    GENERATION_CAPABILITY_SCHEMA,
)
from image_gen.webui.output_details import load_image_file_details
from image_gen.webui.theme import THEME_LIBRARY_SCHEMA_VERSION
from image_gen.webui.upscaler_catalog import UPSCALER_CATALOG_CONTRACT_VERSION
from image_gen.webui.workspace import WORKSPACE_CONTRACT_VERSION


def build_bootstrap_router(*, context, catalog, upscaler_catalog, jobs, model_selection, generation_capabilities, prompt_configuration, selections, store, theme_library, theme_storage_warning, _default_asset_payload, _runtime_startup_status, _visible_recent_outputs, WEBUI_VERSION) -> APIRouter:
    router = APIRouter()

    @router.get("/api/bootstrap")
    async def bootstrap() -> dict[str, Any]:
        raw_defaults = context.generation_defaults()
        default_selection = selections.normalize(
            raw_defaults,
            migrate_legacy_auto_fallback=False,
        )
        defaults = default_selection.payload

        settings = store.load_application_settings()
        settings["theme_effective_palette"] = theme_library.resolve_effective_palette()
        settings["theme_library_activation"] = theme_library.library_payload().get("activation", {})
        if theme_storage_warning:
            settings["theme_storage_warning"] = theme_storage_warning
        session = store.load_session() if settings.get("restore_last_session", True) else {}
        effective_source = {**defaults, **session}
        if session:
            # Do not let the current defaults make an old stored session appear
            # already migrated. Missing metadata means legacy version zero.
            effective_source["_webui_selection_version"] = session.get(
                "_webui_selection_version", 0
            )
            effective_source["_webui_scheduler_user_selected"] = session.get(
                "_webui_scheduler_user_selected", False
            )
        effective_selection = selections.normalize(
            effective_source,
            fallback_payload=defaults,
            migrate_legacy_auto_fallback=True,
            repair_incompatible_explicit=True,
        )
        effective_generation = effective_selection.payload

        if session:
            migrated_session = dict(session)
            for key in (
                "sampler_name",
                "scheduler_name",
                "scheduler_kwargs",
                "_webui_selection_version",
                "_webui_scheduler_user_selected",
            ):
                if key in effective_generation:
                    migrated_session[key] = effective_generation[key]
            if migrated_session != session:
                session = store.save_session(migrated_session)

        return {
            "version": WEBUI_VERSION,
            "application": build_program_metadata(context.project_root),
            "project_root": str(context.project_root),
            "output_root": str(context.txt2img_output_root),
            "runtime_paths": {
                "config_path": str(context.config_path),
                "checkpoints_dir": str(context.checkpoints_dir),
                "vae_dir": str(context.vae_dir),
                "output_dir": str(context.txt2img_output_root),
            },
            "api_contract": {
                "asset_catalog_contract_version": ASSET_CATALOG_CONTRACT_VERSION,
                "prompt_asset_contract_version": PROMPT_ASSET_CONTRACT_VERSION,
                "upscaler_catalog_contract_version": UPSCALER_CATALOG_CONTRACT_VERSION,
                "generation_capability_contract_version": GENERATION_CAPABILITY_CONTRACT_VERSION,
                "generation_capability_schema": GENERATION_CAPABILITY_SCHEMA,
                "generation_capability_routes": [
                    "/api/generation/capabilities",
                    "/api/models/active",
                ],
                "model_activation_routes": [
                    "/api/models/activate",
                    "/api/model/activate",
                    "/api/activate-model",
                ],
                "replay_routes": [
                    "/api/replay/preflight",
                    "/api/replay/submit",
                    "/api/replay/batch/preflight",
                    "/api/replay/batch/submit",
                ],
                "batch_io_routes": [
                    "/api/batch/import/parse",
                    "/api/batch/import/preflight",
                    "/api/batch/import/submit",
                    "/api/batch/export",
                ],
                "variation_routes": [
                    "/api/variations/preflight",
                    "/api/variations/submit",
                    "/api/variations/export",
                ],
                "prompt_parser_routes": [
                    "/api/prompt-parsers",
                    "/api/prompt-shortcut-profiles",
                    "/api/prompt-parser-presets",
                    "/api/prompts/translate",
                    "/api/prompts/preflight",
                ],
                "default_asset_routes": [
                    "/api/default-assets",
                ],
                "upscaler_catalog_routes": [
                    "/api/upscalers",
                    "/api/upscalers/refresh",
                    "/api/hires/dimension-plan",
                ],
                "asset_catalog_routes": [
                    "/api/assets/catalog",
                    "/api/assets/refresh",
                    "/api/assets/checkpoints",
                    "/api/assets/loras",
                    "/api/assets/vaes",
                    "/api/assets/textual-inversions",
                    "/api/civitai/assets/{asset_type}/metadata",
                    "/api/civitai/assets/{asset_type}/{asset_id}/metadata",
                ],
                "asset_hub_contract_version": ASSET_HUB_CONTRACT_VERSION,
                "asset_hub_routes": [
                    "/api/asset-hub/providers",
                    "/api/asset-hub/providers/civitai/status",
                    "/api/asset-hub/search",
                    "/api/asset-hub/models/civitai/{model_id}",
                    "/api/asset-hub/versions/civitai/{version_id}",
                    "/api/asset-hub/hash/civitai/{hash}",
                    "/api/asset-hub/providers/civitai/secret",
                    "/api/asset-hub/providers/civitai/secret/validate",
                    "/api/asset-hub/download-plans",
                    "/api/asset-hub/download-jobs",
                    "/api/asset-hub/download-jobs/{job_id}",
                    "/api/asset-hub/download-jobs/{job_id}/cancel",
                    "/api/asset-hub/download-jobs/{job_id}/resume",
                    "/api/asset-hub/download-jobs/{job_id}/events",
                    "/api/asset-hub/install-plans",
                    "/api/asset-hub/install-jobs",
                    "/api/asset-hub/install-jobs/{job_id}",
                    "/api/asset-hub/installed",
                    "/api/asset-hub/installed/{install_id}",
                    "/api/asset-hub/installed/{install_id}/refresh",
                    "/api/asset-hub/upscalers/compatibility",
                    "/api/asset-hub/upscalers/favorites",
                ],
                "theme_library_contract_version": THEME_LIBRARY_SCHEMA_VERSION,
                "theme_library_routes": [
                    "/api/themes/library",
                    "/api/themes/effective",
                    "/api/themes/import",
                    "/api/themes/{package_id}/enable",
                    "/api/themes/{package_id}/disable",
                    "/api/themes/{package_id}",
                ],
                "help_center_contract_version": HELP_CENTER_CONTRACT_VERSION,
                "help_center_routes": [
                    "/api/help",
                    "/api/help/search",
                    "/api/help/topic/{topic_id}",
                    "/api/help/media/{media_path}",
                ],
                "workspace_layout_routes": [
                    "/api/workspace/layout",
                    "/api/workspace/layout/reset",
                ],
                "workspace_foundation_contract_version": WORKSPACE_CONTRACT_VERSION,
                "workspace_foundation_routes": [
                    "/api/workspaces/definitions",
                    "/api/workspaces/import/validate",
                    "/api/workspaces/{page_id}",
                    "/api/workspaces/{page_id}/active",
                    "/api/workspaces/{page_id}/activate",
                    "/api/workspaces/{page_id}/duplicate",
                    "/api/workspaces/{page_id}/rename",
                    "/api/workspaces/{page_id}/reset",
                ],
                "lora_asset_routes": [
                    "/api/assets/loras",
                    "/api/assets/loras/refresh",
                    "/api/assets/loras/{asset_id}",
                    "/api/assets/loras/{asset_id}/preview",
                    "/api/assets/loras/scan",
                ],
            },
            "defaults": defaults,
            "session": session,
            "effective_generation": effective_generation,
            "selection_notes": [
                *default_selection.notes,
                *effective_selection.notes,
            ],
            "settings": settings,
            "default_assets": _default_asset_payload(),
            "runtime_startup_status": _runtime_startup_status(),
            "plugins": catalog.plugins(),
            "upscalers": upscaler_catalog.payload(),
            **prompt_configuration.bootstrap_payload(),
            "models": catalog.model_payload(),
            "active_model": model_selection.current_payload(),
            "generation_capabilities": generation_capabilities.resolve_active(request=effective_generation),
            "model_runtime": jobs.model_runtime_status(),
            "recent_outputs": _visible_recent_outputs(),
            "prompt_presets": store.list_prompt_presets(),
            "generation_profiles": store.list_profiles("generation"),
            "worker": jobs.status(),
        }


    @router.get("/api/upscalers")
    async def upscaler_catalog_status() -> dict[str, Any]:
        return upscaler_catalog.payload()


    @router.post("/api/hires/dimension-plan")
    async def hires_dimension_plan(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            plan = resolve_hires_dimensions(dict(payload or {}))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"contract_version": plan.contract_version, "plan": plan.to_dict()}


    @router.post("/api/outpaint/prototype/source")
    async def upload_outpaint_prototype_source(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "source.png").suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            raise HTTPException(status_code=400, detail="Outpaint prototype source must be a supported image file.")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Outpaint prototype source is empty.")
        if len(data) > 64 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Outpaint prototype source exceeds the 64 MiB prototype limit.")
        temp_root = context.data_root / "webui" / "temp" / "outpaint-prototype"
        temp_root.mkdir(parents=True, exist_ok=True)
        target = temp_root / f"source-{uuid.uuid4().hex}{suffix}"
        target.write_bytes(data)
        try:
            details = load_image_file_details(
                context, target, display_name=file.filename or target.name
            ).to_dict()
        except (OSError, ValueError) as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Unable to read outpaint source image: {exc}") from exc
        image = dict(details.get("image") or {})
        replay_payload = dict(details.get("replay") or {})
        return {
            "path": str(target.resolve()),
            "filename": file.filename or target.name,
            "width": int(image.get("width") or 0),
            "height": int(image.get("height") or 0),
            "metadata_source": str(details.get("metadata_source") or ""),
            "positive_prompt": str(replay_payload.get("positive_prompt") or ""),
            "negative_prompt": str(replay_payload.get("negative_prompt") or ""),
            "metadata_available": bool(replay_payload),
            "warnings": list(details.get("warnings") or []),
        }


    @router.post("/api/outpaint/prototype/plan")
    async def outpaint_prototype_plan(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            explicit_x = int(payload.get("source_x", -1))
            explicit_y = int(payload.get("source_y", -1))
            plan = plan_outpaint_canvas(
                source_width=int(payload.get("source_width") or 0),
                source_height=int(payload.get("source_height") or 0),
                target_width=int(payload.get("target_width") or 0),
                target_height=int(payload.get("target_height") or 0),
                anchor=str(payload.get("anchor") or "center"),
                feather_px=int(payload.get("feather_px", 24)),
                source_x=None if explicit_x < 0 else explicit_x,
                source_y=None if explicit_y < 0 else explicit_y,
            )
            return plan.to_dict()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/upscalers/refresh")
    async def refresh_upscaler_catalog(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        values = payload or {}
        mode = str(values.get("mode") or "all").strip().casefold()
        selected_file = str(values.get("selected_file") or "").strip() or None
        try:
            return await asyncio.to_thread(
                upscaler_catalog.refresh,
                mode=mode,
                selected_file=selected_file,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    return router
