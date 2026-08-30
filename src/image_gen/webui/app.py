from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

import yaml
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from image_gen.contracts import PROMPT_ASSET_CONTRACT_VERSION
from image_gen.program_metadata import PRODUCT_NAME, build_program_metadata
from image_gen.runtime.hires_sizing import resolve_hires_dimensions
from image_gen.systems.outpainting import plan_outpaint_canvas
from image_gen.systems.asset_hub import (
    ASSET_HUB_CONTRACT_VERSION,
    ArchitectureCompatibilityPolicy,
    AssetHubDownloadManager,
    AssetGalleryCache,
    AssetDiscoveryIndex,
    AssetHubInstaller,
    AssetHubSecretStore,
    AssetHubSelectionStore,
    AssetSearchSessionStore,
    AssetHubService,
    CivitaiProvider,
    DownloadRepository,
    DownloadRuntimeSettings,
    GalleryCacheSettings,
    InstallRepository,
    UpscalerFavoriteStore,
    LocalPresenceResolver,
)
from image_gen.runtime_options import (
    RuntimeStartupOptions,
    build_runtime_startup_status,
    build_runtime_command_from_status,
    merge_runtime_startup_settings,
    resolve_runtime_startup_options,
)
from image_gen.webui.batch_io import BatchIOService
from image_gen.webui.batch_replay import BatchReplayService
from image_gen.webui.bug_reports import BugReportError, BugReportService
from image_gen.webui.diagnostics import write_webui_failure_bundle
from image_gen.webui.default_assets import resolve_default_assets
from image_gen.webui.catalog import ASSET_CATALOG_CONTRACT_VERSION, WebUICatalog
from image_gen.webui.changelog import ChangelogService
from image_gen.webui.help_center import HELP_CENTER_CONTRACT_VERSION, HelpCenterService
from image_gen.webui.upscaler_catalog import (
    UPSCALER_CATALOG_CONTRACT_VERSION,
    WebUIUpscalerCatalog,
)
from image_gen.webui.civitai_asset_metadata import (
    CivitaiAssetMetadataService,
    CivitaiCredentialError,
    CivitaiMetadataNotFound,
    CivitaiRequestError,
    civitai_api_key_status,
    delete_civitai_api_key,
    write_civitai_api_key,
    sync_civitai_api_key_to_secret_store,
)
from image_gen.webui.image_refs import decode_external_image_ref, is_within_root
from image_gen.webui.jobs import GenerationJobManager
from image_gen.webui.generation_capabilities import GenerationCapabilityService
from image_gen.webui.model_selection import ModelSelectionUnavailableError, WebUIModelSelectionState
from modules.registry import ComponentRegistryService, ComponentSelectionService
from image_gen.webui.output_details import load_image_file_details, load_output_details
from image_gen.webui.prompt_configuration import PromptConfigurationService
from image_gen.webui.profile import ImageGenProfileService
from image_gen.webui.replay import ReplayService
from image_gen.webui.routes.asset_hub import build_asset_hub_router
from image_gen.webui.routes.theme_manager import build_theme_manager_router
from image_gen.webui.routes.system import build_system_router
from image_gen.webui.routes.prompt_tools import build_prompt_tools_router
from image_gen.webui.routes.bootstrap import build_bootstrap_router
from image_gen.webui.routes.assets import build_assets_router
from image_gen.webui.routes.models import build_models_router
from image_gen.webui.routes.outputs import build_outputs_router
from image_gen.webui.routes.replay import build_replay_router
from image_gen.webui.routes.workspace import build_workspace_router, _deep_merge_dict
from image_gen.webui.routes.settings import build_settings_router
from image_gen.webui.routes.hires_profiles import build_hires_profiles_router
from image_gen.webui.routes.jobs_api import build_jobs_router, encode_sse_event
from image_gen.webui.routes.static_pages import build_static_pages_router
from image_gen.webui.routes.payloads import (
    NamedPayload,
    PromptPresetPayload,
    ModelActivationPayload,
    OutputFolderPayload,
    CivitaiCredentialPayload,
)
from image_gen.webui.selection import WebUISelectionResolver
from image_gen.webui.store import WebUIStore
from image_gen.webui.theme import (
    THEME_LIBRARY_SCHEMA_VERSION,
    ThemePackageLibrary,
    ThemeStorageConfigurationError,
    ThemeStorageRoots,
)
from image_gen.webui.variation_matrix import VariationMatrixService
from image_gen.webui.workspace import (
    WORKSPACE_CONTRACT_VERSION,
    WORKSPACE_SCHEMA,
    WORKSPACE_SCHEMA_VERSION,
    WorkspaceRenderer,
    WorkspaceStore,
    WorkspaceStoreError,
    build_default_workspace_layouts,
    build_default_workspace_registry,
    review_workspace_import,
    workspace_responsive_contract_payload,
)
from modules.project_context import ProjectContext
from modules.registry.asset_registry import AssetRegistry
from modules.prompt_parsers import (
    PROMPT_MERGE_CONTRACT_VERSION,
    PROMPT_ROUTE_CONTRACT_VERSION,
    PROMPT_SHADOW_CONTRACT_VERSION,
    default_prompt_parser_registry,
)


WEBUI_VERSION = "0.1.74"


def create_app(
    project_root: str | Path | None = None,
    runtime_startup_options: RuntimeStartupOptions | dict[str, Any] | None = None,
    restart_callback: Callable[[], Any] | None = None,
) -> FastAPI:
    context = ProjectContext.load(project_root=project_root)
    server_instance_id = uuid.uuid4().hex
    server_started_at_unix = time.time()
    catalog = WebUICatalog(context)
    upscaler_catalog = WebUIUpscalerCatalog(context)
    civitai_connection = CivitaiAssetMetadataService(context)
    store = WebUIStore(context.data_root / "webui")
    application_settings = store.load_application_settings()
    profile = ImageGenProfileService(
        context.project_root,
        context.data_root,
        context.txt2img_output_root,
        startup_timestamp=server_started_at_unix,
    )
    if isinstance(runtime_startup_options, RuntimeStartupOptions):
        resolved_runtime_startup = merge_runtime_startup_settings(
            runtime_startup_options,
            application_settings,
        )
    elif isinstance(runtime_startup_options, dict):
        resolved_runtime_startup = merge_runtime_startup_settings(
            RuntimeStartupOptions.from_mapping(runtime_startup_options),
            application_settings,
        )
    else:
        resolved_runtime_startup = resolve_runtime_startup_options(
            settings=application_settings,
        )
    jobs = GenerationJobManager(
        context,
        settings_provider=store.load_application_settings,
        recent_output_provider=lambda path: catalog.output_summary_from_path(Path(path)),
        output_record_callback=profile.record_generated_image,
    )
    jobs.runtime_startup_options = resolved_runtime_startup.to_dict()
    model_selection = WebUIModelSelectionState(context)
    component_selection = ComponentSelectionService(context)
    component_registry = ComponentRegistryService(context, registry=component_selection.registry)
    generation_capabilities = GenerationCapabilityService(
        model_selection=model_selection,
        catalog=catalog,
        upscaler_catalog=upscaler_catalog,
        component_selection=component_selection,
        context=context,
    )
    selections = WebUISelectionResolver(jobs.registry)
    replay = ReplayService(context, jobs, model_selection, upscaler_catalog=upscaler_catalog)
    batch_replay = BatchReplayService(replay, jobs, model_selection)
    batch_io = BatchIOService(context, jobs, model_selection)
    variations = VariationMatrixService(context, jobs, model_selection, batch_io)
    prompt_parsers = default_prompt_parser_registry()
    prompt_configuration = PromptConfigurationService(store, parser_registry=prompt_parsers)
    bug_reports = BugReportService(context.project_root)
    changelog = ChangelogService(context.project_root)
    help_center = HelpCenterService(context.project_root)
    static_root = Path(__file__).resolve().parent / "static"
    workspace_registry = build_default_workspace_registry()
    workspace_defaults = build_default_workspace_layouts()
    workspace_store = WorkspaceStore(
        context.data_root / "webui" / "workspaces",
        workspace_registry,
        workspace_defaults,
    )
    workspace_renderer = WorkspaceRenderer(workspace_registry)

    theme_storage_warning = ""
    try:
        theme_roots = ThemeStorageRoots.resolve(
            project_root=context.project_root,
            settings=application_settings.get("theme_storage"),
        )
    except ThemeStorageConfigurationError as exc:
        theme_storage_warning = str(exc)
        theme_roots = ThemeStorageRoots.resolve(project_root=context.project_root, settings={})
    theme_library = ThemePackageLibrary(
        theme_roots,
        legacy_palette_provider=lambda: store.load_application_settings().get("theme_palette", {}),
    )

    def _asset_hub_local_records():
        for asset_kind in ("checkpoint", "lora", "vae", "textual_inversion"):
            for record in catalog.asset_list(asset_kind):
                yield record
        for record in upscaler_catalog.payload().get("neural", []):
            if isinstance(record, Mapping):
                item = dict(record)
                item.setdefault("asset_type", "upscaler")
                yield item

    asset_hub_secrets = AssetHubSecretStore()
    try:
        sync_civitai_api_key_to_secret_store(context, asset_hub_secrets)
    except CivitaiCredentialError:
        # The WebUI connection dialog owns creation/repair of the existing
        # secrets/civitai_api_key.txt credential. Asset Hub stays disconnected
        # until that same file becomes usable.
        pass
    civitai_provider = CivitaiProvider(
        context.cache_root / "asset-hub" / "providers" / "civitai",
        secret_provider=lambda: asset_hub_secrets.get("civitai"),
    )
    asset_hub_presence = LocalPresenceResolver(_asset_hub_local_records)
    asset_hub_policy = ArchitectureCompatibilityPolicy()
    asset_hub = AssetHubService(
        [civitai_provider],
        policy=asset_hub_policy,
        presence=asset_hub_presence,
    )
    asset_hub_discovery_database = context.data_root / "asset-hub" / "asset-discovery.sqlite3"
    asset_hub_discovery_index = AssetDiscoveryIndex(
        asset_hub_discovery_database,
        presence=asset_hub_presence,
        policy=asset_hub_policy,
    )
    asset_hub_search_sessions = AssetSearchSessionStore(asset_hub_discovery_database)
    asset_hub_database = context.data_root / "asset-hub" / "asset-hub.sqlite3"
    asset_hub_download_repository = DownloadRepository(asset_hub_database)
    asset_hub_downloads = AssetHubDownloadManager(
        asset_hub.providers,
        secret_store=asset_hub_secrets,
        repository=asset_hub_download_repository,
        temporary_root=context.temporary_root,
        report_root=context.data_root / "asset-hub" / "reports" / "downloads",
        settings=DownloadRuntimeSettings.from_config(context.config),
    )
    asset_hub_gallery_cache = AssetGalleryCache(
        context.cache_root / "asset-hub" / "gallery",
        asset_hub_database,
        settings=GalleryCacheSettings.from_config(context.config),
    )
    asset_hub_install_repository = InstallRepository(asset_hub_database)
    asset_hub_registry = AssetRegistry(str(context.registry_db_path))
    asset_hub_upscaler_favorites = UpscalerFavoriteStore(context.cache_root)
    asset_hub_selections = AssetHubSelectionStore(context.data_root / "asset-hub" / "selections")
    asset_hub_installer = AssetHubInstaller(
        context=context,
        service=asset_hub,
        downloads=asset_hub_downloads,
        catalog=catalog,
        upscaler_catalog=upscaler_catalog,
        registry=asset_hub_registry,
        repository=asset_hub_install_repository,
        discovery_index=asset_hub_discovery_index,
    )

    asset_hub_downloads.set_completion_handler(asset_hub_installer.auto_install_download)

    def _lora_auto_scan_enabled() -> bool:
        settings = store.load_application_settings()
        return bool(settings.get("lora_auto_scan_unknown_on_startup", True))

    async def _scan_unknown_loras_in_background() -> None:
        try:
            await asyncio.to_thread(catalog.scan_loras, mode="missing")
        except Exception:
            pass

    async def _recover_asset_hub_downloads() -> None:
        try:
            await asyncio.to_thread(asset_hub_downloads.cleanup_stale_partials)
            await asset_hub_downloads.reconcile_completed()
        except Exception:
            pass

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await jobs.start()
        lora_scan_task = None
        asset_hub_recovery_task = asyncio.create_task(_recover_asset_hub_downloads())
        if _lora_auto_scan_enabled():
            lora_scan_task = asyncio.create_task(_scan_unknown_loras_in_background())
        try:
            try:
                bug_profile = (await asyncio.to_thread(bug_reports.refresh_local)).get("profile", {})
                await asyncio.to_thread(profile.publish_presence, bug_profile, active=True)
            except Exception:
                pass
            yield
        finally:
            if asset_hub_recovery_task is not None and not asset_hub_recovery_task.done():
                asset_hub_recovery_task.cancel()
            if lora_scan_task is not None and not lora_scan_task.done():
                lora_scan_task.cancel()
            try:
                await asyncio.to_thread(profile.publish_presence, {}, active=False)
            except Exception:
                pass
            await jobs.stop()

    app = FastAPI(title=f"{PRODUCT_NAME} WebUI", version=WEBUI_VERSION, lifespan=lifespan)
    app.state.context = context
    app.state.catalog = catalog
    app.state.store = store
    app.state.jobs = jobs
    app.state.model_selection = model_selection
    app.state.replay = replay
    app.state.batch_replay = batch_replay
    app.state.batch_io = batch_io
    app.state.variations = variations
    app.state.prompt_parsers = prompt_parsers
    app.state.prompt_configuration = prompt_configuration
    app.state.profile = profile
    app.state.bug_reports = bug_reports
    app.state.changelog = changelog
    app.state.help_center = help_center
    app.state.workspace_registry = workspace_registry
    app.state.workspace_store = workspace_store
    app.state.workspace_renderer = workspace_renderer
    app.state.asset_hub = asset_hub
    app.state.asset_hub_secrets = asset_hub_secrets
    app.state.asset_hub_downloads = asset_hub_downloads
    app.state.asset_hub_gallery_cache = asset_hub_gallery_cache
    app.state.asset_hub_installer = asset_hub_installer
    app.state.asset_hub_registry = asset_hub_registry
    app.state.asset_hub_upscaler_favorites = asset_hub_upscaler_favorites
    app.state.asset_hub_selections = asset_hub_selections
    app.state.asset_hub_discovery_index = asset_hub_discovery_index
    app.state.asset_hub_search_sessions = asset_hub_search_sessions
    app.include_router(build_asset_hub_router(
        asset_hub,
        secrets=asset_hub_secrets,
        downloads=asset_hub_downloads,
        installer=asset_hub_installer,
        upscaler_catalog=upscaler_catalog,
        upscaler_favorites=asset_hub_upscaler_favorites,
        selections=asset_hub_selections,
        discovery_index=asset_hub_discovery_index,
        search_sessions=asset_hub_search_sessions,
        gallery_cache=asset_hub_gallery_cache,
        user_config_path=context.config_path,
    ))
    app.state.theme_library = theme_library
    app.state.theme_storage_warning = theme_storage_warning
    app.include_router(build_theme_manager_router(theme_library))
    app.state.runtime_startup_options = resolved_runtime_startup.to_dict()
    app.state.server_instance_id = server_instance_id
    app.state.server_started_at_unix = server_started_at_unix
    app.state.restart_callback = restart_callback

    def _default_asset_payload(model: dict[str, Any] | None = None, document: dict[str, Any] | None = None) -> dict[str, Any]:
        profiles = document if isinstance(document, dict) else store.load_default_asset_profiles()
        active = model if isinstance(model, dict) else model_selection.current_payload()
        return resolve_default_assets(profiles, active)

    def _runtime_startup_status() -> dict[str, Any]:
        worker_status = jobs.model_runtime.status()
        worker_ready = worker_status.get("ready")
        return build_runtime_startup_status(
            app.state.runtime_startup_options,
            store.load_application_settings(),
            worker_ready=worker_ready if isinstance(worker_ready, dict) else None,
            worker_status=worker_status,
        )

    def _recent_output_browser_settings() -> dict[str, Any]:
        settings = store.load_application_settings()
        browser = settings.get("recent_outputs_browser") or {}
        if not isinstance(browser, dict):
            browser = {}
        return {
            "time_window": str(browser.get("time_window", "72") or "72"),
            "custom_hours": int(browser.get("custom_hours", 24) or 24),
            "include_subfolders": bool(browser.get("include_subfolders", True)),
            "source_paths": [str(item) for item in (browser.get("source_paths") or []) if str(item).strip()],
            "require_metadata_for_external": bool(browser.get("require_metadata_for_external", True)),
        }

    def _resolved_recent_output_hours(raw_value: str | int | None, custom_hours: int) -> int | None:
        value = str(raw_value or "72").strip().lower()
        if value in {"all", "0", "none"}:
            return None
        if value == "custom":
            return max(1, int(custom_hours or 24))
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return 72
        return max(1, numeric)

    def _visible_recent_outputs(
        limit: int | None = None,
        *,
        hours: int | None = None,
        include_subfolders: bool | None = None,
        source_paths: list[str] | None = None,
        require_metadata_for_external: bool | None = None,
    ) -> list[dict[str, Any]]:
        settings = _recent_output_browser_settings()
        visibility = store.load_recent_output_visibility()
        cleared_through = int(visibility.get("cleared_through_modified_ns", 0) or 0)
        effective_hours = _resolved_recent_output_hours(settings["time_window"], settings["custom_hours"]) if hours is None else hours
        effective_include_subfolders = settings["include_subfolders"] if include_subfolders is None else include_subfolders
        effective_paths = settings["source_paths"] if source_paths is None else source_paths
        effective_metadata_gate = settings["require_metadata_for_external"] if require_metadata_for_external is None else require_metadata_for_external
        try:
            catalog_items = catalog.recent_outputs(
                limit=limit,
                hours=effective_hours,
                include_subfolders=effective_include_subfolders,
                extra_paths=effective_paths,
                require_metadata_for_external=effective_metadata_gate,
            )
        except TypeError as exc:
            # Preserve compatibility with tests/extensions that replace the
            # catalog method using the older limit-only callable contract.
            if "unexpected keyword argument" not in str(exc):
                raise
            catalog_items = catalog.recent_outputs(limit=limit)
        return [
            item
            for item in catalog_items
            if int(item.get("modified_ns", 0) or 0) > cleared_through
        ]

    def _allowed_recent_output_roots() -> list[Path]:
        browser = _recent_output_browser_settings()
        return catalog.configured_output_roots(browser.get("source_paths") or [])

    def _resolve_external_image_ref_or_404(image_ref: str) -> Path:
        path = decode_external_image_ref(image_ref)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="The requested image file is no longer available.")
        if not any(is_within_root(path, root) for root in _allowed_recent_output_roots()):
            raise HTTPException(status_code=403, detail="The requested image file is outside the allowed recent-output folders.")
        return path

    def _webui_failure(
        stage: str,
        exc: BaseException,
        *,
        payload: Any = None,
        request_path: str | None = None,
        status_code: int = 400,
        extra: dict[str, Any] | None = None,
    ) -> HTTPException:
        try:
            bundle = write_webui_failure_bundle(
                project_root=context.project_root,
                stage=stage,
                error=exc,
                payload=payload,
                request_path=request_path,
                extra=extra,
            )
            detail = f"{exc} (diagnostic bundle: {bundle})"
        except Exception:
            detail = str(exc)
        return HTTPException(status_code=status_code, detail=detail)

    @app.exception_handler(Exception)
    async def diagnose_unhandled_webui_error(request: Request, exc: Exception) -> JSONResponse:
        try:
            bundle = write_webui_failure_bundle(
                project_root=context.project_root,
                stage="unhandled_request",
                error=exc,
                request_path=request.url.path,
                extra={
                    "method": request.method,
                    "query": str(request.url.query or ""),
                    "client": request.client.host if request.client else None,
                },
            )
            detail = f"Unhandled WebUI error: {exc} (diagnostic bundle: {bundle})"
        except Exception:
            detail = f"Unhandled WebUI error: {exc}"
        return JSONResponse(status_code=500, content={"detail": detail})

    @app.middleware("http")
    async def disable_webui_shell_caching(request: Request, call_next):
        response = await call_next(request)
        if request.url.path in {"/", "/region-builder.html"} or request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.mount("/assets", StaticFiles(directory=static_root / "assets"), name="assets")
    app.mount(
        "/outputs",
        StaticFiles(directory=context.txt2img_output_root, check_dir=False),
        name="outputs",
    )

    app.include_router(build_system_router(
        app=app,
        context=context,
        jobs=jobs,
        model_selection=model_selection,
        prompt_parsers=prompt_parsers,
        server_instance_id=server_instance_id,
        server_started_at_unix=server_started_at_unix,
        changelog=changelog,
        help_center=help_center,
        profile=profile,
        bug_reports=bug_reports,
        write_webui_failure_bundle=lambda **kwargs: write_webui_failure_bundle(**kwargs),
        WEBUI_VERSION=WEBUI_VERSION,
    ))

    def _preview_media_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        return "application/octet-stream"


    app.include_router(build_prompt_tools_router(
        prompt_parsers=prompt_parsers,
        prompt_configuration=prompt_configuration,
    ))


    app.include_router(build_bootstrap_router(
        context=context,
        catalog=catalog,
        upscaler_catalog=upscaler_catalog,
        jobs=jobs,
        model_selection=model_selection,
        generation_capabilities=generation_capabilities,
        prompt_configuration=prompt_configuration,
        selections=selections,
        store=store,
        theme_library=theme_library,
        theme_storage_warning=theme_storage_warning,
        _default_asset_payload=_default_asset_payload,
        _runtime_startup_status=_runtime_startup_status,
        _visible_recent_outputs=_visible_recent_outputs,
        WEBUI_VERSION=WEBUI_VERSION,
    ))


    app.include_router(build_assets_router(
        context=context,
        catalog=catalog,
        upscaler_catalog=upscaler_catalog,
        civitai_connection=civitai_connection,
        asset_hub_secrets=asset_hub_secrets,
        _lora_auto_scan_enabled=_lora_auto_scan_enabled,
        _preview_media_type=_preview_media_type,
    ))


    app.include_router(build_models_router(
        context=context,
        catalog=catalog,
        component_registry=component_registry,
        component_selection=component_selection,
        jobs=jobs,
        model_selection=model_selection,
        generation_capabilities=generation_capabilities,
        _default_asset_payload=_default_asset_payload,
        _webui_failure=_webui_failure,
    ))


    app.include_router(build_outputs_router(
        context=context,
        catalog=catalog,
        store=store,
        prompt_configuration=prompt_configuration,
        upscaler_catalog=upscaler_catalog,
        _preview_media_type=_preview_media_type,
        _recent_output_browser_settings=_recent_output_browser_settings,
        _resolve_external_image_ref_or_404=_resolve_external_image_ref_or_404,
        _visible_recent_outputs=_visible_recent_outputs,
    ))


    app.include_router(build_replay_router(
        replay=replay,
        batch_replay=batch_replay,
        batch_io=batch_io,
        variations=variations,
        _webui_failure=_webui_failure,
    ))


    app.include_router(build_workspace_router(
        store=store,
        workspace_defaults=workspace_defaults,
        workspace_registry=workspace_registry,
        workspace_store=workspace_store,
        _default_asset_payload=_default_asset_payload,
    ))


    app.include_router(build_settings_router(
        context=context,
        store=store,
        theme_library=theme_library,
        registry=jobs.registry,
        jobs=jobs,
        _runtime_startup_status=_runtime_startup_status,
    ))


    app.include_router(build_hires_profiles_router(
        store=store,
        catalog=catalog,
        upscaler_catalog=upscaler_catalog,
        prompt_configuration=prompt_configuration,
    ))


    app.include_router(build_jobs_router(
        jobs=jobs,
        catalog=catalog,
        component_selection=component_selection,
        model_selection=model_selection,
        generation_capabilities=generation_capabilities,
        prompt_configuration=prompt_configuration,
        upscaler_catalog=upscaler_catalog,
        _preview_media_type=_preview_media_type,
        _webui_failure=_webui_failure,
    ))


    app.include_router(build_static_pages_router(static_root=static_root))


    return app
