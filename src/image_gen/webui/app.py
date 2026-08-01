from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from image_gen.runtime_options import (
    RuntimeStartupOptions,
    build_runtime_startup_status,
    build_runtime_command_from_status,
    merge_runtime_startup_settings,
    resolve_runtime_startup_options,
)
from image_gen.webui.batch_io import BatchIOService
from image_gen.webui.batch_replay import BatchReplayService
from image_gen.webui.diagnostics import write_webui_failure_bundle
from image_gen.webui.catalog import WebUICatalog
from image_gen.webui.image_refs import decode_external_image_ref, is_within_root
from image_gen.webui.jobs import GenerationJobManager
from image_gen.webui.model_selection import WebUIModelSelectionState
from image_gen.webui.output_details import load_image_file_details, load_output_details
from image_gen.webui.prompt_configuration import PromptConfigurationService
from image_gen.webui.replay import ReplayService
from image_gen.webui.selection import WebUISelectionResolver
from image_gen.webui.store import WebUIStore
from image_gen.webui.variation_matrix import VariationMatrixService
from modules.project_context import ProjectContext
from modules.prompt_parsers import (
    PROMPT_MERGE_CONTRACT_VERSION,
    PROMPT_ROUTE_CONTRACT_VERSION,
    PROMPT_SHADOW_CONTRACT_VERSION,
    default_prompt_parser_registry,
)


WEBUI_VERSION = "0.1.63"


class NamedPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    values: dict[str, Any] = Field(default_factory=dict)
    plugin_id: str | None = None
    overwrite: bool = True


class PromptPresetPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    positive_prompt: str = ""
    negative_prompt: str = ""
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


class ModelActivationPayload(BaseModel):
    model_path: str = Field(min_length=1)


def encode_sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    project_root: str | Path | None = None,
    runtime_startup_options: RuntimeStartupOptions | dict[str, Any] | None = None,
) -> FastAPI:
    context = ProjectContext.load(project_root=project_root)
    catalog = WebUICatalog(context)
    store = WebUIStore(context.data_root / "webui")
    application_settings = store.load_application_settings()
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
    jobs = GenerationJobManager(context, settings_provider=store.load_application_settings)
    jobs.runtime_startup_options = resolved_runtime_startup.to_dict()
    model_selection = WebUIModelSelectionState(context)
    selections = WebUISelectionResolver(jobs.registry)
    replay = ReplayService(context, jobs, model_selection)
    batch_replay = BatchReplayService(replay, jobs, model_selection)
    batch_io = BatchIOService(context, jobs, model_selection)
    variations = VariationMatrixService(context, jobs, model_selection, batch_io)
    prompt_parsers = default_prompt_parser_registry()
    prompt_configuration = PromptConfigurationService(store, parser_registry=prompt_parsers)
    static_root = Path(__file__).resolve().parent / "static"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await jobs.start()
        startup_model_path = str(
            (store.load_session() or {}).get("model_path")
            or (context.generation_defaults() or {}).get("model_path")
            or ""
        )
        if startup_model_path:
            try:
                selected = model_selection.activate(startup_model_path, source="startup_model_activation")
                await jobs.activate_model(
                    selected.resolved_path,
                    selection=selected.to_dict(),
                )
            except Exception:
                pass
        yield
        await jobs.stop()

    app = FastAPI(title="IMAGE_GEN WebUI", version=WEBUI_VERSION, lifespan=lifespan)
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
    app.state.runtime_startup_options = resolved_runtime_startup.to_dict()

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
    ) -> HTTPException:
        try:
            bundle = write_webui_failure_bundle(
                project_root=context.project_root,
                stage=stage,
                error=exc,
                payload=payload,
                request_path=request_path,
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
        if request.url.path == "/" or request.url.path.startswith("/assets/"):
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

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        # Browsers request this automatically. IMAGE_GEN does not currently
        # ship an icon, so return an intentional empty response instead of a 404.
        return Response(status_code=204)

    def _preview_media_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        return "application/octet-stream"

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "worker": jobs.status(),
            "version": WEBUI_VERSION,
            "active_model": model_selection.current_payload(),
            "prompt_parsers": prompt_parsers.descriptors(),
        }

    @app.get("/api/prompt-parsers")
    async def prompt_parser_catalog() -> dict[str, Any]:
        return {
            "default": "legacy",
            "contract_version": "image-gen-prompt-parser-v1",
            "canonical_contract_version": "image-gen-canonical-prompt-v1",
            "route_contract_version": PROMPT_ROUTE_CONTRACT_VERSION,
            "shadow_contract_version": PROMPT_SHADOW_CONTRACT_VERSION,
            "merge_contract_version": PROMPT_MERGE_CONTRACT_VERSION,
            "parsers": prompt_parsers.descriptors(),
        }

    @app.get("/api/prompt-shortcut-profiles")
    async def list_prompt_shortcut_profiles() -> list[dict[str, Any]]:
        return prompt_configuration.list_profiles()

    @app.post("/api/prompt-shortcut-profiles/validate")
    async def validate_prompt_shortcut_profile(payload: dict[str, Any]) -> dict[str, Any]:
        return prompt_configuration.validate_profile(payload)

    @app.post("/api/prompt-shortcut-profiles")
    async def save_prompt_shortcut_profile(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = prompt_configuration.save_profile(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"profile": saved, "profiles": prompt_configuration.list_profiles()}

    @app.delete("/api/prompt-shortcut-profiles/{profile_id}")
    async def delete_prompt_shortcut_profile(profile_id: str) -> dict[str, Any]:
        try:
            deleted = prompt_configuration.delete_profile(profile_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": deleted, "profiles": prompt_configuration.list_profiles()}

    @app.get("/api/prompt-parser-presets")
    async def list_prompt_parser_presets() -> list[dict[str, Any]]:
        return prompt_configuration.parser_presets()

    @app.post("/api/prompt-parser-presets")
    async def save_prompt_parser_preset(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = prompt_configuration.save_preset(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"preset": saved, "presets": prompt_configuration.parser_presets()}

    @app.delete("/api/prompt-parser-presets/{preset_id}")
    async def delete_prompt_parser_preset(preset_id: str) -> dict[str, Any]:
        try:
            deleted = prompt_configuration.delete_preset(preset_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": deleted, "presets": prompt_configuration.parser_presets()}

    @app.post("/api/prompts/translate")
    async def translate_prompts(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return prompt_configuration.translate_preview(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/prompts/preflight")
    async def preflight_prompts(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return prompt_configuration.preflight_report(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/bootstrap")
    async def bootstrap() -> dict[str, Any]:
        raw_defaults = context.generation_defaults()
        default_selection = selections.normalize(
            raw_defaults,
            migrate_legacy_auto_fallback=False,
        )
        defaults = default_selection.payload

        settings = store.load_application_settings()
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
            "project_root": str(context.project_root),
            "output_root": str(context.txt2img_output_root),
            "runtime_paths": {
                "config_path": str(context.config_path),
                "checkpoints_dir": str(context.checkpoints_dir),
                "vae_dir": str(context.vae_dir),
                "output_dir": str(context.txt2img_output_root),
            },
            "api_contract": {
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
            },
            "defaults": defaults,
            "session": session,
            "effective_generation": effective_generation,
            "selection_notes": [
                *default_selection.notes,
                *effective_selection.notes,
            ],
            "settings": settings,
            "runtime_startup_status": _runtime_startup_status(),
            "plugins": catalog.plugins(),
            **prompt_configuration.bootstrap_payload(),
            "models": catalog.model_payload(),
            "active_model": model_selection.current_payload(),
            "model_runtime": jobs.model_runtime_status(),
            "recent_outputs": _visible_recent_outputs(),
            "prompt_presets": store.list_prompt_presets(),
            "generation_profiles": store.list_profiles("generation"),
            "worker": jobs.status(),
        }

    @app.post("/api/models/refresh")
    async def refresh_models() -> dict[str, Any]:
        return catalog.refresh_models()

    @app.get("/api/models/active")
    async def active_model() -> dict[str, Any]:
        return {
            "active_model": model_selection.current_payload(),
            "model_runtime": jobs.model_runtime_status(),
        }

    @app.get("/api/models/runtime-status")
    async def model_runtime_status() -> dict[str, Any]:
        return jobs.model_runtime_status()

    async def _activate_model_impl(payload: ModelActivationPayload) -> dict[str, Any]:
        try:
            selected = model_selection.activate(payload.model_path)
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
        except (OSError, ValueError, RuntimeError) as exc:
            raise _webui_failure(
                "model_gpu_activation",
                exc,
                payload=payload.model_dump(),
                request_path="/api/models/activate",
                status_code=400,
            ) from exc
        return {
            "active_model": selected.to_dict(),
            "model_runtime": jobs.model_runtime_status(),
            "activation": activation,
        }

    @app.post("/api/models/activate")
    async def activate_model(payload: ModelActivationPayload) -> dict[str, Any]:
        return await _activate_model_impl(payload)

    @app.post("/api/model/activate")
    async def activate_model_alias(payload: ModelActivationPayload) -> dict[str, Any]:
        return await _activate_model_impl(payload)

    @app.post("/api/activate-model")
    async def activate_model_legacy_alias(payload: ModelActivationPayload) -> dict[str, Any]:
        return await _activate_model_impl(payload)

    @app.get("/api/model-activation/debug")
    async def model_activation_debug() -> dict[str, Any]:
        return {
            "project_root": str(context.project_root),
            "config_path": str(context.config_path),
            "checkpoints_dir": str(context.checkpoints_dir),
            "default_model_path": str(context.default_model_path) if context.default_model_path else None,
            "active_model": model_selection.current_payload(),
            "checkpoints_dir_exists": context.checkpoints_dir.is_dir(),
        }

    @app.post("/api/workspace/reload")
    async def reload_workspace() -> dict[str, Any]:
        catalog.reload_plugins()
        catalog.refresh_models()
        return {
            "plugins": catalog.plugins(),
            **prompt_configuration.bootstrap_payload(),
            "models": catalog.model_payload(),
            "recent_outputs": _visible_recent_outputs(),
        }

    @app.get("/api/recent-outputs")
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

    @app.post("/api/recent-outputs/reload")
    async def reload_recent_outputs() -> JSONResponse:
        return JSONResponse({"recent_outputs": _visible_recent_outputs()})

    @app.post("/api/recent-outputs/clear")
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

    @app.get("/api/image-files/{image_ref:path}")
    async def recent_output_image(image_ref: str) -> FileResponse:
        path = _resolve_external_image_ref_or_404(image_ref)
        return FileResponse(path, media_type=_preview_media_type(path))

    @app.get("/api/image-files/{image_ref:path}/details")
    async def recent_output_image_details(image_ref: str) -> dict[str, Any]:
        path = _resolve_external_image_ref_or_404(image_ref)
        return load_image_file_details(context, path, display_name=path.name).to_dict()

    @app.get("/api/outputs/{output_id:path}/details")
    async def output_details(output_id: str) -> dict[str, Any]:
        try:
            return load_output_details(context, output_id).to_dict()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/outputs/inspect-upload")
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

    @app.post("/api/replay/preflight")
    async def replay_preflight(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return replay.preflight(payload).to_dict()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/replay/submit")
    async def replay_submit(payload: dict[str, Any]) -> dict[str, Any]:
        token = str(payload.get("preflight_token") or "").strip()
        if not token:
            raise HTTPException(status_code=400, detail="A replay preflight token is required.")
        try:
            preflight, job = await replay.submit(token)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, OSError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"preflight": preflight.to_dict(), "job": job.to_dict()}

    @app.post("/api/replay/batch/preflight")
    async def batch_replay_preflight(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return batch_replay.preflight(payload).to_dict()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/replay/batch/submit")
    async def batch_replay_submit(payload: dict[str, Any]) -> dict[str, Any]:
        token = str(payload.get("preflight_token") or "").strip()
        if not token:
            raise HTTPException(status_code=400, detail="A batch replay preflight token is required.")
        try:
            preflight, submitted, rejected = await batch_replay.submit(
                token,
                queue_valid_only=bool(payload.get("queue_valid_only", False)),
            )
        except ValueError as exc:
            raise _webui_failure(
                "batch_replay_submit",
                exc,
                payload=payload,
                request_path="/api/replay/batch/submit",
                status_code=409,
            ) from exc
        except (KeyError, OSError, TypeError) as exc:
            raise _webui_failure(
                "batch_replay_submit",
                exc,
                payload=payload,
                request_path="/api/replay/batch/submit",
                status_code=400,
            ) from exc
        return {
            "preflight": preflight.to_dict(),
            "submitted": [job.to_dict() for job in submitted],
            "rejected": rejected,
            "submitted_count": len(submitted),
            "rejected_count": len(rejected),
        }

    @app.post("/api/batch/import/parse")
    async def batch_import_parse(
        file: UploadFile = File(...),
        format_hint: str = Form(""),
        defaults_policy: str = Form("file_only"),
        current_values: str = Form("{}"),
    ) -> dict[str, Any]:
        try:
            content = await file.read()
            try:
                current = json.loads(current_values or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("current_values must be valid JSON.") from exc
            if not isinstance(current, dict):
                raise ValueError("current_values must contain a JSON object.")
            result = batch_io.parse_bytes(
                content,
                filename=file.filename or "queue",
                format_hint=format_hint or None,
                defaults_policy=defaults_policy,
                current_values=current,
            )
            return result.to_dict()
        except (OSError, TypeError, ValueError) as exc:
            raise _webui_failure(
                "batch_import_parse",
                exc,
                payload={
                    "filename": file.filename,
                    "format_hint": format_hint,
                    "defaults_policy": defaults_policy,
                    "current_values": current_values,
                },
                request_path="/api/batch/import/parse",
                status_code=400,
            ) from exc
        finally:
            await file.close()

    @app.post("/api/batch/import/preflight")
    async def batch_import_preflight(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return batch_io.preflight(payload).to_dict()
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise _webui_failure(
                "batch_import_preflight",
                exc,
                payload=payload,
                request_path="/api/batch/import/preflight",
                status_code=400,
            ) from exc

    @app.post("/api/batch/import/submit")
    async def batch_import_submit(payload: dict[str, Any]) -> dict[str, Any]:
        token = str(payload.get("preflight_token") or "").strip()
        if not token:
            raise HTTPException(status_code=400, detail="An import preflight token is required.")
        try:
            preflight, submitted, rejected = await batch_io.submit(
                token,
                queue_valid_only=bool(payload.get("queue_valid_only", False)),
            )
        except ValueError as exc:
            raise _webui_failure(
                "batch_import_submit",
                exc,
                payload=payload,
                request_path="/api/batch/import/submit",
                status_code=409,
            ) from exc
        except (KeyError, OSError, TypeError) as exc:
            raise _webui_failure(
                "batch_import_submit",
                exc,
                payload=payload,
                request_path="/api/batch/import/submit",
                status_code=400,
            ) from exc
        return {
            "preflight": preflight.to_dict(),
            "submitted": [job.to_dict() for job in submitted],
            "rejected": rejected,
            "submitted_count": len(submitted),
            "rejected_count": len(rejected),
        }

    @app.post("/api/batch/export")
    async def batch_export(payload: dict[str, Any]) -> Response:
        try:
            result = batch_io.export(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        headers = {
            "Content-Disposition": f'attachment; filename="{result.filename}"',
            "X-IMAGE-GEN-Export-Warnings": json.dumps(result.warnings, ensure_ascii=True),
            "X-IMAGE-GEN-Job-Count": str(result.job_count),
        }
        return Response(content=result.content, media_type=result.media_type, headers=headers)

    @app.post("/api/variations/preflight")
    async def variation_preflight(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return variations.preflight(payload).to_dict()
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise _webui_failure(
                "variation_preflight",
                exc,
                payload=payload,
                request_path="/api/variations/preflight",
                status_code=400,
            ) from exc

    @app.post("/api/variations/submit")
    async def variation_submit(payload: dict[str, Any]) -> dict[str, Any]:
        token = str(payload.get("preflight_token") or "").strip()
        if not token:
            exc = ValueError("A variation preflight token is required.")
            raise _webui_failure(
                "variation_submit", exc, payload=payload,
                request_path="/api/variations/submit", status_code=400,
            )
        try:
            preflight, submitted, rejected = await variations.submit(token)
        except ValueError as exc:
            raise _webui_failure(
                "variation_submit", exc, payload=payload,
                request_path="/api/variations/submit", status_code=409,
            ) from exc
        return {
            "preflight": preflight.to_dict(),
            "submitted": [job.to_dict() for job in submitted],
            "rejected": rejected,
            "submitted_count": len(submitted),
            "rejected_count": len(rejected),
        }

    @app.post("/api/variations/export")
    async def variation_export(payload: dict[str, Any]) -> Response:
        token = str(payload.get("preflight_token") or "").strip()
        if not token:
            raise HTTPException(status_code=400, detail="A variation preflight token is required.")
        try:
            preflight = variations.preflight_from_token(token)
            if not preflight.jobs:
                raise ValueError("The variation preflight contains no jobs to export.")
            result = batch_io.export({
                "format": payload.get("format") or "native",
                "filename_stem": payload.get("filename_stem") or "variation_matrix",
                "source": "IMAGE_GEN Variation Matrix",
                "jobs": [
                    {
                        "job_id": f"variation-{item['job_index']:04d}",
                        "request": item["request"],
                        "provenance": {
                            "metadata_source": "variation_matrix",
                        },
                    }
                    for item in preflight.jobs
                    if item["valid"]
                ],
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise _webui_failure(
                "variation_export", exc, payload=payload,
                request_path="/api/variations/export", status_code=400,
            ) from exc
        headers = {
            "Content-Disposition": f'attachment; filename="{result.filename}"',
            "X-IMAGE-GEN-Export-Warnings": json.dumps(result.warnings, ensure_ascii=True),
            "X-IMAGE-GEN-Job-Count": str(result.job_count),
        }
        return Response(content=result.content, media_type=result.media_type, headers=headers)

    @app.get("/api/session")
    async def get_session() -> dict[str, Any]:
        return store.load_session()

    @app.put("/api/session")
    async def put_session(payload: dict[str, Any]) -> dict[str, Any]:
        return store.save_session(payload)

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        settings = store.load_application_settings()
        settings["_runtime_startup_status"] = _runtime_startup_status()
        return settings

    @app.put("/api/settings")
    async def put_settings(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resolve_runtime_startup_options(environment={}, settings=payload)
            overrides = payload.get("runtime_job_overrides")
            if overrides is not None:
                if not isinstance(overrides, dict):
                    raise ValueError("runtime_job_overrides must be a JSON object.")
                resolve_runtime_startup_options(environment={}, settings=overrides)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        saved = store.save_application_settings(payload)
        saved["_runtime_startup_status"] = _runtime_startup_status()
        return saved

    @app.get("/api/runtime/startup-status")
    async def runtime_startup_status() -> dict[str, Any]:
        return _runtime_startup_status()

    @app.get("/api/runtime/command")
    async def runtime_command() -> dict[str, Any]:
        return build_runtime_command_from_status(_runtime_startup_status())

    @app.post("/api/runtime/inherit-startup-profile")
    async def inherit_runtime_startup_profile() -> dict[str, Any]:
        saved = store.inherit_runtime_startup_profile()
        saved["_runtime_startup_status"] = _runtime_startup_status()
        return saved

    @app.get("/api/prompt-presets")
    async def get_prompt_presets() -> list[dict[str, Any]]:
        return store.list_prompt_presets()

    @app.post("/api/prompt-presets")
    async def save_prompt_preset(payload: PromptPresetPayload) -> dict[str, Any]:
        return store.save_prompt_preset(payload.name, payload.model_dump())

    @app.delete("/api/prompt-presets/{name}")
    async def delete_prompt_preset(name: str) -> dict[str, Any]:
        return {"deleted": store.delete_prompt_preset(name)}

    @app.get("/api/profiles/{kind}")
    async def get_profiles(kind: str, plugin_id: str | None = None) -> list[dict[str, Any]]:
        return store.list_profiles(kind, plugin_id)

    @app.post("/api/profiles/{kind}")
    async def save_profile(kind: str, payload: NamedPayload) -> dict[str, Any]:
        try:
            return store.save_profile(kind, payload.name, payload.values, payload.plugin_id, overwrite=payload.overwrite)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/profiles/{kind}/{name}")
    async def delete_profile(kind: str, name: str, plugin_id: str | None = None) -> dict[str, Any]:
        return {"deleted": store.delete_profile(kind, name, plugin_id)}

    @app.get("/api/jobs")
    async def list_jobs() -> dict[str, Any]:
        return {"jobs": jobs.list_jobs(), "worker": jobs.status()}

    @app.post("/api/maintenance/live-preview/cleanup")
    async def cleanup_live_previews() -> dict[str, Any]:
        return jobs.cleanup_preview_directories()

    @app.post("/api/maintenance/job-cache/clear")
    async def clear_job_cache() -> dict[str, Any]:
        report = jobs.clear_job_cache(preserve_active=True, startup=False)
        return {
            **report,
            "jobs": jobs.list_jobs(),
            "worker": jobs.status(),
        }

    @app.post("/api/maintenance/queue/dismiss-terminal")
    async def dismiss_terminal_jobs() -> dict[str, Any]:
        report = jobs.dismiss_terminal_jobs()
        return {**report, "jobs": jobs.list_jobs(), "worker": jobs.status()}

    @app.post("/api/maintenance/queue/clear")
    async def clear_queued_jobs(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        report = jobs.clear_queued_jobs(
            reason=str((payload or {}).get("reason") or "Queued jobs were cleared from the WebUI."),
        )
        return {
            **report,
            "jobs": jobs.list_jobs(),
            "worker": jobs.status(),
        }

    @app.post("/api/schedulers/validate")
    async def validate_scheduler(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return jobs.preflight_scheduler(payload)
        except (KeyError, ValueError) as exc:
            raise _webui_failure(
                "scheduler_prequeue_validation",
                exc,
                payload=payload,
                request_path="/api/schedulers/validate",
                status_code=400,
            ) from exc

    @app.post("/api/jobs")
    async def submit_job(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            prepared_payload = prompt_configuration.prepare_generation_payload(payload)
        except ValueError as exc:
            raise _webui_failure(
                "job_submission_prompt_configuration",
                exc,
                payload=payload,
                request_path="/api/jobs",
                status_code=400,
            ) from exc
        try:
            authoritative_payload, selected_model = model_selection.enforce(prepared_payload)
        except ValueError as exc:
            raise _webui_failure(
                "job_submission_model_selection",
                exc,
                payload=prepared_payload,
                request_path="/api/jobs",
                status_code=409,
            ) from exc
        try:
            job = await jobs.submit(
                authoritative_payload,
                model_selection=selected_model.to_dict(),
            )
        except (KeyError, ValueError) as exc:
            raise _webui_failure(
                "job_submission_validation",
                exc,
                payload=authoritative_payload,
                request_path="/api/jobs",
                status_code=400,
            ) from exc
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/primary-output")
    async def job_primary_output(job_id: str) -> dict[str, Any]:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        for raw_path in job.output_paths:
            summary = catalog.output_summary_from_path(Path(raw_path))
            if summary is not None:
                return summary
        raise HTTPException(status_code=404, detail="No generated output is available for this job yet.")

    @app.get("/api/jobs/{job_id}/diagnostics")
    async def job_diagnostics(job_id: str) -> dict[str, Any]:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        return jobs.diagnostics_payload(job)

    @app.get("/api/jobs/{job_id}/log")
    async def job_log(job_id: str) -> FileResponse:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        path = Path(job.console_log_path or "")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Job console log is not available yet.")
        return FileResponse(path, media_type="text/plain", filename=f"{job_id}-console.log")

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = await jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")

        async def event_stream():
            async for payload in jobs.subscribe(job_id):
                event_type = str(payload.get("type") or "job-progress")
                yield encode_sse_event(event_type, payload)
                if event_type in {"job-completed", "job-cancelled", "job-failed"}:
                    break

        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    @app.get("/api/jobs/{job_id}/preview/latest")
    async def job_preview_latest(job_id: str) -> FileResponse:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        path = jobs.live_preview_latest_file(job)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Live preview is not available yet.")
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        }
        return FileResponse(path, media_type=_preview_media_type(path), headers=headers)

    @app.get("/api/jobs/{job_id}/preview/{step_number}")
    async def job_preview_step(job_id: str, step_number: int) -> FileResponse:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        path = jobs.live_preview_step_path(job, step_number)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Requested live preview step was not found.")
        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        return FileResponse(path, media_type=_preview_media_type(path), headers=headers)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    return app
