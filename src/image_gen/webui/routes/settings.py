from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml
from fastapi import APIRouter, HTTPException

from image_gen.runtime_options import build_runtime_command_from_status, resolve_runtime_startup_options
from image_gen.webui.routes.payloads import NamedPayload, PromptPresetPayload
from image_gen.webui.schema_utils import scope_plugin_profile_values


def build_settings_router(*, context, store, theme_library, registry, jobs=None, _runtime_startup_status) -> APIRouter:
    router = APIRouter()

    def _user_config_path() -> Path:
        return (context.project_root / "user_config" / "user-config.yml").resolve()


    def _read_user_config_document() -> dict[str, Any]:
        path = _user_config_path()
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        parsed = yaml.safe_load(text) if text.strip() else {}
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError("user-config.yml must contain a YAML mapping/object at the document root.")
        return {
            "path": str(path),
            "text": text,
            "parsed": parsed,
            "exists": path.is_file(),
            "restart_required": True,
        }


    @router.get("/api/user-config")
    async def get_user_config() -> dict[str, Any]:
        try:
            return _read_user_config_document()
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.put("/api/user-config")
    async def put_user_config(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or "")
        try:
            parsed = yaml.safe_load(text) if text.strip() else {}
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="user-config.yml must contain a YAML mapping/object at the document root.")
        path = _user_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = path.with_suffix(path.suffix + ".bak")
        if path.is_file():
            shutil.copy2(path, backup_path)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
        result = _read_user_config_document()
        result["backup_path"] = str(backup_path) if backup_path.is_file() else ""
        return result


    @router.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        settings = store.load_application_settings()
        settings["theme_effective_palette"] = theme_library.resolve_effective_palette()
        settings["theme_library_activation"] = theme_library.library_payload().get("activation", {})
        settings["_runtime_startup_status"] = _runtime_startup_status()
        return settings


    @router.put("/api/settings")
    async def put_settings(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resolve_runtime_startup_options(environment={}, settings=payload)
            overrides = payload.get("runtime_job_overrides")
            if overrides is not None:
                if not isinstance(overrides, dict):
                    raise ValueError("runtime_job_overrides must be a JSON object.")
                resolve_runtime_startup_options(environment={}, settings=overrides)
            previous = store.load_application_settings() if "model_residency_mode" in payload else {}
            saved = store.save_application_settings(payload)
            residency_transition = None
            if "model_residency_mode" in payload and jobs is not None:
                previous_mode = str(previous.get("model_residency_mode") or "managed")
                saved_mode = str(saved.get("model_residency_mode") or "managed")
                if previous_mode != saved_mode:
                    try:
                        residency_transition = await jobs.apply_model_residency_mode(saved_mode)
                    except RuntimeError as exc:
                        residency_transition = {
                            "deferred": True,
                            "reason": "runtime_apply_failed",
                            "error": str(exc),
                            "status": jobs.model_runtime_status(),
                        }
                else:
                    residency_transition = {"deferred": False, "unchanged": True, "status": jobs.model_runtime_status()}
            if "theme_palette" in payload:
                theme_library.deactivate_global_theme()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        saved["theme_effective_palette"] = theme_library.resolve_effective_palette()
        saved["theme_library_activation"] = theme_library.library_payload().get("activation", {})
        saved["_runtime_startup_status"] = _runtime_startup_status()
        if residency_transition is not None:
            saved["_model_residency_transition"] = residency_transition
            saved["_model_runtime_status"] = dict(residency_transition.get("status") or {})
        return saved


    @router.get("/api/runtime/startup-status")
    async def runtime_startup_status() -> dict[str, Any]:
        return _runtime_startup_status()


    @router.get("/api/runtime/command")
    async def runtime_command() -> dict[str, Any]:
        return build_runtime_command_from_status(_runtime_startup_status())


    @router.post("/api/runtime/inherit-startup-profile")
    async def inherit_runtime_startup_profile() -> dict[str, Any]:
        saved = store.inherit_runtime_startup_profile()
        saved["_runtime_startup_status"] = _runtime_startup_status()
        return saved


    @router.get("/api/prompt-presets")
    async def get_prompt_presets() -> list[dict[str, Any]]:
        return store.list_prompt_presets()


    @router.post("/api/prompt-presets")
    async def save_prompt_preset(payload: PromptPresetPayload) -> dict[str, Any]:
        return store.save_prompt_preset(payload.name, payload.model_dump())


    @router.delete("/api/prompt-presets/{name}")
    async def delete_prompt_preset(name: str) -> dict[str, Any]:
        return {"deleted": store.delete_prompt_preset(name)}


    def _scope_scheduler_profile_values(
        plugin_id: str | None,
        values: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        requested = str(plugin_id or "").strip()
        if not requested:
            raise ValueError(
                "Scheduler profiles require a scheduler plugin_id so their settings can be scoped safely."
            )
        descriptor = registry.resolve_descriptor(requested, kind="scheduler")
        if descriptor is None:
            raise ValueError(f"Unknown scheduler plugin_id: {requested!r}")
        return scope_plugin_profile_values(values, descriptor.config_schema, kind="scheduler")


    @router.get("/api/profiles/{kind}")
    async def get_profiles(kind: str, plugin_id: str | None = None) -> list[dict[str, Any]]:
        records = store.list_profiles(kind, plugin_id)
        if str(kind or "").strip().lower() != "scheduler":
            return records

        cleaned_records: list[dict[str, Any]] = []
        try:
            for record in records:
                record_plugin_id = plugin_id or record.get("plugin_id")
                values = _scope_scheduler_profile_values(record_plugin_id, record.get("values"))
                cleaned = {**record, "values": values}
                cleaned_records.append(cleaned)
                if values != dict(record.get("values") or {}):
                    store.save_profile(
                        kind,
                        str(record.get("name") or "Untitled"),
                        values,
                        record_plugin_id,
                        overwrite=True,
                    )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return cleaned_records


    @router.post("/api/profiles/{kind}")
    async def save_profile(kind: str, payload: NamedPayload) -> dict[str, Any]:
        values = payload.values
        if str(kind or "").strip().lower() == "scheduler":
            try:
                values = _scope_scheduler_profile_values(payload.plugin_id, values)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            return store.save_profile(kind, payload.name, values, payload.plugin_id, overwrite=payload.overwrite)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


    @router.delete("/api/profiles/{kind}/{name}")
    async def delete_profile(kind: str, name: str, plugin_id: str | None = None) -> dict[str, Any]:
        return {"deleted": store.delete_profile(kind, name, plugin_id)}


    return router
