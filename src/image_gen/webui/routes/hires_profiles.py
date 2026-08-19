from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException



_MODEL_FAMILY_CHOICES = (
    {"value": "sd1.x", "label": "SD1.x"},
    {"value": "sd2.x", "label": "SD2.x"},
    {"value": "sdxl", "label": "SDXL"},
    {"value": "sd3.x", "label": "SD3.x"},
)


def _choice_item(value: Any, *, value_keys: tuple[str, ...], label_keys: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    selected = next((value.get(key) for key in value_keys if value.get(key) not in (None, "")), None)
    if selected in (None, ""):
        return None
    label = next((value.get(key) for key in label_keys if value.get(key) not in (None, "")), selected)
    payload = {"value": str(selected), "label": str(label)}
    for key in ("sha256", "available", "selectable", "architecture", "native_scale"):
        if key in value:
            payload[key] = value.get(key)
    return payload


def _choice_overrides(*, catalog, upscaler_catalog, prompt_configuration) -> dict[str, list[dict[str, Any]]]:
    plugins = catalog.plugins()
    prompt_payload = prompt_configuration.bootstrap_payload()
    upscalers = upscaler_catalog.payload()

    samplers = [
        item
        for raw in list(plugins.get("samplers") or [])
        if (item := _choice_item(raw, value_keys=("name", "plugin_id"), label_keys=("label", "name", "plugin_id")))
    ]
    schedulers = [
        item
        for raw in list(plugins.get("schedulers") or [])
        if (item := _choice_item(raw, value_keys=("name", "plugin_id"), label_keys=("label", "name", "plugin_id")))
    ]
    prompt_parsers = [
        item
        for raw in list(prompt_payload.get("prompt_parsers") or [])
        if (item := _choice_item(raw, value_keys=("parser_id", "name"), label_keys=("label", "parser_id", "name")))
    ]
    shortcut_profiles = [
        item
        for raw in list(prompt_payload.get("prompt_shortcut_profiles") or [])
        if (item := _choice_item(raw, value_keys=("profile_id",), label_keys=("label", "profile_id")))
    ]
    neural = []
    for raw in list(upscalers.get("neural") or []):
        item = _choice_item(raw, value_keys=("upscaler_id",), label_keys=("display_name", "file_name", "upscaler_id"))
        if item is None:
            continue
        item["available"] = bool(raw.get("selectable", False))
        item["strategy"] = "pixel_neural"
        neural.append(item)
    for raw in list(upscalers.get("interpolation_baselines") or []):
        item = _choice_item(raw, value_keys=("upscaler_id",), label_keys=("display_name", "upscaler_id"))
        if item is None:
            continue
        item["available"] = bool(raw.get("selectable", True))
        item["strategy"] = str(raw.get("strategy") or "pixel_resize")
        item["builtin"] = True
        neural.append(item)
    return {
        "samplers": samplers,
        "schedulers": schedulers,
        "prompt_parsers": prompt_parsers,
        "shortcut_profiles": shortcut_profiles,
        "upscalers": neural,
    }


def build_hires_profiles_router(*, store, catalog, upscaler_catalog, prompt_configuration) -> APIRouter:
    router = APIRouter()

    def choices() -> dict[str, list[dict[str, Any]]]:
        return _choice_overrides(
            catalog=catalog,
            upscaler_catalog=upscaler_catalog,
            prompt_configuration=prompt_configuration,
        )

    def profile_bundle(profile_id: str) -> dict[str, Any]:
        profile = store.get_hires_profile(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Unknown hires profile: {profile_id}")
        try:
            manifest = store.inspect_hires_profile(profile_id, choice_overrides=choices())
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        assignments = [
            item for item in store.list_hires_default_assignments()
            if str(item.get("profile_id") or "") == str(profile_id)
        ]
        return {"profile": profile, "manifest": manifest, "assignments": assignments}

    @router.get("/api/hires-profiles")
    async def list_hires_profiles() -> dict[str, Any]:
        return {
            "profiles": store.list_hires_profiles(),
            "assignments": store.list_hires_default_assignments(),
            "model_family_choices": list(_MODEL_FAMILY_CHOICES),
        }

    @router.get("/api/hires-profiles/schema")
    async def hires_profile_schema() -> dict[str, Any]:
        service = store.hires_profile_service
        manifest = service.schema.build_save_manifest(
            profile_id="draft",
            profile_name="New Hires Preset",
            values={},
            included_fields=(),
            choice_overrides=choices(),
        )
        return {
            "manifest": manifest.to_dict(),
            "model_family_choices": list(_MODEL_FAMILY_CHOICES),
            "compatibility_choices": {
                "model_families": list(_MODEL_FAMILY_CHOICES),
                "upscalers": choices().get("upscalers", []),
            },
        }

    @router.post("/api/hires-profiles/resolve-auto")
    async def resolve_hires_auto(payload: dict[str, Any]) -> dict[str, Any]:
        result = store.resolve_hires_auto(
            dict(payload.get("context") or payload or {}),
            choice_overrides=choices(),
        )
        if not bool((result.get("validation") or {}).get("valid", False)):
            raise HTTPException(status_code=422, detail=result)
        return result

    @router.get("/api/hires-profiles/{profile_id}")
    async def get_hires_profile(profile_id: str) -> dict[str, Any]:
        return profile_bundle(profile_id)

    @router.post("/api/hires-profiles")
    async def save_hires_profile(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = store.save_hires_profile(
                name=str(payload.get("name") or ""),
                values=dict(payload.get("values") or {}),
                included_fields=list(payload.get("included_fields") or []),
                profile_id=str(payload.get("profile_id") or ""),
                description=str(payload.get("description") or ""),
                compatibility=dict(payload.get("compatibility") or {}),
                baseline_profile_id=str(payload.get("baseline_profile_id") or ""),
                choice_overrides=choices(),
            )
        except ValueError as exc:
            manifest = getattr(exc, "manifest", None)
            if manifest is not None:
                raise HTTPException(
                    status_code=400,
                    detail={"message": str(exc), "manifest": manifest.to_dict()},
                ) from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **saved,
            "profiles": store.list_hires_profiles(),
            "assignments": store.list_hires_default_assignments(),
        }

    @router.post("/api/hires-profiles/{profile_id}/duplicate")
    async def duplicate_hires_profile(profile_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        source = store.get_hires_profile(profile_id)
        if source is None:
            raise HTTPException(status_code=404, detail=f"Unknown hires profile: {profile_id}")
        requested_name = str((payload or {}).get("name") or "").strip()
        name = requested_name or f"{source.get('name') or 'Hires Preset'} Copy"
        try:
            saved = store.duplicate_hires_profile(profile_id, name=name, choice_overrides=choices())
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **saved,
            "profiles": store.list_hires_profiles(),
            "assignments": store.list_hires_default_assignments(),
        }

    @router.delete("/api/hires-profiles/{profile_id}")
    async def delete_hires_profile(profile_id: str) -> dict[str, Any]:
        try:
            result = store.delete_hires_profile(profile_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **result,
            "profiles": store.list_hires_profiles(),
            "assignments": store.list_hires_default_assignments(),
        }

    @router.get("/api/hires-default-assignments")
    async def list_hires_default_assignments() -> dict[str, Any]:
        return {"assignments": store.list_hires_default_assignments()}

    @router.post("/api/hires-default-assignments")
    async def save_hires_default_assignment(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = store.save_hires_default_assignment(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"assignment": saved, "assignments": store.list_hires_default_assignments()}

    @router.delete("/api/hires-default-assignments/{assignment_key:path}")
    async def delete_hires_default_assignment(assignment_key: str) -> dict[str, Any]:
        decoded = unquote(str(assignment_key or ""))
        return {
            "deleted": store.delete_hires_default_assignment(decoded),
            "assignments": store.list_hires_default_assignments(),
        }

    return router


__all__ = ["build_hires_profiles_router"]
