from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from image_gen.webui.workspace import (
    WORKSPACE_CONTRACT_VERSION,
    WORKSPACE_SCHEMA,
    WORKSPACE_SCHEMA_VERSION,
    WorkspaceStoreError,
    review_workspace_import,
    workspace_responsive_contract_payload,
)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge_dict(dict(output[key]), value)
        else:
            output[key] = value
    return output


def build_workspace_router(*, store, workspace_defaults, workspace_registry, workspace_store, _default_asset_payload) -> APIRouter:
    router = APIRouter()

    @router.get("/api/session")
    async def get_session() -> dict[str, Any]:
        return store.load_session()


    @router.put("/api/session")
    async def put_session(payload: dict[str, Any]) -> dict[str, Any]:
        return store.save_session(payload)


    @router.get("/api/default-assets")
    async def get_default_assets() -> dict[str, Any]:
        return _default_asset_payload()


    @router.put("/api/default-assets")
    async def put_default_assets(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = store.save_default_asset_profiles(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _default_asset_payload(document=saved)


    @router.patch("/api/default-assets")
    async def patch_default_assets(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            current = store.load_default_asset_profiles()
            merged = _deep_merge_dict(current, payload)
            saved = store.save_default_asset_profiles(merged)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _default_asset_payload(document=saved)


    def _workspace_layout_payload(settings: dict[str, Any] | None = None) -> dict[str, Any]:
        resolved = settings or store.load_application_settings()
        layout = dict(resolved.get("ui_layout") or {})
        return {
            "workspace_layout_version": int(layout.get("workspace_layout_version") or 1),
            "layout": layout,
            "settings": resolved,
        }


    @router.get("/api/workspace/layout")
    async def get_workspace_layout() -> dict[str, Any]:
        return _workspace_layout_payload()


    @router.patch("/api/workspace/layout")
    async def patch_workspace_layout(payload: dict[str, Any]) -> dict[str, Any]:
        requested = payload.get("layout") if isinstance(payload.get("layout"), dict) else payload
        try:
            saved = store.save_application_settings({"ui_layout": requested})
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _workspace_layout_payload(saved)


    @router.post("/api/workspace/layout/reset")
    async def reset_workspace_layout() -> dict[str, Any]:
        packaged = store.load_packaged_application_defaults()
        default_layout = dict(packaged.get("ui_layout") or {})
        saved = store.save_application_settings({"ui_layout": default_layout})
        return _workspace_layout_payload(saved)


    def _workspace_definitions_payload() -> dict[str, Any]:
        return {
            "contractVersion": WORKSPACE_CONTRACT_VERSION,
            "schema": WORKSPACE_SCHEMA,
            "schemaVersion": WORKSPACE_SCHEMA_VERSION,
            **workspace_registry.definitions_payload(),
            "responsive": workspace_responsive_contract_payload(),
            "shippedDefaults": [layout.to_dict() for layout in workspace_defaults.values()],
        }


    @router.get("/api/workspaces/definitions")
    async def get_workspace_definitions() -> dict[str, Any]:
        return _workspace_definitions_payload()


    @router.post("/api/workspaces/import/validate")
    async def validate_workspace_import(payload: dict[str, Any]) -> dict[str, Any]:
        return review_workspace_import(payload, workspace_registry).to_dict()


    @router.get("/api/workspaces/{page_id}")
    async def list_page_workspaces(page_id: str) -> dict[str, Any]:
        if workspace_registry.page(page_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown workspace page '{page_id}'.")
        return {
            "pageId": page_id,
            "workspaces": workspace_store.list(page_id),
            "active": workspace_store.get_active(page_id).to_dict(),
        }


    @router.post("/api/workspaces/{page_id}")
    async def save_page_workspace(page_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if workspace_registry.page(page_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown workspace page '{page_id}'.")
        requested = dict(payload)
        requested_page = str(requested.get("pageId") or "").strip()
        if requested_page and requested_page.casefold() != page_id.casefold():
            raise HTTPException(status_code=400, detail="Workspace pageId does not match the requested page.")
        requested["pageId"] = page_id
        try:
            saved = workspace_store.save(requested)
        except WorkspaceStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"workspace": saved.to_dict()}


    @router.get("/api/workspaces/{page_id}/active")
    async def get_active_page_workspace(page_id: str) -> dict[str, Any]:
        try:
            return workspace_store.get_active(page_id).to_dict()
        except WorkspaceStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.post("/api/workspaces/{page_id}/activate")
    async def activate_page_workspace(page_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        workspace_id = str(payload.get("workspaceId") or "").strip()
        if not workspace_id:
            raise HTTPException(status_code=400, detail="workspaceId is required.")
        try:
            return workspace_store.set_active(page_id, workspace_id).to_dict()
        except WorkspaceStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/workspaces/{page_id}/duplicate")
    async def duplicate_page_workspace(page_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        workspace_id = str(payload.get("workspaceId") or "").strip()
        if not workspace_id:
            raise HTTPException(status_code=400, detail="workspaceId is required.")
        try:
            duplicated = workspace_store.duplicate(workspace_id, str(payload.get("name") or ""))
        except WorkspaceStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if duplicated.page_id.casefold() != page_id.casefold():
            workspace_store.delete(duplicated.workspace_id)
            raise HTTPException(status_code=400, detail="Source workspace does not belong to the requested page.")
        return {"workspace": duplicated.to_dict()}


    @router.post("/api/workspaces/{page_id}/rename")
    async def rename_page_workspace(page_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        workspace_id = str(payload.get("workspaceId") or "").strip()
        name = str(payload.get("name") or "").strip()
        if not workspace_id or not name:
            raise HTTPException(status_code=400, detail="workspaceId and name are required.")
        existing = workspace_store.get(workspace_id)
        if existing is None or existing.page_id.casefold() != page_id.casefold():
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' was not found on page '{page_id}'.")
        try:
            renamed = workspace_store.rename(workspace_id, name)
        except WorkspaceStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"workspace": renamed.to_dict()}


    @router.delete("/api/workspaces/{page_id}/{workspace_id}")
    async def delete_page_workspace(page_id: str, workspace_id: str) -> dict[str, Any]:
        existing = workspace_store.get(workspace_id)
        if existing is None or existing.page_id.casefold() != page_id.casefold():
            raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' was not found on page '{page_id}'.")
        try:
            deleted = workspace_store.delete(workspace_id)
        except WorkspaceStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"deleted": deleted, "active": workspace_store.get_active(page_id).to_dict()}


    @router.post("/api/workspaces/{page_id}/reset")
    async def reset_page_workspace(page_id: str) -> dict[str, Any]:
        try:
            return workspace_store.reset_active(page_id).to_dict()
        except WorkspaceStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    return router
