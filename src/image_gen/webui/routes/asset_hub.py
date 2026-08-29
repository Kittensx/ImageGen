from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from image_gen.systems.asset_hub import (
    ASSET_HUB_CONTRACT_VERSION,
    AssetHubDownloadManager,
    AssetGalleryCache,
    AssetDiscoveryIndex,
    AssetHubInstaller,
    UpscalerFavoriteStore,
    compatible_upscaler_payload,
    AssetHubError,
    AssetHubSecretStore,
    AssetHubSelectionStore,
    AssetSearchSessionStore,
    AssetHubService,
    ProviderSearchRequest,
    persist_download_settings,
    persist_gallery_settings,
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


async def _request_json(request: Request) -> Any:
    try:
        return await request.json()
    except Exception:
        return {}


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    token = str(value).strip().casefold()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part or "").strip() for part in value if str(part or "").strip())
    return ()


def _session_discovery(session: Mapping[str, Any] | None, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    session_discovery = dict(session.get("discoveryCriteria") or {}) if isinstance(session, Mapping) else {}
    body = dict(fallback or {}) if isinstance(fallback, Mapping) else {}

    def provided(*keys: str) -> str:
        for key in keys:
            value = body.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    provider_text = provided("provider_id", "providerId")
    mode_text = provided("mode")
    query_text = provided("q", "query")
    asset_type_text = provided("asset_kind", "assetType", "type")
    creator_text = provided("creator")
    sort_text = provided("sort")
    period_text = provided("period")
    safe_content_value = body.get("safe_content") if "safe_content" in body else body.get("safeContent")

    provider_id = (provider_text or str(session_discovery.get("providerId") or (session.get("providerId") if isinstance(session, Mapping) else "") or "civitai")).strip().casefold() or "civitai"
    mode = (mode_text or str(session_discovery.get("mode") or (session.get("mode") if isinstance(session, Mapping) else "browse") or "browse")).strip().casefold()
    discovery = {
        "providerId": provider_id,
        "mode": "search" if mode == "search" else "browse",
        "query": query_text or str(session_discovery.get("query") or "").strip(),
        "assetType": (asset_type_text if asset_type_text and asset_type_text.casefold() != "any" else str(session_discovery.get("assetType") or asset_type_text or "any")).strip() or "any",
        "creator": creator_text or str(session_discovery.get("creator") or "").strip(),
        "providerSort": sort_text or str(session_discovery.get("providerSort") or "").strip(),
        "period": period_text or str(session_discovery.get("period") or "").strip(),
        "safeContent": _truthy(safe_content_value, bool(session_discovery.get("safeContent", True))) if safe_content_value is not None else bool(session_discovery.get("safeContent", True)),
    }
    if discovery["mode"] != "search":
        discovery["query"] = ""
    limit = body.get("limit") if body.get("limit") not in (None, "") else session_discovery.get("limit")
    try:
        if limit is not None and str(limit).strip() != "":
            discovery["limit"] = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        pass
    return discovery


def _session_result_filters(
    session: Mapping[str, Any] | None,
    *,
    base_models: Any = None,
    safe_content: Any = True,
    support_filter: str = "supported",
    library_filter: str = "any",
    preview_filter: str = "any",
) -> dict[str, Any]:
    result_filters = dict(session.get("resultFilters") or {}) if isinstance(session, Mapping) else {}
    resolved_base_models = _string_list(base_models)
    if not resolved_base_models:
        resolved_base_models = _string_list(result_filters.get("baseModels"))
    resolved_safe_content = _truthy(safe_content, True)
    if resolved_safe_content is True and result_filters.get("safeContent") is False:
        resolved_safe_content = False
    resolved_support_filter = str(support_filter or "").strip() or "supported"
    if resolved_support_filter == "supported" and result_filters.get("supportFilter"):
        resolved_support_filter = str(result_filters.get("supportFilter") or "supported").strip() or "supported"
    resolved_library_filter = str(library_filter or "").strip() or "any"
    if resolved_library_filter == "any" and result_filters.get("libraryFilter"):
        resolved_library_filter = str(result_filters.get("libraryFilter") or "any").strip() or "any"
    resolved_preview_filter = str(preview_filter or "").strip() or "any"
    if resolved_preview_filter == "any" and result_filters.get("previewFilter"):
        resolved_preview_filter = str(result_filters.get("previewFilter") or "any").strip() or "any"
    return {
        "baseModels": resolved_base_models,
        "safeContent": resolved_safe_content,
        "supportFilter": resolved_support_filter,
        "libraryFilter": resolved_library_filter,
        "previewFilter": resolved_preview_filter,
    }


def build_asset_hub_router(
    service: AssetHubService,
    *,
    secrets: AssetHubSecretStore | None = None,
    downloads: AssetHubDownloadManager | None = None,
    gallery_cache: AssetGalleryCache | None = None,
    installer: AssetHubInstaller | None = None,
    upscaler_catalog: Any | None = None,
    upscaler_favorites: UpscalerFavoriteStore | None = None,
    selections: AssetHubSelectionStore | None = None,
    discovery_index: AssetDiscoveryIndex | None = None,
    search_sessions: AssetSearchSessionStore | None = None,
    user_config_path: str | Path | None = None,
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

    @router.get("/selections/{purpose}")
    async def get_selection(purpose: str) -> Any:
        if selections is None:
            return _error_response(AssetHubError("selection_store_unavailable", "Asset selection storage is unavailable.", status_code=503))
        try:
            selection = selections.get(purpose)
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "selection": selection.to_dict() if selection is not None else None,
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.put("/selections/{purpose}")
    async def set_selection(purpose: str, payload: Any = Body(default={})) -> Any:
        if selections is None:
            return _error_response(AssetHubError("selection_store_unavailable", "Asset selection storage is unavailable.", status_code=503))
        try:
            selection = selections.set(purpose, _body_mapping(payload))
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "selection": selection.to_dict(),
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.delete("/selections/{purpose}")
    async def delete_selection(purpose: str) -> Any:
        if selections is None:
            return _error_response(AssetHubError("selection_store_unavailable", "Asset selection storage is unavailable.", status_code=503))
        try:
            removed = selections.delete(purpose)
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "removed": bool(removed),
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/search-sessions")
    async def list_search_sessions(include_closed: bool = Query(False), limit: int = Query(100, ge=1, le=500)) -> Any:
        if search_sessions is None:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "sessions": []}
        try:
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "sessions": search_sessions.list(include_closed=include_closed, limit=limit),
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/search-sessions")
    async def create_search_session(request: Request) -> Any:
        if search_sessions is None:
            return _error_response(AssetHubError("search_sessions_unavailable", "Persistent search sessions are unavailable.", status_code=503))
        body = _body_mapping(await _request_json(request))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "session": search_sessions.create(body)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/search-sessions/{session_id}")
    async def get_search_session(session_id: str) -> Any:
        if search_sessions is None:
            return _error_response(AssetHubError("search_sessions_unavailable", "Persistent search sessions are unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "session": search_sessions.get(session_id)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.patch("/search-sessions/{session_id}")
    async def update_search_session(session_id: str, request: Request) -> Any:
        if search_sessions is None:
            return _error_response(AssetHubError("search_sessions_unavailable", "Persistent search sessions are unavailable.", status_code=503))
        body = _body_mapping(await _request_json(request))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "session": search_sessions.update(session_id, body)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/search-sessions/{session_id}/pause")
    async def pause_search_session(session_id: str) -> Any:
        if search_sessions is None:
            return _error_response(AssetHubError("search_sessions_unavailable", "Persistent search sessions are unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "session": search_sessions.pause(session_id)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/search-sessions/{session_id}/resume")
    async def resume_search_session(session_id: str) -> Any:
        if search_sessions is None:
            return _error_response(AssetHubError("search_sessions_unavailable", "Persistent search sessions are unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "session": search_sessions.resume(session_id)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/search-sessions/{session_id}/stop")
    async def stop_search_session(session_id: str) -> Any:
        if search_sessions is None:
            return _error_response(AssetHubError("search_sessions_unavailable", "Persistent search sessions are unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "session": search_sessions.stop(session_id)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/search-sessions/{session_id}/close")
    async def close_search_session(session_id: str) -> Any:
        if search_sessions is None:
            return _error_response(AssetHubError("search_sessions_unavailable", "Persistent search sessions are unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "session": search_sessions.close(session_id)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/index/status")
    async def discovery_index_status() -> Any:
        if discovery_index is None:
            return _error_response(AssetHubError("discovery_index_unavailable", "Persistent discovery index is unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "status": discovery_index.status()}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/index/models/{provider_id}/{model_id}")
    async def discovery_index_model(provider_id: str, model_id: str) -> Any:
        if discovery_index is None:
            return _error_response(AssetHubError("discovery_index_unavailable", "Persistent discovery index is unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "model": discovery_index.get_model(provider_id, model_id)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/index/search")
    async def search_index(
        provider_id: str = Query("civitai"),
        q: str = Query(""),
        asset_kind: str = Query("any"),
        base_models: list[str] = Query(default=[]),
        creator: str = Query(""),
        sort: str = Query(""),
        period: str = Query(""),
        safe_content: bool = Query(True),
        support_filter: str = Query("any"),
        library_filter: str = Query("any"),
        preview_filter: str = Query("any"),
        mode: str = Query("search"),
        limit: int = Query(48, ge=1, le=500),
        offset: int = Query(0, ge=0),
        session_id: str = Query("", alias="sessionId"),
    ) -> Any:
        if discovery_index is None:
            return _error_response(AssetHubError("discovery_index_unavailable", "Persistent discovery index is unavailable.", status_code=503))
        try:
            session_payload = search_sessions.get(session_id) if (search_sessions is not None and str(session_id or "").strip()) else None
            discovery = _session_discovery(session_payload, {
                "provider_id": provider_id,
                "q": q,
                "asset_kind": asset_kind,
                "creator": creator,
                "sort": sort,
                "period": period,
                "mode": mode,
                "limit": limit,
            })
            selected_provider = discovery["providerId"]
            selected_session_id = str(session_id or "").strip()
            selected_mode = discovery["mode"]
            resolved_filters = _session_result_filters(
                session_payload,
                base_models=base_models,
                safe_content=safe_content,
                support_filter=support_filter,
                library_filter=library_filter,
                preview_filter=preview_filter,
            )

            if search_sessions is not None and selected_session_id and selected_mode == "search" and search_sessions.candidate_count(selected_session_id) == 0:
                seed_offset = 0
                seed_limit = 500
                while True:
                    seed_page = discovery_index.search(
                        provider_id=selected_provider,
                        query=discovery["query"],
                        creator=discovery["creator"],
                        asset_kind=discovery["assetType"],
                        sort=discovery["providerSort"],
                        period=discovery["period"],
                        mode=selected_mode,
                        limit=seed_limit,
                        offset=seed_offset,
                    )
                    seed_items = list(seed_page.get("items") or [])
                    if seed_items:
                        search_sessions.add_candidates(
                            selected_session_id,
                            selected_provider,
                            [str(item.get("remoteModelId") or "") for item in seed_items],
                            source_kind="local_index",
                        )
                    next_seed = seed_page.get("nextOffset")
                    if next_seed is None or not seed_items:
                        break
                    seed_offset = int(next_seed)

            page = discovery_index.search(
                provider_id=selected_provider,
                query=discovery["query"],
                creator=discovery["creator"],
                asset_kind=discovery["assetType"],
                base_models=resolved_filters["baseModels"],
                safe_content=resolved_filters["safeContent"],
                support_filter=resolved_filters["supportFilter"],
                library_filter=resolved_filters["libraryFilter"],
                preview_filter=resolved_filters["previewFilter"],
                sort=discovery["providerSort"],
                period=discovery["period"],
                mode=selected_mode,
                limit=limit,
                offset=offset,
                candidate_session_id=selected_session_id if selected_mode == "search" else "",
            )
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "page": {
                    **page,
                    "sessionId": selected_session_id or None,
                    "discoveryCriteria": discovery,
                    "resultFilters": dict((session_payload or {}).get("resultFilters") or {}),
                },
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/index/query")
    async def faceted_index_query(request: Request) -> Any:
        if discovery_index is None:
            return _error_response(AssetHubError("discovery_index_unavailable", "Persistent discovery index is unavailable.", status_code=503))
        if search_sessions is None:
            return _error_response(AssetHubError("search_sessions_unavailable", "Persistent search sessions are unavailable.", status_code=503))
        body = _body_mapping(await _request_json(request))
        session_id = str(body.get("sessionId") or "").strip()
        if not session_id:
            return _error_response(AssetHubError("search_session_required", "A search session is required for local faceted filtering.", status_code=400))
        try:
            session_payload = search_sessions.get(session_id)
            if "filters" in body:
                session_payload = search_sessions.update(session_id, {"resultFilters": _body_mapping(body.get("filters"))})
            discovery = dict(session_payload.get("discoveryCriteria") or {})
            provider_id = str(discovery.get("providerId") or session_payload.get("providerId") or "civitai").strip().casefold() or "civitai"
            mode = "search" if str(discovery.get("mode") or session_payload.get("mode") or "browse").strip().casefold() == "search" else "browse"

            # The local candidate seed is the primary-discovery result set only.
            # Secondary filters are deliberately not applied while populating it.
            if mode == "search" and search_sessions.candidate_count(session_id) == 0:
                seed_offset = 0
                while True:
                    seed_page = discovery_index.search(
                        provider_id=provider_id,
                        query=str(discovery.get("query") or ""),
                        creator=str(discovery.get("creator") or ""),
                        asset_kind=str(discovery.get("assetType") or "any"),
                        sort=str(discovery.get("providerSort") or ""),
                        period=str(discovery.get("period") or ""),
                        mode="search",
                        limit=500,
                        offset=seed_offset,
                    )
                    seed_items = list(seed_page.get("items") or [])
                    if seed_items:
                        search_sessions.add_candidates(
                            session_id,
                            provider_id,
                            [str(item.get("remoteModelId") or "") for item in seed_items],
                            source_kind="local_index",
                        )
                    next_seed = seed_page.get("nextOffset")
                    if next_seed is None or not seed_items:
                        break
                    seed_offset = int(next_seed)

            filters = dict(session_payload.get("resultFilters") or {})
            page = discovery_index.query_facets(
                provider_id=provider_id,
                session_id=session_id,
                mode=mode,
                filters=filters,
                sort=str(body.get("sort") or filters.get("localSort") or "candidate_order"),
                offset=max(0, int(body.get("offset") or 0)),
                limit=max(1, min(int(body.get("limit") or 50), 500)),
                facets=tuple(str(value or "") for value in (body.get("facets") or ())),
            )
            session_payload = search_sessions.update(session_id, {
                "resultCount": int(page.get("matchCount") or 0),
                "cachedResultCount": int(page.get("candidateCount") or 0),
                "touchLocal": True,
            })
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "page": page,
                "session": session_payload,
            }
        except (ValueError, TypeError) as exc:
            return _error_response(AssetHubError("facet_query_invalid", str(exc), status_code=400))
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/search")
    async def search(
        provider_id: str = Query("civitai"),
        q: str = Query(""),
        legacy_query: str = Query("", alias="query"),
        asset_kind: str = Query("any"),
        legacy_asset_kind: str = Query("", alias="type"),
        base_models: list[str] = Query(default=[]),
        legacy_base_models: list[str] = Query(default=[], alias="base_model"),
        creator: str = Query(""),
        sort: str = Query(""),
        period: str = Query(""),
        safe_content: bool = Query(True),
        support_filter: str = Query("supported"),
        library_filter: str = Query("any"),
        cursor: str = Query(""),
        limit: int = Query(48, ge=1, le=200),
        session_id: str = Query("", alias="sessionId"),
    ) -> Any:
        try:
            session_payload = search_sessions.get(session_id) if (search_sessions is not None and str(session_id or "").strip()) else None
            direct_query = q or legacy_query
            direct_asset_kind = legacy_asset_kind or asset_kind
            discovery = _session_discovery(session_payload, {
                "provider_id": provider_id,
                "q": direct_query,
                "asset_kind": direct_asset_kind,
                "creator": creator,
                "sort": sort,
                "period": period,
                "mode": "search" if str((session_payload or {}).get("mode") or "").strip().casefold() == "search" or str(q or "").strip() else "browse",
                "limit": limit,
            })
            has_session = bool(str(session_id or "").strip())
            provider_base_models = () if has_session else _string_list([*base_models, *legacy_base_models])
            request = ProviderSearchRequest(
                query=discovery["query"],
                asset_kind=discovery["assetType"],
                base_models=provider_base_models,
                creator=discovery.get("creator", ""),
                sort=discovery["providerSort"],
                period=discovery["period"],
                safe_content=bool(discovery.get("safeContent", True)),
                support_filter="any" if has_session else support_filter,
                library_filter="any" if has_session else library_filter,
                search_mode=discovery["mode"],
                cursor=cursor,
                limit=limit,
            )
            payload = await service.search(discovery["providerId"], request)
            items = [item.to_dict() if hasattr(item, "to_dict") else dict(item) if isinstance(item, Mapping) else item for item in payload.items]
            indexed = 0
            if discovery_index is not None and items:
                indexed = discovery_index.ingest_items(discovery["providerId"], items)
                discovery_index.remember_search_page(
                    provider_id=discovery["providerId"],
                    items=items,
                    query=discovery["query"],
                    creator=discovery["creator"],
                    asset_kind=discovery["assetType"],
                    sort=discovery["providerSort"],
                    period=discovery["period"],
                    mode=discovery["mode"],
                    first_page=not bool(cursor),
                    complete=not bool(payload.next_cursor),
                )
            if search_sessions is not None and str(session_id or "").strip() and items:
                search_sessions.add_candidates(
                    str(session_id).strip(),
                    discovery["providerId"],
                    [str(item.get("remoteModelId") or "") for item in items],
                    source_kind="provider_search",
                )
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "readOnly": True,
                "indexed": int(indexed or 0),
                "page": {
                    **payload.to_dict(),
                    "sessionId": str(session_id or "").strip() or None,
                    "discoveryCriteria": discovery,
                },
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/models/{provider_id}/{model_id}")
    async def model(
        provider_id: str,
        model_id: str,
        refresh: bool = Query(default=False),
        browser: bool = Query(default=False),
    ) -> Any:
        try:
            result = await service.get_model(provider_id, model_id, refresh=refresh, include_unsupported=browser)
            model_payload = result.to_dict()
            if discovery_index is not None:
                try:
                    discovery_index.ingest_model(provider_id, model_payload)
                except (OSError, sqlite3.Error, ValueError, TypeError):
                    pass
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "model": model_payload}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/versions/{provider_id}/{version_id}")
    async def version(
        provider_id: str,
        version_id: str,
        refresh: bool = Query(default=False),
        browser: bool = Query(default=False),
    ) -> Any:
        try:
            result = await service.get_version(provider_id, version_id, refresh=refresh, include_unsupported=browser)
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

    @router.get("/gallery-settings")
    async def get_gallery_settings() -> Any:
        if gallery_cache is None:
            return _error_response(AssetHubError("gallery_cache_unavailable", "Managed gallery caching is unavailable.", status_code=503))
        return {
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "settings": gallery_cache.settings.to_dict(),
            "cache": gallery_cache.status(),
        }

    @router.put("/gallery-settings")
    async def update_gallery_settings(payload: Any = Body(default={})) -> Any:
        if gallery_cache is None:
            return _error_response(AssetHubError("gallery_cache_unavailable", "Managed gallery caching is unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            settings = gallery_cache.update_settings(body)
            if user_config_path is not None:
                persist_gallery_settings(user_config_path, settings)
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "settings": settings.to_dict(),
                "cache": gallery_cache.status(),
                "persisted": user_config_path is not None,
            }
        except (OSError, ValueError) as exc:
            return _error_response(AssetHubError("gallery_settings_error", str(exc), status_code=400))

    @router.get("/gallery-cache/status")
    async def gallery_cache_status() -> Any:
        if gallery_cache is None:
            return _error_response(AssetHubError("gallery_cache_unavailable", "Managed gallery caching is unavailable.", status_code=503))
        return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "cache": gallery_cache.status()}

    @router.post("/gallery-cache/cleanup")
    async def cleanup_gallery_cache() -> Any:
        if gallery_cache is None:
            return _error_response(AssetHubError("gallery_cache_unavailable", "Managed gallery caching is unavailable.", status_code=503))
        return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "cache": gallery_cache.clear_temporary()}

    @router.post("/gallery-cache/fetch")
    async def fetch_gallery_image(payload: Any = Body(default={})) -> Any:
        if gallery_cache is None:
            return _error_response(AssetHubError("gallery_cache_unavailable", "Managed gallery caching is unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            cached = await gallery_cache.fetch(
                provider_id=str(body.get("providerId") or "civitai"),
                remote_model_id=str(body.get("remoteModelId") or ""),
                remote_version_id=str(body.get("remoteVersionId") or ""),
                provider_image_id=str(body.get("providerImageId") or ""),
                image_url=str(body.get("imageUrl") or ""),
                cache_class="detail",
                protected=False,
            )
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "image": {
                    **cached,
                    "imageUrl": f"/api/asset-hub/gallery-cache/images/{cached['cacheKey']}",
                },
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/gallery-cache/images/{cache_key}")
    async def gallery_cache_image(cache_key: str) -> Any:
        if gallery_cache is None:
            return _error_response(AssetHubError("gallery_cache_unavailable", "Managed gallery caching is unavailable.", status_code=503))
        path = gallery_cache.cached_file(cache_key)
        if path is None:
            return _error_response(AssetHubError("gallery_image_not_found", "Cached gallery image was not found.", status_code=404))
        return FileResponse(path)

    @router.post("/gallery-cache/library")
    async def cache_library_gallery(payload: Any = Body(default={})) -> Any:
        if gallery_cache is None:
            return _error_response(AssetHubError("gallery_cache_unavailable", "Managed gallery caching is unavailable.", status_code=503))
        body = _body_mapping(payload)
        provider_id = str(body.get("providerId") or "civitai").strip().casefold() or "civitai"
        model_id = str(body.get("remoteModelId") or "").strip()
        selected_version_id = str(body.get("remoteVersionId") or "").strip()
        if not model_id:
            return _error_response(AssetHubError("gallery_model_required", "A provider model id is required.", status_code=400))
        # Expanded gallery persistence is library-only. Do not allow a search
        # result to turn into a bulk gallery downloader before the model exists
        # in the local library.
        if not service.presence.model_overlay(provider_id, model_id):
            return _error_response(AssetHubError("gallery_library_required", "Expanded gallery caching is available only after the asset is installed in the library.", status_code=409))
        mode = gallery_cache.settings.library_gallery_mode
        if mode == "hero_only":
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "cached": 0, "mode": mode, "cache": gallery_cache.status()}
        try:
            model = await service.get_model(provider_id, model_id, refresh=False, include_unsupported=True)
            versions = list(model.versions)
            if mode == "selected_version":
                versions = [version for version in versions if str(version.remote_version_id) == selected_version_id]
            targets: list[tuple[str, Any]] = []
            for version in versions:
                for preview in version.previews:
                    if str(preview.kind or "image").casefold() != "image" or not str(preview.url or "").strip():
                        continue
                    targets.append((str(version.remote_version_id), preview))
            cached_count = 0
            for version_id, preview in targets:
                await gallery_cache.fetch(
                    provider_id=provider_id,
                    remote_model_id=model_id,
                    remote_version_id=version_id,
                    provider_image_id=str(preview.provider_image_id or ""),
                    image_url=str(preview.url or ""),
                    cache_class="library",
                    protected=True,
                )
                cached_count += 1
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "cached": cached_count,
                "mode": mode,
                "cache": gallery_cache.status(),
            }
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/download-settings")
    async def get_download_settings() -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        return {
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "settings": downloads.settings_payload(),
        }

    @router.put("/download-settings")
    async def update_download_settings(payload: Any = Body(default={})) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            settings = await downloads.update_settings(body)
            if user_config_path is not None:
                persist_download_settings(user_config_path, settings)
            return {
                "contractVersion": ASSET_HUB_CONTRACT_VERSION,
                "settings": downloads.settings_payload(),
                "persisted": user_config_path is not None,
            }
        except (OSError, ValueError) as exc:
            return _error_response(AssetHubError("download_settings_error", str(exc), status_code=400))

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
                file_name_hint=str(body.get("fileName") or ""),
                expected_bytes_hint=body.get("expectedBytes") or 0,
                expected_sha256_hint=str(body.get("expectedSha256") or ""),
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
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": downloads.job_payload(job)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.get("/download-jobs")
    async def list_download_jobs(limit: int = Query(default=100, ge=1, le=500)) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        jobs = []
        for item in downloads.list_jobs(limit=limit):
            payload = downloads.job_payload(item)
            if installer is not None:
                installed = installer.get_by_download_job(item.job_id)
                payload["install"] = installed.to_dict() if installed is not None else None
            jobs.append(payload)
        return {
            "contractVersion": ASSET_HUB_CONTRACT_VERSION,
            "jobs": jobs,
            "settings": downloads.settings_payload(),
        }

    @router.post("/download-jobs/bulk")
    async def bulk_download_jobs(payload: Any = Body(default={})) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            result = await downloads.bulk_action(str(_body_mapping(payload).get("action") or ""))
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, **result}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.delete("/download-jobs/history")
    async def clear_download_history(status: str = Query(default="inactive", max_length=32)) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, **downloads.clear_history(status=status)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/download-jobs/cleanup-stale")
    async def cleanup_stale_downloads(payload: Any = Body(default={})) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        body = _body_mapping(payload)
        try:
            hours = max(0.0, float(body.get("maxAgeHours", 24)))
        except (TypeError, ValueError):
            hours = 24.0
        result = downloads.cleanup_stale_partials(
            max_age_seconds=hours * 3600.0,
            include_recent_unrecoverable=bool(body.get("includeRecentUnrecoverable", True)),
        )
        return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, **result}

    @router.get("/download-jobs/{job_id}")
    async def get_download_job(job_id: str) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            record = downloads.get_job(job_id)
            payload = downloads.job_payload(record)
            if installer is not None:
                installed = installer.get_by_download_job(record.job_id)
                payload["install"] = installed.to_dict() if installed is not None else None
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": payload}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/download-jobs/{job_id}/pause")
    async def pause_download_job(job_id: str) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            job = await downloads.pause(job_id)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": downloads.job_payload(job)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/download-jobs/{job_id}/cancel")
    async def cancel_download_job(job_id: str) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            job = await downloads.cancel(job_id)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": downloads.job_payload(job)}
        except AssetHubError as exc:
            return _error_response(exc)

    @router.post("/download-jobs/{job_id}/resume")
    async def resume_download_job(job_id: str) -> Any:
        if downloads is None:
            return _error_response(AssetHubError("download_runtime_unavailable", "Asset Hub downloads are unavailable.", status_code=503))
        try:
            job = await downloads.resume(job_id)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "job": downloads.job_payload(job)}
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

    @router.post("/installed/{install_id}/open-folder")
    async def open_installed_folder(install_id: str) -> Any:
        if installer is None:
            return _error_response(AssetHubError("install_runtime_unavailable", "Asset Hub installation is unavailable.", status_code=503))
        try:
            path = installer.open_install_folder(install_id)
            return {"contractVersion": ASSET_HUB_CONTRACT_VERSION, "opened": True, "path": path}
        except AssetHubError as exc:
            return _error_response(exc)

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
