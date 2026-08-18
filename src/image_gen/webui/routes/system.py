from __future__ import annotations

import asyncio
import os
import zipfile
from typing import Any, Mapping

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from image_gen.program_metadata import PRODUCT_NAME, build_program_metadata
from image_gen.webui.bug_reports import BugReportError


def build_system_router(
    *,
    app,
    context,
    jobs,
    model_selection,
    prompt_parsers,
    server_instance_id,
    server_started_at_unix,
    changelog,
    help_center,
    profile,
    bug_reports,
    write_webui_failure_bundle,
    WEBUI_VERSION,
) -> APIRouter:
    router = APIRouter()

    @router.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        # Browsers request this automatically. IMAGE_GEN does not currently
        # ship an icon, so return an intentional empty response instead of a 404.
        return Response(status_code=204)


    @router.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "instance_id": server_instance_id,
            "process_id": os.getpid(),
            "started_at_unix": server_started_at_unix,
            "worker": jobs.status(),
            "version": WEBUI_VERSION,
            "application": build_program_metadata(context.project_root),
            "active_model": model_selection.current_payload(),
            "prompt_parsers": prompt_parsers.descriptors(),
        }


    def _cached_bug_profile() -> dict[str, Any]:
        # Profile/Discord controls should never rebuild diagnostic ZIPs. Bug bundles
        # are prepared only by the explicit bug-report catalog/sync flows.
        return dict((bug_reports.payload().get("profile") or {}))


    @router.get("/api/changelog")
    async def changelog_catalog() -> dict[str, Any]:
        try:
            return await asyncio.to_thread(changelog.catalog)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to load the changelog: {exc}") from exc


    @router.get("/api/changelog/{entry_date}")
    async def changelog_entry(entry_date: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(changelog.entry, entry_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Unable to load changelog entry: {exc}") from exc


    @router.get("/api/help")
    async def help_catalog() -> dict[str, Any]:
        try:
            return await asyncio.to_thread(help_center.catalog)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to load Help Center: {exc}") from exc


    @router.get("/api/help/search")
    async def help_search(q: str = "") -> dict[str, Any]:
        try:
            return await asyncio.to_thread(help_center.search, q)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.get("/api/help/topic/{topic_id:path}")
    async def help_topic(topic_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(help_center.topic, topic_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Unable to load help topic: {exc}") from exc


    @router.get("/api/help/media/{media_path:path}")
    async def help_media(media_path: str) -> FileResponse:
        try:
            path = await asyncio.to_thread(help_center.media_path, media_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)


    @router.get("/api/profile")
    async def imagegen_profile() -> dict[str, Any]:
        try:
            return profile.snapshot(_cached_bug_profile())
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to load the local profile: {exc}") from exc


    @router.patch("/api/profile/sharing")
    async def update_profile_sharing(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            await asyncio.to_thread(profile.update_sharing, payload or {})
            bug_profile = _cached_bug_profile()
            presence = await asyncio.to_thread(profile.publish_presence, bug_profile, active=True)
            diagnostic = _discord_presence_diagnostic(presence, request_path="/api/profile/sharing")
            result = profile.snapshot(bug_profile)
            result["presence_publish"] = presence
            result["presence_diagnostic_created"] = bool(diagnostic.get("diagnostic_created"))
            if diagnostic.get("diagnostic_stage"):
                result["presence_diagnostic_stage"] = diagnostic["diagnostic_stage"]
            return result
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to update profile sharing preferences: {exc}") from exc


    @router.get("/api/profile/discord/community")
    async def discord_community_status() -> dict[str, Any]:
        return await asyncio.to_thread(profile.discord_community_status)


    @router.post("/api/profile/discord/connect")
    async def connect_discord_profile() -> dict[str, Any]:
        result = await asyncio.to_thread(profile.connect_discord)
        if not result.get("ok"):
            state = str(result.get("state") or "discord_link_failed")
            message = str(result.get("message") or "Unable to link Discord.")
            status = 503 if state in {"discord_application_required", "helper_required", "native_helper_required", "sdk_unavailable"} else 400
            raise HTTPException(status_code=status, detail=message)
        snapshot = profile.snapshot(_cached_bug_profile())
        snapshot["discord_link"] = result
        return snapshot


    def _discord_presence_diagnostic(result: Mapping[str, Any], *, request_path: str) -> dict[str, Any]:
        """Record unexpected Discord publication failures without treating setup/user states as bugs."""

        state = str(result.get("state") or "unavailable").strip() or "unavailable"
        expected_states = {
            "disabled_by_user",
            "discord_application_required",
            "helper_required",
            "native_helper_required",
            "presence_helper_required",
            "sdk_unavailable",
        }
        if bool(result.get("published")) or state in expected_states:
            return {"diagnostic_created": False}
        message = str(result.get("message") or "Discord did not provide an error message.").strip()
        error = RuntimeError(f"Discord Rich Presence publish failed [{state}]: {message}")
        try:
            bundle = write_webui_failure_bundle(
                project_root=context.project_root,
                stage="discord_presence_refresh",
                error=error,
                payload={"activity": result.get("activity") or {}},
                request_path=request_path,
                extra={
                    "presence_state": state,
                    "presence_message": message,
                    "discord_capabilities": profile.discord_capabilities(),
                },
            )
            return {
                "diagnostic_created": True,
                "diagnostic_stage": "discord_presence_refresh",
                "diagnostic_bundle": str(bundle.relative_to(context.project_root)) if bundle.is_relative_to(context.project_root) else bundle.name,
            }
        except Exception:
            # Presence refresh should still return its original result even if
            # diagnostic persistence itself is unavailable.
            return {"diagnostic_created": False}


    @router.post("/api/profile/discord/presence")
    async def refresh_discord_presence() -> dict[str, Any]:
        bug_profile = _cached_bug_profile()
        result = await asyncio.to_thread(profile.publish_presence, bug_profile, active=True)
        diagnostic = _discord_presence_diagnostic(result, request_path="/api/profile/discord/presence")
        return {"presence": result, "profile": profile.snapshot(bug_profile), **diagnostic}


    @router.post("/api/profile/discord/disconnect")
    async def disconnect_discord_profile() -> dict[str, Any]:
        await asyncio.to_thread(profile.publish_presence, {}, active=False)
        await asyncio.to_thread(profile.disconnect_discord)
        return profile.snapshot(_cached_bug_profile())


    @router.get("/api/bug-reports")
    async def bug_report_catalog() -> dict[str, Any]:
        try:
            return await asyncio.to_thread(bug_reports.refresh_local)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to prepare local bug reports: {exc}") from exc


    @router.post("/api/bug-reports/sync")
    async def sync_bug_reports() -> dict[str, Any]:
        try:
            await asyncio.to_thread(bug_reports.refresh_local)
            return await asyncio.to_thread(bug_reports.sync_github)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to synchronize bug reports: {exc}") from exc


    @router.post("/api/bug-reports/{fingerprint}/issue")
    async def open_bug_report_issue(fingerprint: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(bug_reports.mark_issue_opened, fingerprint)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BugReportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


    @router.get("/api/bug-reports/{fingerprint}/bundle")
    async def download_bug_report_bundle(fingerprint: str) -> FileResponse:
        try:
            path = bug_reports.bundle_path(fingerprint)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="application/zip", filename=path.name)


    @router.post("/api/bug-reports/{fingerprint}/reveal")
    async def reveal_bug_report_bundle(fingerprint: str) -> dict[str, Any]:
        try:
            path = bug_reports.reveal_bundle(fingerprint)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BugReportError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"revealed": True, "filename": path.name}


    @router.post("/api/system/restart")
    async def restart_backend() -> dict[str, Any]:
        callback = app.state.restart_callback
        if not callable(callback):
            raise HTTPException(
                status_code=503,
                detail=f"Backend restart is unavailable because {PRODUCT_NAME} was not launched through run_webui.bat.",
            )
        loop = asyncio.get_running_loop()
        loop.call_later(0.20, callback)
        return {
            "restart_requested": True,
            "previous_instance_id": server_instance_id,
            "message": f"The {PRODUCT_NAME} backend is restarting. The browser will reconnect automatically.",
        }


    return router
