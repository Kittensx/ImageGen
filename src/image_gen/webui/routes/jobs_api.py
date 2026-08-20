from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from image_gen.webui.model_selection import ModelSelectionUnavailableError


def encode_sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_jobs_router(*, jobs, catalog, component_selection, model_selection, prompt_configuration, upscaler_catalog, generation_capabilities, _preview_media_type, _webui_failure) -> APIRouter:
    router = APIRouter()

    @router.get("/api/jobs")
    async def list_jobs() -> dict[str, Any]:
        return {"jobs": jobs.list_jobs(), "worker": jobs.status()}


    @router.post("/api/maintenance/live-preview/cleanup")
    async def cleanup_live_previews() -> dict[str, Any]:
        return jobs.cleanup_preview_directories()


    @router.post("/api/maintenance/job-cache/clear")
    async def clear_job_cache() -> dict[str, Any]:
        report = jobs.clear_job_cache(preserve_active=True, startup=False)
        return {
            **report,
            "jobs": jobs.list_jobs(),
            "worker": jobs.status(),
        }


    @router.post("/api/maintenance/queue/dismiss-terminal")
    async def dismiss_terminal_jobs() -> dict[str, Any]:
        report = jobs.dismiss_terminal_jobs()
        return {**report, "jobs": jobs.list_jobs(), "worker": jobs.status()}


    @router.post("/api/maintenance/queue/clear")
    async def clear_queued_jobs(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        report = jobs.clear_queued_jobs(
            reason=str((payload or {}).get("reason") or "Queued jobs were cleared from the WebUI."),
        )
        return {
            **report,
            "jobs": jobs.list_jobs(),
            "worker": jobs.status(),
        }


    @router.post("/api/maintenance/generation/force-stop")
    async def force_stop_generation() -> dict[str, Any]:
        report = await jobs.force_stop_generation(
            reason="Force stop requested from the WebUI.",
        )
        return {
            **report,
            "jobs": jobs.list_jobs(),
            "worker": jobs.status(),
        }


    @router.post("/api/queue/pause-after-current")
    async def pause_queue_after_current(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            worker = await jobs.pause_after_current(
                str((payload or {}).get("job_id") or "").strip() or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "jobs": jobs.list_jobs(),
            "worker": worker,
        }


    @router.post("/api/queue/resume")
    async def resume_generation_queue() -> dict[str, Any]:
        worker = await jobs.resume_queue()
        return {
            "jobs": jobs.list_jobs(),
            "worker": worker,
        }


    @router.post("/api/schedulers/validate")
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


    @router.post("/api/jobs")
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
            if bool(prepared_payload.get("advanced_models_enabled")):
                resolved_composition = component_selection.resolve_selection(
                    str(prepared_payload.get("advanced_model_family") or ""),
                    prepared_payload.get("advanced_model_components") or {},
                    t5_device=prepared_payload.get("advanced_model_t5_device") or "cpu",
                    allow_digital_components=bool(prepared_payload.get("advanced_model_allow_digital_components", True)),
                )
                authoritative_payload = dict(prepared_payload)
                authoritative_payload["model_path"] = resolved_composition["base_source_path"]
                authoritative_payload["advanced_model_family"] = resolved_composition["family"]
                # Advanced Models owns the VAE choice. Do not apply the normal
                # checkpoint-mode VAE override on top of the selected component.
                authoritative_payload["vae_path"] = None
                authoritative_payload["_advanced_model_resolved"] = resolved_composition
                authoritative_payload["advanced_model_composition_sha256"] = resolved_composition["composition_sha256"]
                authoritative_payload["advanced_model_allow_digital_components"] = bool(
                    resolved_composition.get("digital_components_allowed", True)
                )
                authoritative_payload["text_encoder_3_device"] = resolved_composition["t5_device"]
                selected_model_payload = generation_capabilities.model_context_for_advanced_composition(
                    resolved_composition
                )
                selection_id = str(selected_model_payload["selection_id"])
                authoritative_payload["_webui_model_selection_id"] = selection_id
                authoritative_payload["_webui_model_requested_path"] = ""
            else:
                authoritative_payload, selected_model = model_selection.enforce(prepared_payload)
                selected_model_payload = selected_model.to_dict()
        except ModelSelectionUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise _webui_failure(
                "job_submission_model_selection",
                exc,
                payload=prepared_payload,
                request_path="/api/jobs",
                status_code=409,
            ) from exc
        try:
            authoritative_payload = generation_capabilities.enforce_request(
                authoritative_payload,
                active_model=selected_model_payload,
            )
            authoritative_payload = upscaler_catalog.validate_request(authoritative_payload)
            job = await jobs.submit(
                authoritative_payload,
                model_selection=selected_model_payload,
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


    @router.get("/api/jobs/{job_id}/primary-output")
    async def job_primary_output(job_id: str) -> dict[str, Any]:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        for raw_path in job.output_paths:
            summary = catalog.output_summary_from_path(Path(raw_path))
            if summary is not None:
                return summary
        raise HTTPException(status_code=404, detail="No generated output is available for this job yet.")


    @router.get("/api/jobs/{job_id}/diagnostics")
    async def job_diagnostics(job_id: str) -> dict[str, Any]:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        return jobs.diagnostics_payload(job)


    @router.get("/api/jobs/{job_id}/log")
    async def job_log(job_id: str) -> FileResponse:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        path = Path(job.console_log_path or "")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Job console log is not available yet.")
        return FileResponse(path, media_type="text/plain", filename=f"{job_id}-console.log")


    @router.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        existing = jobs.get_job(job_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        if existing.status == "finalizing":
            raise HTTPException(
                status_code=409,
                detail="Generation is complete and output saving is in progress. The save operation will continue.",
            )
        job = await jobs.cancel(job_id)
        return job.to_dict()


    @router.post("/api/jobs/{job_id}/pause")
    async def pause_job(job_id: str) -> dict[str, Any]:
        try:
            worker = await jobs.pause_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Generation job not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"jobs": jobs.list_jobs(), "worker": worker}


    @router.post("/api/jobs/{job_id}/resume")
    async def resume_job(job_id: str) -> dict[str, Any]:
        try:
            worker = await jobs.resume_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Generation job not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"jobs": jobs.list_jobs(), "worker": worker}


    @router.post("/api/jobs/{job_id}/reorder")
    async def reorder_job(job_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            worker = jobs.reorder_job(job_id, str((payload or {}).get("direction") or ""))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Generation job not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"jobs": jobs.list_jobs(), "worker": worker}


    @router.post("/api/jobs/{job_id}/skip")
    async def skip_job_image(job_id: str) -> dict[str, Any]:
        try:
            job = await jobs.skip_current(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Generation job not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.to_dict()


    @router.get("/api/jobs/{job_id}/events")
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


    def _preview_response(path: Path, *, headers: dict[str, str]) -> Response:
        # Live preview files are atomically replaced as generation advances and the
        # final frame may replace the last step file. FileResponse determines
        # Content-Length from a stat call before streaming the path, so a replace in
        # that window can produce a different byte length and make httptools abort
        # the response. Snapshot the small preview file first; Content-Length then
        # describes the exact immutable bytes being returned.
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Live preview is no longer available.") from exc
        return Response(content=content, media_type=_preview_media_type(path), headers=headers)


    @router.get("/api/jobs/{job_id}/preview/latest")
    async def job_preview_latest(job_id: str) -> Response:
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
        return _preview_response(path, headers=headers)


    @router.get("/api/jobs/{job_id}/preview/{step_number}")
    async def job_preview_step(job_id: str, step_number: int) -> Response:
        job = jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Generation job not found.")
        path = jobs.live_preview_step_path(job, step_number)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Requested live preview step was not found.")
        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        return _preview_response(path, headers=headers)


    return router
