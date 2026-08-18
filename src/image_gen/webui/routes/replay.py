from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from image_gen.program_metadata import PRODUCT_NAME


def build_replay_router(*, replay, batch_replay, batch_io, variations, _webui_failure) -> APIRouter:
    router = APIRouter()

    @router.post("/api/replay/preflight")
    async def replay_preflight(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return replay.preflight(payload).to_dict()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/replay/submit")
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


    @router.post("/api/replay/batch/preflight")
    async def batch_replay_preflight(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return batch_replay.preflight(payload).to_dict()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @router.post("/api/replay/batch/submit")
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


    @router.post("/api/batch/import/parse")
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


    @router.post("/api/batch/import/preflight")
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


    @router.post("/api/batch/import/submit")
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


    @router.post("/api/batch/export")
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


    @router.post("/api/variations/preflight")
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


    @router.post("/api/variations/submit")
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


    @router.post("/api/variations/export")
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
                "source": f"{PRODUCT_NAME} Variation Matrix",
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


    return router
