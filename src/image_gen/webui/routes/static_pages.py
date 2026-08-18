from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


def build_static_pages_router(*, static_root) -> APIRouter:
    router = APIRouter()

    @router.get("/region-builder.html", include_in_schema=False)
    async def region_builder() -> FileResponse:
        return FileResponse(static_root / "region-builder.html")


    @router.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_root / "index.html")


    return router
