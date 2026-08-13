from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from image_gen.webui.theme.library import ThemeLibraryError, ThemePackageLibrary


def _raise(exc: ThemeLibraryError) -> None:
    raise HTTPException(status_code=400, detail=exc.to_dict()) from exc


def build_theme_manager_router(service: ThemePackageLibrary) -> APIRouter:
    router = APIRouter(prefix="/api/themes", tags=["themes"])

    @router.get("/library")
    async def library() -> dict[str, Any]:
        return service.library_payload()

    @router.get("/effective")
    async def effective() -> dict[str, Any]:
        return {"effectivePalette": service.resolve_effective_palette()}

    @router.post("/import")
    async def import_package(file: UploadFile = File(...)) -> dict[str, Any]:
        import_root = service.roots.theme_cache_root / "uploads"
        import_root.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "theme.igtheme.zip").suffix or ".zip"
        temp_path: Path | None = None
        total = 0
        upload_limit = 128 * 1024 * 1024
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=import_root, suffix=suffix) as handle:
                temp_path = Path(handle.name)
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > upload_limit:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "code": "theme_package_too_large",
                                "message": "Theme package upload exceeds the 128 MiB upload limit.",
                            },
                        )
                    handle.write(chunk)
            if total == 0:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "theme_package_empty", "message": "Theme package is empty."},
                )
            record = service.install_archive(temp_path)
            return {"installed": record, "library": service.library_payload()}
        except ThemeLibraryError as exc:
            _raise(exc)
        finally:
            await file.close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        raise RuntimeError("unreachable")

    @router.post("/{package_id}/enable")
    async def enable(package_id: str) -> dict[str, Any]:
        try:
            record = service.enable(package_id)
            return {
                "package": record,
                "effectivePalette": service.resolve_effective_palette(),
                "library": service.library_payload(),
            }
        except ThemeLibraryError as exc:
            _raise(exc)
        raise RuntimeError("unreachable")

    @router.post("/{package_id}/disable")
    async def disable(package_id: str) -> dict[str, Any]:
        try:
            record = service.disable(package_id)
            return {
                "package": record,
                "effectivePalette": service.resolve_effective_palette(),
                "library": service.library_payload(),
            }
        except ThemeLibraryError as exc:
            _raise(exc)
        raise RuntimeError("unreachable")

    @router.delete("/{package_id}")
    async def remove(package_id: str) -> dict[str, Any]:
        try:
            removed = service.remove(package_id)
            return {
                "removed": removed,
                "effectivePalette": service.resolve_effective_palette(),
                "library": service.library_payload(),
            }
        except ThemeLibraryError as exc:
            _raise(exc)
        raise RuntimeError("unreachable")

    return router
