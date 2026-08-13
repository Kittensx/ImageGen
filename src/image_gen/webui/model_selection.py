from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from modules.asset_discovery import resolve_nested_asset
from modules.checkpoint_inspector import CheckpointInspector
from modules.project_context import ProjectContext
from modules.txt2img.model_selector import MODEL_EXTENSIONS
from image_gen.systems.sd21_support import SD21SupportManager


def _canonical_text(value: str | os.PathLike[str]) -> str:
    return str(value or "").strip().replace("\\", "/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_token(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


@dataclass(frozen=True)
class ActiveModelSelection:
    selection_id: str
    requested_path: str
    resolved_path: str
    model_name: str
    extension: str
    size_bytes: int
    modified_ns: int
    selected_at: str
    model_filename: str = ""
    model_name_source: str = ""
    source: str = "webui"
    status: str = "ready"
    architecture: str = ""
    prediction_type: str = ""
    conditioning_dimension: int | None = None
    architecture_summary: str = ""
    architecture_source: str = ""
    checkpoint_kind: str = ""
    architecture_contract: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "requested_path": self.requested_path,
            "resolved_path": self.resolved_path,
            "model_name": self.model_name,
            "extension": self.extension,
            "model_filename": self.model_filename,
            "model_name_source": self.model_name_source,
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
            "selected_at": self.selected_at,
            "source": self.source,
            "status": self.status,
            "architecture": self.architecture,
            "prediction_type": self.prediction_type,
            "conditioning_dimension": self.conditioning_dimension,
            "architecture_summary": self.architecture_summary,
            "architecture_source": self.architecture_source,
            "checkpoint_kind": self.checkpoint_kind,
            "architecture_contract": dict(self.architecture_contract or {}),
        }


class ModelSelectionUnavailableError(ValueError):
    """Expected no-model/stale-model state, not an application failure."""


class WebUIModelSelectionState:
    """Server-authoritative checkpoint selection for the WebUI.

    The browser dropdown is not trusted as the final source of truth. A model
    must be activated through the backend, and every submitted job must carry
    the matching activation token and path. This prevents a restored default or
    stale browser value from silently replacing the user's selected checkpoint.
    """

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self._active: ActiveModelSelection | None = None
        self._lock = threading.RLock()
        self._inspector = CheckpointInspector()
        self._inspection_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._sd21_support: SD21SupportManager | None = None

    def _sd21_support_manager(self) -> SD21SupportManager:
        if self._sd21_support is None:
            self._sd21_support = SD21SupportManager(self.context)
        return self._sd21_support

    def _resolve_existing_model(self, value: str | os.PathLike[str]) -> Path:
        text = _canonical_text(value)
        if not text:
            raise ModelSelectionUnavailableError("No checkpoint is installed or selected. Add a checkpoint before generating.")

        candidates: list[Path] = []
        direct = self.context.resolve_project_path(text).expanduser().resolve()
        candidates.append(direct)

        stripped_leading = text.lstrip("/\\")
        if stripped_leading and stripped_leading != text:
            candidates.append((self.context.project_root / stripped_leading).expanduser().resolve())

        checkpoints_dir = getattr(self.context, "checkpoints_dir", None)
        tail = Path(stripped_leading or text)
        if checkpoints_dir is not None and tail.name:
            candidates.append((checkpoints_dir / tail.name).expanduser().resolve())
            if tail.parts:
                candidates.append((checkpoints_dir.joinpath(*tail.parts)).expanduser().resolve())

        seen: set[str] = set()
        unique_candidates: list[Path] = []
        for candidate in candidates:
            token = _path_token(candidate)
            if token in seen:
                continue
            seen.add(token)
            unique_candidates.append(candidate)

        for path in unique_candidates:
            if not path.is_file():
                continue
            if path.suffix.lower() not in MODEL_EXTENSIONS:
                supported = ", ".join(sorted(MODEL_EXTENSIONS))
                raise ValueError(
                    f"Unsupported checkpoint extension {path.suffix!r}; expected one of: {supported}."
                )
            return path

        if checkpoints_dir is not None and checkpoints_dir.is_dir():
            nested = resolve_nested_asset(
                checkpoints_dir,
                text,
                extensions=MODEL_EXTENSIONS,
            )
            if nested is not None:
                return nested

        configured_root = checkpoints_dir.resolve() if checkpoints_dir is not None else self.context.project_root.resolve()
        attempted = "; ".join(str(item) for item in unique_candidates)
        raise ModelSelectionUnavailableError(
            "Selected checkpoint is not available in the local checkpoint library. "
            f"Requested: {text}. "
            f"Configured checkpoint root: {configured_root}. "
            f"Attempted: {attempted}"
        )

    def _inspect_model_contract(self, path: Path, *, stat_size: int, stat_mtime_ns: int) -> dict[str, Any]:
        cache_key = (_path_token(path), int(stat_size), int(stat_mtime_ns))
        with self._lock:
            cached = self._inspection_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        payload: dict[str, Any] = {}
        if path.suffix.lower() == ".safetensors":
            try:
                report = self._inspector.inspect(str(path))
                contract = report.architecture_contract.to_dict()
                payload = {
                    "architecture": report.architecture,
                    "prediction_type": report.prediction_type,
                    "conditioning_dimension": report.model_dimension,
                    "architecture_summary": report.architecture_summary,
                    "architecture_source": report.architecture_source or report.prediction_type_source,
                    "checkpoint_kind": report.checkpoint_kind,
                    "architecture_contract": contract,
                    "model_name": str(getattr(report, "model_name", path.stem) or path.stem),
                    "model_name_source": str(getattr(report, "model_name_source", "filename") or "filename"),
                }
            except Exception:
                payload = {}
        with self._lock:
            self._inspection_cache = {cache_key: dict(payload)}
        return dict(payload)

    def authorize(self, model_path: str | os.PathLike[str], *, source: str = "webui") -> ActiveModelSelection:
        """Validate and snapshot a checkpoint without changing browser state.

        Replay jobs use this path so they receive the same extension, path,
        size, and modification-time enforcement as normal WebUI selections
        without unexpectedly replacing the checkpoint selected in a separate
        generation form.
        """
        path = self._resolve_existing_model(model_path)
        stat = path.stat()
        inspection = self._inspect_model_contract(
            path,
            stat_size=int(stat.st_size),
            stat_mtime_ns=int(stat.st_mtime_ns),
        )
        return ActiveModelSelection(
            selection_id=uuid.uuid4().hex[:16],
            requested_path=str(model_path),
            resolved_path=str(path),
            model_name=str(inspection.get("model_name") or path.stem),
            extension=path.suffix.lower(),
            model_filename=path.name,
            model_name_source=str(inspection.get("model_name_source") or "filename"),
            size_bytes=int(stat.st_size),
            modified_ns=int(stat.st_mtime_ns),
            selected_at=_utc_now(),
            source=str(source or "webui"),
            architecture=str(inspection.get("architecture") or ""),
            prediction_type=str(inspection.get("prediction_type") or ""),
            conditioning_dimension=inspection.get("conditioning_dimension"),
            architecture_summary=str(inspection.get("architecture_summary") or ""),
            architecture_source=str(inspection.get("architecture_source") or ""),
            checkpoint_kind=str(inspection.get("checkpoint_kind") or ""),
            architecture_contract=dict(inspection.get("architecture_contract") or {}),
        )

    def activate(self, model_path: str | os.PathLike[str], *, source: str = "webui") -> ActiveModelSelection:
        selection = self.authorize(model_path, source=source)
        if str(selection.architecture or "").strip().casefold() == "sd2.x":
            support = self._sd21_support_manager().ensure_for_architecture(
                selection.architecture,
                launch_if_missing=True,
                reason=f"checkpoint_activation:{source or 'webui'}",
            )
            if not support.ready:
                if support.installer_running:
                    raise ValueError(
                        "Stable Diffusion 2.1 runtime support is still installing in a separate setup window. "
                        "Wait for that installer to finish, then activate the checkpoint again."
                    )
                raise ValueError(
                    "Stable Diffusion 2.1 runtime support files are missing. IMAGE_GEN attempted to launch the "
                    "separate SD2.1 support installer. Complete that installer, then activate the checkpoint again."
                )
            if not support.hardware_eligible:
                raise ValueError(
                    "The selected Stable Diffusion 2.x model cannot be activated on the currently qualified GPU. "
                    f"{support.hardware_reason}"
                )
        with self._lock:
            self._active = selection
        return selection

    def current(self) -> ActiveModelSelection | None:
        with self._lock:
            return self._active

    def current_payload(self) -> dict[str, Any] | None:
        current = self.current()
        return current.to_dict() if current is not None else None

    def deactivate(self) -> None:
        with self._lock:
            self._active = None

    def enforce(self, payload: Mapping[str, Any] | None) -> tuple[dict[str, Any], ActiveModelSelection]:
        incoming = dict(payload or {})
        requested_path = str(incoming.get("model_path") or "").strip()
        supplied_selection_id = str(incoming.get("_webui_model_selection_id") or "").strip()

        with self._lock:
            active = self._active

        browser_resolved_path = ""
        browser_matches_active = False
        browser_resolve_error = ""

        if requested_path:
            try:
                requested = self._resolve_existing_model(requested_path)
                browser_resolved_path = str(requested)
            except ValueError as exc:
                browser_resolve_error = str(exc)
                raise ModelSelectionUnavailableError(
                    "The checkpoint selected in the browser is no longer available. "
                    "Refresh the model list and select it again. "
                    f"Details: {exc}"
                ) from exc

            requested_token = _path_token(requested)
            active_token = _path_token(active.resolved_path) if active is not None else ""
            selection_mismatch = bool(supplied_selection_id and active is not None and supplied_selection_id != active.selection_id)
            if active is None or requested_token != active_token or selection_mismatch:
                active = self.activate(str(requested), source="job_submission")
        elif active is None:
            raise ModelSelectionUnavailableError("No checkpoint is installed or selected. Add a checkpoint before generating.")

        active_path = Path(active.resolved_path).expanduser().resolve()
        browser_matches_active = not requested_path or _path_token(active_path) == _path_token(browser_resolved_path)
        if not active_path.is_file():
            raise ValueError(
                "The backend-activated checkpoint is no longer available on disk. "
                "Refresh the model list and select it again."
            )
        stat = active_path.stat()
        if int(stat.st_size) != active.size_bytes or int(stat.st_mtime_ns) != active.modified_ns:
            raise ValueError(
                "The selected checkpoint changed on disk after activation. "
                "Refresh the model list and select it again."
            )

        incoming["model_path"] = str(active_path)
        incoming["_webui_model_selection_id"] = active.selection_id
        incoming["_webui_model_requested_path"] = requested_path
        incoming["_webui_model_active_path"] = str(active_path)
        incoming["_webui_model_size_bytes"] = active.size_bytes
        incoming["_webui_model_modified_ns"] = active.modified_ns
        incoming["_webui_model_browser_resolved_path"] = browser_resolved_path
        incoming["_webui_model_browser_matches_active"] = browser_matches_active
        incoming["_webui_model_browser_resolve_error"] = browser_resolve_error
        return incoming, active


__all__ = ["ActiveModelSelection", "ModelSelectionUnavailableError", "WebUIModelSelectionState"]
