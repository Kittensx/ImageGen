from __future__ import annotations

import json
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .models import WorkspaceLayout, WorkspaceValidationIssue
from .registry import WorkspaceRegistry
from .validation import validate_workspace_layout

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


class WorkspaceStoreError(ValueError):
    pass


@dataclass(frozen=True)
class ActiveWorkspaceResolution:
    layout: WorkspaceLayout
    requested_workspace_id: str
    fallback_used: bool = False
    issues: tuple[WorkspaceValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.to_dict(),
            "requestedWorkspaceId": self.requested_workspace_id,
            "fallbackUsed": self.fallback_used,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_id(value: str) -> str:
    cleaned = _SAFE_ID.sub("-", str(value or "").strip()).strip(".-")
    return cleaned or "workspace"


class WorkspaceStore:
    """JSON persistence for immutable shipped defaults plus migratable user workspaces."""

    def __init__(
        self,
        root: str | Path,
        registry: WorkspaceRegistry,
        shipped_defaults: Mapping[str, WorkspaceLayout],
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.registry = registry
        self.shipped_defaults = dict(shipped_defaults)
        self.user_dir = self.root / "user"
        self.state_path = self.root / "active.json"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._validate_shipped_defaults()

    @staticmethod
    def _read(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            handle.write(text)
            handle.flush()
            temp_path = Path(handle.name)
        temp_path.replace(path)

    def _validate_shipped_defaults(self) -> None:
        for workspace_id, layout in self.shipped_defaults.items():
            result = validate_workspace_layout(layout.to_dict(), self.registry)
            if not result.valid:
                messages = "; ".join(issue.message for issue in result.issues)
                raise WorkspaceStoreError(f"Invalid shipped workspace '{workspace_id}': {messages}")
            policy = self.registry.page(layout.page_id)
            if policy and policy.default_workspace_id == workspace_id:
                continue

    def _user_path(self, workspace_id: str) -> Path:
        return self.user_dir / f"{_safe_id(workspace_id)}.json"

    def _active_state(self) -> dict[str, str]:
        raw = self._read(self.state_path, {})
        return {str(key): str(value) for key, value in raw.items()} if isinstance(raw, dict) else {}

    def list(self, page_id: str) -> list[dict[str, Any]]:
        page_key = page_id.casefold()
        output: list[dict[str, Any]] = []
        for layout in self.shipped_defaults.values():
            if layout.page_id.casefold() == page_key:
                output.append({**layout.to_dict(), "distribution": "shipped", "readOnly": True})
        for path in sorted(self.user_dir.glob("*.json")):
            raw = self._read(path, None)
            if not isinstance(raw, Mapping):
                continue
            result = validate_workspace_layout(raw, self.registry)
            if result.layout and result.layout.page_id.casefold() == page_key:
                output.append({
                    **result.layout.to_dict(),
                    "distribution": "user",
                    "readOnly": False,
                    "valid": result.valid,
                    "issues": [issue.to_dict() for issue in result.issues],
                })
        return output

    def get(self, workspace_id: str) -> WorkspaceLayout | None:
        shipped = self.shipped_defaults.get(workspace_id)
        if shipped is not None:
            return shipped
        raw = self._read(self._user_path(workspace_id), None)
        if not isinstance(raw, Mapping):
            return None
        result = validate_workspace_layout(raw, self.registry)
        return result.layout if result.valid else None

    def validate_saved(self, workspace_id: str):
        if workspace_id in self.shipped_defaults:
            return validate_workspace_layout(self.shipped_defaults[workspace_id].to_dict(), self.registry)
        raw = self._read(self._user_path(workspace_id), None)
        if not isinstance(raw, Mapping):
            return None
        return validate_workspace_layout(raw, self.registry)

    def save(self, raw: Mapping[str, Any]) -> WorkspaceLayout:
        payload = dict(raw)
        workspace_id = str(payload.get("workspaceId") or "").strip()
        if not workspace_id:
            workspace_id = f"user.{uuid.uuid4().hex}"
            payload["workspaceId"] = workspace_id
        if workspace_id in self.shipped_defaults:
            raise WorkspaceStoreError("Shipped workspaces are immutable; duplicate the preset before editing it.")
        previous = self.get(workspace_id)
        now = _utc_now()
        payload.setdefault("createdAt", previous.created_at if previous else now)
        payload["updatedAt"] = now
        result = validate_workspace_layout(payload, self.registry)
        if not result.valid or result.layout is None:
            raise WorkspaceStoreError("; ".join(issue.message for issue in result.issues))
        self._write(self._user_path(result.layout.workspace_id), result.layout.to_dict())
        return result.layout

    def delete(self, workspace_id: str) -> bool:
        if workspace_id in self.shipped_defaults:
            raise WorkspaceStoreError("Shipped workspaces cannot be deleted.")
        path = self._user_path(workspace_id)
        if not path.exists():
            return False
        path.unlink()
        state = self._active_state()
        changed = False
        for page_id, active_id in list(state.items()):
            if active_id.casefold() == workspace_id.casefold():
                del state[page_id]
                changed = True
        if changed:
            self._write(self.state_path, state)
        return True

    def rename(self, workspace_id: str, name: str) -> WorkspaceLayout:
        layout = self.get(workspace_id)
        if layout is None:
            raise WorkspaceStoreError(f"Workspace '{workspace_id}' was not found or is invalid.")
        if workspace_id in self.shipped_defaults:
            raise WorkspaceStoreError("Shipped workspaces cannot be renamed.")
        payload = layout.to_dict()
        payload["name"] = str(name or "").strip()
        return self.save(payload)

    def duplicate(self, workspace_id: str, name: str) -> WorkspaceLayout:
        source = self.get(workspace_id)
        if source is None:
            raise WorkspaceStoreError(f"Workspace '{workspace_id}' was not found or is invalid.")
        payload = source.to_dict()
        payload["workspaceId"] = f"user.{uuid.uuid4().hex}"
        payload["name"] = str(name or "").strip() or f"{source.name} Copy"
        payload.pop("createdAt", None)
        payload.pop("updatedAt", None)
        return self.save(payload)

    def set_active(self, page_id: str, workspace_id: str) -> ActiveWorkspaceResolution:
        layout = self.get(workspace_id)
        if layout is None:
            validation = self.validate_saved(workspace_id)
            if validation is not None:
                raise WorkspaceStoreError("; ".join(issue.message for issue in validation.issues))
            raise WorkspaceStoreError(f"Workspace '{workspace_id}' was not found.")
        if layout.page_id.casefold() != page_id.casefold():
            raise WorkspaceStoreError(f"Workspace '{workspace_id}' belongs to page '{layout.page_id}', not '{page_id}'.")
        state = self._active_state()
        state[page_id] = layout.workspace_id
        self._write(self.state_path, state)
        return ActiveWorkspaceResolution(layout, layout.workspace_id)

    def get_active(self, page_id: str) -> ActiveWorkspaceResolution:
        policy = self.registry.page(page_id)
        if policy is None:
            raise WorkspaceStoreError(f"Unknown workspace page '{page_id}'.")
        state = self._active_state()
        requested = state.get(page_id, policy.default_workspace_id)
        if requested in self.shipped_defaults:
            return ActiveWorkspaceResolution(self.shipped_defaults[requested], requested)
        raw = self._read(self._user_path(requested), None)
        if isinstance(raw, Mapping):
            result = validate_workspace_layout(raw, self.registry)
            if result.valid and result.layout and result.layout.page_id.casefold() == page_id.casefold():
                return ActiveWorkspaceResolution(result.layout, requested)
            default = self.shipped_defaults[policy.default_workspace_id]
            return ActiveWorkspaceResolution(default, requested, True, result.issues)
        default = self.shipped_defaults[policy.default_workspace_id]
        return ActiveWorkspaceResolution(default, requested, requested != policy.default_workspace_id)

    def reset_active(self, page_id: str) -> ActiveWorkspaceResolution:
        policy = self.registry.page(page_id)
        if policy is None:
            raise WorkspaceStoreError(f"Unknown workspace page '{page_id}'.")
        state = self._active_state()
        state.pop(page_id, None)
        self._write(self.state_path, state)
        default = self.shipped_defaults[policy.default_workspace_id]
        return ActiveWorkspaceResolution(default, policy.default_workspace_id)
