from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .models import WorkspaceComponentPlacement, WorkspaceLayout
from .registry import WorkspaceRegistry
from .validation import validate_workspace_layout


@dataclass(frozen=True)
class WorkspaceRenderItem:
    placement: WorkspaceComponentPlacement
    status: str
    output: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement": self.placement.to_dict(),
            "status": self.status,
            "output": self.output,
            "error": self.error,
        }


@dataclass(frozen=True)
class WorkspaceRenderPlan:
    page_id: str
    workspace_id: str
    items: tuple[WorkspaceRenderItem, ...]
    page_blocked: bool = False
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pageId": self.page_id,
            "workspaceId": self.workspace_id,
            "pageBlocked": self.page_blocked,
            "items": [item.to_dict() for item in self.items],
            "diagnostics": list(self.diagnostics),
        }


class WorkspaceRenderer:
    """Framework-neutral renderer foundation that isolates component-level failures."""

    def __init__(self, registry: WorkspaceRegistry) -> None:
        self.registry = registry

    def render(
        self,
        layout: WorkspaceLayout,
        *,
        component_factory: Callable[[WorkspaceComponentPlacement], Any] | None = None,
    ) -> WorkspaceRenderPlan:
        validation = validate_workspace_layout(layout.to_dict(), self.registry)
        if not validation.valid or validation.layout is None:
            messages = tuple(issue.message for issue in validation.issues)
            return WorkspaceRenderPlan(layout.page_id, layout.workspace_id, (), True, messages)
        policy = self.registry.page(layout.page_id)
        required = {value.casefold() for value in policy.required_components} if policy else set()
        items: list[WorkspaceRenderItem] = []
        diagnostics: list[str] = []
        page_blocked = False
        factory = component_factory or (lambda placement: placement.to_dict())
        for placement in validation.layout.components:
            if not placement.visible:
                items.append(WorkspaceRenderItem(placement, "hidden"))
                continue
            try:
                output = factory(placement)
                items.append(WorkspaceRenderItem(placement, "mounted", output=output))
            except Exception as exc:  # component isolation is intentionally broad here
                message = f"Unable to render '{placement.component_id}': {exc}"
                diagnostics.append(message)
                status = "required_error" if placement.component_id.casefold() in required else "optional_error"
                page_blocked = page_blocked or status == "required_error"
                items.append(WorkspaceRenderItem(placement, status, error=message))
        return WorkspaceRenderPlan(
            page_id=validation.layout.page_id,
            workspace_id=validation.layout.workspace_id,
            items=tuple(items),
            page_blocked=page_blocked,
            diagnostics=tuple(diagnostics),
        )
