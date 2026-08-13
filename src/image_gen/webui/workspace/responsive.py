from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TYPE_CHECKING

from .models import WorkspaceComponentDescriptor, WorkspaceLayout

if TYPE_CHECKING:
    from .registry import WorkspaceRegistry


class WorkspaceWidthClass(str, Enum):
    WIDE = "wide"
    STANDARD = "standard"
    COMPACT = "compact"
    NARROW = "narrow"


WORKSPACE_WIDTH_CLASSES = tuple(item.value for item in WorkspaceWidthClass)
WORKSPACE_WIDTH_THRESHOLDS = {
    "narrowMax": 719,
    "compactMax": 1049,
    "standardMax": 1450,
}
WORKSPACE_REPRESENTATIVE_WIDTHS = {
    WorkspaceWidthClass.WIDE.value: 1600,
    WorkspaceWidthClass.STANDARD.value: 1280,
    WorkspaceWidthClass.COMPACT.value: 960,
    WorkspaceWidthClass.NARROW.value: 680,
}


def classify_workspace_width(width: float | int) -> WorkspaceWidthClass:
    value = max(0.0, float(width))
    if value > WORKSPACE_WIDTH_THRESHOLDS["standardMax"]:
        return WorkspaceWidthClass.WIDE
    if value > WORKSPACE_WIDTH_THRESHOLDS["compactMax"]:
        return WorkspaceWidthClass.STANDARD
    if value > WORKSPACE_WIDTH_THRESHOLDS["narrowMax"]:
        return WorkspaceWidthClass.COMPACT
    return WorkspaceWidthClass.NARROW


def responsive_grid_span(
    base_span: int,
    width_class: WorkspaceWidthClass | str,
    *,
    minimum: int = 1,
    maximum: int = 12,
) -> int:
    width_value = WorkspaceWidthClass(str(getattr(width_class, "value", width_class)))
    base = max(minimum, min(maximum, int(base_span)))
    if width_value is WorkspaceWidthClass.WIDE:
        requested = base
    elif width_value is WorkspaceWidthClass.STANDARD:
        requested = 12 if base >= 8 else max(6, base)
    elif width_value is WorkspaceWidthClass.COMPACT:
        requested = 12 if base >= 7 else max(6, base)
    else:
        requested = 12
    return max(minimum, min(maximum, requested))


def responsive_presentation_span(
    effective_span: int,
    width_class: WorkspaceWidthClass | str,
    *,
    shell_state: str = "expanded",
    minimum: int = 1,
    maximum: int = 12,
) -> int:
    width_value = WorkspaceWidthClass(str(getattr(width_class, "value", width_class)))
    base = max(minimum, min(maximum, int(effective_span)))
    if str(shell_state or "expanded").strip().lower() != "side":
        return base
    target = {
        WorkspaceWidthClass.WIDE: 2,
        WorkspaceWidthClass.STANDARD: 3,
        WorkspaceWidthClass.COMPACT: 6,
        WorkspaceWidthClass.NARROW: 12,
    }[width_value]
    return max(minimum, min(maximum, target))


def resolve_responsive_variant(
    descriptor: WorkspaceComponentDescriptor,
    width_class: WorkspaceWidthClass | str,
    *,
    preferred_variant: str = "",
    placement_overrides: Mapping[str, object] | None = None,
) -> tuple[str, str]:
    width_value = WorkspaceWidthClass(str(getattr(width_class, "value", width_class))).value
    preferred = str(preferred_variant or descriptor.default_variant or "standard")
    overrides = placement_overrides or {}
    desired = ""
    if width_value != WorkspaceWidthClass.WIDE.value:
        desired = str(overrides.get(width_value) or descriptor.responsive_variants.get(width_value) or "").strip()
    if not desired:
        desired = preferred

    supported = tuple(descriptor.supported_variants)
    if desired in supported:
        return desired, "preferred"
    if "standard" in supported:
        return "standard", "standard_fallback"
    if preferred in supported:
        return preferred, "preferred_fallback"
    if descriptor.default_variant in supported:
        return descriptor.default_variant, "default_fallback"
    return supported[0], "first_supported_fallback"


@dataclass(frozen=True)
class WorkspaceResponsivePlacement:
    component_id: str
    instance_id: str
    width_class: str
    base_span: int
    effective_span: int
    base_variant: str
    effective_variant: str
    variant_resolution: str

    def to_dict(self) -> dict[str, object]:
        return {
            "componentId": self.component_id,
            "instanceId": self.instance_id,
            "widthClass": self.width_class,
            "baseSpan": self.base_span,
            "effectiveSpan": self.effective_span,
            "baseVariant": self.base_variant,
            "effectiveVariant": self.effective_variant,
            "variantResolution": self.variant_resolution,
        }


def build_responsive_layout_plan(
    layout: WorkspaceLayout,
    registry: "WorkspaceRegistry",
    width: float | int,
) -> tuple[WorkspaceResponsivePlacement, ...]:
    width_class = classify_workspace_width(width)
    planned: list[WorkspaceResponsivePlacement] = []
    for placement in layout.components:
        descriptor = registry.component(placement.component_id)
        if descriptor is None:
            continue
        variant, resolution = resolve_responsive_variant(
            descriptor,
            width_class,
            preferred_variant=placement.variant,
            placement_overrides=placement.responsive,
        )
        responsive_span = responsive_grid_span(
            placement.position.column_span,
            width_class,
            minimum=descriptor.min_grid_span,
            maximum=descriptor.max_grid_span,
        )
        planned.append(
            WorkspaceResponsivePlacement(
                component_id=placement.component_id,
                instance_id=placement.instance_id,
                width_class=width_class.value,
                base_span=placement.position.column_span,
                effective_span=responsive_presentation_span(
                    responsive_span,
                    width_class,
                    shell_state=placement.shell_state,
                    minimum=descriptor.min_grid_span,
                    maximum=descriptor.max_grid_span,
                ),
                base_variant=placement.variant,
                effective_variant=variant,
                variant_resolution=resolution,
            )
        )
    return tuple(planned)


def workspace_responsive_contract_payload() -> dict[str, object]:
    return {
        "widthClasses": list(WORKSPACE_WIDTH_CLASSES),
        "thresholds": dict(WORKSPACE_WIDTH_THRESHOLDS),
        "representativeWidths": dict(WORKSPACE_REPRESENTATIVE_WIDTHS),
        "measurement": "workspace-container",
        "gridColumns": 12,
        "snapGridPixels": 8,
    }
