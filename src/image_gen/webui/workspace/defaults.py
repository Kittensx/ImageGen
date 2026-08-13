from __future__ import annotations

from .models import (
    PageWorkspacePolicy,
    WorkspaceComponentDescriptor,
    WorkspaceComponentPlacement,
    WorkspaceDistribution,
    WorkspaceLayout,
    WorkspacePosition,
)
from .registry import WorkspaceRegistry

HOME_DEFAULT_WORKSPACE_ID = "imagegen.home.default"


def _empty_settings_schema() -> dict[str, object]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def build_default_workspace_registry() -> WorkspaceRegistry:
    registry = WorkspaceRegistry()
    components = (
        ("home.welcome", "Welcome", "navigation", "home", "feature", 12, {"compact": "standard", "narrow": "standard"}, 520, ()),
        ("home.readiness", "Readiness", "status", "info", "standard", 4, {}, 300, ()),
        ("home.quick-launch", "Quick launch", "navigation", "generate", "standard", 5, {}, 320, ()),
        ("home.profile", "Profile", "profile", "home-bug-contribution", "feature", 4, {"compact": "standard", "narrow": "standard"}, 320, ()),
        ("home.discord", "Discord", "community", "discord", "feature", 4, {"compact": "standard", "narrow": "standard"}, 320, ()),
        ("home.developer-updates", "Changelog", "updates", "external-link", "horizontal", 12, {"compact": "standard", "narrow": "standard"}, 420, ("content.markdown",)),
        ("home.help-center", "Help Center", "help", "info", "horizontal", 12, {"compact": "standard", "narrow": "standard"}, 420, ("content.markdown", "content.media")),
    )
    for component_id, title, category, icon, variant, span, responsive, min_width, capabilities in components:
        registry.register_component(
            WorkspaceComponentDescriptor(
                component_id=component_id,
                title=title,
                category=category,
                distribution=WorkspaceDistribution.BUNDLED,
                package_id="image_gen.core.home",
                allowed_pages=("home",),
                default_variant=variant,
                supported_variants=("standard", "feature", "horizontal"),
                default_grid_span=span,
                min_grid_span=2,
                max_grid_span=12,
                settings_schema=_empty_settings_schema(),
                icon=icon,
                responsive_variants=responsive,
                min_useful_width=min_width,
                required_capabilities=capabilities,
            )
        )
    registry.register_page(
        PageWorkspacePolicy(
            page_id="home",
            layout_schema_version=1,
            allowed_components=tuple(item[0] for item in components),
            required_components=("home.welcome", "home.readiness"),
            default_workspace_id=HOME_DEFAULT_WORKSPACE_ID,
            maximum_instances={item[0]: 1 for item in components},
        )
    )
    return registry


def build_default_workspace_layouts() -> dict[str, WorkspaceLayout]:
    specs = (
        ("home.welcome", 1, 1, 12, "feature", True),
        ("home.readiness", 1, 2, 4, "standard", True),
        ("home.profile", 5, 2, 4, "feature", True),
        ("home.discord", 9, 2, 4, "feature", True),
        ("home.quick-launch", 1, 3, 5, "standard", False),
        ("home.developer-updates", 1, 4, 12, "horizontal", True),
        ("home.help-center", 1, 5, 12, "horizontal", True),
    )
    return {
        HOME_DEFAULT_WORKSPACE_ID: WorkspaceLayout(
            workspace_id=HOME_DEFAULT_WORKSPACE_ID,
            name="ImageGen Default",
            page_id="home",
            source_preset=HOME_DEFAULT_WORKSPACE_ID,
            components=tuple(
                WorkspaceComponentPlacement(
                    component_id=component_id,
                    position=WorkspacePosition(column, row, span, 1),
                    variant=variant,
                    visible=visible,
                )
                for component_id, column, row, span, variant, visible in specs
            ),
        )
    }
