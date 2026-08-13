from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .migrations import WorkspaceMigrationError, WorkspaceMigrationRegistry, default_workspace_migrations
from .models import (
    WORKSPACE_SCHEMA,
    WORKSPACE_SCHEMA_VERSION,
    canonical_workspace_variant,
    WorkspaceComponentPlacement,
    WorkspaceIssueSeverity,
    WorkspaceLayout,
    WorkspacePosition,
    WorkspaceValidationIssue,
    WorkspaceValidationResult,
)
from .registry import WorkspaceRegistry
from .responsive import WORKSPACE_WIDTH_CLASSES


class WorkspaceValidationError(ValueError):
    pass


def _issue(path: str, message: str, *, code: str = "invalid", **kwargs: str) -> WorkspaceValidationIssue:
    return WorkspaceValidationIssue(path=path, message=message, code=code, **kwargs)


def _validate_scalar(value: Any, schema: Mapping[str, Any], path: str) -> list[WorkspaceValidationIssue]:
    issues: list[WorkspaceValidationIssue] = []
    expected = str(schema.get("type") or "").strip()
    if expected == "boolean" and not isinstance(value, bool):
        issues.append(_issue(path, "Expected a boolean setting.", code="settings.type"))
    elif expected == "string" and not isinstance(value, str):
        issues.append(_issue(path, "Expected a string setting.", code="settings.type"))
    elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        issues.append(_issue(path, "Expected an integer setting.", code="settings.type"))
    elif expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        issues.append(_issue(path, "Expected a numeric setting.", code="settings.type"))
    elif expected == "array" and not isinstance(value, list):
        issues.append(_issue(path, "Expected an array setting.", code="settings.type"))
    if issues:
        return issues
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(_issue(path, f"Setting must be at least {schema['minimum']}.", code="settings.minimum"))
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(_issue(path, f"Setting must be at most {schema['maximum']}.", code="settings.maximum"))
    if "enum" in schema and value not in schema.get("enum", []):
        issues.append(_issue(path, "Setting is not one of the supported values.", code="settings.enum"))
    return issues


def validate_component_settings(
    settings: Mapping[str, Any], schema: Mapping[str, Any], path: str
) -> list[WorkspaceValidationIssue]:
    if not schema:
        return [] if not settings else [_issue(path, "This component does not accept settings.", code="settings.unsupported")]
    if str(schema.get("type") or "object") != "object":
        return [_issue(path, "Component settingsSchema must describe an object.", code="settings.schema")]
    properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
    required = {str(value) for value in schema.get("required", []) if str(value)}
    issues: list[WorkspaceValidationIssue] = []
    missing = sorted(required - set(settings))
    for key in missing:
        issues.append(_issue(f"{path}.{key}", "Required setting is missing.", code="settings.required"))
    additional = schema.get("additionalProperties", True)
    for key, value in settings.items():
        field_path = f"{path}.{key}"
        field_schema = properties.get(key)
        if field_schema is None:
            if additional is False:
                issues.append(_issue(field_path, "Unsupported component setting.", code="settings.additional"))
            continue
        if isinstance(field_schema, Mapping):
            issues.extend(_validate_scalar(value, field_schema, field_path))
    return issues


def _parse_layout(payload: Mapping[str, Any]) -> tuple[WorkspaceLayout | None, list[WorkspaceValidationIssue]]:
    issues: list[WorkspaceValidationIssue] = []
    workspace_id = str(payload.get("workspaceId") or "").strip()
    name = str(payload.get("name") or "").strip()
    page_id = str(payload.get("pageId") or "").strip()
    if not workspace_id:
        issues.append(_issue("workspaceId", "workspaceId is required.", code="identity.workspace"))
    if not name:
        issues.append(_issue("name", "name is required.", code="identity.name"))
    if not page_id:
        issues.append(_issue("pageId", "pageId is required.", code="identity.page"))
    raw_components = payload.get("components")
    if not isinstance(raw_components, list):
        issues.append(_issue("components", "components must be an array.", code="components.type"))
        return None, issues
    components: list[WorkspaceComponentPlacement] = []
    for index, item in enumerate(raw_components):
        path = f"components[{index}]"
        if not isinstance(item, Mapping):
            issues.append(_issue(path, "Component placement must be an object.", code="placement.type"))
            continue
        component_id = str(item.get("componentId") or "").strip()
        instance_id = str(item.get("instanceId") or "").strip()
        variant = canonical_workspace_variant(item.get("variant"), "standard")
        position = item.get("position")
        if not component_id:
            issues.append(_issue(f"{path}.componentId", "componentId is required.", code="placement.component"))
            continue
        if not isinstance(position, Mapping):
            issues.append(_issue(f"{path}.position", "position must be an object.", code="geometry.position"))
            continue
        try:
            column = int(position.get("column"))
            row = int(position.get("row"))
            column_span = int(position.get("columnSpan"))
            height_units = int(position.get("heightUnits"))
        except (TypeError, ValueError):
            issues.append(_issue(f"{path}.position", "Geometry values must be integers.", code="geometry.type"))
            continue
        settings = item.get("settings") or {}
        responsive = item.get("responsive") or {}
        if not isinstance(settings, Mapping):
            issues.append(_issue(f"{path}.settings", "settings must be an object.", code="settings.type"))
            settings = {}
        if not isinstance(responsive, Mapping):
            issues.append(_issue(f"{path}.responsive", "responsive must be an object.", code="responsive.type"))
            responsive = {}
        components.append(
            WorkspaceComponentPlacement(
                component_id=component_id,
                instance_id=instance_id,
                position=WorkspacePosition(column, row, column_span, height_units),
                variant=variant,
                settings=dict(settings),
                responsive={str(key).strip().lower(): canonical_workspace_variant(value) if isinstance(value, str) else value for key, value in responsive.items()},
                visible=item.get("visible", True) is not False,
                shell_state=str(item.get("shellState") or "expanded"),
            )
        )
    if issues and (not workspace_id or not name or not page_id):
        return None, issues
    layout = WorkspaceLayout(
        schema=str(payload.get("schema") or WORKSPACE_SCHEMA),
        schema_version=int(payload.get("schemaVersion") or WORKSPACE_SCHEMA_VERSION),
        workspace_id=workspace_id,
        name=name,
        page_id=page_id,
        components=tuple(components),
        description=str(payload.get("description") or ""),
        created_at=str(payload.get("createdAt") or ""),
        updated_at=str(payload.get("updatedAt") or ""),
        source_preset=str(payload.get("sourcePreset") or ""),
        application_version_created_with=str(payload.get("applicationVersionCreatedWith") or ""),
    )
    return layout, issues


def validate_workspace_layout(
    raw: Mapping[str, Any],
    registry: WorkspaceRegistry,
    *,
    migrations: WorkspaceMigrationRegistry | None = None,
) -> WorkspaceValidationResult:
    migration_registry = migrations or default_workspace_migrations()
    try:
        migrated = migration_registry.migrate(raw)
    except WorkspaceMigrationError as exc:
        return WorkspaceValidationResult(None, (_issue("schemaVersion", str(exc), code="schema.compatibility"),))
    layout, issues = _parse_layout(migrated.payload)
    if layout is None:
        return WorkspaceValidationResult(None, tuple(issues), migrated.migrated_from_version)
    if layout.schema != WORKSPACE_SCHEMA:
        issues.append(_issue("schema", f"Unsupported workspace schema '{layout.schema}'.", code="schema.identity"))
    if layout.schema_version != WORKSPACE_SCHEMA_VERSION:
        issues.append(_issue("schemaVersion", "Workspace schema was not migrated to the current version.", code="schema.version"))
    policy = registry.page(layout.page_id)
    if policy is None:
        issues.append(_issue("pageId", f"Unknown workspace page '{layout.page_id}'.", code="page.unknown"))
        return WorkspaceValidationResult(layout, tuple(issues), migrated.migrated_from_version)
    allowed = {value.casefold() for value in policy.allowed_components}
    required = {value.casefold() for value in policy.required_components}
    visible_counts: Counter[str] = Counter()
    all_counts: Counter[str] = Counter()
    identities: set[str] = set()
    for index, placement in enumerate(layout.components):
        path = f"components[{index}]"
        component_key = placement.component_id.casefold()
        descriptor = registry.component(placement.component_id)
        identity_key = placement.identity.casefold()
        if identity_key in identities:
            issues.append(_issue(f"{path}.instanceId", f"Duplicate component instance identity '{placement.identity}'.", code="instance.duplicate", component_id=placement.component_id, instance_id=placement.instance_id))
        identities.add(identity_key)
        all_counts[component_key] += 1
        if placement.visible:
            visible_counts[component_key] += 1
        if descriptor is None:
            issues.append(_issue(f"{path}.componentId", f"Unknown workspace component '{placement.component_id}'.", code="component.unknown", component_id=placement.component_id, instance_id=placement.instance_id))
            continue
        if component_key not in allowed:
            issues.append(_issue(f"{path}.componentId", f"Component '{placement.component_id}' is not allowed on page '{layout.page_id}'.", code="page.disallowed_component", component_id=placement.component_id, instance_id=placement.instance_id))
        if descriptor.allowed_pages and layout.page_id.casefold() not in {value.casefold() for value in descriptor.allowed_pages}:
            issues.append(_issue(f"{path}.componentId", f"Component '{placement.component_id}' does not declare compatibility with page '{layout.page_id}'.", code="component.page_compatibility", component_id=placement.component_id, instance_id=placement.instance_id))
        pos = placement.position
        if pos.column < 1 or pos.column > 12:
            issues.append(_issue(f"{path}.position.column", "column must be between 1 and 12.", code="geometry.column", component_id=placement.component_id))
        if pos.row < 1:
            issues.append(_issue(f"{path}.position.row", "row must be at least 1.", code="geometry.row", component_id=placement.component_id))
        if pos.column_span < 1 or pos.column_span > 12 or pos.column + pos.column_span - 1 > 12:
            issues.append(_issue(f"{path}.position.columnSpan", "columnSpan must fit within the 12-column workspace grid.", code="geometry.span", component_id=placement.component_id))
        if pos.height_units < 1:
            issues.append(_issue(f"{path}.position.heightUnits", "heightUnits must be at least 1.", code="geometry.height", component_id=placement.component_id))
        if not (descriptor.min_grid_span <= pos.column_span <= descriptor.max_grid_span):
            issues.append(_issue(f"{path}.position.columnSpan", f"Component '{placement.component_id}' requires a span between {descriptor.min_grid_span} and {descriptor.max_grid_span}.", code="geometry.component_span", component_id=placement.component_id))
        if placement.variant not in descriptor.supported_variants:
            issues.append(_issue(f"{path}.variant", f"Variant '{placement.variant}' is not supported by '{placement.component_id}'.", code="variant.unsupported", component_id=placement.component_id, instance_id=placement.instance_id))
        for width_class, responsive_variant in placement.responsive.items():
            responsive_path = f"{path}.responsive.{width_class}"
            if str(width_class).strip().lower() not in WORKSPACE_WIDTH_CLASSES:
                issues.append(_issue(responsive_path, "Responsive overrides only support wide, standard, compact, and narrow width classes.", code="responsive.width_class", component_id=placement.component_id, instance_id=placement.instance_id))
                continue
            if not isinstance(responsive_variant, str):
                issues.append(_issue(responsive_path, "Responsive variant override must be a string variant name.", code="responsive.type", component_id=placement.component_id, instance_id=placement.instance_id))
                continue
            if responsive_variant not in descriptor.supported_variants:
                issues.append(_issue(responsive_path, f"Responsive variant '{responsive_variant}' is not supported by '{placement.component_id}'.", code="responsive.variant", component_id=placement.component_id, instance_id=placement.instance_id))
        issues.extend(validate_component_settings(placement.settings, descriptor.settings_schema, f"{path}.settings"))
    for component_id in policy.required_components:
        if visible_counts[component_id.casefold()] < 1:
            issues.append(_issue("components", f"Required component '{component_id}' must have at least one visible instance.", code="required.missing", component_id=component_id))
    for component_id, maximum in policy.maximum_instances.items():
        if all_counts[component_id.casefold()] > int(maximum):
            issues.append(_issue("components", f"Component '{component_id}' allows at most {maximum} instance(s).", code="instance.maximum", component_id=component_id))
    for group in policy.mutually_exclusive_components:
        present = [component_id for component_id in group if visible_counts[component_id.casefold()] > 0]
        if len(present) > 1:
            issues.append(_issue("components", f"Mutually exclusive components are simultaneously visible: {', '.join(present)}.", code="component.mutually_exclusive"))
    return WorkspaceValidationResult(layout, tuple(issues), migrated.migrated_from_version)
