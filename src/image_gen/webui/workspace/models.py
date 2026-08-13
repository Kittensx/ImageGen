from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

WORKSPACE_SCHEMA = "imagegen.workspace.layout"
WORKSPACE_SCHEMA_VERSION = 1
WORKSPACE_CONTRACT_VERSION = 1

CANONICAL_WORKSPACE_VARIANTS = frozenset(
    {"compact", "standard", "horizontal", "vertical", "feature", "full-width"}
)
WORKSPACE_VARIANT_ALIASES = {"featured": "feature"}


def canonical_workspace_variant(value: object, fallback: str = "standard") -> str:
    token = str(value or fallback).strip().lower()
    return WORKSPACE_VARIANT_ALIASES.get(token, token or fallback)


class WorkspaceDistribution(str, Enum):
    BUNDLED = "bundled"
    EXTERNAL = "external"


class WorkspaceIssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class WorkspaceInstallReference:
    registry: str
    package_ref: str

    def to_dict(self) -> dict[str, str]:
        return {"registry": self.registry, "packageRef": self.package_ref}


@dataclass(frozen=True)
class WorkspaceComponentDescriptor:
    component_id: str
    title: str
    category: str
    distribution: WorkspaceDistribution
    package_id: str
    allowed_pages: tuple[str, ...]
    default_variant: str = "standard"
    supported_variants: tuple[str, ...] = ("standard",)
    default_grid_span: int = 12
    min_grid_span: int = 1
    max_grid_span: int = 12
    settings_schema: Mapping[str, Any] = field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()
    icon: str = ""
    description: str = ""
    min_height: int | None = None
    max_height: int | None = None
    install: WorkspaceInstallReference | None = None
    links: Mapping[str, str] = field(default_factory=dict)
    responsive_variants: Mapping[str, str] = field(default_factory=dict)
    min_useful_width: int | None = None
    min_useful_height: int | None = None

    @property
    def portable(self) -> bool:
        return self.distribution is WorkspaceDistribution.BUNDLED or self.install is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "componentId": self.component_id,
            "title": self.title,
            "category": self.category,
            "distribution": self.distribution.value,
            "packageId": self.package_id,
            "allowedPages": list(self.allowed_pages),
            "defaultVariant": self.default_variant,
            "supportedVariants": list(self.supported_variants),
            "defaultGridSpan": self.default_grid_span,
            "minGridSpan": self.min_grid_span,
            "maxGridSpan": self.max_grid_span,
            "settingsSchema": dict(self.settings_schema),
            "requiredCapabilities": list(self.required_capabilities),
            "icon": self.icon,
            "description": self.description,
            "minHeight": self.min_height,
            "maxHeight": self.max_height,
            "install": self.install.to_dict() if self.install else None,
            "links": dict(self.links),
            "responsive": dict(self.responsive_variants),
            "minUsefulWidth": self.min_useful_width,
            "minUsefulHeight": self.min_useful_height,
            "portable": self.portable,
        }


@dataclass(frozen=True)
class PageWorkspacePolicy:
    page_id: str
    layout_schema_version: int
    allowed_components: tuple[str, ...]
    required_components: tuple[str, ...]
    default_workspace_id: str
    maximum_instances: Mapping[str, int] = field(default_factory=dict)
    mutually_exclusive_components: tuple[tuple[str, ...], ...] = ()
    locked_regions: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pageId": self.page_id,
            "layoutSchemaVersion": self.layout_schema_version,
            "allowedComponents": list(self.allowed_components),
            "requiredComponents": list(self.required_components),
            "defaultWorkspaceId": self.default_workspace_id,
            "maximumInstances": dict(self.maximum_instances),
            "mutuallyExclusiveComponents": [list(group) for group in self.mutually_exclusive_components],
            "lockedRegions": list(self.locked_regions),
            "requiredCapabilities": list(self.required_capabilities),
        }


@dataclass(frozen=True)
class WorkspacePosition:
    column: int
    row: int
    column_span: int
    height_units: int

    def to_dict(self) -> dict[str, int]:
        return {
            "column": self.column,
            "row": self.row,
            "columnSpan": self.column_span,
            "heightUnits": self.height_units,
        }


@dataclass(frozen=True)
class WorkspaceComponentPlacement:
    component_id: str
    position: WorkspacePosition
    variant: str = "standard"
    settings: Mapping[str, Any] = field(default_factory=dict)
    instance_id: str = ""
    responsive: Mapping[str, Any] = field(default_factory=dict)
    visible: bool = True
    shell_state: str = "expanded"

    @property
    def identity(self) -> str:
        return self.instance_id or self.component_id

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "componentId": self.component_id,
            "position": self.position.to_dict(),
            "variant": self.variant,
            "settings": dict(self.settings),
            "responsive": dict(self.responsive),
            "visible": self.visible,
            "shellState": self.shell_state,
        }
        if self.instance_id:
            payload["instanceId"] = self.instance_id
        return payload


@dataclass(frozen=True)
class WorkspaceLayout:
    workspace_id: str
    name: str
    page_id: str
    components: tuple[WorkspaceComponentPlacement, ...]
    schema: str = WORKSPACE_SCHEMA
    schema_version: int = WORKSPACE_SCHEMA_VERSION
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    source_preset: str = ""
    application_version_created_with: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "schemaVersion": self.schema_version,
            "workspaceId": self.workspace_id,
            "name": self.name,
            "pageId": self.page_id,
            "components": [placement.to_dict() for placement in self.components],
        }
        optional = {
            "description": self.description,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "sourcePreset": self.source_preset,
            "applicationVersionCreatedWith": self.application_version_created_with,
        }
        payload.update({key: value for key, value in optional.items() if value})
        return payload


@dataclass(frozen=True)
class WorkspaceValidationIssue:
    path: str
    message: str
    severity: WorkspaceIssueSeverity = WorkspaceIssueSeverity.ERROR
    code: str = "invalid"
    component_id: str = ""
    instance_id: str = ""
    package_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
        }
        if self.component_id:
            payload["componentId"] = self.component_id
        if self.instance_id:
            payload["instanceId"] = self.instance_id
        if self.package_ref:
            payload["packageRef"] = self.package_ref
        return payload


@dataclass(frozen=True)
class WorkspaceValidationResult:
    layout: WorkspaceLayout | None
    issues: tuple[WorkspaceValidationIssue, ...] = ()
    migrated_from_version: int | None = None

    @property
    def valid(self) -> bool:
        return self.layout is not None and not any(
            issue.severity is WorkspaceIssueSeverity.ERROR for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "layout": self.layout.to_dict() if self.layout else None,
            "issues": [issue.to_dict() for issue in self.issues],
            "migratedFromVersion": self.migrated_from_version,
        }
