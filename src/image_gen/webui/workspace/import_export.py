from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .models import WorkspaceIssueSeverity, WorkspaceValidationIssue, WorkspaceValidationResult
from .registry import WorkspaceRegistry
from .validation import validate_workspace_layout


@dataclass(frozen=True)
class WorkspaceImportReview:
    validation: WorkspaceValidationResult
    missing_packages: tuple[str, ...] = ()

    @property
    def activatable(self) -> bool:
        return self.validation.valid and not self.missing_packages

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.validation.to_dict(),
            "activatable": self.activatable,
            "missingPackages": list(self.missing_packages),
        }


def review_workspace_import(raw: Mapping[str, Any], registry: WorkspaceRegistry) -> WorkspaceImportReview:
    result = validate_workspace_layout(raw, registry)
    if result.layout is None:
        return WorkspaceImportReview(result)
    missing: list[str] = []
    issues = list(result.issues)
    for placement in result.layout.components:
        descriptor = registry.component(placement.component_id)
        if descriptor is None or descriptor.distribution.value != "external":
            continue
        if descriptor.install is None:
            continue
        package_ref = descriptor.install.package_ref
        if package_ref not in missing:
            missing.append(package_ref)
            issues.append(
                WorkspaceValidationIssue(
                    path="components",
                    message=f"External component package '{package_ref}' must be resolved before activation.",
                    severity=WorkspaceIssueSeverity.WARNING,
                    code="package.resolve",
                    component_id=placement.component_id,
                    package_ref=package_ref,
                )
            )
    if tuple(issues) != result.issues:
        result = WorkspaceValidationResult(result.layout, tuple(issues), result.migrated_from_version)
    return WorkspaceImportReview(result, tuple(missing))


def export_workspace(layout) -> dict[str, Any]:
    return layout.to_dict()
