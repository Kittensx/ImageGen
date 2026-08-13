from __future__ import annotations

from typing import Iterable

from .models import PageWorkspacePolicy, WorkspaceComponentDescriptor, WorkspaceDistribution
from .responsive import WORKSPACE_WIDTH_CLASSES


class WorkspaceRegistryError(ValueError):
    pass


def _key(value: str) -> str:
    return str(value or "").strip().casefold()


class WorkspaceRegistry:
    """Canonical stable-identity registry for workspace components and page policies."""

    def __init__(self) -> None:
        self._components: dict[str, WorkspaceComponentDescriptor] = {}
        self._pages: dict[str, PageWorkspacePolicy] = {}

    def register_component(self, descriptor: WorkspaceComponentDescriptor) -> WorkspaceComponentDescriptor:
        component_id = str(descriptor.component_id or "").strip()
        if not component_id:
            raise WorkspaceRegistryError("componentId is required.")
        key = _key(component_id)
        if key in self._components:
            raise WorkspaceRegistryError(f"Workspace component '{component_id}' is already registered.")
        if not descriptor.package_id.strip():
            raise WorkspaceRegistryError(f"Workspace component '{component_id}' requires packageId.")
        if descriptor.distribution is WorkspaceDistribution.EXTERNAL and descriptor.install is None:
            raise WorkspaceRegistryError(
                f"External workspace component '{component_id}' requires canonical install metadata."
            )
        if descriptor.install is not None and (
            not descriptor.install.registry.strip() or not descriptor.install.package_ref.strip()
        ):
            raise WorkspaceRegistryError(
                f"Workspace component '{component_id}' has incomplete install metadata."
            )
        variants = tuple(dict.fromkeys(str(value).strip() for value in descriptor.supported_variants if str(value).strip()))
        if not variants:
            raise WorkspaceRegistryError(f"Workspace component '{component_id}' requires supportedVariants.")
        if descriptor.default_variant not in variants:
            raise WorkspaceRegistryError(
                f"Workspace component '{component_id}' defaultVariant is not supported."
            )
        if not (1 <= descriptor.min_grid_span <= descriptor.default_grid_span <= descriptor.max_grid_span <= 12):
            raise WorkspaceRegistryError(
                f"Workspace component '{component_id}' grid spans must satisfy 1 <= min <= default <= max <= 12."
            )
        for width_class, variant in descriptor.responsive_variants.items():
            if str(width_class).strip().lower() not in WORKSPACE_WIDTH_CLASSES:
                raise WorkspaceRegistryError(
                    f"Workspace component '{component_id}' has unknown responsive width class '{width_class}'."
                )
            if str(variant).strip() not in variants:
                raise WorkspaceRegistryError(
                    f"Workspace component '{component_id}' responsive variant '{variant}' is not supported."
                )
        for label, value in (("minUsefulWidth", descriptor.min_useful_width), ("minUsefulHeight", descriptor.min_useful_height)):
            if value is not None and int(value) < 1:
                raise WorkspaceRegistryError(
                    f"Workspace component '{component_id}' {label} must be at least 1 when provided."
                )
        self._components[key] = descriptor
        return descriptor

    def register_page(self, policy: PageWorkspacePolicy) -> PageWorkspacePolicy:
        page_id = str(policy.page_id or "").strip()
        if not page_id:
            raise WorkspaceRegistryError("pageId is required.")
        key = _key(page_id)
        if key in self._pages:
            raise WorkspaceRegistryError(f"Workspace page '{page_id}' is already registered.")
        allowed = {_key(value) for value in policy.allowed_components}
        required = {_key(value) for value in policy.required_components}
        unknown = [
            value
            for value in (*policy.allowed_components, *policy.required_components)
            if _key(value) not in self._components
        ]
        if unknown:
            raise WorkspaceRegistryError(
                f"Workspace page '{page_id}' references unregistered component(s): {', '.join(unknown)}."
            )
        if not required.issubset(allowed):
            raise WorkspaceRegistryError(
                f"Workspace page '{page_id}' requiredComponents must be included in allowedComponents."
            )
        for component_id in policy.allowed_components:
            descriptor = self.component(component_id)
            if descriptor and descriptor.allowed_pages and _key(page_id) not in {
                _key(item) for item in descriptor.allowed_pages
            }:
                raise WorkspaceRegistryError(
                    f"Workspace component '{component_id}' does not allow page '{page_id}'."
                )
        for component_id, maximum in policy.maximum_instances.items():
            if _key(component_id) not in allowed:
                raise WorkspaceRegistryError(
                    f"maximumInstances references component '{component_id}' outside the page allowlist."
                )
            if int(maximum) < 1:
                raise WorkspaceRegistryError("maximumInstances values must be at least 1.")
        self._pages[key] = policy
        return policy

    def component(self, component_id: str) -> WorkspaceComponentDescriptor | None:
        return self._components.get(_key(component_id))

    def page(self, page_id: str) -> PageWorkspacePolicy | None:
        return self._pages.get(_key(page_id))

    def components(self) -> tuple[WorkspaceComponentDescriptor, ...]:
        return tuple(self._components.values())

    def pages(self) -> tuple[PageWorkspacePolicy, ...]:
        return tuple(self._pages.values())

    def compatible_components(self, page_id: str) -> tuple[WorkspaceComponentDescriptor, ...]:
        policy = self.page(page_id)
        if policy is None:
            return ()
        allowed = {_key(value) for value in policy.allowed_components}
        return tuple(descriptor for key, descriptor in self._components.items() if key in allowed)

    def definitions_payload(self) -> dict[str, object]:
        return {
            "components": [descriptor.to_dict() for descriptor in self.components()],
            "pages": [policy.to_dict() for policy in self.pages()],
        }

    def extend_components(self, descriptors: Iterable[WorkspaceComponentDescriptor]) -> None:
        for descriptor in descriptors:
            self.register_component(descriptor)
