from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from image_gen.contracts import SamplerCapabilities, require_adapter_conformance
from image_gen.systems.registry.descriptors import (
    PluginDescriptor,
    PluginKind,
    normalize_identity,
)
from image_gen.systems.registry.discovery import PluginDiscovery
from image_gen.systems.registry.errors import (
    DuplicatePluginIdentityError,
    PluginCompatibilityError,
    PluginDescriptorError,
    PluginInstantiationError,
)


@dataclass(frozen=True)
class PluginCompatibilityResult:
    sampler_id: str
    scheduler_id: str
    is_compatible: bool = True
    reasons: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    negotiated_pipeline_mode: str | None = None
    step_expansion_clamped: bool = False
    tail_metadata_clamped: bool = False

    def raise_if_incompatible(self) -> None:
        if not self.is_compatible:
            raise PluginCompatibilityError(
                f"Sampler {self.sampler_id!r} is incompatible with scheduler "
                f"{self.scheduler_id!r}: " + "; ".join(self.reasons)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler_id": self.sampler_id,
            "scheduler_id": self.scheduler_id,
            "is_compatible": self.is_compatible,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "negotiated_pipeline_mode": self.negotiated_pipeline_mode,
            "step_expansion_clamped": self.step_expansion_clamped,
            "tail_metadata_clamped": self.tail_metadata_clamped,
        }


class PluginRegistry:
    """Validated, in-memory registry for sampler and scheduler descriptors."""

    def __init__(self) -> None:
        self._by_id: dict[str, PluginDescriptor] = {}
        self._identity_index: dict[tuple[str, str], str] = {}
        self._adapter_classes: dict[str, type[Any]] = {}

    @classmethod
    def discover(cls, registry_root: str | Any) -> "PluginRegistry":
        registry = cls()
        discovery = PluginDiscovery(registry_root)
        for candidate in discovery.candidates():
            module = discovery.import_candidate(candidate)
            descriptor = discovery.descriptor_from_module(module, candidate)
            registry.register(descriptor, module=module)
        return registry

    def register(
        self,
        descriptor: PluginDescriptor | Mapping[str, Any],
        *,
        module: Any | None = None,
    ) -> PluginDescriptor:
        if not isinstance(descriptor, PluginDescriptor):
            descriptor = PluginDescriptor.from_mapping(descriptor)

        collisions: list[str] = []
        for identity_kind, value in descriptor.identities:
            token = normalize_identity(value)
            owner = self._identity_index.get((descriptor.kind, token))
            if owner is not None and owner != descriptor.plugin_id:
                collisions.append(
                    f"{identity_kind} {value!r} is already owned by {owner!r}"
                )
        if descriptor.plugin_id in self._by_id:
            collisions.append(f"plugin_id {descriptor.plugin_id!r} is already registered")
        if collisions:
            raise DuplicatePluginIdentityError(
                f"Cannot register {descriptor.kind} plugin {descriptor.plugin_id!r}: "
                + "; ".join(collisions)
            )

        imported = module or importlib.import_module(descriptor.module)
        adapter_class = getattr(imported, descriptor.adapter_class, None)
        if not inspect.isclass(adapter_class):
            raise PluginDescriptorError(
                f"Plugin {descriptor.plugin_id!r} adapter_class {descriptor.adapter_class!r} "
                f"was not found as a class in module {descriptor.module!r}."
            )
        require_adapter_conformance(adapter_class, descriptor.kind)

        if descriptor.kind == "sampler":
            capabilities = SamplerCapabilities.from_value(
                descriptor.capabilities,
                default_name=descriptor.name,
            )
            if normalize_identity(capabilities.sampler_name) != normalize_identity(descriptor.name):
                raise PluginDescriptorError(
                    f"Plugin {descriptor.plugin_id!r} capabilities.sampler_name "
                    f"{capabilities.sampler_name!r} must match descriptor name {descriptor.name!r}."
                )

        self._by_id[descriptor.plugin_id] = descriptor
        self._adapter_classes[descriptor.plugin_id] = adapter_class
        for _, value in descriptor.identities:
            self._identity_index[(descriptor.kind, normalize_identity(value))] = descriptor.plugin_id
        return descriptor

    def descriptors(self, kind: PluginKind | None = None) -> tuple[PluginDescriptor, ...]:
        values = self._by_id.values()
        if kind is not None:
            values = (descriptor for descriptor in values if descriptor.kind == kind)
        return tuple(sorted(values, key=lambda item: (item.kind, item.label.casefold(), item.plugin_id)))

    def resolve(
        self,
        value: str | PluginDescriptor | None,
        *,
        kind: PluginKind | None = None,
    ) -> PluginDescriptor | None:
        if value is None:
            return None
        if isinstance(value, PluginDescriptor):
            if kind is not None and value.kind != kind:
                return None
            return value
        token = normalize_identity(value)
        if not token:
            return None
        if kind is not None:
            plugin_id = self._identity_index.get((kind, token))
            return self._by_id.get(plugin_id) if plugin_id else None
        matches = {
            plugin_id
            for (candidate_kind, candidate_token), plugin_id in self._identity_index.items()
            if candidate_token == token
        }
        if len(matches) == 1:
            return self._by_id[next(iter(matches))]
        return None

    def require(self, value: str | PluginDescriptor, *, kind: PluginKind) -> PluginDescriptor:
        descriptor = self.resolve(value, kind=kind)
        if descriptor is None:
            available = ", ".join(item.name for item in self.descriptors(kind)) or "<none>"
            raise KeyError(f"Unknown {kind} plugin {value!r}. Available: {available}")
        return descriptor

    def instantiate(
        self,
        value: str | PluginDescriptor,
        *,
        kind: PluginKind | None = None,
        state: Any | None = None,
    ) -> Any:
        descriptor = value if isinstance(value, PluginDescriptor) else self.resolve(value, kind=kind)
        if descriptor is None:
            raise KeyError(f"Unknown plugin {value!r}.")
        adapter_class = self._adapter_classes[descriptor.plugin_id]
        try:
            signature = inspect.signature(adapter_class)
            parameters = signature.parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if "state" in parameters or accepts_kwargs:
                return adapter_class(state=state)
            if "shared_state" in parameters:
                return adapter_class(shared_state=state)
            return adapter_class()
        except Exception as exc:
            raise PluginInstantiationError(
                f"Failed to instantiate {descriptor.kind} plugin {descriptor.plugin_id!r} "
                f"using {descriptor.module}.{descriptor.adapter_class}: {exc}"
            ) from exc

    def legacy_map(self, kind: PluginKind) -> dict[str, dict[str, Any]]:
        return {
            descriptor.plugin_id: descriptor.to_legacy_entry()
            for descriptor in self.descriptors(kind)
        }

    def validate_pair(
        self,
        sampler: str | PluginDescriptor,
        scheduler: str | PluginDescriptor,
    ) -> PluginCompatibilityResult:
        sampler_descriptor = self.require(sampler, kind="sampler")
        scheduler_descriptor = self.require(scheduler, kind="scheduler")
        sampler_caps = SamplerCapabilities.from_value(
            sampler_descriptor.capabilities,
            default_name=sampler_descriptor.name,
        )
        scheduler_caps = dict(scheduler_descriptor.capabilities)
        reasons: list[str] = []
        warnings: list[str] = []

        supports_fixed = bool(scheduler_caps.get("supports_fixed_steps", False))
        supports_expansion = bool(scheduler_caps.get("supports_step_expansion", False))
        supports_tail = bool(scheduler_caps.get("supports_tail_metadata", False))
        scheduler_family = normalize_identity(scheduler_caps.get("scheduler_family"))
        modes = {
            normalize_identity(mode).replace(" ", "_")
            for mode in scheduler_caps.get("pipeline_modes", [])
        }

        sampler_token = normalize_identity(sampler_caps.sampler_name).replace(" ", "_")
        is_kes_sampler = sampler_token in {
            "kes",
            "kes_sampler",
            "kes_style",
            "kes_style_sampler",
            "simple_kes_sampler",
        }
        fixed_mode_available = supports_fixed or "fixed_steps" in modes or "compatible" in modes
        clamp_kes_to_fixed = (
            is_kes_sampler
            and scheduler_family != "kes"
            and fixed_mode_available
        )
        step_expansion_clamped = bool(
            clamp_kes_to_fixed and sampler_caps.supports_step_expansion and not supports_expansion
        )
        tail_metadata_clamped = bool(
            clamp_kes_to_fixed and sampler_caps.supports_tail_metadata and not supports_tail
        )
        negotiated_pipeline_mode = (
            "fixed_steps" if clamp_kes_to_fixed else sampler_caps.forced_pipeline_mode
        )

        if sampler_caps.requires_requested_step_schedule and not fixed_mode_available:
            reasons.append("sampler requires a fixed requested-step schedule")

        if clamp_kes_to_fixed:
            if step_expansion_clamped:
                warnings.append(
                    "KES step expansion was clamped because the selected non-KES scheduler "
                    "provides a fixed requested-step schedule."
                )
            if tail_metadata_clamped:
                warnings.append(
                    "KES tail metadata was disabled because the selected non-KES scheduler "
                    "does not provide KES tail semantics."
                )
        else:
            if sampler_caps.supports_step_expansion and not supports_expansion:
                reasons.append("sampler supports expanded steps but scheduler cannot produce them")
            if sampler_caps.supports_tail_metadata and not supports_tail:
                reasons.append("sampler requires scheduler tail metadata")
            if sampler_caps.forced_pipeline_mode == "extended_steps":
                if not supports_expansion and "extended_steps" not in modes and "compatible" not in modes:
                    reasons.append("sampler requires extended_steps compatibility")

        if sampler_caps.forced_pipeline_mode == "fixed_steps":
            if not fixed_mode_available:
                reasons.append("sampler requires fixed_steps mode")

        return PluginCompatibilityResult(
            sampler_id=sampler_descriptor.plugin_id,
            scheduler_id=scheduler_descriptor.plugin_id,
            is_compatible=not reasons,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            negotiated_pipeline_mode=negotiated_pipeline_mode,
            step_expansion_clamped=step_expansion_clamped,
            tail_metadata_clamped=tail_metadata_clamped,
        )

    def to_diagnostic_dict(self) -> dict[str, Any]:
        return {
            "schema": "image-gen-plugin-registry-v1",
            "plugin_count": len(self._by_id),
            "plugins": [descriptor.to_dict() for descriptor in self.descriptors()],
        }
