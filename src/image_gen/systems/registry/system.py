from __future__ import annotations

import importlib
import inspect
import threading
from pathlib import Path
from typing import Any

from image_gen.systems.registry.descriptors import PluginDescriptor, PluginKind, normalize_identity
from image_gen.systems.registry.registry import PluginCompatibilityResult, PluginRegistry
from image_gen.runtime.scheduler_settings import normalize_scheduler_payload
from image_gen.systems.image_conditioning import require_qualified_hires_pair


class RuntimeRegistrySystem:
    """Session-scoped plugin registry and adapter construction boundary.

    Descriptor discovery is lazy and performed at most once for each system
    instance. The same instance should be shared by the CLI and Txt2ImgRunner.
    """

    def __init__(
        self,
        state: Any | None = None,
        *,
        project_context: Any | None = None,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.state = state
        self.project_context = project_context
        self._registry = registry
        self._lock = threading.Lock()
        self._construction_count = 1 if registry is not None else 0

    @property
    def construction_count(self) -> int:
        return self._construction_count

    @property
    def registry(self) -> PluginRegistry:
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    root = self._registry_root()
                    self._registry = PluginRegistry.discover(root)
                    self._construction_count += 1
        return self._registry

    def _registry_root(self) -> Path:
        configured_root = getattr(self.project_context, "registry_root", None)
        if configured_root is not None:
            return Path(configured_root).resolve()
        return (Path(__file__).resolve().parents[4] / "modules" / "ss_registry").resolve()

    def bind_state(self, state: Any) -> None:
        self.state = state

    def descriptors(self, kind: PluginKind | None = None) -> tuple[PluginDescriptor, ...]:
        return self.registry.descriptors(kind)

    def legacy_map(self, kind: PluginKind) -> dict[str, dict[str, Any]]:
        return self.registry.legacy_map(kind)

    @staticmethod
    def _entries(registry: Any) -> list[tuple[Any, dict[str, Any]]]:
        if isinstance(registry, dict):
            return [(key, value) for key, value in registry.items() if isinstance(value, dict)]
        for attr in ("new_map", "sampler_map", "sched_map_index", "scheduler_map", "map", "registry"):
            candidate = getattr(registry, attr, None)
            if isinstance(candidate, dict):
                return [(key, value) for key, value in candidate.items() if isinstance(value, dict)]
        return []

    def resolve_descriptor(self, requested: Any, *, kind: PluginKind) -> PluginDescriptor | None:
        if kind == "scheduler" and normalize_identity(requested) == "karras exponential":
            requested = "simple_kes"
        return self.registry.resolve(str(requested or ""), kind=kind)

    def resolve_entry(
        self,
        requested: Any,
        registry: Any | None = None,
        *,
        kind: PluginKind,
    ) -> dict[str, Any] | None:
        if registry is None:
            descriptor = self.resolve_descriptor(requested, kind=kind)
            return descriptor.to_legacy_entry() if descriptor else None

        needle = normalize_identity(requested)
        if not needle:
            return None
        if kind == "scheduler" and needle == "karras exponential":
            needle = "simple_kes"
        for key, entry in self._entries(registry):
            candidates = {
                normalize_identity(key),
                normalize_identity(entry.get("plugin_id")),
                normalize_identity(entry.get("id")),
                normalize_identity(entry.get("name")),
                normalize_identity(entry.get("label")),
                *(normalize_identity(alias) for alias in entry.get("aliases", []) or []),
            }
            if needle in candidates:
                return entry
        return None

    def validate_pair(
        self,
        sampler: str | PluginDescriptor,
        scheduler: str | PluginDescriptor,
    ) -> PluginCompatibilityResult:
        return self.registry.validate_pair(sampler, scheduler)

    @staticmethod
    def _apply_compatibility_scheduler_kwargs(
        scheduler_kwargs: dict[str, Any],
        *,
        sampler_descriptor: PluginDescriptor,
        scheduler_descriptor: PluginDescriptor,
        compatibility: PluginCompatibilityResult,
    ) -> dict[str, Any]:
        output = dict(scheduler_kwargs or {})
        if compatibility.step_expansion_clamped or compatibility.tail_metadata_clamped:
            output["pipeline_mode"] = compatibility.negotiated_pipeline_mode or "fixed_steps"
            negotiated = dict(output.get("compatibility", {}) or {})
            negotiated.update(
                {
                    "requested_by_sampler": sampler_descriptor.name,
                    "scheduler_family": scheduler_descriptor.capabilities.get(
                        "scheduler_family", "unknown"
                    ),
                    "negotiated_pipeline_mode": compatibility.negotiated_pipeline_mode,
                    "step_expansion_clamped": compatibility.step_expansion_clamped,
                    "tail_metadata_clamped": compatibility.tail_metadata_clamped,
                    "warnings": list(compatibility.warnings),
                }
            )
            output["compatibility"] = negotiated
        return output

    def apply_compatibility_to_payload(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized = dict(payload or {})
        sampler_descriptor = self.resolve_descriptor(
            normalized.get("sampler_name"), kind="sampler"
        )
        scheduler_descriptor = self.resolve_descriptor(
            normalized.get("scheduler_name"), kind="scheduler"
        )
        if sampler_descriptor is None or scheduler_descriptor is None:
            return normalized, {}
        compatibility = self.validate_pair(sampler_descriptor, scheduler_descriptor)
        compatibility.raise_if_incompatible()
        normalized["sampler_name"] = sampler_descriptor.name
        normalized["scheduler_name"] = scheduler_descriptor.name
        normalized["scheduler_kwargs"] = self._apply_compatibility_scheduler_kwargs(
            dict(normalized.get("scheduler_kwargs") or {}),
            sampler_descriptor=sampler_descriptor,
            scheduler_descriptor=scheduler_descriptor,
            compatibility=compatibility,
        )
        diagnostics = dict(normalized.get("diagnostics") or {})
        diagnostics["plugin_compatibility"] = compatibility.to_dict()
        normalized["diagnostics"] = diagnostics
        return normalized, compatibility.to_dict()

    def apply_resolution(self, request: Any, extras: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        sampler_descriptor = self.resolve_descriptor(
            extras.get("sampler_label") or request.sampler_name,
            kind="sampler",
        )
        scheduler_descriptor = self.resolve_descriptor(
            extras.get("scheduler_label") or request.scheduler_name,
            kind="scheduler",
        )

        # Transitional fallback for callers injecting an old registry map.
        sampler_entry = None
        scheduler_entry = None
        if sampler_descriptor is None:
            sampler_entry = self.resolve_entry(
                extras.get("sampler_label") or request.sampler_name,
                extras.get("live_sampler_map") or extras.get("sampler_registry"),
                kind="sampler",
            )
        if scheduler_descriptor is None:
            scheduler_entry = self.resolve_entry(
                extras.get("scheduler_label") or request.scheduler_name,
                extras.get("live_scheduler_map") or extras.get("scheduler_registry"),
                kind="scheduler",
            )

        if sampler_descriptor is not None:
            request.sampler_name = sampler_descriptor.name
            sampler_entry = sampler_descriptor.to_legacy_entry()
            extras["resolved_sampler_descriptor"] = sampler_descriptor
        if scheduler_descriptor is not None:
            request.scheduler_name = scheduler_descriptor.name
            scheduler_entry = scheduler_descriptor.to_legacy_entry()
            extras["resolved_scheduler_descriptor"] = scheduler_descriptor

        if sampler_entry:
            request.sampler_name = sampler_entry.get("name") or request.sampler_name
            extras["resolved_sampler_entry"] = sampler_entry
            extras["resolved_sampler_name"] = sampler_entry.get("name")
            extras["resolved_sampler_label"] = sampler_entry.get("label")
        if scheduler_entry:
            request.scheduler_name = scheduler_entry.get("name") or request.scheduler_name
            extras["resolved_scheduler_entry"] = scheduler_entry
            extras["resolved_scheduler_name"] = scheduler_entry.get("name")
            extras["resolved_scheduler_label"] = scheduler_entry.get("label")

        if sampler_descriptor is not None and scheduler_descriptor is not None:
            compatibility = self.validate_pair(sampler_descriptor, scheduler_descriptor)
            compatibility.raise_if_incompatible()
            extras["plugin_compatibility"] = compatibility.to_dict()
            request.scheduler_kwargs = self._apply_compatibility_scheduler_kwargs(
                dict(getattr(request, "scheduler_kwargs", {}) or {}),
                sampler_descriptor=sampler_descriptor,
                scheduler_descriptor=scheduler_descriptor,
                compatibility=compatibility,
            )

        hires_sampler_requested = str(getattr(request, "hires_sampler_name", "") or "").strip()
        hires_scheduler_requested = str(getattr(request, "hires_scheduler_name", "") or "").strip()
        hires_sampler_descriptor = self.resolve_descriptor(
            hires_sampler_requested or request.sampler_name,
            kind="sampler",
        )
        hires_scheduler_descriptor = self.resolve_descriptor(
            hires_scheduler_requested or request.scheduler_name,
            kind="scheduler",
        )
        if hires_sampler_descriptor is None:
            raise KeyError(
                "Unknown hires sampler plugin "
                f"{hires_sampler_requested or request.sampler_name!r}."
            )
        if hires_scheduler_descriptor is None:
            raise KeyError(
                "Unknown hires scheduler plugin "
                f"{hires_scheduler_requested or request.scheduler_name!r}."
            )
        hires_compatibility = self.validate_pair(
            hires_sampler_descriptor,
            hires_scheduler_descriptor,
        )
        hires_compatibility.raise_if_incompatible()
        hires_qualification = require_qualified_hires_pair(
            hires_sampler_descriptor.name,
            hires_scheduler_descriptor.name,
            compatibility=hires_compatibility.to_dict(),
        )
        extras["resolved_hires_sampler_descriptor"] = hires_sampler_descriptor
        extras["resolved_hires_scheduler_descriptor"] = hires_scheduler_descriptor
        extras["hires_plugin_compatibility"] = hires_compatibility.to_dict()
        extras["hires_pair_qualification"] = hires_qualification.to_serializable_dict()
        extras["hires_sampler_inherited"] = not bool(hires_sampler_requested)
        extras["hires_scheduler_inherited"] = not bool(hires_scheduler_requested)
        request.hires_sampler_name = (
            "" if not hires_sampler_requested else hires_sampler_descriptor.name
        )
        request.hires_scheduler_name = (
            "" if not hires_scheduler_requested else hires_scheduler_descriptor.name
        )
        base_scheduler_name = str(request.scheduler_name or "").strip().lower()
        base_sampler_name = str(request.sampler_name or "").strip().lower()
        hires_scheduler_kwargs = (
            dict(getattr(request, "scheduler_kwargs", {}) or {})
            if hires_scheduler_descriptor.name.lower() == base_scheduler_name
            else {}
        )
        hires_sampler_kwargs = (
            dict(getattr(request, "sampler_kwargs", {}) or {})
            if hires_sampler_descriptor.name.lower() == base_sampler_name
            else {}
        )
        hires_scheduler_kwargs = self._apply_compatibility_scheduler_kwargs(
            hires_scheduler_kwargs,
            sampler_descriptor=hires_sampler_descriptor,
            scheduler_descriptor=hires_scheduler_descriptor,
            compatibility=hires_compatibility,
        )
        setattr(request, "_hires_resolved_scheduler_kwargs", hires_scheduler_kwargs)
        setattr(request, "_hires_resolved_sampler_kwargs", hires_sampler_kwargs)

        # Resolve Simple KES through the same canonical validator used by the
        # WebUI prequeue path and CLI effective-request writer. This is
        # intentionally idempotent so direct runtime callers cannot bypass it.
        scheduler_payload = {
            "scheduler_name": request.scheduler_name,
            "steps": request.steps,
            "scheduler_kwargs": dict(getattr(request, "scheduler_kwargs", {}) or {}),
            "diagnostics": dict(getattr(request, "diagnostics", {}) or {}),
        }
        scheduler_payload, scheduler_resolution = normalize_scheduler_payload(scheduler_payload)
        if scheduler_resolution is not None:
            request.scheduler_name = scheduler_payload["scheduler_name"]
            request.steps = int(scheduler_payload["steps"])
            request.scheduler_kwargs = dict(scheduler_payload["scheduler_kwargs"] or {})
            request.diagnostics = dict(scheduler_payload.get("diagnostics") or {})
            extras["scheduler_settings_resolution"] = scheduler_resolution.to_dict()

        return request, extras

    def instantiate_adapter(self, entry: PluginDescriptor | dict[str, Any] | None) -> Any | None:
        if not entry:
            return None
        if isinstance(entry, PluginDescriptor):
            return self.registry.instantiate(entry, state=self.state)

        plugin_id = entry.get("plugin_id") or entry.get("id")
        if plugin_id:
            descriptor = self.registry.resolve(str(plugin_id), kind=entry.get("kind"))
            if descriptor is not None:
                return self.registry.instantiate(descriptor, state=self.state)

        module_name = entry.get("module")
        class_name = entry.get("adapter_class") or entry.get("entry_class")
        if not module_name or not class_name:
            raise ValueError(f"Registry entry missing module/class info: {entry}")
        adapter_cls = getattr(importlib.import_module(module_name), class_name)
        signature = inspect.signature(adapter_cls)
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if "state" in parameters or accepts_kwargs:
            return adapter_cls(state=self.state)
        if "shared_state" in parameters:
            return adapter_cls(shared_state=self.state)
        return adapter_cls()
