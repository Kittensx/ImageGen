# File: modules/ss_registry/master_sampler.py
"""Compatibility facade for the Phase 05 sampler plugin registry.

Runtime code uses :class:`image_gen.systems.registry.RuntimeRegistrySystem`.
This class remains for older diagnostics and imports; generated maps are JSON
snapshots only and are never imported by the runtime.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from image_gen.contracts import SamplerCapabilities
from image_gen.systems.registry import PluginDescriptor, PluginRegistry, RuntimeRegistrySystem
from modules.project_context import ProjectContext


def _safe_name(value: Any, fallback: str) -> str:
    name = str(value or fallback).strip().casefold().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_.+-]+", "_", name).strip("_")
    return name or fallback


class SamplerMap:
    """Expose the canonical sampler descriptors in the historical map shape."""

    def __init__(
        self,
        shared_state: Optional[Any] = None,
        base_path: Optional[str] = None,
        verbose: bool = False,
        output_to_file: bool = False,
        project_context: ProjectContext | None = None,
        auto_generate: bool = True,
    ) -> None:
        self.project_context = project_context
        if base_path is None:
            base_path = str(
                project_context.registry_root
                if project_context is not None
                else Path(__file__).resolve().parent
            )
        self.base_path = str(Path(base_path).expanduser().resolve())
        self.scan_path = str((Path(self.base_path) / "samplers").resolve())
        self.output_file = str((Path(self.base_path) / "sampler_registry.diagnostic.json").resolve())
        self.state = shared_state
        self.verbose = verbose
        self.output_to_file = bool(output_to_file)
        self.sampler_map: Dict[str, Dict[str, Any]] = {}
        self._registry = PluginRegistry()
        if auto_generate:
            self.generate()

    def generate(self) -> dict[str, dict[str, Any]]:
        system = RuntimeRegistrySystem(
            self.state,
            project_context=self.project_context,
        )
        if self.project_context is None:
            system._registry = PluginRegistry.discover(self.base_path)
        self._registry = system.registry
        self.sampler_map = self._registry.legacy_map("sampler")
        if self.output_to_file:
            self._write_output_file()
        if self.verbose:
            print(f"[Registry] Discovered {len(self.sampler_map)} sampler plugin(s).")
        return self.sampler_map

    def _register_sampler_module(
        self,
        mod: Any,
        entry: str,
        module_name: str,
        config_entry_name: str,
        source_type: str,
    ) -> None:
        value = getattr(mod, "PLUGIN_DESCRIPTOR", None)
        adapter = getattr(mod, "SAMPLER_ADAPTER_CLASS", None)
        if value is None:
            meta = dict(getattr(mod, "meta", {}) or {})
            name = _safe_name(meta.get("name"), entry)
            capabilities = getattr(adapter, "SAMPLER_CAPABILITIES", None)
            if capabilities is None:
                is_kes = name in {"kes", "kes_sampler"}
                capabilities = SamplerCapabilities(
                    sampler_name=name,
                    guidance_owner="sampler" if is_kes else "pipeline",
                    uses_raw_model_fn=is_kes,
                    uses_guided_model_fn=not is_kes,
                    supports_step_expansion=is_kes,
                    supports_tail_metadata=is_kes,
                    requires_requested_step_schedule=not is_kes,
                    forced_pipeline_mode="extended_steps" if is_kes else "fixed_steps",
                )
            if isinstance(capabilities, SamplerCapabilities):
                capabilities = capabilities.to_serializable_dict()
            value = {
                "plugin_id": f"sampler.{name}",
                "kind": "sampler",
                "name": name,
                "label": meta.get("label", name),
                "description": meta.get("description", ""),
                "module": module_name,
                "adapter_class": getattr(adapter, "__name__", "MissingSamplerAdapter"),
                "aliases": [],
                "capabilities": capabilities,
                "config_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": True,
                },
            }
        if adapter is not None and not hasattr(mod, adapter.__name__):
            setattr(mod, adapter.__name__, adapter)
        descriptor = (
            value
            if isinstance(value, PluginDescriptor)
            else PluginDescriptor.from_mapping(value, default_kind="sampler", default_module=module_name)
        )
        self._registry.register(descriptor, module=mod)
        self.sampler_map[descriptor.plugin_id] = descriptor.to_legacy_entry()

    def _write_output_file(self) -> None:
        path = Path(self.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "image-gen-sampler-registry-diagnostic-v1",
                    "runtime_dependency": False,
                    "samplers": self.sampler_map,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.verbose:
            print(f"[Registry] Wrote diagnostic snapshot: {path}")
