# File: modules/ss_registry/master_scheduler.py
"""Compatibility facade for the Phase 05 scheduler plugin registry.

Runtime code discovers descriptors in memory. Diagnostic snapshots are JSON
artifacts and are never imported by the generation path.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from image_gen.systems.registry import PluginDescriptor, PluginRegistry, RuntimeRegistrySystem
from modules.project_context import ProjectContext


def _safe_name(value: Any, fallback: str) -> str:
    name = str(value or fallback).strip().casefold().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_.+-]+", "_", name).strip("_")
    return name or fallback


class SchedulerMap:
    """Expose canonical scheduler descriptors in the historical map shape."""

    def __init__(
        self,
        shared_state=None,
        base_path: str | None = None,
        scan_path: str | None = None,
        verbose: bool = False,
        output_to_file: bool = False,
        project_context: ProjectContext | None = None,
        auto_generate: bool = True,
    ) -> None:
        self.state = shared_state
        self.project_context = project_context
        if base_path is None:
            base_path = str(
                project_context.registry_root
                if project_context is not None
                else Path(__file__).resolve().parent
            )
        self.base_path = str(Path(base_path).expanduser().resolve())
        self.scan_path = str(
            Path(scan_path).expanduser().resolve()
            if scan_path is not None
            else (Path(self.base_path) / "schedulers").resolve()
        )
        self.output_file = str((Path(self.base_path) / "scheduler_registry.diagnostic.json").resolve())
        self.verbose = verbose
        self.output_to_file = bool(output_to_file)
        self.sched_map_index: dict[str, dict[str, Any]] = {}
        self.new_map: dict[str, dict[str, Any]] = {}
        self._id_to_name: dict[str, str] = {}
        self._label_to_name: dict[str, str] = {}
        self._registry = PluginRegistry()
        if auto_generate:
            self.generate()

    def generate(self) -> dict[str, dict[str, Any]]:
        system = RuntimeRegistrySystem(self.state, project_context=self.project_context)
        if self.project_context is None:
            system._registry = PluginRegistry.discover(self.base_path)
        self._registry = system.registry
        self.sched_map_index = self._registry.legacy_map("scheduler")
        self.new_map = dict(self.sched_map_index)
        self._id_to_name = {
            key: entry.get("name", key) for key, entry in self.sched_map_index.items()
        }
        self._label_to_name = {
            entry.get("label", key): entry.get("name", key)
            for key, entry in self.sched_map_index.items()
        }
        if self.output_to_file:
            self._write_reference_file()
        if self.verbose:
            print(f"[Registry] Discovered {len(self.sched_map_index)} scheduler plugin(s).")
        return self.sched_map_index

    def _register_scheduler_module(
        self,
        mod: Any,
        entry: str,
        module_name: str,
        seen_labels: set[str],
        subdir: str,
    ) -> None:
        value = getattr(mod, "PLUGIN_DESCRIPTOR", None)
        if value is None:
            meta = dict(getattr(mod, "meta", {}) or {})
            adapter = getattr(mod, "SCHEDULER_ADAPTER_CLASS", None)
            name = _safe_name(meta.get("name"), entry)
            value = {
                "plugin_id": f"scheduler.{name}",
                "kind": "scheduler",
                "name": name,
                "label": meta.get("label", name),
                "description": meta.get("summary_text", ""),
                "module": module_name,
                "adapter_class": getattr(adapter, "__name__", "MissingSchedulerAdapter"),
                "aliases": [],
                "capabilities": {
                    "pipeline_modes": list(meta.get("supports_pipeline_modes", ["fixed_steps"])),
                    "supports_fixed_steps": True,
                    "supports_step_expansion": bool(meta.get("supports_step_expansion", False)),
                    "supports_tail_metadata": bool(meta.get("supports_tail_steps", False)),
                    "schedule_domain": str(meta.get("schedule_domain", "vp_sigma")),
                },
                "config_schema": {
                    "type": "object",
                    "properties": dict(meta.get("args", {})),
                    "required": [],
                    "additionalProperties": True,
                },
            }
        adapter = getattr(mod, "SCHEDULER_ADAPTER_CLASS", None)
        if adapter is not None and not hasattr(mod, adapter.__name__):
            setattr(mod, adapter.__name__, adapter)
        descriptor = (
            value
            if isinstance(value, PluginDescriptor)
            else PluginDescriptor.from_mapping(value, default_kind="scheduler", default_module=module_name)
        )
        label_token = descriptor.label.casefold()
        if label_token in {item.casefold() for item in seen_labels}:
            from image_gen.systems.registry import DuplicatePluginIdentityError

            raise DuplicatePluginIdentityError(
                f"Cannot register scheduler plugin {descriptor.plugin_id!r}: "
                f"label {descriptor.label!r} is duplicated."
            )
        self._registry.register(descriptor, module=mod)
        seen_labels.add(descriptor.label)
        self.sched_map_index[descriptor.plugin_id] = descriptor.to_legacy_entry()
        self.new_map = dict(self.sched_map_index)

    def _write_reference_file(self) -> None:
        path = Path(self.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "image-gen-scheduler-registry-diagnostic-v1",
                    "runtime_dependency": False,
                    "schedulers": self.sched_map_index,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.verbose:
            print(f"[Registry] Wrote diagnostic snapshot: {path}")

    def get_scheduler_by_id(self, id_or_name):
        if id_or_name in self.sched_map_index:
            return self.sched_map_index[id_or_name].get("name")
        return self._id_to_name.get(id_or_name)

    def get_by_label_or_id_or_name(self, value):
        if value in self.sched_map_index:
            return self.sched_map_index[value].get("name")
        return self._id_to_name.get(value) or self._label_to_name.get(value)
