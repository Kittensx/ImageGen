from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable

from image_gen.systems.registry.descriptors import PluginDescriptor, PluginKind
from image_gen.systems.registry.errors import PluginDescriptorError, PluginDiscoveryError


@dataclass(frozen=True)
class PluginCandidate:
    kind: PluginKind
    module_name: str
    source_path: Path


class PluginDiscovery:
    """Discover descriptor-bearing sampler and scheduler modules from project roots."""

    def __init__(self, registry_root: str | Path) -> None:
        self.registry_root = Path(registry_root).expanduser().resolve()

    def candidates(self, kind: PluginKind | None = None) -> tuple[PluginCandidate, ...]:
        selected: list[PluginCandidate] = []
        if kind in (None, "sampler"):
            selected.extend(self._sampler_candidates())
        if kind in (None, "scheduler"):
            selected.extend(self._scheduler_candidates())
        return tuple(sorted(selected, key=lambda item: (item.kind, item.module_name.casefold())))

    def _sampler_candidates(self) -> Iterable[PluginCandidate]:
        root = self.registry_root / "samplers"
        if not root.is_dir():
            raise PluginDiscoveryError(f"Sampler plugin root does not exist: {root}")

        for child in root.iterdir():
            if child.name in {"config", "__pycache__", "simple_samplers"}:
                continue
            init_file = child / "__init__.py"
            if child.is_dir() and init_file.is_file():
                yield PluginCandidate(
                    "sampler",
                    f"modules.ss_registry.samplers.{child.name}",
                    init_file,
                )

        simple_root = root / "simple_samplers"
        if simple_root.is_dir():
            for source in simple_root.glob("*.py"):
                if source.name == "__init__.py":
                    continue
                yield PluginCandidate(
                    "sampler",
                    f"modules.ss_registry.samplers.simple_samplers.{source.stem}",
                    source,
                )

    def _scheduler_candidates(self) -> Iterable[PluginCandidate]:
        root = self.registry_root / "schedulers"
        if not root.is_dir():
            raise PluginDiscoveryError(f"Scheduler plugin root does not exist: {root}")
        for child in root.iterdir():
            if child.name in {"config", "__pycache__"}:
                continue
            init_file = child / "__init__.py"
            if child.is_dir() and init_file.is_file():
                yield PluginCandidate(
                    "scheduler",
                    f"modules.ss_registry.schedulers.{child.name}",
                    init_file,
                )

    @staticmethod
    def import_candidate(candidate: PluginCandidate) -> ModuleType:
        try:
            return importlib.import_module(candidate.module_name)
        except Exception as exc:
            raise PluginDiscoveryError(
                f"Failed to import {candidate.kind} plugin module {candidate.module_name!r} "
                f"from {candidate.source_path}: {exc}"
            ) from exc

    @staticmethod
    def descriptor_from_module(
        module: ModuleType,
        candidate: PluginCandidate,
    ) -> PluginDescriptor:
        value = getattr(module, "PLUGIN_DESCRIPTOR", None)
        if value is None:
            raise PluginDescriptorError(
                f"{candidate.kind.title()} plugin module {candidate.module_name!r} "
                "must export PLUGIN_DESCRIPTOR."
            )
        if isinstance(value, PluginDescriptor):
            descriptor = value
        else:
            descriptor = PluginDescriptor.from_mapping(
                value,
                default_kind=candidate.kind,
                default_module=candidate.module_name,
                source_path=str(candidate.source_path),
            )
        if descriptor.kind != candidate.kind:
            raise PluginDescriptorError(
                f"Plugin {descriptor.plugin_id!r} declares kind {descriptor.kind!r}, "
                f"but was discovered under the {candidate.kind} root."
            )
        if descriptor.module != candidate.module_name:
            raise PluginDescriptorError(
                f"Plugin {descriptor.plugin_id!r} declares module {descriptor.module!r}; "
                f"discovery imported {candidate.module_name!r}."
            )
        return descriptor
