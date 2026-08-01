from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from image_gen.systems.registry import RuntimeRegistrySystem
from image_gen.systems.registry.descriptors import PluginDescriptor

WEBUI_SELECTION_VERSION = 2
_WEBUI_SELECTION_VERSION_KEY = "_webui_selection_version"
_WEBUI_SCHEDULER_EXPLICIT_KEY = "_webui_scheduler_user_selected"

# The original WebUI silently fell back to the alphabetically first registry
# entries when configured names could not be resolved. With the current
# registry ordering that produced KES + Standard Karras. That pair is valid as
# a control experiment, but Standard Karras is model-bounded (~14.6 max sigma)
# and is not the production-quality KES schedule used by the working CLI path.
_LEGACY_AUTO_FALLBACK_PAIR = ("kes", "standard_karras")


@dataclass(frozen=True)
class SelectionNormalization:
    payload: dict[str, Any]
    notes: tuple[str, ...] = field(default_factory=tuple)
    changed: bool = False


class WebUISelectionResolver:
    """Resolve WebUI sampler/scheduler selections without silent first-item fallback."""

    def __init__(self, registry: RuntimeRegistrySystem) -> None:
        self.registry = registry

    def _resolve(self, kind: str, requested: Any) -> PluginDescriptor | None:
        return self.registry.resolve_descriptor(requested, kind=kind)

    def _first(self, kind: str) -> PluginDescriptor | None:
        values = self.registry.descriptors(kind)
        return values[0] if values else None

    def _safe_sampler(self) -> PluginDescriptor | None:
        return self._resolve("sampler", "kes") or self._first("sampler")

    def _preferred_scheduler(self, sampler: PluginDescriptor | None) -> PluginDescriptor | None:
        if sampler is None:
            return self._resolve("scheduler", "simple_kes") or self._first("scheduler")

        capabilities = dict(sampler.capabilities or {})
        candidates: list[Any] = []
        preferred = capabilities.get("preferred_scheduler")
        if preferred:
            candidates.append(preferred)
        preferred_many = capabilities.get("preferred_schedulers")
        if isinstance(preferred_many, (list, tuple)):
            candidates.extend(preferred_many)

        # Stable fallback policy for descriptors created before preferred
        # scheduler metadata was added.
        if sampler.name == "kes":
            candidates.append("simple_kes")
        elif sampler.name in {"dpmpp_2m", "simple_euler"}:
            candidates.append("standard_karras")

        for candidate in candidates:
            descriptor = self._resolve("scheduler", candidate)
            if descriptor is not None:
                return descriptor

        for descriptor in self.registry.descriptors("scheduler"):
            try:
                if self.registry.validate_pair(sampler, descriptor).is_compatible:
                    return descriptor
            except Exception:
                continue
        return self._first("scheduler")

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _as_version(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def normalize(
        self,
        payload: Mapping[str, Any] | None,
        *,
        fallback_payload: Mapping[str, Any] | None = None,
        migrate_legacy_auto_fallback: bool = True,
        reject_unknown: bool = False,
    ) -> SelectionNormalization:
        original = dict(payload or {})
        fallback = dict(fallback_payload or {})
        normalized = dict(original)
        notes: list[str] = []

        requested_sampler = normalized.get("sampler_name")
        sampler = self._resolve("sampler", requested_sampler)
        if sampler is None:
            sampler = self._resolve("sampler", fallback.get("sampler_name"))
        if sampler is None:
            if reject_unknown and requested_sampler:
                raise ValueError(f"Unknown sampler selection: {requested_sampler!r}.")
            sampler = self._safe_sampler()
            if requested_sampler:
                notes.append(
                    f"Sampler {requested_sampler!r} was not registered; selected "
                    f"{sampler.name!r} instead." if sampler else
                    f"Sampler {requested_sampler!r} was not registered."
                )
        if sampler is None:
            raise ValueError("No sampler plugins are available.")
        normalized["sampler_name"] = sampler.name

        requested_scheduler = normalized.get("scheduler_name")
        scheduler = self._resolve("scheduler", requested_scheduler)
        if scheduler is None:
            scheduler = self._resolve("scheduler", fallback.get("scheduler_name"))
        if scheduler is None:
            if reject_unknown and requested_scheduler:
                raise ValueError(f"Unknown scheduler selection: {requested_scheduler!r}.")
            scheduler = self._preferred_scheduler(sampler)
            if requested_scheduler:
                notes.append(
                    f"Scheduler {requested_scheduler!r} was not registered; selected "
                    f"{scheduler.name!r} instead." if scheduler else
                    f"Scheduler {requested_scheduler!r} was not registered."
                )
        if scheduler is None:
            raise ValueError("No scheduler plugins are available.")

        explicit_scheduler = self._as_bool(
            normalized.get(_WEBUI_SCHEDULER_EXPLICIT_KEY, False)
        )
        selection_version = self._as_version(
            normalized.get(_WEBUI_SELECTION_VERSION_KEY)
        )

        if (
            migrate_legacy_auto_fallback
            and selection_version < WEBUI_SELECTION_VERSION
            and not explicit_scheduler
            and (sampler.name, scheduler.name) == _LEGACY_AUTO_FALLBACK_PAIR
        ):
            replacement = self._preferred_scheduler(sampler)
            if replacement is not None and replacement.name != scheduler.name:
                scheduler = replacement
                normalized["scheduler_kwargs"] = {}
                notes.append(
                    "Replaced the legacy automatic KES + Standard Karras fallback "
                    "with KES + Simple KES so the WebUI uses the same high-sigma "
                    "generation path as the working CLI run."
                )

        compatibility = self.registry.validate_pair(sampler, scheduler)
        if not compatibility.is_compatible:
            if explicit_scheduler or reject_unknown:
                compatibility.raise_if_incompatible()
            replacement = self._preferred_scheduler(sampler)
            if replacement is None:
                compatibility.raise_if_incompatible()
            scheduler = replacement
            normalized["scheduler_kwargs"] = {}
            notes.append(
                f"Selected scheduler was incompatible with {sampler.label}; "
                f"switched to {scheduler.label}."
            )

        normalized["scheduler_name"] = scheduler.name
        normalized[_WEBUI_SELECTION_VERSION_KEY] = WEBUI_SELECTION_VERSION
        normalized[_WEBUI_SCHEDULER_EXPLICIT_KEY] = explicit_scheduler

        return SelectionNormalization(
            payload=normalized,
            notes=tuple(notes),
            changed=normalized != original,
        )

    @staticmethod
    def strip_webui_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        return {
            key: value
            for key, value in dict(payload or {}).items()
            if not str(key).startswith("_webui_")
        }


__all__ = [
    "SelectionNormalization",
    "WEBUI_SELECTION_VERSION",
    "WebUISelectionResolver",
]
