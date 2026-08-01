from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

from image_gen.systems.registry.errors import PluginDescriptorError

PluginKind = Literal["sampler", "scheduler"]

_STABLE_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_CANONICAL_NAME = re.compile(r"^[a-z0-9][a-z0-9_.+-]*$")
_DOTTED_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_identity(value: Any) -> str:
    """Return a case-insensitive registry identity token."""
    return " ".join(str(value or "").strip().casefold().split())


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): item for key, item in dict(value or {}).items()}


def _normalize_aliases(values: Iterable[Any] | None) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values or ():
        value = str(raw or "").strip()
        token = normalize_identity(value)
        if not token or token in seen:
            continue
        seen.add(token)
        output.append(value)
    return tuple(output)


def validate_config_schema(schema: Mapping[str, Any], *, plugin_id: str) -> dict[str, Any]:
    payload = _copy_mapping(schema)
    if payload.get("type") != "object":
        raise PluginDescriptorError(
            f"Plugin {plugin_id!r} config_schema.type must be exactly 'object'."
        )
    properties = payload.get("properties")
    if not isinstance(properties, Mapping):
        raise PluginDescriptorError(
            f"Plugin {plugin_id!r} config_schema.properties must be a mapping."
        )
    required = payload.get("required", [])
    if not isinstance(required, (list, tuple)) or not all(isinstance(x, str) for x in required):
        raise PluginDescriptorError(
            f"Plugin {plugin_id!r} config_schema.required must be a list of field names."
        )
    unknown_required = sorted(set(required) - set(properties))
    if unknown_required:
        raise PluginDescriptorError(
            f"Plugin {plugin_id!r} config_schema.required references undefined fields: "
            + ", ".join(unknown_required)
        )
    additional = payload.get("additionalProperties", True)
    if not isinstance(additional, bool):
        raise PluginDescriptorError(
            f"Plugin {plugin_id!r} config_schema.additionalProperties must be boolean."
        )
    payload["properties"] = dict(properties)
    payload["required"] = list(required)
    payload["additionalProperties"] = additional
    return payload


@dataclass(frozen=True)
class PluginDescriptor:
    """Stable, serializable metadata for one sampler or scheduler plugin."""

    plugin_id: str
    kind: PluginKind
    name: str
    label: str
    module: str
    adapter_class: str
    capabilities: Mapping[str, Any]
    config_schema: Mapping[str, Any]
    aliases: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    version: str = "1"
    source_path: str | None = None

    def __post_init__(self) -> None:
        plugin_id = str(self.plugin_id or "").strip()
        kind = str(self.kind or "").strip().lower()
        name = str(self.name or "").strip()
        label = str(self.label or "").strip()
        module = str(self.module or "").strip()
        adapter_class = str(self.adapter_class or "").strip()
        version = str(self.version or "").strip()

        if kind not in {"sampler", "scheduler"}:
            raise PluginDescriptorError(
                f"Plugin {plugin_id or '<unknown>'!r} kind must be 'sampler' or 'scheduler'."
            )
        if not _STABLE_ID.fullmatch(plugin_id):
            raise PluginDescriptorError(
                f"Plugin ID {plugin_id!r} must match {_STABLE_ID.pattern}."
            )
        if not plugin_id.startswith(kind + "."):
            raise PluginDescriptorError(
                f"Plugin ID {plugin_id!r} must start with '{kind}.'."
            )
        if not _CANONICAL_NAME.fullmatch(name):
            raise PluginDescriptorError(
                f"Plugin {plugin_id!r} name {name!r} must be lowercase and filesystem-safe."
            )
        if not label:
            raise PluginDescriptorError(f"Plugin {plugin_id!r} label cannot be empty.")
        if not _DOTTED_MODULE.fullmatch(module):
            raise PluginDescriptorError(
                f"Plugin {plugin_id!r} module {module!r} is not a valid dotted module path."
            )
        if not _CLASS_NAME.fullmatch(adapter_class):
            raise PluginDescriptorError(
                f"Plugin {plugin_id!r} adapter_class {adapter_class!r} is not a valid class name."
            )
        if not version:
            raise PluginDescriptorError(f"Plugin {plugin_id!r} version cannot be empty.")
        if not isinstance(self.capabilities, Mapping):
            raise PluginDescriptorError(
                f"Plugin {plugin_id!r} capabilities must be a mapping."
            )
        if not self.capabilities:
            raise PluginDescriptorError(
                f"Plugin {plugin_id!r} capabilities cannot be empty."
            )
        if not isinstance(self.config_schema, Mapping):
            raise PluginDescriptorError(
                f"Plugin {plugin_id!r} config_schema must be a mapping."
            )

        aliases = _normalize_aliases(self.aliases)
        own_tokens = {
            normalize_identity(plugin_id),
            normalize_identity(name),
            normalize_identity(label),
        }
        aliases = tuple(alias for alias in aliases if normalize_identity(alias) not in own_tokens)
        schema = validate_config_schema(self.config_schema, plugin_id=plugin_id)

        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "adapter_class", adapter_class)
        object.__setattr__(self, "capabilities", MappingProxyType(_copy_mapping(self.capabilities)))
        object.__setattr__(self, "config_schema", MappingProxyType(schema))
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "version", version)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_kind: PluginKind | None = None,
        default_module: str | None = None,
        source_path: str | None = None,
    ) -> "PluginDescriptor":
        if not isinstance(value, Mapping):
            raise PluginDescriptorError("PLUGIN_DESCRIPTOR must be a mapping or PluginDescriptor.")
        payload = dict(value)
        if default_kind is not None:
            payload.setdefault("kind", default_kind)
        if default_module is not None:
            payload.setdefault("module", default_module)
        if source_path is not None:
            payload.setdefault("source_path", source_path)
        missing = [
            field_name
            for field_name in (
                "plugin_id",
                "kind",
                "name",
                "label",
                "module",
                "adapter_class",
                "capabilities",
                "config_schema",
            )
            if field_name not in payload
        ]
        if missing:
            identity = payload.get("plugin_id") or payload.get("name") or default_module or "<unknown>"
            raise PluginDescriptorError(
                f"Plugin descriptor {identity!r} is missing required field(s): "
                + ", ".join(missing)
            )
        return cls(
            plugin_id=payload["plugin_id"],
            kind=payload["kind"],
            name=payload["name"],
            label=payload["label"],
            module=payload["module"],
            adapter_class=payload["adapter_class"],
            capabilities=payload["capabilities"],
            config_schema=payload["config_schema"],
            aliases=tuple(payload.get("aliases") or ()),
            description=payload.get("description", ""),
            version=payload.get("version", "1"),
            source_path=payload.get("source_path"),
        )

    @property
    def identities(self) -> tuple[tuple[str, str], ...]:
        values: list[tuple[str, str]] = [
            ("plugin_id", self.plugin_id),
            ("name", self.name),
            ("label", self.label),
        ]
        values.extend(("alias", alias) for alias in self.aliases)
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "kind": self.kind,
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "version": self.version,
            "module": self.module,
            "adapter_class": self.adapter_class,
            "aliases": list(self.aliases),
            "capabilities": dict(self.capabilities),
            "config_schema": dict(self.config_schema),
            "source_path": self.source_path,
        }

    def to_legacy_entry(self) -> dict[str, Any]:
        """Return the transitional map shape used by the current CLI and manifests."""
        entry = self.to_dict()
        entry.update(
            {
                "id": self.plugin_id,
                "preferred_entry": "adapter",
                "entry_type": "class",
                "entry_class": self.adapter_class,
                "source_type": "plugin_descriptor",
            }
        )
        return entry
