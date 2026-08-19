from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


_LINKED_SCHEDULER_FIELDS = {"steps", "device"}


def _normalized_type_name(raw: Any) -> str:
    token = raw
    if isinstance(raw, (list, tuple)):
        token = next((item for item in raw if str(item).lower() != "null"), None)
    text = str(token or "string").strip().lower()
    if text in {"int", "integer", "optional[int]", "optional[integer]"}:
        return "integer"
    if text in {"float", "double", "number", "optional[float]", "optional[number]", "optional[double]"}:
        return "number"
    if text in {"bool", "boolean", "optional[bool]", "optional[boolean]"}:
        return "boolean"
    if text in {"object", "dict", "mapping"}:
        return "object"
    if text in {"array", "list", "tuple"}:
        return "array"
    return "string"


def normalize_property_schema(name: str, schema: Mapping[str, Any] | None, *, kind: str | None = None) -> dict[str, Any]:
    source = dict(schema or {})
    output = deepcopy(source)

    output_type = _normalized_type_name(source.get("type"))
    output["type"] = output_type

    choices = source.get("enum")
    if not isinstance(choices, list):
        choices = source.get("choices")
    if isinstance(choices, (list, tuple)):
        output["enum"] = list(choices)

    range_value = source.get("range")
    if isinstance(range_value, (list, tuple)) and len(range_value) >= 2:
        minimum, maximum = range_value[0], range_value[1]
        if minimum not in (None, ""):
            output["minimum"] = minimum
        if maximum not in (None, ""):
            output["maximum"] = maximum

    if "short_desc" in source and "title" not in output:
        output["title"] = source.get("short_desc")
    if "long_desc" in source and "description" not in output:
        output["description"] = source.get("long_desc")

    if output_type == "object":
        properties = source.get("properties") if isinstance(source.get("properties"), Mapping) else {}
        output["properties"] = {
            str(child_name): normalize_property_schema(
                str(child_name),
                child_schema if isinstance(child_schema, Mapping) else {},
                kind=kind,
            )
            for child_name, child_schema in properties.items()
        }
    elif output_type == "array":
        items = source.get("items") if isinstance(source.get("items"), Mapping) else {}
        output["items"] = normalize_property_schema("item", items, kind=kind)

    if kind == "scheduler" and name in _LINKED_SCHEDULER_FIELDS:
        output["x_linked"] = True

    # WebUI should not submit unchanged advanced values back into the runtime.
    output["x_omit_if_default"] = True
    return output


def normalize_config_schema(schema: Mapping[str, Any] | None, *, kind: str | None = None) -> dict[str, Any]:
    payload = dict(schema or {})
    properties = payload.get("properties") if isinstance(payload.get("properties"), Mapping) else {}
    normalized_properties = {
        str(name): normalize_property_schema(str(name), spec if isinstance(spec, Mapping) else {}, kind=kind)
        for name, spec in properties.items()
    }
    return {
        "type": "object",
        "properties": normalized_properties,
        "required": list(payload.get("required") or []),
        "additionalProperties": bool(payload.get("additionalProperties", False)),
    }


def _scope_profile_value(value: Any, schema: Mapping[str, Any] | None) -> Any:
    """Return only values explicitly owned by a profile schema branch."""

    spec = normalize_property_schema("", schema or {})
    kind = spec.get("type", "string")
    if kind == "object":
        if not isinstance(value, Mapping):
            return {}
        properties = spec.get("properties") if isinstance(spec.get("properties"), Mapping) else {}
        return {
            str(name): _scope_profile_value(value[name], child_schema)
            for name, child_schema in properties.items()
            if name in value
        }
    if kind == "array":
        if not isinstance(value, (list, tuple)):
            value = [value]
        item_schema = spec.get("items") if isinstance(spec.get("items"), Mapping) else {}
        return [_scope_profile_value(item, item_schema) for item in value]
    return coerce_value_by_schema(value, spec)


def scope_plugin_profile_values(
    values: Mapping[str, Any] | None,
    schema: Mapping[str, Any] | None,
    *,
    kind: str,
) -> dict[str, Any]:
    """Restrict a named advanced profile to fields explicitly owned by its plugin.

    Runtime schemas may accept additional properties for adapter compatibility, but
    a saved advanced profile is intentionally narrower. Scheduler fields linked to
    the main generation form (currently ``steps`` and ``device``) are also excluded
    so a scheduler profile cannot capture unrelated generation state.
    """

    incoming = dict(values or {})
    normalized = normalize_config_schema(schema, kind=kind)
    properties = normalized.get("properties") if isinstance(normalized.get("properties"), Mapping) else {}
    scoped: dict[str, Any] = {}
    for name, field_schema in properties.items():
        if name not in incoming:
            continue
        if isinstance(field_schema, Mapping) and field_schema.get("x_linked"):
            continue
        scoped[str(name)] = _scope_profile_value(incoming[name], field_schema)
    return scoped

def coerce_value_by_schema(value: Any, schema: Mapping[str, Any] | None) -> Any:
    spec = normalize_property_schema("", schema or {})
    kind = spec.get("type", "string")
    if value is None:
        return None
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
        return bool(value)
    if kind == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        text = str(value).strip()
        if text == "":
            return None
        return int(float(text))
    if kind == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    if kind == "array":
        if isinstance(value, (list, tuple)):
            return [coerce_value_by_schema(item, spec.get("items") or {}) for item in value]
        return [coerce_value_by_schema(value, spec.get("items") or {})]
    if kind == "object":
        if not isinstance(value, Mapping):
            return {}
        properties = spec.get("properties") or {}
        return {
            str(key): coerce_value_by_schema(item, properties.get(str(key)) or {})
            for key, item in value.items()
        }
    if isinstance(value, str):
        return value
    return str(value)
