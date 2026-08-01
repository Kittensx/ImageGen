from __future__ import annotations

from typing import Any, Iterator

from .reports import json_safe, stable_hash


def is_attention_module(module: Any) -> bool:
    return all(hasattr(module, name) for name in ("to_q", "to_k", "to_v", "heads"))


def iter_attention_modules(unet: Any) -> Iterator[tuple[str, Any]]:
    named_modules = getattr(unet, "named_modules", None)
    if not callable(named_modules):
        return
    for path, module in named_modules():
        if path and is_attention_module(module):
            yield str(path), module


def _int_attr(value: Any, name: str) -> int | None:
    result = getattr(value, name, None)
    if isinstance(result, bool):
        return None
    if isinstance(result, int):
        return int(result)
    return None


def _projection_metadata(projection: Any) -> dict[str, int | None]:
    return {
        "in_features": _int_attr(projection, "in_features"),
        "out_features": _int_attr(projection, "out_features"),
    }


def _head_dimension(out_features: int | None, heads: int | None) -> tuple[int | None, str | None]:
    if out_features is None:
        return None, "projection out_features is unavailable"
    if heads is None or heads <= 0:
        return None, "attention head count is unavailable or invalid"
    quotient, remainder = divmod(out_features, heads)
    if remainder:
        return None, f"projection dimension {out_features} is not divisible by {heads} heads"
    return quotient, None


def _processor_path(module: Any) -> str | None:
    processor = getattr(module, "processor", None)
    if processor is None:
        return None
    return f"{type(processor).__module__}.{type(processor).__name__}"


def _classify_attention(path: str, module: Any, q_in: int | None, k_in: int | None) -> str:
    lowered = path.lower()
    if lowered.endswith("attn1") or ".attn1." in lowered:
        return "self"
    if lowered.endswith("attn2") or ".attn2." in lowered:
        return "cross"
    if bool(getattr(module, "only_cross_attention", False)):
        return "cross"
    if q_in is not None and k_in is not None and q_in != k_in:
        return "cross"
    return "self_or_cross"


def attention_module_record(path: str, module: Any) -> dict[str, Any]:
    heads = _int_attr(module, "heads")
    q = _projection_metadata(getattr(module, "to_q", None))
    k = _projection_metadata(getattr(module, "to_k", None))
    v = _projection_metadata(getattr(module, "to_v", None))
    q_head_dim, q_error = _head_dimension(q["out_features"], heads)
    k_head_dim, k_error = _head_dimension(k["out_features"], heads)
    v_head_dim, v_error = _head_dimension(v["out_features"], heads)
    errors = [item for item in (q_error, k_error, v_error) if item]
    kind = _classify_attention(path, module, q["in_features"], k["in_features"])
    cross_attention_dim = getattr(module, "cross_attention_dim", None)
    if not isinstance(cross_attention_dim, (int, float, str, list, tuple, type(None))):
        cross_attention_dim = str(cross_attention_dim)
    return {
        "module_path": path,
        "module_class": f"{type(module).__module__}.{type(module).__name__}",
        "processor_path": _processor_path(module),
        "attention_kind": kind,
        "heads": heads,
        "to_q": q,
        "to_k": k,
        "to_v": v,
        "q_head_dim": q_head_dim,
        "k_head_dim": k_head_dim,
        "v_head_dim": v_head_dim,
        "cross_attention_dim": json_safe(cross_attention_dim),
        "valid": not errors,
        "errors": errors,
    }


def _model_identity(unet: Any, explicit_identity: str | None) -> dict[str, Any]:
    config = getattr(unet, "config", None)
    selected: dict[str, Any] = {
        "explicit_identity": explicit_identity,
        "class": f"{type(unet).__module__}.{type(unet).__name__}",
    }
    for name in (
        "_name_or_path",
        "model_type",
        "sample_size",
        "in_channels",
        "out_channels",
        "cross_attention_dim",
        "block_out_channels",
        "down_block_types",
        "up_block_types",
    ):
        value = getattr(config, name, None) if config is not None else None
        if value is not None:
            selected[name] = json_safe(value)
    return selected


def build_model_attention_signature(
    unet: Any,
    *,
    model_identity: str | None = None,
) -> dict[str, Any]:
    modules = [attention_module_record(path, module) for path, module in iter_attention_modules(unet)]
    unique_layout_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    unique_head_dimensions: set[int] = set()
    validation_errors: list[dict[str, Any]] = []
    for record in modules:
        for value in (record["q_head_dim"], record["k_head_dim"], record["v_head_dim"]):
            if isinstance(value, int):
                unique_head_dimensions.add(value)
        key = (
            record["attention_kind"],
            record["heads"],
            record["q_head_dim"],
            record["k_head_dim"],
            record["v_head_dim"],
        )
        unique_layout_map.setdefault(
            key,
            {
                "attention_kind": record["attention_kind"],
                "heads": record["heads"],
                "q_head_dim": record["q_head_dim"],
                "k_head_dim": record["k_head_dim"],
                "v_head_dim": record["v_head_dim"],
                "module_paths": [],
            },
        )["module_paths"].append(record["module_path"])
        if record["errors"]:
            validation_errors.append(
                {
                    "module_path": record["module_path"],
                    "errors": list(record["errors"]),
                }
            )

    signature: dict[str, Any] = {
        "schema_version": 1,
        "model_identity": _model_identity(unet, model_identity),
        "architecture": type(unet).__name__,
        "attention_module_count": len(modules),
        "modules": modules,
        "unique_layouts": sorted(
            unique_layout_map.values(),
            key=lambda item: (
                str(item["attention_kind"]),
                int(item["q_head_dim"] or -1),
                int(item["heads"] or -1),
            ),
        ),
        "unique_head_dimensions": sorted(unique_head_dimensions),
        "validation_errors": validation_errors,
    }
    signature["signature_hash"] = stable_hash(signature)
    return signature
