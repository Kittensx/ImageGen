from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


_VALID_MODES = {"same_as_base", "scale_from_base", "explicit_dimensions"}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


def _normalize_dimension(value: int, *, multiple: int = 8) -> int:
    value = max(64, min(16384, int(value)))
    if multiple <= 1:
        return value
    return max(multiple, int(round(value / multiple)) * multiple)


@dataclass(frozen=True)
class HiresDimensionPlan:
    mode: str
    base_width: int
    base_height: int
    requested_scale: float
    requested_width: int
    requested_height: int
    effective_width: int
    effective_height: int
    effective_scale_x: float
    effective_scale_y: float
    dimension_multiple: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_hires_dimensions(values: Mapping[str, Any] | Any) -> HiresDimensionPlan:
    source: Mapping[str, Any]
    if isinstance(values, Mapping):
        source = values
    else:
        source = vars(values)
    base_width = _normalize_dimension(_positive_int(source.get("width"), 512))
    base_height = _normalize_dimension(_positive_int(source.get("height"), 512))
    mode = str(source.get("hires_size_mode") or "same_as_base").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            "hires_size_mode must be one of: same_as_base, scale_from_base, explicit_dimensions."
        )
    scale = max(1.0, min(8.0, _positive_float(source.get("hires_scale"), 2.0)))
    requested_width = _positive_int(source.get("hires_width"), 0)
    requested_height = _positive_int(source.get("hires_height"), 0)
    if mode == "same_as_base":
        effective_width, effective_height = base_width, base_height
    elif mode == "scale_from_base":
        effective_width = _normalize_dimension(int(round(base_width * scale)))
        effective_height = _normalize_dimension(int(round(base_height * scale)))
    else:
        if requested_width <= 0 or requested_height <= 0:
            raise ValueError(
                "hires_width and hires_height must both be positive when hires_size_mode is explicit_dimensions."
            )
        effective_width = _normalize_dimension(requested_width)
        effective_height = _normalize_dimension(requested_height)
    return HiresDimensionPlan(
        mode=mode,
        base_width=base_width,
        base_height=base_height,
        requested_scale=scale,
        requested_width=requested_width,
        requested_height=requested_height,
        effective_width=effective_width,
        effective_height=effective_height,
        effective_scale_x=round(effective_width / base_width, 6),
        effective_scale_y=round(effective_height / base_height, 6),
    )


def apply_hires_dimensions(payload: dict[str, Any]) -> dict[str, Any]:
    plan = resolve_hires_dimensions(payload)
    payload["hires_size_mode"] = plan.mode
    payload["hires_scale"] = plan.requested_scale
    payload["hires_width"] = plan.requested_width
    payload["hires_height"] = plan.requested_height
    payload["hires_dimension_plan"] = plan.to_dict()
    return payload


__all__ = ["HiresDimensionPlan", "apply_hires_dimensions", "resolve_hires_dimensions"]
