from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


_VALID_MODES = {"same_as_base", "scale_from_base", "explicit_dimensions"}
_SCALE_TOLERANCE = 1e-6
HIRES_DIMENSION_PLAN_VERSION = "phase14n12-dimension-plan-v2"


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(math.floor(float(value) + 0.5))
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


def _clamp_dimension(value: int) -> int:
    return max(64, min(16384, int(math.floor(float(value) + 0.5))))


def _normalize_base_dimension(value: int, *, multiple: int = 8) -> int:
    value = _clamp_dimension(value)
    if multiple <= 1:
        return value
    return max(multiple, int(math.floor((value / multiple) + 0.5)) * multiple)


def _align_dimension(value: int, *, multiple: int = 8) -> int:
    value = _clamp_dimension(value)
    if multiple <= 1:
        return value
    return min(16384, max(multiple, int(math.ceil(value / multiple)) * multiple))


@dataclass(frozen=True)
class HiresDimensionPlan:
    contract_version: str
    mode: str
    base_width: int
    base_height: int
    requested_scale: float | None
    requested_width: int
    requested_height: int
    internal_width: int
    internal_height: int
    final_width: int
    final_height: int
    effective_width: int
    effective_height: int
    effective_scale_x: float
    effective_scale_y: float
    axis_scale_width: float
    axis_scale_height: float
    uniform_scale: float | None
    is_uniform_scale: bool
    aspect_ratio_changed: bool
    alignment_applied: bool
    alignment_correction_required: bool
    dimension_multiple: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_hires_dimensions(
    values: Mapping[str, Any] | Any,
    *,
    dimension_multiple: int = 8,
) -> HiresDimensionPlan:
    source: Mapping[str, Any]
    if isinstance(values, Mapping):
        source = values
    else:
        source = vars(values)

    dimension_multiple = max(1, int(dimension_multiple or 8))
    # Keep the historical base-dimension normalization on the VAE's 8-pixel
    # grid. The architecture-specific multiple applies to the *internal*
    # second-pass canvas; it must not silently change the user's requested
    # base size or a scale-from-base target (for example 360 -> 368 -> 552).
    base_width = _normalize_base_dimension(_positive_int(source.get("width"), 512), multiple=8)
    base_height = _normalize_base_dimension(_positive_int(source.get("height"), 512), multiple=8)
    mode = str(source.get("hires_size_mode") or "same_as_base").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            "hires_size_mode must be one of: same_as_base, scale_from_base, explicit_dimensions."
        )

    requested_scale_input = max(1.0, min(8.0, _positive_float(source.get("hires_scale"), 2.0)))
    requested_width_input = _positive_int(source.get("hires_width"), 0)
    requested_height_input = _positive_int(source.get("hires_height"), 0)

    if mode == "same_as_base":
        requested_width, requested_height = base_width, base_height
    elif mode == "scale_from_base":
        requested_width = _clamp_dimension(base_width * requested_scale_input)
        requested_height = _clamp_dimension(base_height * requested_scale_input)
    else:
        if requested_width_input <= 0 or requested_height_input <= 0:
            raise ValueError(
                "hires_width and hires_height must both be positive when hires_size_mode is explicit_dimensions."
            )
        requested_width = _clamp_dimension(requested_width_input)
        requested_height = _clamp_dimension(requested_height_input)

    internal_width = _align_dimension(requested_width, multiple=dimension_multiple)
    internal_height = _align_dimension(requested_height, multiple=dimension_multiple)
    axis_scale_width = round(requested_width / base_width, 6)
    axis_scale_height = round(requested_height / base_height, 6)
    is_uniform_scale = abs(axis_scale_width - axis_scale_height) <= _SCALE_TOLERANCE
    uniform_scale = round((axis_scale_width + axis_scale_height) / 2.0, 6) if is_uniform_scale else None
    aspect_ratio_changed = abs((requested_width / requested_height) - (base_width / base_height)) > _SCALE_TOLERANCE
    alignment_applied = internal_width != requested_width or internal_height != requested_height

    return HiresDimensionPlan(
        contract_version=HIRES_DIMENSION_PLAN_VERSION,
        mode=mode,
        base_width=base_width,
        base_height=base_height,
        requested_scale=requested_scale_input if mode == "scale_from_base" else uniform_scale,
        requested_width=requested_width,
        requested_height=requested_height,
        internal_width=internal_width,
        internal_height=internal_height,
        final_width=requested_width,
        final_height=requested_height,
        # Legacy runtime readers still use effective_* as the aligned second-pass dimensions.
        effective_width=internal_width,
        effective_height=internal_height,
        effective_scale_x=axis_scale_width,
        effective_scale_y=axis_scale_height,
        axis_scale_width=axis_scale_width,
        axis_scale_height=axis_scale_height,
        uniform_scale=uniform_scale,
        is_uniform_scale=is_uniform_scale,
        aspect_ratio_changed=aspect_ratio_changed,
        alignment_applied=alignment_applied,
        alignment_correction_required=alignment_applied,
        dimension_multiple=dimension_multiple,
    )


def apply_hires_dimensions(payload: dict[str, Any]) -> dict[str, Any]:
    plan = resolve_hires_dimensions(payload)
    payload["hires_size_mode"] = plan.mode
    payload["hires_scale"] = plan.requested_scale
    payload["hires_width"] = plan.requested_width
    payload["hires_height"] = plan.requested_height
    payload["hires_axis_scale_width"] = plan.axis_scale_width
    payload["hires_axis_scale_height"] = plan.axis_scale_height
    payload["hires_uniform_scale"] = plan.uniform_scale
    payload["hires_aspect_ratio_changed"] = plan.aspect_ratio_changed
    payload["hires_dimension_plan_version"] = plan.contract_version
    payload["hires_dimension_plan"] = plan.to_dict()
    return payload


__all__ = ["HIRES_DIMENSION_PLAN_VERSION", "HiresDimensionPlan", "apply_hires_dimensions", "resolve_hires_dimensions"]
