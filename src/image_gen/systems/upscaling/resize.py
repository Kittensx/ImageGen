from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

SUPPORTED_EXACT_RESIZE_FILTERS = frozenset({"nearest", "bilinear", "bicubic", "area"})
SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS = frozenset({"auto", *SUPPORTED_EXACT_RESIZE_FILTERS})
SUPPORTED_ASPECT_POLICIES = frozenset({"stretch", "crop_to_fill", "pad_to_fit"})
SUPPORTED_PADDING_MODES = frozenset({"reflect", "replicate", "blurred_edge", "black"})
TARGET_CORRECTION_CONTRACT_VERSION = "phase14n12b-target-correction-v1"


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def resolve_target_dimensions(
    *,
    source_width: int,
    source_height: int,
    target_width: int | None,
    target_height: int | None,
    scale: float | None,
) -> tuple[int, int]:
    width = int(source_width)
    height = int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError("Source dimensions must be positive.")
    requested_width = int(target_width or 0)
    requested_height = int(target_height or 0)
    if requested_width > 0 and requested_height > 0:
        return requested_width, requested_height
    if requested_width > 0:
        return requested_width, max(1, _round_half_up(height * (requested_width / width)))
    if requested_height > 0:
        return max(1, _round_half_up(width * (requested_height / height))), requested_height
    if scale is None or float(scale) <= 0:
        raise ValueError("Specify target dimensions or a positive scale.")
    factor = float(scale)
    return max(1, _round_half_up(width * factor)), max(1, _round_half_up(height * factor))


def compute_native_output_dimensions(
    *,
    source_width: int,
    source_height: int,
    native_scale: int,
) -> tuple[int, int]:
    return max(1, int(source_width) * int(native_scale)), max(1, int(source_height) * int(native_scale))


def resize_exact(
    images: torch.Tensor,
    *,
    target_width: int,
    target_height: int,
    resize_filter: str = "bicubic",
) -> torch.Tensor:
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("Exact resize requires a BCHW tensor.")
    width = int(target_width)
    height = int(target_height)
    if width <= 0 or height <= 0:
        raise ValueError("Exact resize dimensions must be positive.")
    if tuple(images.shape[-2:]) == (height, width):
        return images
    mode = str(resize_filter or "bicubic").strip().casefold()
    if mode not in SUPPORTED_EXACT_RESIZE_FILTERS:
        raise ValueError(
            f"Unsupported exact resize filter {resize_filter!r}; expected one of {sorted(SUPPORTED_EXACT_RESIZE_FILTERS)}."
        )
    kwargs: dict[str, Any] = {"size": (height, width), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    original_dtype = images.dtype
    source = images
    if images.device.type == "cpu" and images.dtype in {torch.float16, torch.bfloat16}:
        source = images.float()
    resized = F.interpolate(source, **kwargs)
    if resized.dtype != original_dtype:
        resized = resized.to(dtype=original_dtype)
    return resized.clamp(0.0, 1.0)


def _resolve_filter_for_scale(requested_filter: str, scale: float) -> str | None:
    selected = str(requested_filter or "auto").strip().casefold()
    if selected not in SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS:
        raise ValueError(
            "Final size correction filter must be auto, nearest, bilinear, bicubic, or area."
        )
    if abs(float(scale) - 1.0) <= 1e-12:
        return None
    if selected != "auto":
        return selected
    return "area" if float(scale) < 1.0 else "bicubic"


def _resolve_stretch_filter(
    requested_filter: str,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> str | None:
    selected = str(requested_filter or "auto").strip().casefold()
    if selected not in SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS:
        raise ValueError(
            "Final size correction filter must be auto, nearest, bilinear, bicubic, or area."
        )
    if source_width == target_width and source_height == target_height:
        return None
    if selected != "auto":
        return selected
    shrinking_or_equal = source_width >= target_width and source_height >= target_height
    enlarging_or_equal = source_width <= target_width and source_height <= target_height
    if shrinking_or_equal and not enlarging_or_equal:
        return "area"
    # Mixed-axis correction contains an enlargement axis, for which bicubic is
    # safer than area. Pure enlargement follows the same automatic default.
    return "bicubic"


@dataclass(frozen=True)
class TargetCorrectionPlan:
    contract_version: str
    aspect_policy: str
    padding_mode: str
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    source_aspect_ratio: float
    target_aspect_ratio: float
    aspect_ratio_changed: bool
    correction_required: bool
    final_size_correction_filter_requested: str
    final_size_correction_filter_resolved: str
    non_uniform_geometry_applied: bool
    resize_change_ratio: float = 0.0
    canvas_change_fraction: float = 0.0
    correction_severity: float = 0.0
    resize_scale: float | None = None
    pre_crop_width: int = 0
    pre_crop_height: int = 0
    crop_left: int = 0
    crop_top: int = 0
    crop_right: int = 0
    crop_bottom: int = 0
    fitted_width: int = 0
    fitted_height: int = 0
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_target_correction(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    aspect_policy: str = "stretch",
    final_size_correction_filter: str = "auto",
    padding_mode: str = "reflect",
) -> TargetCorrectionPlan:
    source_width = int(source_width)
    source_height = int(source_height)
    target_width = int(target_width)
    target_height = int(target_height)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Source and target dimensions must be positive.")

    policy = str(aspect_policy or "stretch").strip().casefold()
    if policy not in SUPPORTED_ASPECT_POLICIES:
        raise ValueError(
            f"Unsupported aspect policy {aspect_policy!r}; expected one of {sorted(SUPPORTED_ASPECT_POLICIES)}."
        )
    pad_mode = str(padding_mode or "reflect").strip().casefold()
    if pad_mode not in SUPPORTED_PADDING_MODES:
        raise ValueError(
            f"Unsupported padding mode {padding_mode!r}; expected one of {sorted(SUPPORTED_PADDING_MODES)}."
        )
    requested_filter = str(final_size_correction_filter or "auto").strip().casefold()
    if requested_filter not in SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS:
        raise ValueError(
            "Final size correction filter must be auto, nearest, bilinear, bicubic, or area."
        )

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    aspect_changed = abs(source_aspect - target_aspect) > 1e-12
    exact_match = source_width == target_width and source_height == target_height

    if policy == "stretch":
        resolved = _resolve_stretch_filter(
            requested_filter,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
        )
        return TargetCorrectionPlan(
            contract_version=TARGET_CORRECTION_CONTRACT_VERSION,
            aspect_policy=policy,
            padding_mode=pad_mode,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            source_aspect_ratio=source_aspect,
            target_aspect_ratio=target_aspect,
            aspect_ratio_changed=aspect_changed,
            correction_required=not exact_match,
            final_size_correction_filter_requested=requested_filter,
            final_size_correction_filter_resolved=resolved or "none",
            non_uniform_geometry_applied=bool(aspect_changed and not exact_match),
            resize_change_ratio=max(
                abs((target_width / source_width) - 1.0),
                abs((target_height / source_height) - 1.0),
            ),
            canvas_change_fraction=(
                abs((target_width / target_height) - (source_width / source_height))
                / max(source_width / source_height, 1e-12)
                if aspect_changed else 0.0
            ),
            correction_severity=max(
                max(abs((target_width / source_width) - 1.0), abs((target_height / source_height) - 1.0)),
                (abs((target_width / target_height) - (source_width / source_height)) / max(source_width / source_height, 1e-12)) if aspect_changed else 0.0,
            ),
        )

    if policy == "crop_to_fill":
        scale = max(target_width / source_width, target_height / source_height)
        pre_width = max(target_width, int(math.ceil((source_width * scale) - 1e-12)))
        pre_height = max(target_height, int(math.ceil((source_height * scale) - 1e-12)))
        overflow_width = max(0, pre_width - target_width)
        overflow_height = max(0, pre_height - target_height)
        crop_left = overflow_width // 2
        crop_top = overflow_height // 2
        crop_right = crop_left + target_width
        crop_bottom = crop_top + target_height
        resolved = _resolve_filter_for_scale(requested_filter, scale)
        return TargetCorrectionPlan(
            contract_version=TARGET_CORRECTION_CONTRACT_VERSION,
            aspect_policy=policy,
            padding_mode=pad_mode,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            source_aspect_ratio=source_aspect,
            target_aspect_ratio=target_aspect,
            aspect_ratio_changed=aspect_changed,
            correction_required=(not exact_match) or overflow_width > 0 or overflow_height > 0,
            final_size_correction_filter_requested=requested_filter,
            final_size_correction_filter_resolved=resolved or "none",
            non_uniform_geometry_applied=False,
            resize_change_ratio=abs(float(scale) - 1.0),
            canvas_change_fraction=(1.0 - ((target_width * target_height) / max(1, pre_width * pre_height))),
            correction_severity=max(abs(float(scale) - 1.0), 1.0 - ((target_width * target_height) / max(1, pre_width * pre_height))),
            resize_scale=float(scale),
            pre_crop_width=pre_width,
            pre_crop_height=pre_height,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
        )

    scale = min(target_width / source_width, target_height / source_height)
    fitted_width = min(target_width, max(1, int(math.floor((source_width * scale) + 1e-12))))
    fitted_height = min(target_height, max(1, int(math.floor((source_height * scale) + 1e-12))))
    remaining_width = max(0, target_width - fitted_width)
    remaining_height = max(0, target_height - fitted_height)
    pad_left = remaining_width // 2
    pad_right = remaining_width - pad_left
    pad_top = remaining_height // 2
    pad_bottom = remaining_height - pad_top
    resolved = _resolve_filter_for_scale(requested_filter, scale)
    return TargetCorrectionPlan(
        contract_version=TARGET_CORRECTION_CONTRACT_VERSION,
        aspect_policy=policy,
        padding_mode=pad_mode,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        source_aspect_ratio=source_aspect,
        target_aspect_ratio=target_aspect,
        aspect_ratio_changed=aspect_changed,
        correction_required=(not exact_match) or remaining_width > 0 or remaining_height > 0,
        final_size_correction_filter_requested=requested_filter,
        final_size_correction_filter_resolved=resolved or "none",
        non_uniform_geometry_applied=False,
        resize_change_ratio=abs(float(scale) - 1.0),
        canvas_change_fraction=(1.0 - ((fitted_width * fitted_height) / max(1, target_width * target_height))),
        correction_severity=max(abs(float(scale) - 1.0), 1.0 - ((fitted_width * fitted_height) / max(1, target_width * target_height))),
        resize_scale=float(scale),
        fitted_width=fitted_width,
        fitted_height=fitted_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def _reflect_indices(length: int, before: int, after: int, *, device: torch.device) -> torch.Tensor:
    length = int(length)
    before = max(0, int(before))
    after = max(0, int(after))
    if length <= 1:
        return torch.zeros(length + before + after, dtype=torch.long, device=device)
    positions = torch.arange(-before, length + after, dtype=torch.long, device=device)
    period = 2 * (length - 1)
    folded = torch.remainder(positions, period)
    return torch.where(folded < length, folded, period - folded)


def _pad_reflect_unbounded(
    images: torch.Tensor,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> torch.Tensor:
    y_index = _reflect_indices(int(images.shape[-2]), top, bottom, device=images.device)
    x_index = _reflect_indices(int(images.shape[-1]), left, right, device=images.device)
    return images.index_select(-2, y_index).index_select(-1, x_index)


def _pad_blurred_edge(
    images: torch.Tensor,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> torch.Tensor:
    padded = F.pad(images, (left, right, top, bottom), mode="replicate")
    if not any((left, right, top, bottom)):
        return padded
    max_pad = max(left, right, top, bottom)
    kernel = min(31, max(3, (max_pad // 4) * 2 + 1))
    radius = kernel // 2
    blurred_source = F.pad(padded, (radius, radius, radius, radius), mode="replicate")
    blurred = F.avg_pool2d(blurred_source, kernel_size=kernel, stride=1)
    blurred[..., top : top + images.shape[-2], left : left + images.shape[-1]] = images
    return blurred.clamp(0.0, 1.0)


def apply_target_correction(
    images: torch.Tensor,
    *,
    target_width: int,
    target_height: int,
    aspect_policy: str = "stretch",
    final_size_correction_filter: str = "auto",
    padding_mode: str = "reflect",
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("Target correction requires a BCHW tensor.")
    source_height = int(images.shape[-2])
    source_width = int(images.shape[-1])
    plan = plan_target_correction(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        aspect_policy=aspect_policy,
        final_size_correction_filter=final_size_correction_filter,
        padding_mode=padding_mode,
    )
    resolved_filter = plan.final_size_correction_filter_resolved

    if plan.aspect_policy == "stretch":
        corrected = images if resolved_filter == "none" else resize_exact(
            images,
            target_width=plan.target_width,
            target_height=plan.target_height,
            resize_filter=resolved_filter,
        )
    elif plan.aspect_policy == "crop_to_fill":
        resized = images if (
            plan.pre_crop_width == source_width and plan.pre_crop_height == source_height
        ) else resize_exact(
            images,
            target_width=plan.pre_crop_width,
            target_height=plan.pre_crop_height,
            resize_filter=resolved_filter if resolved_filter != "none" else "bicubic",
        )
        corrected = resized[
            ...,
            plan.crop_top : plan.crop_bottom,
            plan.crop_left : plan.crop_right,
        ]
    else:
        resized = images if (
            plan.fitted_width == source_width and plan.fitted_height == source_height
        ) else resize_exact(
            images,
            target_width=plan.fitted_width,
            target_height=plan.fitted_height,
            resize_filter=resolved_filter if resolved_filter != "none" else "bicubic",
        )
        if plan.padding_mode == "reflect":
            corrected = _pad_reflect_unbounded(
                resized,
                left=plan.pad_left,
                right=plan.pad_right,
                top=plan.pad_top,
                bottom=plan.pad_bottom,
            )
        elif plan.padding_mode == "replicate":
            corrected = F.pad(
                resized,
                (plan.pad_left, plan.pad_right, plan.pad_top, plan.pad_bottom),
                mode="replicate",
            )
        elif plan.padding_mode == "black":
            corrected = F.pad(
                resized,
                (plan.pad_left, plan.pad_right, plan.pad_top, plan.pad_bottom),
                mode="constant",
                value=0.0,
            )
        else:
            corrected = _pad_blurred_edge(
                resized,
                left=plan.pad_left,
                right=plan.pad_right,
                top=plan.pad_top,
                bottom=plan.pad_bottom,
            )

    if tuple(corrected.shape[-2:]) != (plan.target_height, plan.target_width):
        raise RuntimeError(
            "Target correction did not produce the exact requested correction canvas: "
            f"expected {(plan.target_height, plan.target_width)}, got {tuple(corrected.shape[-2:])}."
        )
    metadata = plan.to_dict()
    metadata["output_width"] = int(corrected.shape[-1])
    metadata["output_height"] = int(corrected.shape[-2])
    return corrected.clamp(0.0, 1.0), metadata


resize_to_exact_dimensions = resize_exact

__all__ = [
    "SUPPORTED_ASPECT_POLICIES",
    "SUPPORTED_EXACT_RESIZE_FILTERS",
    "SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS",
    "SUPPORTED_PADDING_MODES",
    "TARGET_CORRECTION_CONTRACT_VERSION",
    "TargetCorrectionPlan",
    "apply_target_correction",
    "compute_native_output_dimensions",
    "plan_target_correction",
    "resolve_target_dimensions",
    "resize_exact",
    "resize_to_exact_dimensions",
]
