from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

SUPPORTED_EXACT_RESIZE_FILTERS = frozenset({"nearest", "bilinear", "bicubic", "area"})
SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS = frozenset({"auto", *SUPPORTED_EXACT_RESIZE_FILTERS})
SUPPORTED_ASPECT_POLICIES = frozenset({"stretch", "crop_to_fill", "pad_to_fit"})
SUPPORTED_PADDING_MODES = frozenset({"reflect", "replicate", "blurred_edge", "black"})
SUPPORTED_BLURRED_EDGE_METHODS = frozenset({"box", "gaussian_1d"})
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


def _build_blurred_edge_mask(
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    pad_top: int,
    pad_left: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.ones((1, 1, target_height, target_width), device=device, dtype=dtype)
    mask[:, :, pad_top : pad_top + source_height, pad_left : pad_left + source_width] = 0.0
    return mask


def _edge_band_stats(image: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    sample = image.detach().to(device="cpu", dtype=torch.float32)
    band = mask.detach().to(device="cpu", dtype=torch.float32).expand(sample.shape[0], sample.shape[1], -1, -1) > 0.5
    values = sample[band]
    if values.numel() == 0:
        return {"available": False, "pixel_count": 0}
    horizontal = float((sample[..., 1:] - sample[..., :-1]).abs().mean().item()) if sample.shape[-1] > 1 else 0.0
    vertical = float((sample[..., 1:, :] - sample[..., :-1, :]).abs().mean().item()) if sample.shape[-2] > 1 else 0.0
    return {
        "available": True,
        "pixel_count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "spatial_delta_mean": (horizontal + vertical) / 2.0,
    }


def _compare_blurred_edge_outputs(
    selected: torch.Tensor,
    alternate: torch.Tensor,
    *,
    mask: torch.Tensor,
) -> dict[str, Any]:
    lhs = selected.detach().to(device="cpu", dtype=torch.float32)
    rhs = alternate.detach().to(device="cpu", dtype=torch.float32)
    diff = (lhs - rhs).abs()
    mse = float(((lhs - rhs) ** 2).mean().item())
    psnr = float("inf") if mse == 0.0 else float(20.0 * math.log10(1.0 / max(math.sqrt(mse), 1e-12)))
    expanded_mask = mask.detach().to(device="cpu", dtype=torch.float32).expand(lhs.shape[0], lhs.shape[1], -1, -1) > 0.5
    edge_values = diff[expanded_mask]
    edge_rmse = 0.0
    if edge_values.numel() > 0:
        edge_rmse = math.sqrt(float(((lhs[expanded_mask] - rhs[expanded_mask]) ** 2).mean().item()))
    return {
        "available": True,
        "mae": float(diff.mean().item()),
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "max_abs": float(diff.max().item()),
        "psnr": psnr,
        "edge_band_mae": float(edge_values.mean().item()) if edge_values.numel() > 0 else 0.0,
        "edge_band_rmse": edge_rmse,
    }


def _gaussian_sigma_for_kernel(kernel_size: int) -> float:
    return max(0.8, float(max(1, kernel_size)) / 6.0)


def _blur_tensor_box(image: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    radius = max(0, int(kernel_size) // 2)
    if radius < 1:
        return image
    padded = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def _blur_tensor_gaussian_1d(image: torch.Tensor, *, kernel_size: int, sigma: float) -> torch.Tensor:
    radius = max(0, int(kernel_size) // 2)
    if radius < 1:
        return image
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    channels = int(image.shape[1])
    padded_x = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    weight_x = kernel_1d.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
    blurred_x = F.conv2d(padded_x, weight_x, groups=channels)
    padded_y = F.pad(blurred_x, (0, 0, radius, radius), mode="replicate")
    weight_y = kernel_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
    return F.conv2d(padded_y, weight_y, groups=channels)


def _render_blurred_edge(
    image: torch.Tensor,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    method: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    padded = F.pad(image, (left, right, top, bottom), mode="replicate")
    if not any((left, right, top, bottom)):
        return padded, {
            "method": method,
            "kernel_size": 1,
            "radius": 0,
            "sigma": None,
            "duration_ms": 0.0,
        }
    kernel = max(3, max(left, right, top, bottom) * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    sigma = _gaussian_sigma_for_kernel(kernel) if method == "gaussian_1d" else None
    started = time.perf_counter()
    if method == "gaussian_1d":
        blurred = _blur_tensor_gaussian_1d(padded, kernel_size=kernel, sigma=float(sigma))
    else:
        blurred = _blur_tensor_box(padded, kernel_size=kernel)
    blurred[..., top : top + image.shape[-2], left : left + image.shape[-1]] = image
    return blurred.clamp(0.0, 1.0), {
        "method": method,
        "kernel_size": int(kernel),
        "radius": int(kernel // 2),
        "sigma": float(sigma) if sigma is not None else None,
        "duration_ms": float((time.perf_counter() - started) * 1000.0),
    }


def _pad_blurred_edge(
    images: torch.Tensor,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    method: str = "box",
    compare_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected_method = str(method or "box").strip().casefold()
    if selected_method not in SUPPORTED_BLURRED_EDGE_METHODS:
        raise ValueError(f"Unsupported blurred-edge method: {method!r}.")
    result, selected_runtime = _render_blurred_edge(
        images,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        method=selected_method,
    )
    mask = _build_blurred_edge_mask(
        source_height=int(images.shape[-2]),
        source_width=int(images.shape[-1]),
        target_height=int(result.shape[-2]),
        target_width=int(result.shape[-1]),
        pad_top=top,
        pad_left=left,
        device=images.device,
        dtype=images.dtype,
    )
    metadata: dict[str, Any] = {
        "mode": "blurred_edge",
        "selected_method": selected_method,
        "pad_left": int(left),
        "pad_right": int(right),
        "pad_top": int(top),
        "pad_bottom": int(bottom),
        "selected_runtime": selected_runtime,
        "selected_quality_proxy": _edge_band_stats(result, mask),
        "comparison_enabled": bool(compare_diagnostics),
    }
    if compare_diagnostics:
        alternate_method = "gaussian_1d" if selected_method == "box" else "box"
        alternate, alternate_runtime = _render_blurred_edge(
            images,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            method=alternate_method,
        )
        metadata["comparison"] = {
            "mode": "same_input_dual_render",
            "selected_method": selected_method,
            "alternate_method": alternate_method,
            "selected_runtime": selected_runtime,
            "alternate_runtime": alternate_runtime,
            "selected_quality_proxy": metadata["selected_quality_proxy"],
            "alternate_quality_proxy": _edge_band_stats(alternate, mask),
            "selected_vs_alternate": _compare_blurred_edge_outputs(result, alternate, mask=mask),
        }
    return result, metadata


def apply_target_correction(
    images: torch.Tensor,
    *,
    target_width: int,
    target_height: int,
    aspect_policy: str = "stretch",
    final_size_correction_filter: str = "auto",
    padding_mode: str = "reflect",
    blurred_edge_method: str = "box",
    blurred_edge_compare_diagnostics: bool = False,
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
    blurred_edge_runtime: dict[str, Any] | None = None

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
            corrected, blurred_edge_runtime = _pad_blurred_edge(
                resized,
                left=plan.pad_left,
                right=plan.pad_right,
                top=plan.pad_top,
                bottom=plan.pad_bottom,
                method=blurred_edge_method,
                compare_diagnostics=bool(blurred_edge_compare_diagnostics),
            )
        if plan.padding_mode != "blurred_edge":
            blurred_edge_runtime = None

    if tuple(corrected.shape[-2:]) != (plan.target_height, plan.target_width):
        raise RuntimeError(
            "Target correction did not produce the exact requested correction canvas: "
            f"expected {(plan.target_height, plan.target_width)}, got {tuple(corrected.shape[-2:])}."
        )
    metadata = plan.to_dict()
    metadata["output_width"] = int(corrected.shape[-1])
    metadata["output_height"] = int(corrected.shape[-2])
    metadata["blurred_edge_method"] = str(blurred_edge_method or "box")
    metadata["blurred_edge_compare_diagnostics"] = bool(blurred_edge_compare_diagnostics)
    if blurred_edge_runtime is not None:
        metadata["blurred_edge_runtime"] = blurred_edge_runtime
    return corrected.clamp(0.0, 1.0), metadata


resize_to_exact_dimensions = resize_exact

__all__ = [
    "SUPPORTED_ASPECT_POLICIES",
    "SUPPORTED_BLURRED_EDGE_METHODS",
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
