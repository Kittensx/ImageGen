from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

SUPPORTED_EXACT_RESIZE_FILTERS = frozenset({"nearest", "bilinear", "bicubic", "area"})


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
        return requested_width, max(1, int(round(height * (requested_width / width))))
    if requested_height > 0:
        return max(1, int(round(width * (requested_height / height)))), requested_height
    if scale is None or float(scale) <= 0:
        raise ValueError("Specify target dimensions or a positive scale.")
    factor = float(scale)
    return max(1, int(round(width * factor))), max(1, int(round(height * factor)))


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


resize_to_exact_dimensions = resize_exact

__all__ = [
    "SUPPORTED_EXACT_RESIZE_FILTERS",
    "compute_native_output_dimensions",
    "resolve_target_dimensions",
    "resize_exact",
    "resize_to_exact_dimensions",
]
