from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

BLEND_WINDOW_VERSION = "linear_overlap_v1"
TILE_COORDINATE_VERSION = "source_pixel_row_major_v1"
OVERLAP_UNIT = "source_pixels"


@dataclass(frozen=True)
class RuntimeSizeRequirements:
    minimum: int = 0
    multiple_of: int = 1
    square: bool = False

    def __post_init__(self) -> None:
        if int(self.minimum) < 0:
            raise ValueError("Model minimum input size must be non-negative.")
        if int(self.multiple_of) < 1:
            raise ValueError("Model input-size multiple must be positive.")

    def normalized_tile_size(self, value: int) -> int:
        size = max(int(value), int(self.minimum), 1)
        multiple = max(1, int(self.multiple_of))
        remainder = size % multiple
        if remainder:
            size += multiple - remainder
        return size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeModelMetadata:
    architecture_id: str
    native_scale: int
    purpose: str
    input_channels: int
    output_channels: int
    supports_half: bool
    supports_bfloat16: bool
    tiling_recommendation: str
    size_requirements: RuntimeSizeRequirements

    @property
    def external_tiling_allowed(self) -> bool:
        return self.tiling_recommendation == "supported"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["external_tiling_allowed"] = self.external_tiling_allowed
        return payload


@dataclass(frozen=True)
class TileRegion:
    tile_index: int
    row_index: int
    column_index: int
    source_x: int
    source_y: int
    source_width: int
    source_height: int
    output_x: int
    output_y: int
    output_width: int
    output_height: int
    top_overlap: int
    bottom_overlap: int
    left_overlap: int
    right_overlap: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TilePlan:
    source_width: int
    source_height: int
    native_scale: int
    requested_tile_size: int
    effective_tile_size: int
    overlap_source_pixels: int
    overlap_output_pixels: int
    rows: int
    columns: int
    regions: tuple[TileRegion, ...]
    blend_window_version: str = BLEND_WINDOW_VERSION
    coordinate_version: str = TILE_COORDINATE_VERSION
    overlap_unit: str = OVERLAP_UNIT

    @property
    def tiled(self) -> bool:
        return len(self.regions) > 1

    def to_dict(self, *, include_regions: bool = True) -> dict[str, Any]:
        payload = {
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "native_scale": int(self.native_scale),
            "requested_tile_size": int(self.requested_tile_size),
            "effective_tile_size": int(self.effective_tile_size),
            "overlap_source_pixels": int(self.overlap_source_pixels),
            "overlap_output_pixels": int(self.overlap_output_pixels),
            "rows": int(self.rows),
            "columns": int(self.columns),
            "tile_count": len(self.regions),
            "tiled": self.tiled,
            "blend_window_version": self.blend_window_version,
            "coordinate_version": self.coordinate_version,
            "overlap_unit": self.overlap_unit,
        }
        if include_regions:
            payload["regions"] = [item.to_dict() for item in self.regions]
        return payload


def _enum_token(value: Any) -> str:
    token = str(getattr(value, "name", value) or "supported").strip().casefold()
    token = token.rsplit(".", 1)[-1]
    return token if token in {"supported", "discouraged", "internal"} else "unknown"


def normalize_spandrel_metadata(model_descriptor: Any) -> RuntimeModelMetadata:
    requirements = getattr(model_descriptor, "size_requirements", None)
    size_requirements = RuntimeSizeRequirements(
        minimum=int(getattr(requirements, "minimum", 0) or 0),
        multiple_of=int(getattr(requirements, "multiple_of", 1) or 1),
        square=bool(getattr(requirements, "square", False)),
    )
    architecture = getattr(model_descriptor, "architecture", None)
    return RuntimeModelMetadata(
        architecture_id=str(getattr(architecture, "id", "") or ""),
        native_scale=int(getattr(model_descriptor, "scale", 0) or 0),
        purpose=str(getattr(model_descriptor, "purpose", "") or ""),
        input_channels=int(getattr(model_descriptor, "input_channels", 0) or 0),
        output_channels=int(getattr(model_descriptor, "output_channels", 0) or 0),
        supports_half=bool(getattr(model_descriptor, "supports_half", False)),
        supports_bfloat16=bool(getattr(model_descriptor, "supports_bfloat16", False)),
        tiling_recommendation=_enum_token(getattr(model_descriptor, "tiling", "supported")),
        size_requirements=size_requirements,
    )


def tile_starts(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    length = int(length)
    tile_size = min(max(1, int(tile_size)), length)
    overlap = int(overlap)
    if length <= 0:
        raise ValueError("Tile planning requires positive source dimensions.")
    if overlap < 0:
        raise ValueError("Tile overlap must be non-negative.")
    if overlap >= tile_size:
        raise ValueError("Tile overlap must be smaller than the effective tile size.")
    if tile_size >= length:
        return (0,)
    step = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def _neighbor_overlap(starts: tuple[int, ...], index: int, tile_size: int, *, before: bool) -> int:
    if before:
        if index <= 0:
            return 0
        return max(0, starts[index - 1] + tile_size - starts[index])
    if index >= len(starts) - 1:
        return 0
    return max(0, starts[index] + tile_size - starts[index + 1])


def plan_tiles(
    *,
    source_width: int,
    source_height: int,
    native_scale: int,
    requested_tile_size: int,
    overlap_source_pixels: int,
    metadata: RuntimeModelMetadata,
) -> TilePlan:
    width = int(source_width)
    height = int(source_height)
    scale = int(native_scale)
    if width <= 0 or height <= 0:
        raise ValueError("Tile planning requires positive source dimensions.")
    if scale <= 0:
        raise ValueError("Tile planning requires a positive native model scale.")

    requested = int(requested_tile_size)
    if requested <= 0:
        effective = max(width, height)
        overlap = 0
    else:
        if not metadata.external_tiling_allowed:
            raise ValueError(
                "External tiling was requested, but Spandrel recommends "
                f"{metadata.tiling_recommendation!r} tiling for this model."
            )
        effective = metadata.size_requirements.normalized_tile_size(requested)
        overlap = int(overlap_source_pixels)
        if overlap < 0:
            raise ValueError("Tile overlap must be non-negative.")
        if overlap >= effective:
            raise ValueError("Tile overlap must be smaller than the effective tile size.")

    effective_x = min(effective, width)
    effective_y = min(effective, height)
    x_starts = tile_starts(width, effective_x, min(overlap, max(0, effective_x - 1)))
    y_starts = tile_starts(height, effective_y, min(overlap, max(0, effective_y - 1)))

    regions: list[TileRegion] = []
    tile_index = 0
    for row_index, source_y in enumerate(y_starts):
        source_h = min(effective_y, height - source_y)
        for column_index, source_x in enumerate(x_starts):
            source_w = min(effective_x, width - source_x)
            top_source = _neighbor_overlap(y_starts, row_index, effective_y, before=True)
            bottom_source = _neighbor_overlap(y_starts, row_index, effective_y, before=False)
            left_source = _neighbor_overlap(x_starts, column_index, effective_x, before=True)
            right_source = _neighbor_overlap(x_starts, column_index, effective_x, before=False)
            regions.append(
                TileRegion(
                    tile_index=tile_index,
                    row_index=row_index,
                    column_index=column_index,
                    source_x=source_x,
                    source_y=source_y,
                    source_width=source_w,
                    source_height=source_h,
                    output_x=source_x * scale,
                    output_y=source_y * scale,
                    output_width=source_w * scale,
                    output_height=source_h * scale,
                    top_overlap=min(top_source * scale, source_h * scale),
                    bottom_overlap=min(bottom_source * scale, source_h * scale),
                    left_overlap=min(left_source * scale, source_w * scale),
                    right_overlap=min(right_source * scale, source_w * scale),
                )
            )
            tile_index += 1

    return TilePlan(
        source_width=width,
        source_height=height,
        native_scale=scale,
        requested_tile_size=requested,
        effective_tile_size=effective,
        overlap_source_pixels=overlap,
        overlap_output_pixels=overlap * scale,
        rows=len(y_starts),
        columns=len(x_starts),
        regions=tuple(regions),
    )


def linear_blend_weight(
    height: int,
    width: int,
    *,
    top: int,
    bottom: int,
    left: int,
    right: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height = int(height)
    width = int(width)
    weight_y = torch.ones((height,), device=device, dtype=dtype)
    weight_x = torch.ones((width,), device=device, dtype=dtype)
    epsilon = 1.0e-3
    if top > 0:
        weight_y[:top] = torch.linspace(epsilon, 1.0, top, device=device, dtype=dtype)
    if bottom > 0:
        weight_y[-bottom:] = torch.linspace(1.0, epsilon, bottom, device=device, dtype=dtype)
    if left > 0:
        weight_x[:left] = torch.linspace(epsilon, 1.0, left, device=device, dtype=dtype)
    if right > 0:
        weight_x[-right:] = torch.linspace(1.0, epsilon, right, device=device, dtype=dtype)
    return weight_y[:, None] * weight_x[None, :]


def tile_plan_regions(plan: TilePlan) -> Iterable[TileRegion]:
    return plan.regions



def tile_plan_regions(plan: TilePlan) -> Iterable[TileRegion]:
    return plan.regions


NormalizedSpandrelMetadata = RuntimeModelMetadata
TileSlice = TileRegion
blend_weight = linear_blend_weight
normalize_spandrel_runtime_metadata = normalize_spandrel_metadata
build_tile_plan = plan_tiles

__all__ = [
    "BLEND_WINDOW_VERSION",
    "OVERLAP_UNIT",
    "TILE_COORDINATE_VERSION",
    "RuntimeModelMetadata",
    "RuntimeSizeRequirements",
    "NormalizedSpandrelMetadata",
    "TilePlan",
    "TileRegion",
    "TileSlice",
    "blend_weight",
    "build_tile_plan",
    "linear_blend_weight",
    "normalize_spandrel_metadata",
    "normalize_spandrel_runtime_metadata",
    "plan_tiles",
    "tile_plan_regions",
    "tile_starts",
]
