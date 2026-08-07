from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from modules.txt2img.seed_utils import create_torch_generator, offset_seed

OUTPAINT_PROTOTYPE_CONTRACT_VERSION = "phase14n13p-outpaint-prototype-v1"
OUTPAINT_MASK_CONTRACT_VERSION = "phase14n13p-outpaint-mask-v1"
OUTPAINT_NOISE_POLICY_ID = "phase14n13p-offset-seed-plus-2000003-v1"
OUTPAINT_NOISE_SEED_OFFSET = 2_000_003
OUTPAINT_PRESERVATION_STRATEGY = "strict_latent_restore_each_step_plus_exact_core_composite_v1"
OUTPAINT_PROMPT_OVERLAY_CONTRACT_VERSION = "phase14n13p1-prompt-overlay-v1"
OUTPAINT_CONTEXT_SEED_CONTRACT_VERSION = "phase14n13p2-context-seed-v1"
OUTPAINT_PROMPT_MODES = {
    "source_prompt_v1",
    "overlay_only_v1",
    "source_plus_overlay_v1",
}
OUTPAINT_LATENT_STRATEGIES = {
    "noise_only_new_regions_v1",
    "canvas_regional_noise_v1",
}
OUTPAINT_CONTEXT_SEED_MODES = {
    "neutral_gray_v1",
    "edge_pad_v1",
    "reflect_pad_v1",
}
OUTPAINT_ANCHORS = {"center", "left", "right", "top", "bottom"}
OUTPAINT_FAILURE_STAGES = {
    "outpaint_source_decode",
    "outpaint_canvas_planning",
    "outpaint_mask_build",
    "outpaint_conditioning",
    "outpaint_vae_encode",
    "outpaint_noise_initialization",
    "outpaint_sampling",
    "outpaint_decode",
    "outpaint_source_composite",
    "outpaint_live_source_capture",
    "outpaint_source_handoff",
    "outpaint_context_seed",
    "outpaint_latent_canvas_build",
}


def compose_outpaint_prompt_overlay(
    *,
    mode: str,
    source_positive_prompt: str,
    source_negative_prompt: str,
    overlay_positive_prompt: str,
    overlay_negative_prompt: str,
) -> dict[str, Any]:
    """Build the single global prompt pair used by the P-1 outpaint pass.

    P-1 deliberately does not create left/right REGION branches.  The returned
    positive and negative prompts are each encoded once as the normal global
    conditioning pair for the expanded canvas.  Spatial preservation remains
    the responsibility of the latent/pixel preservation masks.
    """

    normalized_mode = str(mode or "source_prompt_v1").strip().lower()
    if normalized_mode not in OUTPAINT_PROMPT_MODES:
        raise ValueError(f"Unsupported outpaint prompt mode: {normalized_mode!r}.")

    source_positive = str(source_positive_prompt or "").strip()
    source_negative = str(source_negative_prompt or "").strip()
    overlay_positive = str(overlay_positive_prompt or "").strip()
    overlay_negative = str(overlay_negative_prompt or "").strip()

    def _join(first: str, second: str) -> str:
        if first and second:
            return f"{first}, {second}"
        return first or second

    if normalized_mode == "overlay_only_v1":
        effective_positive = overlay_positive
        effective_negative = overlay_negative
        source_prompt_participates = False
    elif normalized_mode == "source_plus_overlay_v1":
        effective_positive = _join(source_positive, overlay_positive)
        effective_negative = _join(source_negative, overlay_negative)
        source_prompt_participates = True
    else:
        effective_positive = source_positive
        effective_negative = source_negative
        source_prompt_participates = True

    if normalized_mode == "overlay_only_v1" and not effective_positive:
        raise ValueError(
            "Overlay-only outpaint conditioning requires a non-empty overlay positive prompt."
        )

    return {
        "contract_version": OUTPAINT_PROMPT_OVERLAY_CONTRACT_VERSION,
        "mode": normalized_mode,
        "branch_policy": "single_global_conditioning_v1",
        "source_prompt_participates": source_prompt_participates,
        "source_positive_prompt": source_positive,
        "source_negative_prompt": source_negative,
        "overlay_positive_prompt": overlay_positive,
        "overlay_negative_prompt": overlay_negative,
        "effective_positive_prompt": effective_positive,
        "effective_negative_prompt": effective_negative,
    }


def _positive_int(value: Any, *, field: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{field} must be positive.")
    return result


@dataclass(frozen=True)
class OutpaintPrototypePlan:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    source_x: int
    source_y: int
    anchor: str
    feather_px: int

    @property
    def source_right(self) -> int:
        return self.source_x + self.source_width

    @property
    def source_bottom(self) -> int:
        return self.source_y + self.source_height

    @property
    def left_expansion(self) -> int:
        return self.source_x

    @property
    def right_expansion(self) -> int:
        return self.target_width - self.source_right

    @property
    def top_expansion(self) -> int:
        return self.source_y

    @property
    def bottom_expansion(self) -> int:
        return self.target_height - self.source_bottom

    @property
    def feather_left(self) -> int:
        return min(self.feather_px, self.source_width) if self.left_expansion > 0 else 0

    @property
    def feather_right(self) -> int:
        return min(self.feather_px, self.source_width) if self.right_expansion > 0 else 0

    @property
    def feather_top(self) -> int:
        return min(self.feather_px, self.source_height) if self.top_expansion > 0 else 0

    @property
    def feather_bottom(self) -> int:
        return min(self.feather_px, self.source_height) if self.bottom_expansion > 0 else 0

    @property
    def protected_left(self) -> int:
        return self.source_x + self.feather_left

    @property
    def protected_right(self) -> int:
        return self.source_right - self.feather_right

    @property
    def protected_top(self) -> int:
        return self.source_y + self.feather_top

    @property
    def protected_bottom(self) -> int:
        return self.source_bottom - self.feather_bottom

    def to_dict(self) -> dict[str, Any]:
        generated_pixels = (
            self.target_width * self.target_height
            - self.source_width * self.source_height
        )
        return {
            "contract_version": OUTPAINT_PROTOTYPE_CONTRACT_VERSION,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "source_x": self.source_x,
            "source_y": self.source_y,
            "source_bounds": {
                "left": self.source_x,
                "top": self.source_y,
                "right": self.source_right,
                "bottom": self.source_bottom,
            },
            "anchor": self.anchor,
            "left_expansion": self.left_expansion,
            "right_expansion": self.right_expansion,
            "top_expansion": self.top_expansion,
            "bottom_expansion": self.bottom_expansion,
            "feather_px": self.feather_px,
            "feather_sides": {
                "left": self.feather_left,
                "right": self.feather_right,
                "top": self.feather_top,
                "bottom": self.feather_bottom,
            },
            "protected_bounds": {
                "left": self.protected_left,
                "top": self.protected_top,
                "right": self.protected_right,
                "bottom": self.protected_bottom,
            },
            "generative_pixel_area": generated_pixels,
            "source_pixel_area": self.source_width * self.source_height,
            "target_pixel_area": self.target_width * self.target_height,
        }


def plan_outpaint_canvas(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    anchor: str = "center",
    feather_px: int = 24,
    source_x: int | None = None,
    source_y: int | None = None,
) -> OutpaintPrototypePlan:
    sw = _positive_int(source_width, field="source_width")
    sh = _positive_int(source_height, field="source_height")
    tw = _positive_int(target_width, field="target_width")
    th = _positive_int(target_height, field="target_height")
    if sw > tw or sh > th:
        raise ValueError(
            "Canvas expansion does not resize or crop the source; source dimensions must fit inside the target canvas."
        )
    selected_anchor = str(anchor or "center").strip().lower()
    if selected_anchor not in OUTPAINT_ANCHORS:
        raise ValueError(f"outpaint anchor must be one of: {', '.join(sorted(OUTPAINT_ANCHORS))}.")
    feather = int(feather_px)
    if feather < 0 or feather > 64:
        raise ValueError("Blend width must be between 0 and 64 pixels.")

    max_x = tw - sw
    max_y = th - sh
    if source_x is None:
        x = max_x // 2
        if selected_anchor == "left":
            x = 0
        elif selected_anchor == "right":
            x = max_x
    else:
        x = int(source_x)
    if source_y is None:
        y = max_y // 2
        if selected_anchor == "top":
            y = 0
        elif selected_anchor == "bottom":
            y = max_y
    else:
        y = int(source_y)
    if x < 0 or x > max_x or y < 0 or y > max_y:
        raise ValueError(
            f"Source placement ({x}, {y}) falls outside the {tw}x{th} canvas for a {sw}x{sh} source."
        )

    plan = OutpaintPrototypePlan(
        source_width=sw,
        source_height=sh,
        target_width=tw,
        target_height=th,
        source_x=x,
        source_y=y,
        anchor=selected_anchor,
        feather_px=feather,
    )
    if plan.protected_left > plan.protected_right or plan.protected_top > plan.protected_bottom:
        raise ValueError("Feather width consumes the entire protected source region.")
    if (plan.target_width * plan.target_height - plan.source_width * plan.source_height) < 1:
        raise ValueError("Expansion target must add at least one pixel beyond the source image.")
    return plan


def load_source_image(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Outpaint source image does not exist: {source_path}")
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(source_path) as opened:
        original_size = tuple(int(item) for item in opened.size)
        normalized = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = normalized.size
        tensor = torch.from_numpy(__import__("numpy").array(normalized, dtype="float32"))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous().div_(255.0)
    return tensor, {
        "path": str(source_path),
        "sha256": digest,
        "original_file_dimensions": [original_size[0], original_size[1]],
        "normalized_dimensions": [int(width), int(height)],
        "exif_orientation_normalized": original_size != (width, height),
        "channel_order": "RGB",
        "range": [0.0, 1.0],
    }


def _reflect_indices(indices: torch.Tensor, length: int) -> torch.Tensor:
    if length <= 1:
        return torch.zeros_like(indices)
    period = (2 * int(length)) - 2
    reflected = torch.remainder(indices, period)
    return torch.where(reflected < length, reflected, period - reflected)


def build_seeded_outpaint_canvas(
    source: torch.Tensor,
    plan: OutpaintPrototypePlan,
    *,
    mode: str = "neutral_gray_v1",
) -> tuple[torch.Tensor, dict[str, Any]]:
    if source.ndim != 4 or source.shape[0] != 1 or source.shape[1] != 3:
        raise ValueError("Outpaint source must be one BCHW RGB image.")
    if tuple(source.shape[-2:]) != (plan.source_height, plan.source_width):
        raise ValueError("Outpaint source tensor dimensions do not match the canvas plan.")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("Outpaint source contains non-finite values.")

    selected = str(mode or "neutral_gray_v1").strip().lower()
    if selected not in OUTPAINT_CONTEXT_SEED_MODES:
        raise ValueError(f"Unsupported outpaint context seed mode: {mode}")

    device = source.device
    dtype = source.dtype
    if selected == "neutral_gray_v1":
        canvas = torch.zeros((1, 3, plan.target_height, plan.target_width), device=device, dtype=dtype)
        canvas.fill_(0.5)
    else:
        local_x = torch.arange(int(plan.target_width), device=device, dtype=torch.long) - int(plan.source_x)
        local_y = torch.arange(int(plan.target_height), device=device, dtype=torch.long) - int(plan.source_y)
        if selected == "edge_pad_v1":
            x_index = local_x.clamp(0, int(plan.source_width) - 1)
            y_index = local_y.clamp(0, int(plan.source_height) - 1)
        else:
            x_index = _reflect_indices(local_x, int(plan.source_width))
            y_index = _reflect_indices(local_y, int(plan.source_height))
        canvas = source.index_select(-2, y_index).index_select(-1, x_index)
    canvas[:, :, plan.source_y:plan.source_bottom, plan.source_x:plan.source_right] = source
    metadata = {
        "contract_version": OUTPAINT_CONTEXT_SEED_CONTRACT_VERSION,
        "mode": selected,
        "preserves_source_region_exactly": True,
        "target_width": int(plan.target_width),
        "target_height": int(plan.target_height),
        "source_width": int(plan.source_width),
        "source_height": int(plan.source_height),
    }
    return canvas, metadata


def build_canvas_and_masks(
    source: torch.Tensor,
    plan: OutpaintPrototypePlan,
    *,
    context_seed_mode: str = "neutral_gray_v1",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    canvas, canvas_metadata = build_seeded_outpaint_canvas(
        source,
        plan,
        mode=context_seed_mode,
    )

    device = source.device
    dtype = source.dtype
    generate = torch.ones((1, 1, plan.target_height, plan.target_width), device=device, dtype=dtype)
    generate[:, :, plan.source_y:plan.source_bottom, plan.source_x:plan.source_right] = 0.0
    weight = generate.clone()

    def _ramp(length: int, *, reverse: bool = False) -> torch.Tensor:
        if length <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        values = torch.linspace(1.0, 0.0, steps=length + 2, device=device, dtype=dtype)[1:-1]
        return values.flip(0) if reverse else values

    if plan.feather_left:
        sl = slice(plan.source_x, plan.source_x + plan.feather_left)
        values = _ramp(plan.feather_left).view(1, 1, 1, -1)
        weight[:, :, plan.source_y:plan.source_bottom, sl] = torch.maximum(
            weight[:, :, plan.source_y:plan.source_bottom, sl], values
        )
    if plan.feather_right:
        sl = slice(plan.source_right - plan.feather_right, plan.source_right)
        values = _ramp(plan.feather_right, reverse=True).view(1, 1, 1, -1)
        weight[:, :, plan.source_y:plan.source_bottom, sl] = torch.maximum(
            weight[:, :, plan.source_y:plan.source_bottom, sl], values
        )
    if plan.feather_top:
        sl = slice(plan.source_y, plan.source_y + plan.feather_top)
        values = _ramp(plan.feather_top).view(1, 1, -1, 1)
        weight[:, :, sl, plan.source_x:plan.source_right] = torch.maximum(
            weight[:, :, sl, plan.source_x:plan.source_right], values
        )
    if plan.feather_bottom:
        sl = slice(plan.source_bottom - plan.feather_bottom, plan.source_bottom)
        values = _ramp(plan.feather_bottom, reverse=True).view(1, 1, -1, 1)
        weight[:, :, sl, plan.source_x:plan.source_right] = torch.maximum(
            weight[:, :, sl, plan.source_x:plan.source_right], values
        )

    weight = weight.clamp(0.0, 1.0)
    preserve = (weight <= 0.0).to(dtype=dtype)
    feather = ((weight > 0.0) & (weight < 1.0)).to(dtype=dtype)
    masks = {
        "generation_weight": weight,
        "preserve": preserve,
        "feather": feather,
        "generate": generate,
    }
    metadata = {
        "contract_version": OUTPAINT_MASK_CONTRACT_VERSION,
        "context_seed": dict(canvas_metadata),
        "generation_weight_range": [float(weight.min().item()), float(weight.max().item())],
        # Pixel counts are diagnostic integers, not mask-weight sums.  P-3 can
        # carry a freshly decoded FP16 source into this path; summing a large
        # 0/1 FP16 mask can overflow above 65504 and become ``inf`` before the
        # value is converted to Python ``int``.  Count nonzero entries using
        # integer accumulation so metadata generation is dtype-independent.
        "preserve_pixels": int(torch.count_nonzero(preserve).item()),
        "feather_pixels": int(torch.count_nonzero(feather).item()),
        "generate_pixels": int(torch.count_nonzero(generate).item()),
        "soft_mask": True,
        "semantics": {
            "0": "strictly_preserved",
            "0_to_1": "feather_overlap",
            "1": "generative",
        },
    }
    return canvas, masks, metadata


def resize_mask_to_latent(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Outpaint pixel mask must be BCHW with one channel.")
    if latent.ndim != 4:
        raise ValueError("Outpaint latent must be BCHW.")
    resized = F.interpolate(
        mask.to(device=latent.device, dtype=latent.dtype),
        size=(int(latent.shape[-2]), int(latent.shape[-1])),
        mode="bilinear",
        align_corners=False,
    )
    return resized.clamp(0.0, 1.0)


def build_outpaint_noise(reference: torch.Tensor, *, seeds: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(seeds) != int(reference.shape[0]):
        raise ValueError("Outpaint noise seeds must match latent batch size.")
    derived = [offset_seed(int(seed), OUTPAINT_NOISE_SEED_OFFSET) for seed in seeds]
    noise_items = []
    for seed in derived:
        generator = create_torch_generator(seed, device=reference.device)
        noise_items.append(
            torch.randn(
                (1, *reference.shape[1:]),
                generator=generator,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
    return torch.cat(noise_items, dim=0), {
        "policy_id": OUTPAINT_NOISE_POLICY_ID,
        "base_seeds": [int(seed) for seed in seeds],
        "derived_seeds": [int(seed) for seed in derived],
        "shape": list(reference.shape),
    }


def initialize_outpaint_latents(
    encoded_canvas: torch.Tensor,
    *,
    generation_weight: torch.Tensor,
    noise: torch.Tensor,
    initial_sigma: float,
    strategy: str,
) -> torch.Tensor:
    selected = str(strategy or "noise_only_new_regions_v1").strip().lower()
    if selected not in OUTPAINT_LATENT_STRATEGIES:
        raise ValueError(f"Unsupported outpaint latent strategy: {strategy}")
    if encoded_canvas.shape != noise.shape:
        raise ValueError("Outpaint encoded canvas and noise tensors must have identical shape.")
    if generation_weight.shape[-2:] != encoded_canvas.shape[-2:]:
        raise ValueError("Outpaint latent mask dimensions must match encoded canvas dimensions.")
    weight = generation_weight.to(device=encoded_canvas.device, dtype=encoded_canvas.dtype)
    sigma = float(initial_sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Outpaint initialization requires a positive finite schedule start sigma.")
    noisy = noise * sigma
    if selected == "canvas_regional_noise_v1":
        noisy = encoded_canvas + noisy
    return encoded_canvas * (1.0 - weight) + noisy * weight


class StrictLatentPreservationHook:
    """Restore protected source latents after every sampler transition.

    A zero generation weight is exact latent preservation. Values in the
    feather band blend restored source and newly sampled latent state. A value
    of one leaves the generated latent untouched.
    """

    def __init__(self, reference_latent: torch.Tensor, generation_weight: torch.Tensor) -> None:
        self.reference_latent = reference_latent.detach().clone()
        self.generation_weight = generation_weight.detach().clone()
        self.call_count = 0

    def __call__(self, latent: torch.Tensor, **_: Any) -> torch.Tensor:
        reference = self.reference_latent.to(device=latent.device, dtype=latent.dtype)
        weight = self.generation_weight.to(device=latent.device, dtype=latent.dtype)
        self.call_count += 1
        return reference * (1.0 - weight) + latent * weight

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy": OUTPAINT_PRESERVATION_STRATEGY,
            "sampler_step_restore_calls": int(self.call_count),
            "strict_core_restoration": True,
            "soft_feather_restoration": True,
        }


def composite_exact_protected_core(
    decoded: torch.Tensor,
    source: torch.Tensor,
    plan: OutpaintPrototypePlan,
) -> torch.Tensor:
    if tuple(decoded.shape[-2:]) != (plan.target_height, plan.target_width):
        raise ValueError("Decoded image dimensions do not match the outpaint target canvas.")
    if tuple(source.shape[-2:]) != (plan.source_height, plan.source_width):
        raise ValueError("Source image dimensions do not match the outpaint plan.")
    result = decoded.clone()
    x0, x1 = plan.protected_left, plan.protected_right
    y0, y1 = plan.protected_top, plan.protected_bottom
    if x1 > x0 and y1 > y0:
        sx0 = x0 - plan.source_x
        sx1 = x1 - plan.source_x
        sy0 = y0 - plan.source_y
        sy1 = y1 - plan.source_y
        result[:, :, y0:y1, x0:x1] = source[:, :, sy0:sy1, sx0:sx1].to(
            device=result.device, dtype=result.dtype
        )
    return result


OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION = "phase14n13p3-live-shape-expansion-v1"
OUTPAINT_SHAPE_TARGET_MODES = {"square", "landscape", "portrait", "custom"}
OUTPAINT_SOURCE_HANDOFF_MODES = {"auto", "pixel_vae_reencode", "live_latent"}


def resolve_outpaint_shape_target(
    *,
    base_width: int,
    base_height: int,
    target_mode: str,
    target_width: int = 0,
    target_height: int = 0,
) -> dict[str, Any]:
    bw = _positive_int(base_width, field="base_width")
    bh = _positive_int(base_height, field="base_height")
    mode = str(target_mode or "square").strip().lower()
    if mode not in OUTPAINT_SHAPE_TARGET_MODES:
        raise ValueError(f"Unsupported outpaint shape target mode: {mode!r}.")

    tw = int(target_width or 0)
    th = int(target_height or 0)
    if mode == "square":
        if tw > 0 or th > 0:
            side = max(tw, th)
            if tw > 0 and th > 0 and tw != th:
                raise ValueError("Square shape expansion requires equal target width and height.")
        else:
            side = max(bw, bh)
        tw = th = int(side)
    else:
        if tw < 1 or th < 1:
            raise ValueError(f"{mode} shape expansion requires explicit target width and height.")
        if mode == "landscape" and tw <= th:
            raise ValueError("Landscape shape expansion requires target width greater than target height.")
        if mode == "portrait" and th <= tw:
            raise ValueError("Portrait shape expansion requires target height greater than target width.")

    if tw < bw or th < bh:
        raise ValueError(
            "Post-generation expansion cannot shrink or crop the base generation; target dimensions must contain the full base image."
        )
    if tw == bw and th == bh:
        raise ValueError("Post-generation expansion target must add canvas space beyond the base generation.")
    if tw % 8 or th % 8:
        raise ValueError("Post-generation expansion target width and height must be divisible by 8.")

    return {
        "contract_version": OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION,
        "target_mode": mode,
        "base_width": bw,
        "base_height": bh,
        "target_width": tw,
        "target_height": th,
    }


def inspect_live_latent_alignment(
    *,
    plan: OutpaintPrototypePlan,
    source_latent: torch.Tensor,
    expanded_latent: torch.Tensor,
    latent_scale_factor: int,
) -> dict[str, Any]:
    factor = _positive_int(latent_scale_factor, field="latent_scale_factor")
    reasons: list[str] = []
    if plan.source_x % factor:
        reasons.append(f"source_x={plan.source_x} is not divisible by latent scale factor {factor}")
    if plan.source_y % factor:
        reasons.append(f"source_y={plan.source_y} is not divisible by latent scale factor {factor}")
    if plan.source_width % factor:
        reasons.append(f"source_width={plan.source_width} is not divisible by latent scale factor {factor}")
    if plan.source_height % factor:
        reasons.append(f"source_height={plan.source_height} is not divisible by latent scale factor {factor}")

    expected_source = (plan.source_height // factor, plan.source_width // factor)
    expected_target = (plan.target_height // factor, plan.target_width // factor)
    if tuple(source_latent.shape[-2:]) != expected_source:
        reasons.append(
            f"source latent shape {tuple(source_latent.shape[-2:])} does not match expected {expected_source}"
        )
    if tuple(expanded_latent.shape[-2:]) != expected_target:
        reasons.append(
            f"expanded latent shape {tuple(expanded_latent.shape[-2:])} does not match expected {expected_target}"
        )
    if int(source_latent.shape[0]) != int(expanded_latent.shape[0]):
        reasons.append("source and expanded latent batch sizes differ")
    if int(source_latent.shape[1]) != int(expanded_latent.shape[1]):
        reasons.append("source and expanded latent channel counts differ")

    aligned = not reasons
    return {
        "contract_version": OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION,
        "latent_scale_factor": factor,
        "aligned": aligned,
        "reasons": reasons,
        "source_pixel_placement": {"x": int(plan.source_x), "y": int(plan.source_y)},
        "source_latent_placement": {
            "x": int(plan.source_x // factor) if plan.source_x % factor == 0 else None,
            "y": int(plan.source_y // factor) if plan.source_y % factor == 0 else None,
        },
        "source_latent_shape": list(source_latent.shape),
        "expanded_latent_shape": list(expanded_latent.shape),
    }


def resolve_outpaint_source_handoff(
    *,
    requested_mode: str,
    alignment: Mapping[str, Any],
    live_latent_available: bool,
) -> dict[str, Any]:
    requested = str(requested_mode or "auto").strip().lower()
    if requested not in OUTPAINT_SOURCE_HANDOFF_MODES:
        raise ValueError(f"Unsupported outpaint source handoff mode: {requested!r}.")

    aligned = bool(alignment.get("aligned", False))
    available = bool(live_latent_available)
    fallback_reason = ""
    if requested == "pixel_vae_reencode":
        actual = "pixel_vae_reencode"
    elif requested == "live_latent":
        if not available:
            raise ValueError("Live-latent source handoff was requested, but no live base latent is available.")
        if not aligned:
            details = "; ".join(str(item) for item in alignment.get("reasons", []) if item)
            raise ValueError(
                "Live-latent source handoff requires exact latent-grid alignment. " + details
            )
        actual = "live_latent"
    else:
        if available and aligned:
            actual = "live_latent"
        else:
            actual = "pixel_vae_reencode"
            if not available:
                fallback_reason = "live base latent unavailable"
            elif not aligned:
                fallback_reason = "; ".join(str(item) for item in alignment.get("reasons", []) if item)

    return {
        "contract_version": OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION,
        "requested": requested,
        "actual": actual,
        "fallback_reason": fallback_reason,
        "live_latent_available": available,
        "latent_grid_aligned": aligned,
    }


def embed_live_source_latent(
    expanded_latent: torch.Tensor,
    source_latent: torch.Tensor,
    *,
    plan: OutpaintPrototypePlan,
    latent_scale_factor: int,
) -> torch.Tensor:
    alignment = inspect_live_latent_alignment(
        plan=plan,
        source_latent=source_latent,
        expanded_latent=expanded_latent,
        latent_scale_factor=latent_scale_factor,
    )
    if not alignment["aligned"]:
        raise ValueError("Cannot embed live source latent: " + "; ".join(alignment["reasons"]))
    factor = int(latent_scale_factor)
    x0 = int(plan.source_x // factor)
    y0 = int(plan.source_y // factor)
    x1 = x0 + int(source_latent.shape[-1])
    y1 = y0 + int(source_latent.shape[-2])
    result = expanded_latent.clone()
    result[:, :, y0:y1, x0:x1] = source_latent.to(device=result.device, dtype=result.dtype)
    return result

def format_outpaint_failure(stage: str, message: str, **context: Any) -> str:
    code = str(stage or "").strip().lower()
    if code not in OUTPAINT_FAILURE_STAGES:
        raise ValueError(f"Unknown outpaint failure stage: {stage}")
    suffix = " ".join(f"{key}={value}" for key, value in context.items() if value not in (None, ""))
    return f"[OUTPAINT_STAGE:{code}] {message}" + (f" [{suffix}]" if suffix else "")


def extract_outpaint_failure_stage(error: BaseException | str) -> str:
    text = str(error)
    marker = "[OUTPAINT_STAGE:"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = text.find("]", start)
    if end < 0:
        return ""
    code = text[start:end].strip().lower()
    return code if code in OUTPAINT_FAILURE_STAGES else ""


def outpaint_failure_label(stage: str) -> str:
    labels = {
        "outpaint_source_decode": "Source image decode",
        "outpaint_canvas_planning": "Canvas planning",
        "outpaint_mask_build": "Mask build",
        "outpaint_conditioning": "Prompt overlay conditioning",
        "outpaint_vae_encode": "Source/canvas VAE encode",
        "outpaint_noise_initialization": "Regional noise initialization",
        "outpaint_sampling": "Masked diffusion sampling",
        "outpaint_decode": "Outpaint decode",
        "outpaint_source_composite": "Protected source composite",
        "outpaint_live_source_capture": "Live txt2img source capture",
        "outpaint_source_handoff": "Outpaint source handoff",
        "outpaint_context_seed": "Outpaint context seed",
        "outpaint_latent_canvas_build": "Expanded latent canvas build",
    }
    return labels.get(str(stage or ""), str(stage or "Outpaint"))


def _tensor_to_image(tensor: torch.Tensor, *, grayscale: bool = False) -> Image.Image:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if value.ndim == 4:
        value = value[0]
    if grayscale:
        if value.ndim == 3:
            value = value[0]
        array = value.mul(255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(array, mode="L")
    array = value.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def write_outpaint_diagnostic_artifacts(
    *,
    root: str | Path,
    run_id: str,
    source: torch.Tensor,
    canvas: torch.Tensor,
    generation_weight: torch.Tensor,
    decoded_before_composite: torch.Tensor,
    final_composite: torch.Tensor,
    metadata: Mapping[str, Any],
) -> Path:
    target = Path(root) / str(run_id) / "outpaint_prototype"
    target.mkdir(parents=True, exist_ok=True)
    _tensor_to_image(source).save(target / "source.png")
    _tensor_to_image(canvas).save(target / "canvas_before_sampling.png")
    _tensor_to_image(generation_weight, grayscale=True).save(target / "mask.png")
    _tensor_to_image(decoded_before_composite).save(target / "decoded_before_source_composite.png")
    _tensor_to_image(final_composite).save(target / "final_composite.png")
    (target / "outpaint_diagnostic.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return target


__all__ = [
    "OUTPAINT_ANCHORS",
    "OUTPAINT_CONTEXT_SEED_CONTRACT_VERSION",
    "OUTPAINT_CONTEXT_SEED_MODES",
    "OUTPAINT_FAILURE_STAGES",
    "OUTPAINT_LATENT_STRATEGIES",
    "OUTPAINT_MASK_CONTRACT_VERSION",
    "OUTPAINT_NOISE_POLICY_ID",
    "OUTPAINT_PRESERVATION_STRATEGY",
    "OUTPAINT_PROMPT_MODES",
    "OUTPAINT_PROMPT_OVERLAY_CONTRACT_VERSION",
    "OUTPAINT_SHAPE_EXPANSION_CONTRACT_VERSION",
    "OUTPAINT_SHAPE_TARGET_MODES",
    "OUTPAINT_SOURCE_HANDOFF_MODES",
    "OUTPAINT_PROTOTYPE_CONTRACT_VERSION",
    "OutpaintPrototypePlan",
    "StrictLatentPreservationHook",
    "build_canvas_and_masks",
    "build_seeded_outpaint_canvas",
    "build_outpaint_noise",
    "composite_exact_protected_core",
    "compose_outpaint_prompt_overlay",
    "extract_outpaint_failure_stage",
    "format_outpaint_failure",
    "initialize_outpaint_latents",
    "inspect_live_latent_alignment",
    "resolve_outpaint_shape_target",
    "resolve_outpaint_source_handoff",
    "embed_live_source_latent",
    "load_source_image",
    "outpaint_failure_label",
    "plan_outpaint_canvas",
    "resize_mask_to_latent",
    "write_outpaint_diagnostic_artifacts",
]
