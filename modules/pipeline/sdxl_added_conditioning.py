from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from image_gen.contracts.model_conditioning import BranchModelConditioningKwargs


SDXL_ADDITION_TIME_EMBED_DIM = 256
SDXL_TIME_ID_LENGTH = 6
SDXL_POOLED_DIM = 1280
SDXL_PROJECTION_CLASS_EMBEDDINGS_INPUT_DIM = 2816


@dataclass(frozen=True)
class SDXLTimeIdPlan:
    original_height: int
    original_width: int
    crop_top: int
    crop_left: int
    target_height: int
    target_width: int
    source: str

    @property
    def values(self) -> tuple[int, int, int, int, int, int]:
        return (
            int(self.original_height),
            int(self.original_width),
            int(self.crop_top),
            int(self.crop_left),
            int(self.target_height),
            int(self.target_width),
        )

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "original_size": [int(self.original_height), int(self.original_width)],
            "crop_coords_top_left": [int(self.crop_top), int(self.crop_left)],
            "target_size": [int(self.target_height), int(self.target_width)],
            "source": str(self.source),
        }


def _positive_int(value: Any, *, name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SDXL {name} must be an integer.") from exc
    if resolved <= 0:
        raise ValueError(f"SDXL {name} must be greater than zero; got {resolved}.")
    return resolved


def _nonnegative_int(value: Any, *, name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SDXL {name} must be an integer.") from exc
    if resolved < 0:
        raise ValueError(f"SDXL {name} must be non-negative; got {resolved}.")
    return resolved


def _dimension_plan_from_request(request: Any) -> Mapping[str, Any] | None:
    value = getattr(request, "dimension_plan", None)
    return value if isinstance(value, Mapping) else None


def resolve_sdxl_time_id_plan(
    *,
    request: Any,
    latents: torch.Tensor,
) -> SDXLTimeIdPlan:
    """Resolve SDXL micro-conditioning geometry from the canonical dimension plan.

    For IMAGE_GEN's arbitrary output dimensions, the SDXL "original" size is
    the model-compatible generation canvas, the crop coordinates describe the
    center crop applied after decode, and the target size is the exact requested
    output. Exact-grid requests therefore produce the canonical
    ``[H, W, 0, 0, H, W]`` form.
    """

    if not torch.is_tensor(latents) or latents.ndim != 4:
        raise ValueError("SDXL time IDs require BCHW latent tensors.")

    plan = _dimension_plan_from_request(request)
    if plan is not None:
        original_width = _positive_int(plan.get("generation_width"), name="generation_width")
        original_height = _positive_int(plan.get("generation_height"), name="generation_height")
        target_width = _positive_int(plan.get("requested_width"), name="requested_width")
        target_height = _positive_int(plan.get("requested_height"), name="requested_height")
        crop_left = _nonnegative_int(plan.get("crop_left", 0), name="crop_left")
        crop_top = _nonnegative_int(plan.get("crop_top", 0), name="crop_top")
        latent_scale = _positive_int(plan.get("latent_scale_factor", 8), name="latent_scale_factor")
        expected_height = int(latents.shape[-2]) * latent_scale
        expected_width = int(latents.shape[-1]) * latent_scale
        if (expected_height, expected_width) != (original_height, original_width):
            raise ValueError(
                "SDXL dimension plan does not match the active latent canvas: "
                f"plan={(original_height, original_width)}, "
                f"latents={(expected_height, expected_width)}."
            )
        source = "generation_dimension_plan"
    else:
        # Direct model-callback tests and older callers may not have executed
        # LatentPreparationSystem.plan_dimensions(). Accept that only when the
        # requested size exactly matches the latent canvas; arbitrary crops must
        # use the canonical dimension plan so there is no duplicate crop policy.
        latent_scale = 8
        original_height = int(latents.shape[-2]) * latent_scale
        original_width = int(latents.shape[-1]) * latent_scale
        target_width = _positive_int(getattr(request, "width", 0), name="request.width")
        target_height = _positive_int(getattr(request, "height", 0), name="request.height")
        if (target_height, target_width) != (original_height, original_width):
            raise ValueError(
                "SDXL arbitrary-size time IDs require request.dimension_plan from "
                "LatentPreparationSystem; requested size does not match latent canvas."
            )
        crop_left = 0
        crop_top = 0
        source = "exact_request_fallback"

    if crop_top + target_height > original_height:
        raise ValueError("SDXL crop_top + target_height exceeds generation_height.")
    if crop_left + target_width > original_width:
        raise ValueError("SDXL crop_left + target_width exceeds generation_width.")

    return SDXLTimeIdPlan(
        original_height=original_height,
        original_width=original_width,
        crop_top=crop_top,
        crop_left=crop_left,
        target_height=target_height,
        target_width=target_width,
        source=source,
    )


def build_sdxl_time_ids(
    *,
    request: Any,
    latents: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, SDXLTimeIdPlan]:
    plan = resolve_sdxl_time_id_plan(request=request, latents=latents)
    time_ids = torch.tensor([plan.values], device=device, dtype=dtype)
    if int(batch_size) > 1:
        time_ids = time_ids.expand(int(batch_size), SDXL_TIME_ID_LENGTH).contiguous()
    return time_ids, plan


def validate_sdxl_added_embedding_contract() -> None:
    computed = SDXL_ADDITION_TIME_EMBED_DIM * SDXL_TIME_ID_LENGTH + SDXL_POOLED_DIM
    if computed != SDXL_PROJECTION_CLASS_EMBEDDINGS_INPUT_DIM:
        raise RuntimeError(
            "SDXL added-conditioning dimensions are inconsistent: "
            f"{SDXL_ADDITION_TIME_EMBED_DIM} * {SDXL_TIME_ID_LENGTH} + "
            f"{SDXL_POOLED_DIM} = {computed}, expected "
            f"{SDXL_PROJECTION_CLASS_EMBEDDINGS_INPUT_DIM}."
        )


def build_sdxl_branch_model_conditioning(
    *,
    pooled_cond: torch.Tensor,
    pooled_uncond: torch.Tensor,
    request: Any,
    latents: torch.Tensor,
) -> tuple[BranchModelConditioningKwargs, SDXLTimeIdPlan]:
    validate_sdxl_added_embedding_contract()
    if not torch.is_tensor(pooled_cond) or not torch.is_tensor(pooled_uncond):
        raise TypeError("SDXL added conditioning requires pooled cond and uncond tensors.")
    if pooled_cond.ndim != 2 or pooled_uncond.ndim != 2:
        raise ValueError("SDXL pooled conditioning must have shape [batch, 1280].")
    if int(pooled_cond.shape[-1]) != SDXL_POOLED_DIM:
        raise ValueError(
            f"SDXL pooled_cond must be {SDXL_POOLED_DIM}-wide; got {tuple(pooled_cond.shape)}."
        )
    if int(pooled_uncond.shape[-1]) != SDXL_POOLED_DIM:
        raise ValueError(
            f"SDXL pooled_uncond must be {SDXL_POOLED_DIM}-wide; got {tuple(pooled_uncond.shape)}."
        )

    batch_size = int(latents.shape[0])
    pooled_cond = pooled_cond.to(device=latents.device, dtype=latents.dtype)
    pooled_uncond = pooled_uncond.to(device=latents.device, dtype=latents.dtype)

    def _align(value: torch.Tensor, label: str) -> torch.Tensor:
        current = int(value.shape[0])
        if current == batch_size:
            return value
        if current == 1:
            return value.expand(batch_size, value.shape[1]).contiguous()
        raise ValueError(
            f"SDXL {label} batch size {current} does not match latent batch size {batch_size}; "
            "only singleton pooled conditioning can be broadcast."
        )

    pooled_cond = _align(pooled_cond, "pooled_cond")
    pooled_uncond = _align(pooled_uncond, "pooled_uncond")
    time_ids, plan = build_sdxl_time_ids(
        request=request,
        latents=latents,
        batch_size=batch_size,
        device=latents.device,
        dtype=latents.dtype,
    )

    return (
        BranchModelConditioningKwargs(
            conditional={
                "added_cond_kwargs": {
                    "text_embeds": pooled_cond,
                    "time_ids": time_ids,
                }
            },
            unconditional={
                "added_cond_kwargs": {
                    "text_embeds": pooled_uncond,
                    "time_ids": time_ids,
                }
            },
        ),
        plan,
    )
