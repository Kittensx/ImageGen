from __future__ import annotations

from typing import Any, Optional

import torch

from image_gen.contracts.model_conditioning import (
    BranchModelConditioningKwargs,
    ModelConditioningKwargs,
    select_model_conditioning_branch,
)
from modules.pipeline.sdxl_added_conditioning import (
    build_sdxl_branch_model_conditioning,
)


def align_conditioning_batch(
    tensor: torch.Tensor,
    target_batch_size: int,
    *,
    name: str,
) -> torch.Tensor:
    """Align a conditioning tensor's leading batch dimension.

    Prompt text is intentionally encoded once for the common case where one
    prompt generates multiple images. The resulting conditioning batch of one
    is broadcast to the latent batch here so CLIP is not redundantly executed.

    Only an exact match or a singleton conditioning batch is accepted. A
    non-singleton mismatch is ambiguous and therefore fails with a clear error.
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} conditioning must be a torch.Tensor.")
    if tensor.ndim < 1:
        raise ValueError(f"{name} conditioning must include a batch dimension.")

    target = int(target_batch_size)
    if target < 1:
        raise ValueError("target conditioning batch size must be at least 1.")

    current = int(tensor.shape[0])
    if current == target:
        return tensor
    if current == 1:
        expanded_shape = (target, *tensor.shape[1:])
        return tensor.expand(expanded_shape).contiguous()

    raise ValueError(
        f"{name} conditioning batch size {current} does not match latent "
        f"batch size {target}; only singleton conditioning can be broadcast."
    )


def _align_to_latents(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(latents) or latents.ndim < 1:
        raise ValueError("latents must be a tensor with a batch dimension.")

    target_batch_size = int(latents.shape[0])
    cond = cond.to(device=latents.device, dtype=latents.dtype)
    uncond = uncond.to(device=latents.device, dtype=latents.dtype)
    cond = align_conditioning_batch(
        cond, target_batch_size, name="positive"
    )
    uncond = align_conditioning_batch(
        uncond, target_batch_size, name="negative"
    )
    return cond, uncond


def get_conditioning_resolver(conditioning: Any):
    """
    Safely fetch a stepwise conditioning resolver from a ConditioningOutput-like object.

    Expected location:
        conditioning.extra["resolver"]

    Returns:
        resolver object if present, else None
    """
    conditioning_extra = getattr(conditioning, "extra", None)
    if isinstance(conditioning_extra, dict):
        return conditioning_extra.get("resolver")
    return None


def has_stepwise_conditioning(conditioning: Any) -> bool:
    """
    Returns True if conditioning provides a stepwise resolver.
    """
    return get_conditioning_resolver(conditioning) is not None


def resolve_step_conditioning(
    conditioning: Any,
    step_index: int,
    latents: Optional[torch.Tensor] = None,
    state: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resolve conditioning tensors for a given sampler step.

    Behavior:
    - Falls back to static conditioning.cond / conditioning.uncond
    - If conditioning.extra["resolver"] exists, uses resolver.resolve(step_index)
    - If latents is provided, aligns tensors to latents.device / latents.dtype
    - If state is provided and has a .conditioning namespace, syncs resolved tensors there

    Args:
        conditioning:
            ConditioningOutput-like object with:
                cond
                uncond
                extra (optional dict containing "resolver")

        step_index:
            0-based sampler loop index

        latents:
            Optional tensor used to align device / dtype

        state:
            Optional shared state to receive resolved cond/uncond for debugging/runtime sync

    Returns:
        (cond, uncond)
    """
    cond = getattr(conditioning, "cond", None)
    uncond = getattr(conditioning, "uncond", None)

    if cond is None or uncond is None:
        raise ValueError("conditioning must provide both `cond` and `uncond`.")

    resolver = get_conditioning_resolver(conditioning)
    if resolver is not None:
        cond, uncond = resolver.resolve(step_index=step_index)

    if latents is not None:
        cond, uncond = _align_to_latents(cond, uncond, latents)

    if state is not None and hasattr(state, "conditioning"):
        if hasattr(state.conditioning, "cond"):
            state.conditioning.cond = cond
        if hasattr(state.conditioning, "uncond"):
            state.conditioning.uncond = uncond

    return cond, uncond



def resolve_step_model_conditioning(
    conditioning: Any,
    step_index: int,
    *,
    latents: torch.Tensor,
    request: Any,
) -> ModelConditioningKwargs:
    """Resolve optional model-specific kwargs for the active sampler step.

    Legacy SD1/SD2 conditioning returns ``None``. SDXL resolves pooled
    positive/negative schedules plus geometry time IDs. SD3 resolves the
    branch-specific pooled projections required by the transformer.
    """

    conditioning_extra = getattr(conditioning, "extra", None)
    architecture = (
        str(conditioning_extra.get("conditioning_architecture") or "").strip().lower()
        if isinstance(conditioning_extra, dict)
        else ""
    )
    if architecture not in {"sdxl", "sd3.x"}:
        return None

    pooled_cond = getattr(conditioning, "pooled_cond", None)
    pooled_uncond = getattr(conditioning, "pooled_uncond", None)
    pooled_resolver = (
        conditioning_extra.get("pooled_resolver")
        if isinstance(conditioning_extra, dict)
        else None
    )
    if pooled_resolver is not None:
        resolve_pooled = getattr(pooled_resolver, "resolve_pooled", None)
        if not callable(resolve_pooled):
            raise TypeError("SDXL pooled_resolver must provide resolve_pooled(step_index).")
        pooled_cond, pooled_uncond = resolve_pooled(step_index=step_index)

    if pooled_cond is None or pooled_uncond is None:
        label = "SD3" if architecture == "sd3.x" else "SDXL"
        raise ValueError(f"{label} conditioning is missing pooled_cond/pooled_uncond.")

    if architecture == "sd3.x":
        if pooled_cond.ndim != 2 or pooled_uncond.ndim != 2:
            raise ValueError("SD3 pooled conditioning must be rank-2 [batch, width].")
        if int(pooled_cond.shape[-1]) != 2048 or int(pooled_uncond.shape[-1]) != 2048:
            raise ValueError(
                "SD3 pooled conditioning must be 2048-wide for CLIP-L + CLIP-G."
            )
        batch = int(latents.shape[0])
        if int(pooled_cond.shape[0]) == 1 and batch > 1:
            pooled_cond = pooled_cond.expand(batch, -1)
        if int(pooled_uncond.shape[0]) == 1 and batch > 1:
            pooled_uncond = pooled_uncond.expand(batch, -1)
        if int(pooled_cond.shape[0]) != batch or int(pooled_uncond.shape[0]) != batch:
            raise ValueError("SD3 pooled conditioning batch does not match latent batch.")
        if isinstance(conditioning_extra, dict):
            conditioning_extra["model_conditioning_contract"] = "sd3_pooled_projection_v1"
        return BranchModelConditioningKwargs(
            conditional={"pooled_projections": pooled_cond},
            unconditional={"pooled_projections": pooled_uncond},
        )

    model_conditioning, time_id_plan = build_sdxl_branch_model_conditioning(
        pooled_cond=pooled_cond,
        pooled_uncond=pooled_uncond,
        request=request,
        latents=latents,
    )

    # Metadata only; tensors remain on the branch contract. Keeping this small
    # makes diagnostics/replay JSON-safe without storing prompt embeddings.
    if isinstance(conditioning_extra, dict):
        conditioning_extra["sdxl_time_id_plan"] = time_id_plan.to_serializable_dict()
        conditioning_extra["model_conditioning_contract"] = "sdxl_text_time_v1"

    return model_conditioning


def select_step_model_conditioning_branch(
    model_conditioning: ModelConditioningKwargs,
    branch: str,
) -> dict[str, Any] | None:
    """Resolve one branch for raw-model samplers such as KES."""

    return select_model_conditioning_branch(model_conditioning, branch)


def call_with_optional_model_conditioning(
    callback: Any,
    *args: Any,
    model_conditioning: ModelConditioningKwargs,
    **kwargs: Any,
) -> Any:
    """Call a model callback without changing legacy call arity when unused."""

    if model_conditioning is None:
        return callback(*args, **kwargs)
    return callback(*args, model_conditioning, **kwargs)

def resolve_static_conditioning(
    conditioning: Any,
    latents: Optional[torch.Tensor] = None,
    state: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resolve only the static conditioning tensors, ignoring any resolver.

    Useful for debugging or for samplers that intentionally do not support
    scheduled prompt updates.
    """
    cond = getattr(conditioning, "cond", None)
    uncond = getattr(conditioning, "uncond", None)

    if cond is None or uncond is None:
        raise ValueError("conditioning must provide both `cond` and `uncond`.")

    if latents is not None:
        cond, uncond = _align_to_latents(cond, uncond, latents)

    if state is not None and hasattr(state, "conditioning"):
        if hasattr(state.conditioning, "cond"):
            state.conditioning.cond = cond
        if hasattr(state.conditioning, "uncond"):
            state.conditioning.uncond = uncond

    return cond, uncond


def build_step_conditioning_metadata(conditioning: Any) -> dict[str, Any]:
    """
    Small metadata helper for debug/UI/reporting.
    """
    resolver = get_conditioning_resolver(conditioning)
    return {
        "has_resolver": resolver is not None,
        "uses_stepwise_conditioning": resolver is not None,
    }