from __future__ import annotations

from typing import Any, Optional

import torch


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