from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

CANONICAL_GUIDANCE_MATH_VERSION = "phase11g_shared_guidance_v1"


@dataclass(frozen=True)
class GuidanceSemantics:
    guidance_owner: str
    guidance_mode: str
    guidance_math_version: str = CANONICAL_GUIDANCE_MATH_VERSION
    canonical_cfg_rescale: float = 0.0
    legacy_clamp_guidance: bool = False
    legacy_guidance_multiplier: float = 1.0
    legacy_clamp_range: tuple[float, float] = (-1.0, 1.0)
    guidance_path: str = "canonical_cfg"

    @property
    def canonical_cfg_rescale_applied(self) -> bool:
        return float(self.canonical_cfg_rescale) > 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "guidance_owner": str(self.guidance_owner),
            "guidance_mode": str(self.guidance_mode),
            "guidance_math_version": str(self.guidance_math_version),
            "guidance_path": str(self.guidance_path),
            "cfg_rescale": float(self.canonical_cfg_rescale),
            "cfg_rescale_applied": self.canonical_cfg_rescale_applied,
            "legacy_clamp_guidance": bool(self.legacy_clamp_guidance),
            "legacy_guidance_multiplier": float(self.legacy_guidance_multiplier),
            "legacy_clamp_range": [
                float(self.legacy_clamp_range[0]),
                float(self.legacy_clamp_range[1]),
            ],
        }


@dataclass
class GuidanceComputationResult:
    guided: torch.Tensor
    conditional: torch.Tensor
    unconditional: torch.Tensor
    delta: torch.Tensor
    guided_pre_rescale: torch.Tensor
    guided_post_rescale: torch.Tensor
    metadata: dict[str, Any]



def _normalize_clamp_range(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start = float(value[0])
            end = float(value[1])
            if start <= end:
                return (start, end)
        except (TypeError, ValueError):
            pass
    return (-1.0, 1.0)



def apply_canonical_cfg_rescale(
    guided: torch.Tensor,
    conditional: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    if float(factor) <= 0.0:
        return guided
    dimensions = tuple(range(1, guided.ndim))
    guided_std = guided.float().std(dim=dimensions, keepdim=True, unbiased=False)
    conditional_std = conditional.float().std(
        dim=dimensions, keepdim=True, unbiased=False
    )
    eps = torch.finfo(torch.float32).eps
    rescaled = guided.float() * (conditional_std / torch.clamp(guided_std, min=eps))
    blended = float(factor) * rescaled + (1.0 - float(factor)) * guided.float()
    return blended.to(dtype=guided.dtype)



def combine_guidance_outputs(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    *,
    effective_cfg_scale: float,
    semantics: GuidanceSemantics,
) -> GuidanceComputationResult:
    delta = conditional - unconditional
    guided_pre_rescale = unconditional + (
        float(effective_cfg_scale) * float(semantics.legacy_guidance_multiplier) * delta
    )
    guided_post_rescale = apply_canonical_cfg_rescale(
        guided_pre_rescale,
        conditional,
        float(semantics.canonical_cfg_rescale),
    )
    guided = guided_post_rescale
    if semantics.legacy_clamp_guidance:
        clamp_min, clamp_max = _normalize_clamp_range(semantics.legacy_clamp_range)
        guided = torch.clamp(guided, clamp_min, clamp_max)

    metadata = semantics.to_metadata()
    metadata.update(
        {
            "requested_cfg_scale": float(effective_cfg_scale),
            "effective_cfg_scale": float(effective_cfg_scale)
            * float(semantics.legacy_guidance_multiplier),
        }
    )
    return GuidanceComputationResult(
        guided=guided,
        conditional=conditional,
        unconditional=unconditional,
        delta=delta,
        guided_pre_rescale=guided_pre_rescale,
        guided_post_rescale=guided_post_rescale,
        metadata=metadata,
    )
