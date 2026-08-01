from __future__ import annotations

from typing import Any


def compare_tensors(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_equal": False,
            "reference_shape": [int(item) for item in reference.shape],
            "candidate_shape": [int(item) for item in candidate.shape],
            "finite": False,
        }
    difference = (candidate.float() - reference.float()).abs()
    denominator = reference.float().abs().clamp_min(torch.finfo(torch.float32).eps)
    relative = difference / denominator
    return {
        "shape_equal": True,
        "reference_shape": [int(item) for item in reference.shape],
        "candidate_shape": [int(item) for item in candidate.shape],
        "finite": bool(torch.isfinite(candidate).all().item()),
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "max_relative_error": float(relative.max().item()),
        "mean_relative_error": float(relative.mean().item()),
    }
