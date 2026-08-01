from __future__ import annotations

import hashlib
from typing import Any

import torch


def tensor_digest(tensor: torch.Tensor | None) -> str | None:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def tensor_statistics(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None or not torch.is_tensor(tensor):
        return {}
    value = tensor.detach().to(device="cpu", dtype=torch.float32)
    finite = torch.isfinite(value)
    finite_values = value[finite]
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "finite": bool(finite.all().item()),
        "finite_count": int(finite.sum().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()),
                "norm": float(torch.linalg.vector_norm(finite_values).item()),
            }
        )
    return result


def compare_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "materially_equivalent": False,
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    a = left.detach().to(device="cpu", dtype=torch.float32)
    b = right.detach().to(device="cpu", dtype=torch.float32)
    delta = (a - b).abs()
    rmse = torch.sqrt(torch.mean((a - b) ** 2))
    cosine = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0), dim=1
    )
    return {
        "materially_equivalent": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "shape_equal": True,
        "exact_digest_equal": tensor_digest(a) == tensor_digest(b),
        "max_abs_difference": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_difference": float(delta.mean().item()) if delta.numel() else 0.0,
        "rmse": float(rmse.item()),
        "cosine_similarity": float(cosine.item()),
        "atol": float(atol),
        "rtol": float(rtol),
    }
