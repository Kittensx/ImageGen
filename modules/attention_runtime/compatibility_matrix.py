from __future__ import annotations

import os
import time
from typing import Any, Callable

from .reports import module_device_dtype


def _unique_test_layouts(signature: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for layout in signature.get("unique_layouts", []):
        if not all(isinstance(layout.get(name), int) for name in ("heads", "q_head_dim", "k_head_dim", "v_head_dim")):
            continue
        result.append(dict(layout))
    return result


def run_xformers_layout_matrix(
    unet: Any,
    signature: dict[str, Any],
    *,
    executor: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute representative model-derived layouts through the xFormers API.

    This is a pre-sampling compatibility gate. Runtime sequence lengths are
    captured separately during the first real UNet call.
    """

    import torch

    identity = module_device_dtype(unet)
    device = torch.device(identity["device"])
    dtype_name = identity["dtype"]
    dtype = getattr(torch, dtype_name, None)
    if device.type != "cuda":
        raise RuntimeError(
            "xFormers layout validation requires the UNet to be on its final CUDA device; "
            f"found {device}."
        )
    if dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "xFormers layout validation requires the UNet's final FP16/BF16 dtype; "
            f"found {dtype_name}."
        )
    if executor is None:
        import xformers.ops as xops

        executor = xops.memory_efficient_attention

    layouts = _unique_test_layouts(signature)
    if not layouts:
        raise RuntimeError(
            "No valid projection-derived attention layouts were available for xFormers validation."
        )

    results: list[dict[str, Any]] = []
    for layout in layouts:
        heads = int(layout["heads"])
        q_dim = int(layout["q_head_dim"])
        k_dim = int(layout["k_head_dim"])
        v_dim = int(layout["v_head_dim"])
        kind = str(layout.get("attention_kind") or "self_or_cross")
        q_length = 2
        kv_length = 77 if kind == "cross" else 2
        test_result: dict[str, Any] = {
            "attention_kind": kind,
            "heads": heads,
            "q_head_dim": q_dim,
            "k_head_dim": k_dim,
            "v_head_dim": v_dim,
            "query_length": q_length,
            "key_value_length": kv_length,
            "dtype": dtype_name,
            "device": str(device),
            "module_paths": list(layout.get("module_paths") or []),
            "passed": False,
            "output_shape": None,
            "error": None,
        }
        try:
            query = torch.randn((1, q_length, heads, q_dim), device=device, dtype=dtype)
            key = torch.randn((1, kv_length, heads, k_dim), device=device, dtype=dtype)
            value = torch.randn((1, kv_length, heads, v_dim), device=device, dtype=dtype)
            capture_performance = str(
                os.environ.get("IMAGE_GEN_CAPTURE_ATTENTION_PERFORMANCE", "")
            ).strip().lower() in {"1", "true", "yes", "on", "enabled"}
            if capture_performance:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                first_started = time.perf_counter()
                output = executor(query, key, value, attn_bias=None, p=0.0)
                torch.cuda.synchronize(device)
                test_result["first_call_duration_ms"] = round(
                    (time.perf_counter() - first_started) * 1000.0, 3
                )
                test_result["first_call_peak_allocated_vram_bytes"] = int(
                    torch.cuda.max_memory_allocated(device)
                )
                warm_started = time.perf_counter()
                warm_output = executor(query, key, value, attn_bias=None, p=0.0)
                torch.cuda.synchronize(device)
                test_result["warm_call_duration_ms"] = round(
                    (time.perf_counter() - warm_started) * 1000.0, 3
                )
                test_result["warm_output_shape"] = [
                    int(item) for item in warm_output.shape
                ]
            else:
                output = executor(query, key, value, attn_bias=None, p=0.0)
            test_result["output_shape"] = [int(item) for item in output.shape]
            test_result["passed"] = True
        except Exception as exc:
            test_result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(test_result)

    failed = [item for item in results if not item["passed"]]
    return {
        "schema_version": 1,
        "test_kind": "projection_derived_pre_sampling_matrix",
        "passed": not failed,
        "device": str(device),
        "dtype": dtype_name,
        "layout_count": len(results),
        "results": results,
        "failure_count": len(failed),
    }
