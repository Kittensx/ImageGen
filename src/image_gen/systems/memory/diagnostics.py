from __future__ import annotations

from typing import Any


def memory_failure_bundle(
    *,
    request: Any,
    stage: str,
    error: BaseException,
    manager_summary: dict[str, Any],
    dimension_plan: Any = None,
) -> dict[str, Any]:
    request_payload = (
        request.to_serializable_dict()
        if hasattr(request, "to_serializable_dict")
        else {
            key: getattr(request, key, None)
            for key in (
                "width", "height", "steps", "cfg_scale", "seed",
                "sampler_name", "scheduler_name", "batch_size",
            )
        }
    )
    dimension_payload = (
        dimension_plan.to_serializable_dict()
        if hasattr(dimension_plan, "to_serializable_dict")
        else dimension_plan
    )
    return {
        "format": "image-gen-memory-failure-v1",
        "active_stage": str(stage),
        "error_type": type(error).__name__,
        "error": str(error),
        "request": request_payload,
        "dimension_plan": dimension_payload,
        "memory_manager": manager_summary,
    }
