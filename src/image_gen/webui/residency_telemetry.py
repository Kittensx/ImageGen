from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from image_gen.systems.memory.telemetry import normalize_cuda_memory_payload


_RESIDENCY_LABELS = {
    "empty": "Unloaded",
    "managed_resident": "Managed Resident",
    "hot_gpu": "Hot GPU",
    "hot_staged": "Hot Staged",
    "switching": "Switching",
    "recovering": "Recovering",
}

_REUSE_LABELS = {
    "cold_load": "Cold load",
    "model_switch": "Model switch",
    "resident_managed_reuse": "Managed resident reuse",
    "managed_resident_reuse": "Managed resident reuse",
    "managed_reuse": "Managed resident reuse",
    "hot_reuse": "Hot reuse",
    "hot_staged_reuse": "Hot staged reuse",
}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_ms(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, 3)


def _model_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "\\" in text:
        return PureWindowsPath(text).name
    return Path(text).name


def _component_groups(runtime: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    component_devices = _dict(runtime.get("component_devices"))
    gpu: list[str] = []
    cpu_ready: list[str] = []
    for component, device in component_devices.items():
        name = str(component)
        normalized = str(device or "").strip().lower()
        if normalized.startswith("cuda"):
            gpu.append(name)
        elif normalized.startswith("cpu"):
            cpu_ready.append(name)
    return sorted(gpu), sorted(cpu_ready)


def latest_runtime_job_report(runtime_status: Mapping[str, Any] | None, *, job_id: str | None = None) -> dict[str, Any]:
    runtime = _dict(runtime_status)
    reports = runtime.get("recent_job_reports")
    if not isinstance(reports, list):
        return {}
    requested = str(job_id or "").strip()
    for raw in reversed(reports):
        report = _dict(raw)
        if not report:
            continue
        if not requested or str(report.get("job_id") or "").strip() == requested:
            return report
    return {}


def normalize_runtime_residency_status(runtime_status: Mapping[str, Any] | None) -> dict[str, Any]:
    runtime = _dict(runtime_status)
    state = str(runtime.get("residency_state_effective") or "empty").strip().lower() or "empty"
    requested = str(runtime.get("residency_mode_requested") or "managed").strip().lower() or "managed"
    stage = str(runtime.get("stage") or "idle").strip() or "idle"
    current_model_path = str(runtime.get("current_model_path") or runtime.get("model_path") or "").strip()
    active_gpu, cpu_ready = _component_groups(runtime)
    lease = _dict(runtime.get("composition_execution_lease"))
    prepared = _dict(lease.get("prepared_compositions"))
    warm_states = _dict(lease.get("warm_states"))
    latest_report = latest_runtime_job_report(runtime)
    report_timings = _dict(latest_report.get("timings"))
    runtime_timings = _dict(runtime.get("timings"))
    classification = str(
        latest_report.get("generation_residency_classification")
        or runtime.get("last_generation_residency_classification")
        or ""
    ).strip().lower()
    activation_ms = _finite_ms(
        runtime_timings.get("activate_time_ms")
        if runtime_timings.get("activate_time_ms") is not None
        else runtime_timings.get("initial_activation_time_ms")
    )
    preparation_ms = _finite_ms(
        report_timings.get("request_setup_time_ms")
        if report_timings.get("request_setup_time_ms") is not None
        else report_timings.get("next_job_preparation_time_ms")
    )
    checkpoint_hydration_ms = _finite_ms(report_timings.get("checkpoint_hydration_time_ms"))
    if classification in {"hot_reuse", "hot_staged_reuse", "resident_managed_reuse", "managed_resident_reuse", "managed_reuse"}:
        hydration_state = "none"
    elif checkpoint_hydration_ms is not None and checkpoint_hydration_ms > 0:
        hydration_state = "performed"
    elif checkpoint_hydration_ms == 0:
        hydration_state = "none"
    else:
        hydration_state = "unknown"

    return {
        "schema_version": 1,
        "requested_mode": requested,
        "effective_state": state,
        "effective_label": _RESIDENCY_LABELS.get(state, state.replace("_", " ").title()),
        "stage": stage,
        "current_model_path": current_model_path or None,
        "current_model_name": _model_name(current_model_path) or None,
        "active_gpu_components": active_gpu,
        "cpu_ready_components": cpu_ready,
        "lease_state": str(lease.get("state") or "inactive"),
        "lease_generation": int(lease.get("generation") or 0),
        "lease_component_pool_count": int(lease.get("component_pool_count") or 0),
        "prepared_composition_count": len(prepared),
        "warm_states": warm_states,
        "last_generation_residency_classification": classification or None,
        "last_generation_label": _REUSE_LABELS.get(classification, classification.replace("_", " ").title() if classification else "None"),
        "last_activation_ms": activation_ms,
        "last_preparation_ms": preparation_ms,
        "checkpoint_hydration_state": hydration_state,
        "checkpoint_hydration_time_ms": checkpoint_hydration_ms,
        "hot_reuse_count": int(runtime.get("hot_reuse_count") or 0),
        "cold_or_switch_load_count": int(runtime.get("cold_or_switch_load_count") or 0),
        "last_residency_reason": str(runtime.get("last_residency_reason") or "").strip() or None,
        "last_residency_transition": runtime.get("last_residency_transition"),
    }


def merge_runtime_into_memory_status(
    memory_status: Mapping[str, Any] | None,
    runtime_status: Mapping[str, Any] | None,
    *,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Preserve raw memory-manager telemetry while layering current residency truth.

    HMR/CNRR residency and AdaptiveComponentMemoryManager telemetry answer different
    questions. This helper deliberately keeps the raw memory snapshot/peaks and adds
    normalized resident-runtime semantics instead of allowing one stream to replace
    the other.
    """

    previous = _dict(memory_status)
    runtime = _dict(runtime_status)
    semantic = normalize_runtime_residency_status(runtime)
    runtime_memory = _dict(runtime.get("memory"))
    previous_snapshot = _dict(previous.get("latest_snapshot"))
    previous_cuda = _dict(previous_snapshot.get("cuda"))
    component_devices = _dict(runtime.get("component_devices"))
    runtime_cleared = str(semantic.get("effective_state") or "").lower() in {"empty", "recovering"}
    active_gpu = [] if runtime_cleared else list(semantic.get("active_gpu_components") or [])
    cpu_ready = [] if runtime_cleared else list(semantic.get("cpu_ready_components") or [])

    raw_offloaded = [str(value) for value in (previous.get("offloaded_components") or [])]
    cpu_ready_set = set(cpu_ready)
    offloaded = [] if runtime_cleared else sorted({value for value in raw_offloaded if value not in cpu_ready_set})
    if runtime_cleared:
        component_devices = {}

    if runtime_memory:
        normalized_cuda = normalize_cuda_memory_payload(
            {
                **previous_cuda,
                "available": True,
                "device_name": runtime_memory.get("device_name") or previous_cuda.get("device_name"),
                "allocated_vram_bytes": runtime_memory.get("allocated_bytes"),
                "reserved_vram_bytes": runtime_memory.get("reserved_bytes"),
                "free_vram_bytes": runtime_memory.get("free_bytes"),
                "total_vram_bytes": runtime_memory.get("total_bytes"),
            }
        )
    else:
        normalized_cuda = normalize_cuda_memory_payload(previous_cuda)

    current_allocated = normalized_cuda.get("allocated_vram_bytes")
    current_reserved = normalized_cuda.get("reserved_vram_bytes")
    previous_peak_allocated = previous.get("peak_allocated_vram_bytes")
    previous_peak_reserved = previous.get("peak_reserved_vram_bytes")

    merged = {
        **previous,
        "event": "model_runtime_status",
        "stage": runtime.get("stage") or previous.get("stage"),
        "active_stage": runtime.get("active_stage") or runtime.get("stage") or previous.get("active_stage"),
        "active_gpu_components": active_gpu,
        "cpu_ready_components": cpu_ready,
        "offloaded_components": offloaded,
        "component_devices": component_devices,
        "runtime_residency": semantic,
        "composition_execution_lease": _dict(runtime.get("composition_execution_lease")),
        "latest_snapshot": {
            **previous_snapshot,
            "pipeline_stage": runtime.get("stage") or previous_snapshot.get("pipeline_stage"),
            "cuda": normalized_cuda,
        },
        "telemetry_source": "memory_manager+resident_model_runtime",
    }
    if updated_at is not None:
        merged["updated_at"] = updated_at

    def _max_int(*values: Any) -> int:
        parsed: list[int] = []
        for value in values:
            try:
                parsed.append(max(0, int(value or 0)))
            except (TypeError, ValueError):
                continue
        return max(parsed or [0])

    merged["peak_allocated_vram_bytes"] = _max_int(previous_peak_allocated, current_allocated)
    merged["peak_reserved_vram_bytes"] = _max_int(previous_peak_reserved, current_reserved)
    merged["job_peak_allocated_vram_bytes"] = _max_int(
        previous.get("job_peak_allocated_vram_bytes"), previous_peak_allocated, current_allocated
    )
    merged["job_peak_reserved_vram_bytes"] = _max_int(
        previous.get("job_peak_reserved_vram_bytes"), previous_peak_reserved, current_reserved
    )
    return merged
