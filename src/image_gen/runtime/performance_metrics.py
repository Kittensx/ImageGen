from __future__ import annotations

from contextlib import contextmanager
import os
import time
from typing import Any, Callable, Iterator, TypeVar

import torch

T = TypeVar("T")

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def performance_capture_requested(request: Any | None = None) -> bool:
    diagnostics = getattr(request, "diagnostics", None)
    if isinstance(diagnostics, dict) and _truthy(
        diagnostics.get("capture_attention_performance")
        or diagnostics.get("capture_performance")
    ):
        return True
    return _truthy(os.environ.get("IMAGE_GEN_CAPTURE_ATTENTION_PERFORMANCE"))


class GenerationPerformanceRecorder:
    """Opt-in, failure-isolated timing and CUDA-memory recorder.

    CUDA synchronization and peak-stat resets add measurement overhead, so this
    recorder is disabled unless the request or environment explicitly enables it.
    Resetting peak statistics does not free or move tensors; it only scopes the
    reported peak to the measured stage.
    """

    FORMAT = "image-gen-attention-performance-v1"

    def __init__(
        self,
        *,
        enabled: bool,
        device: str | torch.device | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.started = time.perf_counter()
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.completed = False

    @classmethod
    def from_request(
        cls,
        request: Any | None,
        *,
        device: str | torch.device | None = None,
    ) -> "GenerationPerformanceRecorder":
        return cls(enabled=performance_capture_requested(request), device=device)

    def _cuda_enabled(self) -> bool:
        return self.enabled and self.device.type == "cuda" and torch.cuda.is_available()

    def _synchronize(self) -> None:
        if not self._cuda_enabled():
            return
        try:
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # telemetry must not break generation
            self.errors.append(f"synchronize: {type(exc).__name__}: {exc}")

    def _cuda_snapshot(self) -> dict[str, Any]:
        if not self._cuda_enabled():
            return {
                "available": False,
                "allocated_bytes": None,
                "reserved_bytes": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "free_bytes": None,
                "total_bytes": None,
            }
        try:
            index = self.device.index
            if index is None:
                index = int(torch.cuda.current_device())
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            peak_reserved = getattr(torch.cuda, "max_memory_reserved", None)
            return {
                "available": True,
                "device_index": int(index),
                "device_name": str(torch.cuda.get_device_name(index)),
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "peak_reserved_bytes": (
                    int(peak_reserved(index)) if callable(peak_reserved) else None
                ),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
        except Exception as exc:
            self.errors.append(f"cuda_snapshot: {type(exc).__name__}: {exc}")
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def _reset_peak(self) -> None:
        if not self._cuda_enabled():
            return
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except Exception as exc:
            self.errors.append(f"reset_peak: {type(exc).__name__}: {exc}")

    @contextmanager
    def measure(self, stage: str, *, operation: str | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._synchronize()
        before = self._cuda_snapshot()
        self._reset_peak()
        started = time.perf_counter()
        status = "passed"
        error: str | None = None
        try:
            yield
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._synchronize()
            duration_ms = (time.perf_counter() - started) * 1000.0
            after = self._cuda_snapshot()
            self.records.append(
                {
                    "stage": str(stage),
                    "operation": str(operation or stage),
                    "status": status,
                    "error": error,
                    "duration_ms": round(duration_ms, 3),
                    "cuda_before": before,
                    "cuda_after": after,
                    "peak_allocated_vram_bytes": after.get("peak_allocated_bytes"),
                    "peak_reserved_vram_bytes": after.get("peak_reserved_bytes"),
                }
            )

    def run(
        self,
        stage: str,
        operation: Callable[[], T],
        *,
        operation_name: str | None = None,
    ) -> T:
        with self.measure(stage, operation=operation_name):
            return operation()

    def finish(
        self,
        *,
        attention_backend: dict[str, Any] | None = None,
        memory_management: dict[str, Any] | None = None,
        hires_enabled: bool = False,
    ) -> dict[str, Any]:
        self.completed = True
        total_ms = (time.perf_counter() - self.started) * 1000.0
        records = list(self.records)
        by_stage = {str(item["stage"]): dict(item) for item in records}
        peaks = [
            int(item["peak_allocated_vram_bytes"])
            for item in records
            if isinstance(item.get("peak_allocated_vram_bytes"), int)
        ]
        memory = dict(memory_management or {})
        attention = dict(attention_backend or {})
        attempts = list(attention.get("activation_attempts") or [])
        fallback_count = sum(1 for item in attempts if not item.get("verified"))
        compatibility = dict(attention.get("xformers_compatibility") or {})
        matrix = dict(compatibility.get("compatibility_matrix") or {})
        layout_timings = [
            {
                "attention_kind": item.get("attention_kind"),
                "heads": item.get("heads"),
                "head_dimension": item.get("q_head_dim"),
                "first_call_duration_ms": item.get("first_call_duration_ms"),
                "warm_call_duration_ms": item.get("warm_call_duration_ms"),
                "first_call_peak_allocated_vram_bytes": item.get(
                    "first_call_peak_allocated_vram_bytes"
                ),
            }
            for item in matrix.get("results", [])
            if item.get("first_call_duration_ms") is not None
        ]
        return {
            "format": self.FORMAT,
            "enabled": self.enabled,
            "measurement_overhead": (
                "CUDA synchronization and stage-scoped peak-stat resets were enabled."
                if self.enabled
                else "Performance capture was disabled."
            ),
            "completed": self.completed,
            "device": str(self.device),
            "hires_enabled": bool(hires_enabled),
            "total_generation_duration_ms": round(total_ms, 3),
            "stages": records,
            "stage_index": by_stage,
            "overall_peak_allocated_vram_bytes": max(peaks) if peaks else None,
            "base_pass_peak_allocated_vram_bytes": (
                by_stage.get("base_sampling", {}).get("peak_allocated_vram_bytes")
            ),
            "hires_pass_peak_allocated_vram_bytes": (
                by_stage.get("hires_second_pass", {}).get("peak_allocated_vram_bytes")
            ),
            "final_decode_peak_allocated_vram_bytes": (
                by_stage.get("final_decode", {}).get("peak_allocated_vram_bytes")
            ),
            "component_transfer_count": int(memory.get("component_transfer_count") or len(memory.get("transfers") or [])),
            "fallback_count": fallback_count,
            "attention_initialization": dict(attention.get("initialization_metrics") or {}),
            "layout_first_and_warm_calls": layout_timings,
            "errors": list(self.errors),
        }
