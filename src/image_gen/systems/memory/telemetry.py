from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from typing import Any, Callable

import torch

from .contracts import MemorySnapshot

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _system_memory() -> dict[str, Any]:
    if psutil is not None:
        try:
            memory = psutil.virtual_memory()
            return {
                "total_bytes": int(memory.total),
                "available_bytes": int(memory.available),
                "used_bytes": int(memory.used),
                "percent": float(memory.percent),
            }
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return {
            "total_bytes": page_size * total_pages,
            "available_bytes": page_size * available_pages,
            "used_bytes": None,
            "percent": None,
        }
    except Exception:
        return {
            "total_bytes": None,
            "available_bytes": None,
            "used_bytes": None,
            "percent": None,
        }


def _process_memory() -> dict[str, Any]:
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            info = process.memory_info()
            return {
                "pid": int(process.pid),
                "working_set_bytes": int(info.rss),
                "virtual_memory_bytes": int(info.vms),
            }
        except Exception:
            pass
    return {
        "pid": os.getpid(),
        "working_set_bytes": None,
        "virtual_memory_bytes": None,
    }


def normalize_cuda_memory_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Add explicit physical-VRAM and allocator semantics to a CUDA reading.

    ``torch.cuda.memory_reserved`` can exceed physical VRAM on Windows when the
    allocator is oversubscribed or backed by shared GPU memory. It must never be
    presented as physical VRAM usage. Physical usage is derived only from the
    atomic ``mem_get_info`` total/free pair and is clamped to the device total.
    """

    result = dict(payload or {})

    def _nonnegative_int(key: str) -> int | None:
        try:
            value = int(result.get(key))
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    total = _nonnegative_int("total_vram_bytes")
    free = _nonnegative_int("free_vram_bytes")
    allocated = _nonnegative_int("allocated_vram_bytes")
    reserved = _nonnegative_int("reserved_vram_bytes")

    if total is not None and total > 0 and free is not None:
        physical_free = min(free, total)
        physical_used = max(0, total - physical_free)
        result.update(
            {
                "physical_measurement_available": True,
                "physical_total_vram_bytes": total,
                "physical_free_vram_bytes": physical_free,
                "physical_used_vram_bytes": physical_used,
                "physical_measurement_source": "cuda_mem_get_info",
            }
        )
    else:
        result.update(
            {
                "physical_measurement_available": False,
                "physical_total_vram_bytes": total,
                "physical_free_vram_bytes": None,
                "physical_used_vram_bytes": None,
                "physical_measurement_source": None,
            }
        )

    allocator_committed = max(
        allocated if allocated is not None else 0,
        reserved if reserved is not None else 0,
    )
    overcommit = (
        max(0, allocator_committed - total)
        if total is not None and total > 0
        else None
    )
    result.update(
        {
            "allocator_committed_vram_bytes": allocator_committed,
            "allocator_overcommit_bytes": overcommit,
            "allocator_oversubscribed": bool(overcommit and overcommit > 0),
            "allocator_measurement_semantics": (
                "PyTorch allocated/reserved bytes are allocator measurements and may "
                "exceed physical VRAM under Windows shared-memory oversubscription."
            ),
        }
    )
    return result


class MemoryTelemetry:
    """Failure-isolated CUDA/system telemetry used by the memory manager."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        optional_gpu_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.optional_gpu_provider = optional_gpu_provider

    def cuda_payload(self) -> dict[str, Any]:
        unavailable = {
            "available": False,
            "device_name": None,
            "device_index": None,
            "total_vram_bytes": None,
            "free_vram_bytes": None,
            "allocated_vram_bytes": None,
            "reserved_vram_bytes": None,
            "peak_allocated_vram_bytes": None,
            "peak_reserved_vram_bytes": None,
        }
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return normalize_cuda_memory_payload(unavailable)
        try:
            index = self.device.index
            if index is None:
                index = int(torch.cuda.current_device())
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            return normalize_cuda_memory_payload({
                "available": True,
                "device_name": str(torch.cuda.get_device_name(index)),
                "device_index": int(index),
                "total_vram_bytes": int(total_bytes),
                "free_vram_bytes": int(free_bytes),
                "allocated_vram_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_vram_bytes": int(torch.cuda.memory_reserved(index)),
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(index)),
                "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(index)),
            })
        except Exception as exc:
            return normalize_cuda_memory_payload(
                {**unavailable, "error": f"{type(exc).__name__}: {exc}"}
            )

    def capture(
        self,
        stage: str,
        *,
        component_residency: list[dict[str, Any]] | None = None,
    ) -> MemorySnapshot:
        optional: dict[str, Any] = {}
        if callable(self.optional_gpu_provider):
            try:
                optional = dict(self.optional_gpu_provider() or {})
            except Exception as exc:
                optional = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        return MemorySnapshot(
            timestamp=_utc_now(),
            monotonic_ns=time.monotonic_ns(),
            pipeline_stage=str(stage),
            cuda=self.cuda_payload(),
            system=_system_memory(),
            process=_process_memory(),
            component_residency=list(component_residency or []),
            active_cuda_stream_count=None,
            optional_gpu_telemetry=optional,
        )

    def reset_peak(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass
