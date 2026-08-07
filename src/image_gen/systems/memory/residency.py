from __future__ import annotations

import time
from typing import Any, Iterable

import torch

from modules.component_placement import component_placement_report, place_component

from .contracts import ComponentTransferRecord, ManagedComponent


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def estimate_module_bytes(module: torch.nn.Module) -> tuple[int, int]:
    parameter_bytes = sum(_tensor_bytes(parameter) for parameter in module.parameters())
    buffer_bytes = sum(_tensor_bytes(buffer) for buffer in module.buffers())
    return int(parameter_bytes), int(buffer_bytes)


def module_device(module: torch.nn.Module) -> str:
    for parameter in module.parameters():
        return str(parameter.device)
    for buffer in module.buffers():
        return str(buffer.device)
    return "cpu"


def module_dtype(module: torch.nn.Module) -> str:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return str(parameter.dtype)
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return str(buffer.dtype)
    return "torch.float32"


def parse_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if isinstance(value, torch.dtype):
        return value
    lookup = {
        "torch.float16": torch.float16,
        "float16": torch.float16,
        "torch.float32": torch.float32,
        "float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return lookup.get(str(value or "").strip().lower())


class ComponentResidencyRegistry:
    def __init__(self) -> None:
        self.components: dict[str, ManagedComponent] = {}
        self.transfer_records: list[ComponentTransferRecord] = []

    def register(
        self,
        *,
        component_id: str,
        component_kind: str,
        model_identity: str,
        module: torch.nn.Module,
        preferred_dtype: str | torch.dtype | None = None,
        required_by_stages: Iterable[str] = (),
        estimated_runtime_overhead_bytes: int = 0,
        pinned_cpu_capable: bool = False,
        supports_non_blocking_transfer: bool = False,
        unload_callback: Any = None,
    ) -> ManagedComponent:
        parameter_bytes, buffer_bytes = estimate_module_bytes(module)
        component = ManagedComponent(
            component_id=str(component_id),
            component_kind=str(component_kind),
            model_identity=str(model_identity or "unknown"),
            module=module,
            current_device=module_device(module),
            preferred_dtype=str(preferred_dtype or module_dtype(module)),
            estimated_parameter_bytes=parameter_bytes,
            estimated_buffer_bytes=buffer_bytes,
            estimated_runtime_overhead_bytes=max(0, int(estimated_runtime_overhead_bytes)),
            pinned_cpu_capable=bool(pinned_cpu_capable),
            supports_non_blocking_transfer=bool(supports_non_blocking_transfer),
            last_used_monotonic_ns=time.monotonic_ns(),
            required_by_stages={str(stage) for stage in required_by_stages},
            unload_callback=unload_callback,
        )
        self.components[component.component_id] = component
        return component

    def remove(
        self,
        component_id: str,
        *,
        require_unleased: bool = True,
        call_unload: bool = True,
    ) -> ManagedComponent | None:
        selected = str(component_id)
        component = self.components.get(selected)
        if component is None:
            return None
        if require_unleased and component.leased:
            raise RuntimeError(
                f"Cannot remove leased component {selected!r}."
            )
        if call_unload and callable(component.unload_callback):
            try:
                component.unload_callback()
            except Exception:
                pass
        self.components.pop(selected, None)
        return component

    def replace_scoped(
        self,
        *,
        component_id: str,
        component_kind: str,
        model_identity: str,
        module: torch.nn.Module,
        preferred_dtype: str | torch.dtype | None = None,
        required_by_stages: Iterable[str] = (),
        estimated_runtime_overhead_bytes: int = 0,
        pinned_cpu_capable: bool = False,
        supports_non_blocking_transfer: bool = False,
        unload_callback: Any = None,
    ) -> ManagedComponent:
        selected = str(component_id)
        existing = self.components.get(selected)
        if existing is not None:
            if existing.leased:
                raise RuntimeError(
                    f"Cannot replace leased component {selected!r}."
                )
            self.remove(selected, require_unleased=True, call_unload=True)
        return self.register(
            component_id=selected,
            component_kind=component_kind,
            model_identity=model_identity,
            module=module,
            preferred_dtype=preferred_dtype,
            required_by_stages=required_by_stages,
            estimated_runtime_overhead_bytes=estimated_runtime_overhead_bytes,
            pinned_cpu_capable=pinned_cpu_capable,
            supports_non_blocking_transfer=supports_non_blocking_transfer,
            unload_callback=unload_callback,
        )

    def invalidate_incompatible(self, model_identity: str) -> list[str]:
        identity = str(model_identity or "unknown")
        removed: list[str] = []
        for component_id, component in list(self.components.items()):
            if component.model_identity == identity:
                continue
            if component.leased:
                raise RuntimeError(
                    f"Cannot invalidate leased component {component_id!r} for model switch."
                )
            if callable(component.unload_callback):
                try:
                    component.unload_callback()
                except Exception:
                    pass
            removed.append(component_id)
            self.components.pop(component_id, None)
        return removed

    def get(self, component_id: str) -> ManagedComponent:
        if component_id not in self.components:
            raise KeyError(f"Unknown managed component: {component_id}")
        return self.components[component_id]

    def snapshot(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for component in self.components.values():
            component.current_device = module_device(component.module)
            report = component.to_dict()
            report["placement"] = component_placement_report(component.module)
            values.append(report)
        return sorted(values, key=lambda item: item["component_id"])

    def acquire(self, component_ids: Iterable[str]) -> None:
        now = time.monotonic_ns()
        for component_id in component_ids:
            component = self.get(component_id)
            component.active_leases += 1
            component.last_used_monotonic_ns = now

    def release(self, component_ids: Iterable[str]) -> None:
        now = time.monotonic_ns()
        for component_id in component_ids:
            component = self.get(component_id)
            component.active_leases = max(0, component.active_leases - 1)
            component.last_used_monotonic_ns = now

    def move(
        self,
        component_id: str,
        *,
        device: str | torch.device,
        stage: str,
        reason: str,
        allow_leased: bool = False,
    ) -> ComponentTransferRecord:
        component = self.get(component_id)
        target = str(torch.device(device))
        source = module_device(component.module)
        if source == target or (source.startswith("cuda") and target == "cuda"):
            component.current_device = source
            record = ComponentTransferRecord(
                component_id=component.component_id,
                component_kind=component.component_kind,
                stage=str(stage),
                reason=str(reason),
                from_device=source,
                to_device=source,
                dtype=component.preferred_dtype,
                duration_ms=0.0,
                estimated_bytes=component.estimated_total_bytes,
                success=True,
                monotonic_ns=time.monotonic_ns(),
            )
            return record
        if component.leased and not allow_leased:
            raise RuntimeError(
                f"Cannot evict component {component.component_id!r} while an active stage lease holds it."
            )
        started = time.perf_counter()
        success = False
        error: str | None = None
        try:
            place_component(
                component.module,
                device=target,
                dtype=parse_dtype(component.preferred_dtype),
                owner="AdaptiveComponentMemoryManager",
                component_name=component.component_id,
            )
            component.current_device = module_device(component.module)
            component.last_used_monotonic_ns = time.monotonic_ns()
            success = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record = ComponentTransferRecord(
                component_id=component.component_id,
                component_kind=component.component_kind,
                stage=str(stage),
                reason=str(reason),
                from_device=source,
                to_device=target,
                dtype=component.preferred_dtype,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                estimated_bytes=component.estimated_total_bytes,
                success=success,
                error=error,
                monotonic_ns=time.monotonic_ns(),
            )
            self.transfer_records.append(record)
        return record

    def gpu_component_ids(self) -> list[str]:
        return [
            component_id
            for component_id, component in self.components.items()
            if module_device(component.module).startswith("cuda")
        ]

    def cpu_component_ids(self) -> list[str]:
        return [
            component_id
            for component_id, component in self.components.items()
            if not module_device(component.module).startswith("cuda")
        ]
