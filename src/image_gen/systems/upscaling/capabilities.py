from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CAPABILITY_DISPOSITIONS = frozenset({"supported", "deferred", "failed"})

DEFERRED_DISCOVERY_STATES = frozenset(
    {
        "deferred_architecture",
        "deferred_scale",
        "deferred_hardware_validation",
        "unsupported_architecture",
        "unsupported_scale",
        "unclassified",
    }
)
FAILED_DISCOVERY_STATES = frozenset(
    {
        "unsupported_channels",
        "inspection_failed",
        "corrupt",
    }
)
DEFERRED_RUNTIME_STATES = frozenset(
    {
        "unqualified",
        "deferred_not_tested",
        "deferred_hardware_limit",
        "deferred_hardware_unavailable",
        "deferred_backend_support",
        "deferred_architecture",
        "deferred_scale",
    }
)
FAILED_RUNTIME_STATES = frozenset(
    {
        "backend_unavailable",
        "backend_unqualified",
        "hash_mismatch",
        "metadata_mismatch",
        "load_failed",
        "runtime_contract_failed",
        "security_boundary_failed",
        "corrupt",
    }
)


def discovery_disposition(status: str) -> str:
    normalized = str(status or "unclassified").strip().casefold()
    if normalized == "supported":
        return "supported"
    if normalized in DEFERRED_DISCOVERY_STATES:
        return "deferred"
    if normalized in FAILED_DISCOVERY_STATES:
        return "failed"
    return "failed"


def runtime_disposition(status: str) -> str:
    normalized = str(status or "deferred_not_tested").strip().casefold()
    if normalized in {"qualified_cpu", "qualified_cuda"}:
        return "supported"
    if normalized in DEFERRED_RUNTIME_STATES:
        return "deferred"
    if normalized in FAILED_RUNTIME_STATES:
        return "failed"
    return "failed"


def is_memory_limit_error(value: BaseException | str) -> bool:
    if isinstance(value, MemoryError):
        return True
    try:
        import torch

        if isinstance(value, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = str(value or "").casefold()
    markers = (
        "out of memory",
        "cannot allocate memory",
        "not enough memory",
        "cuda error: out of memory",
        "defaultcpuallocator",
        "bad allocation",
        "std::bad_alloc",
    )
    return any(marker in text for marker in markers)


@dataclass(frozen=True)
class CapabilityAssessment:
    status: str
    disposition: str
    reason_code: str
    reason: str
    hardware_scope: str = ""
    blocking: bool = False

    def __post_init__(self) -> None:
        disposition = str(self.disposition or "failed").strip().casefold()
        if disposition not in CAPABILITY_DISPOSITIONS:
            raise ValueError(f"Unsupported capability disposition: {self.disposition!r}")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "blocking", disposition == "failed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deferred_assessment(
    status: str,
    *,
    reason_code: str,
    reason: str,
    hardware_scope: str = "",
) -> CapabilityAssessment:
    return CapabilityAssessment(
        status=status,
        disposition="deferred",
        reason_code=reason_code,
        reason=reason,
        hardware_scope=hardware_scope,
    )


__all__ = [
    "CAPABILITY_DISPOSITIONS",
    "CapabilityAssessment",
    "DEFERRED_DISCOVERY_STATES",
    "DEFERRED_RUNTIME_STATES",
    "FAILED_DISCOVERY_STATES",
    "FAILED_RUNTIME_STATES",
    "deferred_assessment",
    "discovery_disposition",
    "is_memory_limit_error",
    "runtime_disposition",
]
