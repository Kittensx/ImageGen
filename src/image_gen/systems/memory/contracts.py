from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

from .preview_policy import normalize_preview_policy
from .oom_recovery import normalize_oom_recovery_profile


@dataclass
class ManagedComponent:
    component_id: str
    component_kind: str
    model_identity: str
    module: torch.nn.Module
    current_device: str
    preferred_dtype: str
    estimated_parameter_bytes: int
    estimated_buffer_bytes: int
    estimated_runtime_overhead_bytes: int = 0
    pinned_cpu_capable: bool = False
    supports_non_blocking_transfer: bool = False
    last_used_monotonic_ns: int = 0
    required_by_stages: set[str] = field(default_factory=set)
    unload_callback: Callable[[], Any] | None = None
    active_leases: int = 0

    @property
    def estimated_total_bytes(self) -> int:
        return int(
            self.estimated_parameter_bytes
            + self.estimated_buffer_bytes
            + self.estimated_runtime_overhead_bytes
        )

    @property
    def leased(self) -> bool:
        return self.active_leases > 0

    def to_dict(self) -> dict[str, Any]:
        # Do not use dataclasses.asdict() here.  asdict() deep-copies every field
        # before callers can remove ``module`` and ``unload_callback``.  During
        # CUDA OOM recovery that deep copy can clone torch Parameters and trigger
        # a second allocation failure while merely trying to capture diagnostics.
        # Build the serializable report explicitly so recovery remains allocation-
        # light and never traverses live model objects.
        return {
            "component_id": self.component_id,
            "component_kind": self.component_kind,
            "model_identity": self.model_identity,
            "current_device": self.current_device,
            "preferred_dtype": self.preferred_dtype,
            "estimated_parameter_bytes": int(self.estimated_parameter_bytes),
            "estimated_buffer_bytes": int(self.estimated_buffer_bytes),
            "estimated_runtime_overhead_bytes": int(self.estimated_runtime_overhead_bytes),
            "pinned_cpu_capable": bool(self.pinned_cpu_capable),
            "supports_non_blocking_transfer": bool(self.supports_non_blocking_transfer),
            "last_used_monotonic_ns": int(self.last_used_monotonic_ns),
            "required_by_stages": sorted(self.required_by_stages),
            "active_leases": int(self.active_leases),
            "estimated_total_bytes": self.estimated_total_bytes,
            "leased": self.leased,
        }


@dataclass(frozen=True)
class ComponentTransferRecord:
    component_id: str
    component_kind: str
    stage: str
    reason: str
    from_device: str
    to_device: str
    dtype: str
    duration_ms: float
    estimated_bytes: int
    success: bool
    error: str | None = None
    monotonic_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemorySnapshot:
    timestamp: str
    monotonic_ns: int
    pipeline_stage: str
    cuda: dict[str, Any]
    system: dict[str, Any]
    process: dict[str, Any]
    component_residency: list[dict[str, Any]]
    active_cuda_stream_count: int | None = None
    optional_gpu_telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryEstimate:
    stage: str
    estimated_minimum_bytes: int
    estimated_expected_bytes: int
    safety_adjusted_required_bytes: int
    available_bytes: int | None
    headroom_bytes: int | None
    confidence: str
    major_contributors: dict[str, int]
    feasible: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidencyPlan:
    stage: str
    requested_profile: str
    effective_profile: str
    target_device: str
    required: tuple[str, ...]
    preferred: tuple[str, ...]
    optional: tuple[str, ...]
    selected_for_target: tuple[str, ...]
    selected_for_cpu: tuple[str, ...]
    estimated_stage_bytes: int
    available_bytes: int | None
    safety_margin_bytes: int
    preview_image_decode_suspended: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryManagerSettings:
    policy: str = "auto"
    safety_margin_mb: int = 1024
    retain_checkpoint_between_jobs: bool = True
    retain_vae_between_jobs: bool = False
    pinned_cpu_memory: bool = False
    allow_tiled_vae_fallback: bool = True
    allow_preview_suspension_on_oom: bool = True
    hires_memory_profile: str = "inherit"
    pre_hires_cleanup: bool = False
    preview_policy: str = "normal"
    attention_slicing: str = "off"
    vae_tiling: bool = False
    vae_slicing: bool = False
    vae_device: str = "auto"
    oom_retry_profile: str = "cleanup"
    oom_retry_limit: int = 1

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "MemoryManagerSettings":
        raw = dict(values or {})
        policy = str(raw.get("memory_policy", "auto") or "auto").strip().lower().replace(" ", "_")
        aliases = {
            "high": "high_vram",
            "high_vram": "high_vram",
            "balanced": "balanced",
            "low": "low_vram",
            "low_vram": "low_vram",
            "cpu": "cpu_fallback",
            "cpu_fallback": "cpu_fallback",
            "auto": "auto",
        }
        policy = aliases.get(policy, "auto")
        attention_slicing = str(raw.get("attention_slicing", "off") or "off").strip().lower()
        if attention_slicing not in {"off", "auto", "max"}:
            raise ValueError("attention_slicing must be one of: off, auto, max.")
        vae_device = str(raw.get("vae_device", "auto") or "auto").strip().lower()
        if vae_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("vae_device must be one of: auto, cuda, cpu.")
        return cls(
            policy=policy,
            safety_margin_mb=max(
                0,
                int(
                    raw.get("memory_vram_safety_margin_mb", 1024)
                    if raw.get("memory_vram_safety_margin_mb") is not None
                    else 1024
                ),
            ),
            retain_checkpoint_between_jobs=bool(raw.get("memory_retain_checkpoint_between_jobs", True)),
            retain_vae_between_jobs=bool(raw.get("memory_retain_vae_between_jobs", False)),
            pinned_cpu_memory=bool(raw.get("memory_pinned_cpu_memory", False)),
            allow_tiled_vae_fallback=bool(raw.get("memory_allow_tiled_vae_fallback", True)),
            allow_preview_suspension_on_oom=bool(raw.get("memory_allow_preview_suspension_on_oom", True)),
            hires_memory_profile=str(raw.get("hires_memory_profile", "inherit") or "inherit").strip().lower().replace("-", "_"),
            pre_hires_cleanup=bool(raw.get("pre_hires_cleanup", False)),
            preview_policy=normalize_preview_policy(raw.get("preview_policy", "normal")),
            attention_slicing=attention_slicing,
            vae_tiling=bool(raw.get("vae_tiling", False)),
            vae_slicing=bool(raw.get("vae_slicing", False)),
            vae_device=vae_device,
            oom_retry_profile=normalize_oom_recovery_profile(
                raw.get("oom_retry_profile", "cleanup")
            ),
            oom_retry_limit=max(0, int(raw.get("oom_retry_limit", 1) or 0)),
        )

    @property
    def safety_margin_bytes(self) -> int:
        return int(self.safety_margin_mb) * 1024 * 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
