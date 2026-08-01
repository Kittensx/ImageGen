from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional


GuidanceOwner = Literal["pipeline", "sampler"]


@dataclass(frozen=True)
class SamplerCapabilities:
    sampler_name: str = "unknown"
    guidance_owner: GuidanceOwner = "pipeline"
    uses_raw_model_fn: bool = False
    uses_guided_model_fn: bool = True
    supports_step_expansion: bool = False
    supports_tail_metadata: bool = False
    requires_requested_step_schedule: bool = True
    strict_validation: bool = True
    forced_pipeline_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.guidance_owner not in {"pipeline", "sampler"}:
            raise ValueError("guidance_owner must be 'pipeline' or 'sampler'.")
        if self.guidance_owner == "sampler":
            if not self.uses_raw_model_fn or self.uses_guided_model_fn:
                raise ValueError(
                    "Sampler-owned guidance must use raw_model_fn and must not use guided_model_fn."
                )
        else:
            if not self.uses_guided_model_fn:
                raise ValueError("Pipeline-owned guidance must use guided_model_fn.")

    @classmethod
    def from_value(
        cls,
        value: "SamplerCapabilities | Mapping[str, Any]",
        *,
        default_name: str = "unknown",
    ) -> "SamplerCapabilities":
        if isinstance(value, cls):
            return value
        payload = dict(value)
        payload.setdefault("sampler_name", default_name)
        if "guidance_owner" not in payload:
            name = str(payload.get("sampler_name", default_name)).lower()
            is_kes = name in {"kes", "kes_sampler", "kes_style_sampler", "simple_kes_sampler"}
            payload["guidance_owner"] = "sampler" if is_kes else "pipeline"
            payload.setdefault("uses_raw_model_fn", is_kes)
            payload.setdefault("uses_guided_model_fn", not is_kes)
        return cls(**payload)

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "sampler_name": self.sampler_name,
            "guidance_owner": self.guidance_owner,
            "uses_raw_model_fn": self.uses_raw_model_fn,
            "uses_guided_model_fn": self.uses_guided_model_fn,
            "supports_step_expansion": self.supports_step_expansion,
            "supports_tail_metadata": self.supports_tail_metadata,
            "requires_requested_step_schedule": self.requires_requested_step_schedule,
            "strict_validation": self.strict_validation,
            "forced_pipeline_mode": self.forced_pipeline_mode,
        }


@dataclass
class SchedulerCompatibilityResult:
    sampler_name: str = "unknown"
    is_compatible: bool = True
    pipeline_mode: str = "fixed_steps"
    requires_fixed_schedule: bool = True
    supports_step_expansion: bool = False
    supports_tail_metadata: bool = False
    requested_steps: Optional[int] = None
    effective_steps: Optional[int] = None
    sigma_transitions: Optional[int] = None
    compatibility_mode: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def raise_if_incompatible(self) -> None:
        if not self.is_compatible:
            joined = "; ".join(self.reasons) if self.reasons else "unknown compatibility error"
            raise ValueError(f"Schedule is incompatible with sampler expectations: {joined}")

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "sampler_name": self.sampler_name,
            "is_compatible": self.is_compatible,
            "pipeline_mode": self.pipeline_mode,
            "requires_fixed_schedule": self.requires_fixed_schedule,
            "supports_step_expansion": self.supports_step_expansion,
            "supports_tail_metadata": self.supports_tail_metadata,
            "requested_steps": self.requested_steps,
            "effective_steps": self.effective_steps,
            "sigma_transitions": self.sigma_transitions,
            "compatibility_mode": self.compatibility_mode,
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
        }
