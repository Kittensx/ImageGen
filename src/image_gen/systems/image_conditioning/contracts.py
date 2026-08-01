from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from image_gen.contracts import SchedulerOutput


@dataclass(frozen=True)
class ImageConditionedStepPlan:
    """Policy result used before constructing the scheduler's full schedule."""

    step_policy: str
    requested_refinement_steps: int
    requested_denoising_strength: float
    normalized_denoising_strength: float
    safe_denoising_strength: float
    internal_schedule_steps: int
    effective_refinement_steps: int
    minimum_supported_strength: float
    maximum_internal_schedule_steps: int
    denoising_strength_was_clamped: bool = False

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "step_policy": str(self.step_policy),
            "requested_refinement_steps": int(self.requested_refinement_steps),
            "requested_denoising_strength": float(self.requested_denoising_strength),
            "normalized_denoising_strength": float(self.normalized_denoising_strength),
            "safe_denoising_strength": float(self.safe_denoising_strength),
            "internal_schedule_steps": int(self.internal_schedule_steps),
            "effective_refinement_steps": int(self.effective_refinement_steps),
            "minimum_supported_strength": float(self.minimum_supported_strength),
            "maximum_internal_schedule_steps": int(self.maximum_internal_schedule_steps),
            "denoising_strength_was_clamped": bool(self.denoising_strength_was_clamped),
        }


@dataclass(frozen=True)
class ImageConditionedSchedule:
    """Validated image-conditioned schedule selection.

    ``full_schedule`` is the scheduler adapter's complete schedule.
    ``active_schedule`` is the validated schedule region that will actually be
    sampled for the image-conditioned refinement pass.
    """

    full_schedule: SchedulerOutput
    active_schedule: SchedulerOutput
    step_policy: str
    requested_refinement_steps: int
    internal_schedule_steps: int
    effective_refinement_steps: int
    denoising_strength: float
    start_index: int
    start_sigma: float
    start_timestep: float | None
    step_plan: ImageConditionedStepPlan

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "step_policy": str(self.step_policy),
            "requested_refinement_steps": int(self.requested_refinement_steps),
            "internal_schedule_steps": int(self.internal_schedule_steps),
            "effective_refinement_steps": int(self.effective_refinement_steps),
            "denoising_strength": float(self.denoising_strength),
            "start_index": int(self.start_index),
            "start_sigma": float(self.start_sigma),
            "start_timestep": (
                float(self.start_timestep)
                if self.start_timestep is not None
                else None
            ),
            "step_plan": self.step_plan.to_serializable_dict(),
        }


@dataclass(frozen=True)
class ScheduleRehydrationResult:
    """Exact schedule restored from a recorded replay payload."""

    schedule: ImageConditionedSchedule
    expected_fingerprint: str
    actual_fingerprint: str
    fingerprint_match: bool
    replay_format: str

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "replay_format": str(self.replay_format),
            "expected_fingerprint": str(self.expected_fingerprint),
            "actual_fingerprint": str(self.actual_fingerprint),
            "fingerprint_match": bool(self.fingerprint_match),
            "schedule": self.schedule.to_serializable_dict(),
        }
