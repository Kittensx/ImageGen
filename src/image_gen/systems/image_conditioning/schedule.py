from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from image_gen.contracts import SchedulerOutput

from .contracts import ImageConditionedSchedule, ImageConditionedStepPlan

PROPORTIONAL_TAIL_V1 = "proportional_tail_v1"
A1111_FIXED_STEPS_V1 = "a1111_fixed_steps_v1"
DEFAULT_HIRES_STEP_POLICY = A1111_FIXED_STEPS_V1
SUPPORTED_HIRES_STEP_POLICIES = frozenset(
    {A1111_FIXED_STEPS_V1, PROPORTIONAL_TAIL_V1}
)
MINIMUM_SUPPORTED_DENOISING_STRENGTH = 0.01
MAXIMUM_FIXED_POLICY_STRENGTH = 0.999
MAXIMUM_INTERNAL_SCHEDULE_STEPS = 20_000
MAXIMUM_REQUESTED_REFINEMENT_STEPS = 200


def _scalar_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() < 1:
            return None
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def _resolve_sigma_bounds(
    bounds: Mapping[str, Any] | tuple[float | None, float | None] | None,
) -> tuple[float | None, float | None]:
    if bounds is None:
        return None, None
    if isinstance(bounds, Mapping):
        minimum = bounds.get("min_sigma")
        if minimum is None:
            minimum = bounds.get("minimum")
        maximum = bounds.get("max_sigma")
        if maximum is None:
            maximum = bounds.get("maximum")
        return (
            float(minimum) if minimum is not None else None,
            float(maximum) if maximum is not None else None,
        )
    if isinstance(bounds, tuple) and len(bounds) == 2:
        minimum, maximum = bounds
        return (
            float(minimum) if minimum is not None else None,
            float(maximum) if maximum is not None else None,
        )
    raise TypeError("model_sigma_bounds must be a mapping, tuple, or None.")


def _validate_terminal_zero(schedule: SchedulerOutput, required: bool | None) -> None:
    if required is not True:
        return
    terminal = float(schedule.sigmas[-1].detach().cpu().item())
    if abs(terminal) > 1.0e-8:
        raise ValueError(
            "Image-conditioned schedule requires a terminal zero sigma, but the full schedule does not end at zero."
        )


def _slice_timesteps(timesteps: torch.Tensor, start_index: int) -> torch.Tensor:
    return timesteps[start_index:].clone()


def resolve_image_conditioned_step_plan(
    *,
    requested_refinement_steps: int,
    denoising_strength: float,
    step_policy: str = DEFAULT_HIRES_STEP_POLICY,
    minimum_supported_strength: float = MINIMUM_SUPPORTED_DENOISING_STRENGTH,
    maximum_internal_schedule_steps: int = MAXIMUM_INTERNAL_SCHEDULE_STEPS,
) -> ImageConditionedStepPlan:
    policy = str(step_policy or DEFAULT_HIRES_STEP_POLICY).strip().lower()
    if policy not in SUPPORTED_HIRES_STEP_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_HIRES_STEP_POLICIES))
        raise ValueError(f"hires_step_policy must be one of: {supported}.")

    requested_steps = int(requested_refinement_steps)
    if requested_steps < 1 or requested_steps > MAXIMUM_REQUESTED_REFINEMENT_STEPS:
        raise ValueError(
            "Requested image-conditioned refinement steps must be between 1 and "
            f"{MAXIMUM_REQUESTED_REFINEMENT_STEPS}; received {requested_steps}."
        )

    requested_strength = float(denoising_strength)
    if not math.isfinite(requested_strength):
        raise ValueError("Denoising strength must be finite.")
    normalized_strength = min(max(requested_strength, minimum_supported_strength), 1.0)
    strength_was_clamped = normalized_strength != requested_strength

    if policy == A1111_FIXED_STEPS_V1:
        safe_strength = min(normalized_strength, MAXIMUM_FIXED_POLICY_STRENGTH)
        internal_steps = max(requested_steps, int(requested_steps / safe_strength))
        effective_steps = requested_steps
    else:
        safe_strength = normalized_strength
        internal_steps = requested_steps
        effective_steps = max(1, min(requested_steps, int(round(requested_steps * safe_strength))))

    cap = int(maximum_internal_schedule_steps)
    if cap < requested_steps:
        raise ValueError(
            "Maximum internal schedule steps cannot be lower than requested refinement steps."
        )
    if internal_steps > cap:
        raise ValueError(
            "Image-conditioned internal schedule requires "
            f"{internal_steps} steps, exceeding the explicit safety cap of {cap}. "
            "Increase denoising strength or reduce requested hires steps."
        )

    return ImageConditionedStepPlan(
        step_policy=policy,
        requested_refinement_steps=requested_steps,
        requested_denoising_strength=requested_strength,
        normalized_denoising_strength=normalized_strength,
        safe_denoising_strength=safe_strength,
        internal_schedule_steps=internal_steps,
        effective_refinement_steps=effective_steps,
        minimum_supported_strength=float(minimum_supported_strength),
        maximum_internal_schedule_steps=cap,
        denoising_strength_was_clamped=strength_was_clamped,
    )


def build_image_conditioned_schedule(
    full_schedule: SchedulerOutput,
    *,
    requested_refinement_steps: int,
    denoising_strength: float,
    step_policy: str = DEFAULT_HIRES_STEP_POLICY,
    scheduler_identifier: str = "",
    scheduler_configuration: Mapping[str, Any] | None = None,
    model_sigma_bounds: Mapping[str, Any] | tuple[float | None, float | None] | None = None,
    requires_terminal_zero: bool | None = None,
    experimental_allow_out_of_bounds: bool = False,
    sampler_requires_timestep: bool = True,
    maximum_internal_schedule_steps: int = MAXIMUM_INTERNAL_SCHEDULE_STEPS,
) -> ImageConditionedSchedule:
    del scheduler_identifier, scheduler_configuration
    full_schedule.validate()
    _validate_terminal_zero(full_schedule, requires_terminal_zero)

    step_plan = resolve_image_conditioned_step_plan(
        requested_refinement_steps=requested_refinement_steps,
        denoising_strength=denoising_strength,
        step_policy=step_policy,
        maximum_internal_schedule_steps=maximum_internal_schedule_steps,
    )
    transitions = int(full_schedule.sigma_transitions)
    if transitions < 1:
        raise ValueError("Full schedule must contain at least one transition.")

    if step_plan.step_policy == A1111_FIXED_STEPS_V1:
        selected_steps = int(step_plan.requested_refinement_steps)
        if transitions < selected_steps:
            raise ValueError(
                "The scheduler produced fewer transitions than the fixed-step refinement request. "
                f"Requested {selected_steps}, received {transitions}."
            )
    else:
        selected_steps = max(
            1,
            min(
                transitions,
                int(round(transitions * step_plan.normalized_denoising_strength)),
            ),
        )

    start_index = transitions - selected_steps
    sigmas = full_schedule.sigmas[start_index:].clone()

    timesteps = full_schedule.timesteps
    if timesteps is None:
        raise ValueError("Image-conditioned denoising requires scheduler timesteps.")
    sliced_timesteps = _slice_timesteps(timesteps, start_index)

    start_sigma = _scalar_float(sigmas[0])
    if start_sigma is None or not math.isfinite(start_sigma) or start_sigma <= 0.0:
        raise ValueError("Image-conditioned schedule requires a finite positive starting sigma.")

    start_timestep = _scalar_float(sliced_timesteps[0])
    if sampler_requires_timestep and start_timestep is None:
        raise ValueError(
            "Image-conditioned schedule requires a starting timestep for the active schedule."
        )

    min_sigma, max_sigma = _resolve_sigma_bounds(model_sigma_bounds)
    if not experimental_allow_out_of_bounds:
        if min_sigma is not None and start_sigma < min_sigma:
            raise ValueError(
                "Selected image-conditioned start sigma lies below model-supported bounds."
            )
        if max_sigma is not None and start_sigma > max_sigma:
            raise ValueError(
                "Selected image-conditioned start sigma lies above model-supported bounds."
            )

    metadata = dict(full_schedule.metadata)
    metadata.update(
        {
            "hires_second_pass": True,
            "hires_step_policy": step_plan.step_policy,
            "hires_requested_steps": int(step_plan.requested_refinement_steps),
            "hires_planned_internal_schedule_steps": int(step_plan.internal_schedule_steps),
            "hires_full_schedule_steps": transitions,
            "hires_full_schedule_transition_count": transitions,
            "hires_effective_second_pass_transition_count": selected_steps,
            "hires_schedule_start_index": start_index,
            "hires_requested_denoising_strength": float(
                step_plan.requested_denoising_strength
            ),
            "hires_denoising_strength": float(
                step_plan.normalized_denoising_strength
            ),
            "hires_safe_denoising_strength": float(
                step_plan.safe_denoising_strength
            ),
            "hires_denoising_strength_was_clamped": bool(
                step_plan.denoising_strength_was_clamped
            ),
            "hires_starting_sigma": start_sigma,
            "hires_starting_timestep": start_timestep,
            "hires_schedule_counts_source": "validated_schedule_tensors",
            "image_conditioned_step_plan": step_plan.to_serializable_dict(),
            "image_conditioned_schedule_contract": {
                "requested_refinement_steps": int(step_plan.requested_refinement_steps),
                "planned_internal_schedule_steps": int(step_plan.internal_schedule_steps),
                "internal_schedule_steps": transitions,
                "effective_refinement_steps": selected_steps,
                "start_index": int(start_index),
                "start_sigma": float(start_sigma),
                "start_timestep": float(start_timestep) if start_timestep is not None else None,
            },
        }
    )
    active_schedule = SchedulerOutput(
        sigmas=sigmas,
        timesteps=sliced_timesteps,
        requested_steps=selected_steps,
        effective_steps=int(sigmas.numel() - 1),
        scheduler_step_override_applied=selected_steps != transitions,
        compatibility_mode=full_schedule.compatibility_mode,
        metadata=metadata,
    )
    if active_schedule.sigma_transitions != selected_steps:
        raise ValueError(
            "Image-conditioned active schedule transition count diverged from the selected transition count."
        )

    expected_timestep_lengths = {
        active_schedule.sigma_transitions,
        int(active_schedule.sigmas.numel()),
    }
    if (
        active_schedule.timesteps is None
        or int(active_schedule.timesteps.numel()) not in expected_timestep_lengths
    ):
        raise ValueError(
            "Image-conditioned active schedule timesteps do not satisfy the scheduler contract."
        )

    return ImageConditionedSchedule(
        full_schedule=full_schedule,
        active_schedule=active_schedule,
        step_policy=step_plan.step_policy,
        requested_refinement_steps=int(step_plan.requested_refinement_steps),
        internal_schedule_steps=transitions,
        effective_refinement_steps=selected_steps,
        denoising_strength=float(step_plan.normalized_denoising_strength),
        start_index=int(start_index),
        start_sigma=float(start_sigma),
        start_timestep=float(start_timestep) if start_timestep is not None else None,
        step_plan=step_plan,
    )
