from __future__ import annotations

import copy
from typing import Any, Optional

import torch

from modules.contracts import SamplerCapabilities, SchedulerCompatibilityResult

# Historical names retained as aliases during migration. They are not separate
# dataclass definitions and therefore cannot drift from the canonical contracts.
SamplerScheduleCapabilities = SamplerCapabilities
ScheduleCompatibilityDecision = SchedulerCompatibilityResult
ScheduleValidationResult = SchedulerCompatibilityResult


def get_sampler_schedule_capabilities(
    sampler_name: Optional[str] = None,
    sampler_obj: Any = None,
) -> SamplerCapabilities:
    """Resolve the canonical capability declaration for a sampler."""
    if sampler_obj is not None:
        caps = getattr(sampler_obj, "SAMPLER_CAPABILITIES", None)
        if caps is None:
            caps = getattr(sampler_obj, "SAMPLER_SCHEDULE_CAPABILITIES", None)
        if caps is not None:
            return SamplerCapabilities.from_value(
                caps,
                default_name=str(sampler_name or getattr(sampler_obj, "SAMPLER_NAME", "unknown")),
            )

    normalized = str(
        sampler_name or getattr(sampler_obj, "SAMPLER_NAME", "unknown")
    ).strip().lower()
    is_kes = normalized in {
        "kes",
        "kes_sampler",
        "kes_style_sampler",
        "kes_style",
        "simple_kes_sampler",
    }
    if is_kes:
        return SamplerCapabilities(
            sampler_name=normalized,
            guidance_owner="sampler",
            uses_raw_model_fn=True,
            uses_guided_model_fn=False,
            supports_step_expansion=True,
            supports_tail_metadata=True,
            requires_requested_step_schedule=False,
            strict_validation=True,
            forced_pipeline_mode="extended_steps",
        )

    return SamplerCapabilities(
        sampler_name=normalized,
        guidance_owner="pipeline",
        uses_raw_model_fn=False,
        uses_guided_model_fn=True,
        supports_step_expansion=False,
        supports_tail_metadata=False,
        requires_requested_step_schedule=True,
        strict_validation=True,
        forced_pipeline_mode="fixed_steps",
    )


def resolve_schedule_compatibility_decision(
    capabilities: SamplerCapabilities,
) -> SchedulerCompatibilityResult:
    warnings: list[str] = []

    if capabilities.forced_pipeline_mode:
        pipeline_mode = capabilities.forced_pipeline_mode
    elif capabilities.requires_requested_step_schedule:
        pipeline_mode = "fixed_steps"
    elif capabilities.supports_step_expansion:
        pipeline_mode = "extended_steps"
    else:
        pipeline_mode = "fixed_steps"

    if capabilities.requires_requested_step_schedule and capabilities.supports_step_expansion:
        warnings.append(
            "Sampler declares both requires_requested_step_schedule=True and "
            "supports_step_expansion=True. Using fixed_steps mode."
        )
        pipeline_mode = "fixed_steps"

    return SchedulerCompatibilityResult(
        sampler_name=capabilities.sampler_name,
        is_compatible=True,
        pipeline_mode=pipeline_mode,
        requires_fixed_schedule=capabilities.requires_requested_step_schedule,
        supports_step_expansion=capabilities.supports_step_expansion,
        supports_tail_metadata=capabilities.supports_tail_metadata,
        warnings=warnings,
    )


def build_scheduler_enforcement_kwargs(
    request: Any,
    sampler_name: Optional[str] = None,
    sampler_obj: Any = None,
) -> dict[str, Any]:
    capabilities = get_sampler_schedule_capabilities(
        sampler_name=sampler_name,
        sampler_obj=sampler_obj,
    )
    decision = resolve_schedule_compatibility_decision(capabilities)

    scheduler_kwargs = dict(getattr(request, "scheduler_kwargs", {}) or {})
    scheduler_kwargs["pipeline_mode"] = decision.pipeline_mode

    compatibility = dict(scheduler_kwargs.get("compatibility", {}) or {})
    compatibility["requested_by_sampler"] = capabilities.sampler_name
    compatibility["guidance_owner"] = capabilities.guidance_owner
    compatibility["requires_fixed_schedule"] = decision.requires_fixed_schedule
    compatibility["supports_step_expansion"] = decision.supports_step_expansion
    compatibility["supports_tail_metadata"] = decision.supports_tail_metadata
    scheduler_kwargs["compatibility"] = compatibility
    return scheduler_kwargs


def _extract_schedule_counts(
    schedule: Any,
    request: Any,
) -> tuple[int, int, int, Optional[str], dict[str, Any]]:
    sigmas = getattr(schedule, "sigmas", None)
    if sigmas is None:
        raise ValueError("schedule must provide `sigmas`")
    if callable(sigmas):
        sigmas = sigmas()
    if not torch.is_tensor(sigmas):
        sigmas = torch.tensor(sigmas, dtype=torch.float32)

    sigma_transitions = max(int(sigmas.numel()) - 1, 0)
    schedule_extra = getattr(schedule, "extra", None)
    if not isinstance(schedule_extra, dict):
        schedule_extra = {}

    requested_steps = getattr(schedule, "requested_steps", None)
    if requested_steps is None:
        requested_steps = getattr(request, "steps", sigma_transitions)

    effective_steps = getattr(schedule, "effective_steps", None)
    if effective_steps is None:
        effective_steps = sigma_transitions

    compatibility_mode = getattr(schedule, "compatibility_mode", None)
    try:
        requested_steps = int(requested_steps)
    except (TypeError, ValueError):
        requested_steps = sigma_transitions
    try:
        effective_steps = int(effective_steps)
    except (TypeError, ValueError):
        effective_steps = sigma_transitions

    return requested_steps, effective_steps, sigma_transitions, compatibility_mode, schedule_extra


def validate_schedule_for_sampler(
    schedule: Any,
    request: Any,
    sampler_name: Optional[str] = None,
    sampler_obj: Any = None,
) -> SchedulerCompatibilityResult:
    capabilities = get_sampler_schedule_capabilities(
        sampler_name=sampler_name,
        sampler_obj=sampler_obj,
    )
    decision = resolve_schedule_compatibility_decision(capabilities)
    requested_steps, effective_steps, sigma_transitions, compatibility_mode, schedule_extra = (
        _extract_schedule_counts(schedule=schedule, request=request)
    )

    reasons: list[str] = []
    if capabilities.requires_requested_step_schedule:
        if sigma_transitions != requested_steps:
            reasons.append(
                f"sampler requires requested-step schedule, but sigma_transitions={sigma_transitions} "
                f"and requested_steps={requested_steps}"
            )
        if effective_steps != requested_steps:
            reasons.append(
                f"sampler requires requested-step schedule, but effective_steps={effective_steps} "
                f"and requested_steps={requested_steps}"
            )

    if not capabilities.supports_tail_metadata:
        tail_features = schedule_extra.get("tail_features_used", {})
        if isinstance(tail_features, dict):
            active_tail_features = [key for key, value in tail_features.items() if bool(value)]
            if active_tail_features and effective_steps != requested_steps:
                reasons.append(
                    "sampler does not support tail/expanded schedule semantics, but scheduler reports active "
                    f"tail features: {active_tail_features}"
                )

    if not capabilities.supports_step_expansion and effective_steps != requested_steps:
        reasons.append(
            f"sampler does not support step expansion, but effective_steps={effective_steps} "
            f"and requested_steps={requested_steps}"
        )

    return SchedulerCompatibilityResult(
        sampler_name=capabilities.sampler_name,
        is_compatible=not reasons,
        pipeline_mode=decision.pipeline_mode,
        requires_fixed_schedule=decision.requires_fixed_schedule,
        supports_step_expansion=decision.supports_step_expansion,
        supports_tail_metadata=decision.supports_tail_metadata,
        requested_steps=requested_steps,
        effective_steps=effective_steps,
        sigma_transitions=sigma_transitions,
        compatibility_mode=compatibility_mode,
        warnings=list(decision.warnings),
        reasons=reasons,
    )


def annotate_schedule_with_compatibility(
    schedule: Any,
    validation: SchedulerCompatibilityResult,
    sampler_name: Optional[str] = None,
) -> Any:
    if not hasattr(schedule, "extra") or not isinstance(getattr(schedule, "extra", None), dict):
        return schedule

    extra = dict(schedule.extra)
    extra["validated_for_sampler"] = sampler_name
    extra["schedule_is_compatible"] = validation.is_compatible
    extra["schedule_validation_reasons"] = list(validation.reasons)
    schedule.extra = extra
    return schedule


def ensure_schedule_for_sampler(
    schedule: Any,
    request: Any,
    sampler_name: Optional[str] = None,
    sampler_obj: Any = None,
    rebuild_schedule_fn=None,
) -> Any:
    capabilities = get_sampler_schedule_capabilities(
        sampler_name=sampler_name,
        sampler_obj=sampler_obj,
    )
    initial_validation = validate_schedule_for_sampler(
        schedule=schedule,
        request=request,
        sampler_name=sampler_name,
        sampler_obj=sampler_obj,
    )
    schedule = annotate_schedule_with_compatibility(
        schedule=schedule,
        validation=initial_validation,
        sampler_name=capabilities.sampler_name,
    )

    if initial_validation.is_compatible:
        return schedule

    if rebuild_schedule_fn is None:
        if capabilities.strict_validation:
            initial_validation.raise_if_incompatible()
        return schedule

    enforced_request = copy.copy(request)
    enforced_request.scheduler_kwargs = build_scheduler_enforcement_kwargs(
        request=request,
        sampler_name=sampler_name,
        sampler_obj=sampler_obj,
    )
    rebuilt_schedule = rebuild_schedule_fn(enforced_request)
    rebuilt_validation = validate_schedule_for_sampler(
        schedule=rebuilt_schedule,
        request=enforced_request,
        sampler_name=sampler_name,
        sampler_obj=sampler_obj,
    )
    rebuilt_schedule = annotate_schedule_with_compatibility(
        schedule=rebuilt_schedule,
        validation=rebuilt_validation,
        sampler_name=capabilities.sampler_name,
    )
    if capabilities.strict_validation:
        rebuilt_validation.raise_if_incompatible()
    return rebuilt_schedule
