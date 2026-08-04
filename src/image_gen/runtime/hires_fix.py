from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from image_gen.contracts import GenerationRequest, SchedulerOutput
from image_gen.runtime.hires_sizing import HiresDimensionPlan, resolve_hires_dimensions
from image_gen.systems.image_conditioning import (
    A1111_FIXED_STEPS_V1,
    DEFAULT_HIRES_STEP_POLICY,
    HIRES_NOISE_POLICY_ID,
    HIRES_NOISE_SEED_OFFSET,
    MAXIMUM_REQUESTED_REFINEMENT_STEPS,
    PROPORTIONAL_TAIL_V1,
    SUPPORTED_HIRES_STEP_POLICIES,
    build_image_conditioned_schedule,
    build_schedule_fingerprint_record,
    build_schedule_replay_record,
    resolve_image_conditioned_step_plan,
)
from modules.txt2img.seed_utils import create_torch_generator, offset_seed


HIRES_ALGORITHM_VERSION = "image-gen-hires-refinement-v2"
_VALID_UPSCALERS = {"latent_nearest", "latent_bilinear", "latent_bicubic"}


def _scalar_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() < 1:
            return None
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


@dataclass(frozen=True)
class HiresExecutionPlan:
    enabled: bool
    dimensions: HiresDimensionPlan
    steps: int
    internal_steps: int
    effective_steps: int
    denoising_strength: float
    safe_denoising_strength: float
    upscaler: str
    sampler_name: str
    scheduler_name: str
    cfg_scale: float
    cfg_rescale: float
    step_policy: str = DEFAULT_HIRES_STEP_POLICY
    noise_policy: str = HIRES_NOISE_POLICY_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_version": HIRES_ALGORITHM_VERSION,
            "enabled": bool(self.enabled),
            "dimensions": self.dimensions.to_dict(),
            "steps": int(self.steps),
            "internal_steps": int(self.internal_steps),
            "effective_steps": int(self.effective_steps),
            "denoising_strength": float(self.denoising_strength),
            "safe_denoising_strength": float(self.safe_denoising_strength),
            "upscaler": self.upscaler,
            "sampler_name": self.sampler_name,
            "scheduler_name": self.scheduler_name,
            "cfg_scale": float(self.cfg_scale),
            "cfg_rescale": float(self.cfg_rescale),
            "step_policy": self.step_policy,
            "noise_policy": self.noise_policy,
        }



def resolve_hires_execution_plan(request: GenerationRequest) -> HiresExecutionPlan:
    dimensions = resolve_hires_dimensions(request)
    enabled = bool(getattr(request, "hires_enabled", False))
    raw_steps = getattr(request, "hires_steps", 20)
    steps = int(20 if raw_steps is None else raw_steps)
    if steps < 1 or steps > MAXIMUM_REQUESTED_REFINEMENT_STEPS:
        raise ValueError(
            "hires_steps must be between 1 and "
            f"{MAXIMUM_REQUESTED_REFINEMENT_STEPS}; received {steps}."
        )
    raw_strength = getattr(request, "hires_denoising_strength", 0.45)
    strength = float(0.45 if raw_strength is None else raw_strength)
    upscaler = str(getattr(request, "hires_upscaler", "latent_bilinear") or "latent_bilinear").strip().lower()
    if upscaler not in _VALID_UPSCALERS:
        raise ValueError(
            "hires_upscaler must be one of: latent_nearest, latent_bilinear, latent_bicubic."
        )
    step_policy = str(
        getattr(request, "hires_step_policy", DEFAULT_HIRES_STEP_POLICY)
        or DEFAULT_HIRES_STEP_POLICY
    ).strip().lower()
    step_plan = resolve_image_conditioned_step_plan(
        requested_refinement_steps=steps,
        denoising_strength=strength,
        step_policy=step_policy,
    )
    sampler_name = str(
        getattr(request, "hires_sampler_name", "")
        or getattr(request, "sampler_name", "")
        or ""
    ).strip()
    scheduler_name = str(
        getattr(request, "hires_scheduler_name", "")
        or getattr(request, "scheduler_name", "")
        or ""
    ).strip()
    raw_hires_cfg = getattr(request, "hires_cfg_scale", None)
    cfg_scale = float(
        getattr(request, "cfg_scale", 7.0) if raw_hires_cfg is None else raw_hires_cfg
    )
    raw_hires_rescale = getattr(request, "hires_cfg_rescale", None)
    cfg_rescale = float(
        getattr(request, "cfg_rescale", 0.0)
        if raw_hires_rescale is None
        else raw_hires_rescale
    )
    if enabled and (
        dimensions.effective_width == dimensions.base_width
        and dimensions.effective_height == dimensions.base_height
    ):
        raise ValueError(
            "Hires fix is enabled, but the target dimensions are the same as the base dimensions. "
            "Use hires_size_mode=scale_from_base or explicit_dimensions."
        )
    return HiresExecutionPlan(
        enabled=enabled,
        dimensions=dimensions,
        steps=steps,
        internal_steps=step_plan.internal_schedule_steps,
        effective_steps=step_plan.effective_refinement_steps,
        denoising_strength=step_plan.normalized_denoising_strength,
        safe_denoising_strength=step_plan.safe_denoising_strength,
        upscaler=upscaler,
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale,
        step_policy=step_policy,
        noise_policy=HIRES_NOISE_POLICY_ID,
    )


def build_hires_request(
    request: GenerationRequest,
    plan: HiresExecutionPlan,
) -> GenerationRequest:
    hires_request = replace(
        request,
        positive_prompt=str(request.hires_positive_prompt or request.positive_prompt),
        negative_prompt=str(
            request.hires_negative_prompt
            if request.hires_negative_prompt not in (None, "")
            else request.negative_prompt
        ),
        width=int(plan.dimensions.effective_width),
        height=int(plan.dimensions.effective_height),
        steps=int(plan.internal_steps),
        sampler_name=str(plan.sampler_name),
        scheduler_name=str(plan.scheduler_name),
        sampler_kwargs=dict(
            getattr(request, "_hires_resolved_sampler_kwargs", {}) or {}
        ),
        scheduler_kwargs=dict(
            getattr(request, "_hires_resolved_scheduler_kwargs", {}) or {}
        ),
        cfg_scale=float(plan.cfg_scale),
        cfg_rescale=float(plan.cfg_rescale),
        prompt_cfg_schedule={},
        prompt_cfg_pass_schedules=dict(request.prompt_cfg_pass_schedules or {}),
        prompt_cfg_recorded_schedules=dict(request.prompt_cfg_recorded_schedules or {}),
        prompt_cfg_replay_mode=str(request.prompt_cfg_replay_mode or "reconstruct"),
        prompt_expansion_record={},
        prompt_expansion_pass_records=dict(request.prompt_expansion_pass_records or {}),
        prompt_expansion_recorded=dict(request.prompt_expansion_recorded or {}),
        prompt_expansion_replay_mode=str(request.prompt_expansion_replay_mode or "reconstruct"),
        prompt_semantic_pass_records=dict(request.prompt_semantic_pass_records or {}),
        prompt_semantic_recorded=dict(request.prompt_semantic_recorded or {}),
        prompt_semantic_replay_mode=str(request.prompt_semantic_replay_mode or "reconstruct"),
        region_pass_records=dict(request.region_pass_records or {}),
        region_recorded=dict(request.region_recorded or {}),
        region_replay_mode=str(request.region_replay_mode or "reconstruct"),
        prompt_parser_name=str(request.hires_prompt_parser_name or request.prompt_parser_name),
        prompt_parser_kwargs=dict(request.hires_prompt_parser_kwargs or request.prompt_parser_kwargs),
        prompt_shortcut_profile_name=str(
            request.hires_shortcut_profile_name or request.prompt_shortcut_profile_name
        ),
        prompt_shortcut_profile_snapshot=dict(
            request.hires_shortcut_profile_snapshot or request.prompt_shortcut_profile_snapshot
        ),
        base_prompt_parser_name=str(request.hires_prompt_parser_name or request.prompt_parser_name),
        base_shortcut_profile_name=str(
            request.hires_shortcut_profile_name or request.prompt_shortcut_profile_name
        ),
        hires_enabled=False,
        hires_step_policy=plan.step_policy,
        return_latents=True,
        save_images=False,
        output_dir=None,
        resolved_seeds=list(request.resolved_seeds),
    )
    setattr(hires_request, "_is_hires_request", True)
    return hires_request


def hires_schedule_baseline_metadata(schedule: SchedulerOutput) -> dict[str, Any]:
    """Return the Phase 14M-1 compatibility record from validated tensors.

    Counts and start coordinates are intentionally derived from the active
    tensors. Request values remain separate so scheduler overrides cannot be
    mistaken for actual executed transitions.
    """

    metadata = dict(schedule.metadata or {})
    timesteps = schedule.timesteps
    starting_timestep = None
    if timesteps is not None and timesteps.numel() > 0:
        starting_timestep = _scalar_float(timesteps[0])
    return {
        "schema_version": 1,
        "step_policy": str(metadata.get("hires_step_policy") or DEFAULT_HIRES_STEP_POLICY),
        "requested_hires_steps": int(metadata.get("hires_requested_steps", schedule.requested_steps)),
        "planned_internal_schedule_steps": int(
            metadata.get(
                "hires_planned_internal_schedule_steps",
                metadata.get("hires_full_schedule_transition_count", schedule.sigma_transitions),
            )
        ),
        "full_schedule_transition_count": int(
            metadata.get("hires_full_schedule_transition_count", schedule.sigma_transitions)
        ),
        "effective_second_pass_transition_count": int(schedule.sigma_transitions),
        "requested_denoising_strength": float(
            metadata.get("hires_requested_denoising_strength", metadata.get("hires_denoising_strength", 1.0))
        ),
        "denoising_strength": float(metadata.get("hires_denoising_strength", 1.0)),
        "safe_denoising_strength": float(
            metadata.get("hires_safe_denoising_strength", metadata.get("hires_denoising_strength", 1.0))
        ),
        "selected_start_index": int(metadata.get("hires_schedule_start_index", 0)),
        "selected_starting_sigma": _scalar_float(schedule.sigmas[0]),
        "selected_starting_timestep": starting_timestep,
        "sampler_name": str(metadata.get("hires_sampler_name") or ""),
        "scheduler_name": str(metadata.get("hires_scheduler_name") or ""),
        "noise_policy_identifier": str(
            metadata.get("hires_noise_policy") or HIRES_NOISE_POLICY_ID
        ),
        "counts_are_tensor_derived": True,
    }


def slice_schedule_for_denoising(
    schedule: SchedulerOutput,
    denoising_strength: float,
    *,
    step_policy: str = PROPORTIONAL_TAIL_V1,
    sampler_name: str | None = None,
    scheduler_name: str | None = None,
    noise_policy: str = HIRES_NOISE_POLICY_ID,
) -> SchedulerOutput:
    conditioned = build_image_conditioned_schedule(
        schedule,
        requested_refinement_steps=int(schedule.requested_steps),
        denoising_strength=denoising_strength,
        step_policy=step_policy,
        scheduler_identifier=str(scheduler_name or ""),
        scheduler_configuration=dict((schedule.metadata or {}).get("validated_settings") or {}),
        requires_terminal_zero=None,
        sampler_requires_timestep=True,
    )
    output = conditioned.active_schedule
    output.metadata["hires_sampler_name"] = str(sampler_name or "")
    output.metadata["hires_scheduler_name"] = str(scheduler_name or "")
    output.metadata["hires_noise_policy"] = str(noise_policy or HIRES_NOISE_POLICY_ID)
    output.metadata["hires_schedule_baseline"] = hires_schedule_baseline_metadata(output)
    replay_record = build_schedule_replay_record(
        conditioned,
        scheduler_identifier=str(scheduler_name or ""),
        scheduler_configuration=dict((schedule.metadata or {}).get("validated_settings") or {}),
        sampler_name=str(sampler_name or ""),
        requires_terminal_zero=None,
    )
    output.metadata["hires_schedule_replay"] = replay_record
    output.metadata["hires_schedule_fingerprint"] = build_schedule_fingerprint_record(
        conditioned,
        replay_record=replay_record,
    )
    return output


def upscale_latents(
    latents: torch.Tensor,
    *,
    target_width: int,
    target_height: int,
    latent_scale_factor: int,
    upscaler: str,
) -> torch.Tensor:
    if not torch.is_tensor(latents) or latents.ndim != 4:
        raise ValueError("Hires latent upscaling requires a BCHW latent tensor.")
    latent_width = int(target_width) // int(latent_scale_factor)
    latent_height = int(target_height) // int(latent_scale_factor)
    if latent_width < 1 or latent_height < 1:
        raise ValueError("Hires target dimensions are too small for the latent scale factor.")
    mode = str(upscaler).replace("latent_", "", 1)
    kwargs: dict[str, Any] = {
        "size": (latent_height, latent_width),
        "mode": mode,
    }
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(latents, **kwargs)


def add_hires_noise(
    latents: torch.Tensor,
    *,
    schedule: SchedulerOutput,
    seeds: list[int],
) -> torch.Tensor:
    if len(seeds) != int(latents.shape[0]):
        raise ValueError("Hires noise seeds must match the latent batch size.")
    noise_items = []
    for index, seed in enumerate(seeds):
        generator = create_torch_generator(
            offset_seed(int(seed), HIRES_NOISE_SEED_OFFSET),
            device=latents.device,
        )
        noise_items.append(
            torch.randn(
                (1, *latents.shape[1:]),
                generator=generator,
                device=latents.device,
                dtype=latents.dtype,
            )
        )
    noise = torch.cat(noise_items, dim=0)
    return latents + noise * float(schedule.initial_sigma)


__all__ = [
    "A1111_FIXED_STEPS_V1",
    "DEFAULT_HIRES_STEP_POLICY",
    "HIRES_NOISE_POLICY_ID",
    "HIRES_NOISE_SEED_OFFSET",
    "HiresExecutionPlan",
    "PROPORTIONAL_TAIL_V1",
    "SUPPORTED_HIRES_STEP_POLICIES",
    "add_hires_noise",
    "build_hires_request",
    "hires_schedule_baseline_metadata",
    "resolve_hires_execution_plan",
    "slice_schedule_for_denoising",
    "upscale_latents",
]
