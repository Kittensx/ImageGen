from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Any, Mapping

import torch

from image_gen.contracts import GenerationRequest, SchedulerOutput
from image_gen.runtime.hires_sizing import HiresDimensionPlan, resolve_hires_dimensions
from image_gen.systems.image_conditioning import VAEEncodeResult, VAERoundTripResult
from image_gen.systems.upscaling import (
    SUPPORTED_ASPECT_POLICIES,
    SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS,
    SUPPORTED_PADDING_MODES,
    UpscalerDescriptor,
    compute_native_output_dimensions,
)
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
_VALID_STRATEGIES = {"pixel_neural"}


def _scalar_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() < 1:
            return None
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


@dataclass(frozen=True)
class HiresUpscalePlan:
    strategy: str
    legacy_value: str
    latent_interpolation: str | None
    upscaler_id: str | None
    descriptor: UpscalerDescriptor | None
    target_width: int
    target_height: int
    exact_resize_filter: str
    final_size_correction_filter: str
    aspect_policy: str
    padding_mode: str
    allow_tiling: bool
    native_scale: int
    predicted_native_width: int
    predicted_native_height: int
    requested_final_width: int
    requested_final_height: int
    tile_size: int
    tile_overlap: int
    tile_batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "legacy_value": self.legacy_value,
            "latent_interpolation": self.latent_interpolation,
            "upscaler_id": self.upscaler_id,
            "descriptor": self.descriptor.to_dict() if self.descriptor is not None else None,
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "exact_resize_filter": self.exact_resize_filter,
            "final_size_correction_filter": self.final_size_correction_filter,
            "aspect_policy": self.aspect_policy,
            "padding_mode": self.padding_mode,
            "allow_tiling": bool(self.allow_tiling),
            "native_scale": int(self.native_scale),
            "predicted_native_width": int(self.predicted_native_width),
            "predicted_native_height": int(self.predicted_native_height),
            "requested_final_width": int(self.requested_final_width),
            "requested_final_height": int(self.requested_final_height),
            "tile_size": int(self.tile_size),
            "tile_overlap": int(self.tile_overlap),
            "tile_batch_size": int(self.tile_batch_size),
        }


@dataclass(frozen=True)
class PixelNeuralHiresSourceResult:
    """Immutable source-preparation boundary before Phase 14M noise."""

    exact_target_images: torch.Tensor
    upscale_metadata: Mapping[str, Any]
    vae_encode_result: VAEEncodeResult
    vae_round_trip: VAERoundTripResult | None = None
    diagnostic_artifacts: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        images = self.exact_target_images
        if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError("Pixel-neural source result requires an exact RGB BCHW tensor.")
        metadata = deepcopy(dict(self.upscale_metadata or {}))
        object.__setattr__(self, "upscale_metadata", metadata)
        object.__setattr__(
            self,
            "diagnostic_artifacts",
            deepcopy(dict(self.diagnostic_artifacts or {})),
        )
        target_width = int(metadata.get("target_width") or 0)
        target_height = int(metadata.get("target_height") or 0)
        if target_width < 1 or target_height < 1:
            raise ValueError("Pixel-neural source result requires exact target dimensions in upscaler metadata.")
        if tuple(images.shape[-2:]) != (target_height, target_width):
            raise ValueError("Pixel-neural source tensor does not match upscaler target metadata.")
        latents = self.vae_encode_result.latents
        if int(latents.shape[0]) != int(images.shape[0]):
            raise ValueError("Pixel-neural source result must preserve batch ordering into VAE latents.")
        encode_metadata = dict(self.vae_encode_result.metadata or {})
        exact_contract = dict(encode_metadata.get("exact_image_contract") or {})
        if (
            int(exact_contract.get("target_width") or 0) != target_width
            or int(exact_contract.get("target_height") or 0) != target_height
        ):
            raise ValueError("Pixel-neural VAE encode metadata does not match the exact target tensor.")
        vae_identity = dict(encode_metadata.get("vae") or {})
        vae_hash = str(vae_identity.get("sha256") or "").casefold()
        if len(vae_hash) != 64 or any(character not in "0123456789abcdef" for character in vae_hash):
            raise ValueError("Pixel-neural source result requires a complete VAE SHA-256 identity.")
        encode_upscaler = dict(encode_metadata.get("upscale_provenance") or {})
        if (
            str(encode_upscaler.get("upscaler_id") or "")
            != str(metadata.get("upscaler_id") or "")
            or str(encode_upscaler.get("upscaler_sha256") or "").casefold()
            != str(metadata.get("upscaler_sha256") or "").casefold()
        ):
            raise ValueError("Pixel-neural VAE encode provenance does not match the executed upscaler.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": "phase14n5-pixel-neural-source-result-v1",
            "exact_target_shape": [int(value) for value in self.exact_target_images.shape],
            "upscale_metadata": dict(self.upscale_metadata),
            "vae_encode": self.vae_encode_result.to_serializable_dict(),
            "vae_round_trip": (
                self.vae_round_trip.to_serializable_dict()
                if self.vae_round_trip is not None
                else None
            ),
            "diagnostic_artifacts": dict(self.diagnostic_artifacts or {}),
        }


def resolve_hires_upscale_plan(
    request: GenerationRequest,
    *,
    dimensions: HiresDimensionPlan | None = None,
    registry: Any | None = None,
) -> HiresUpscalePlan:
    """Resolve the active `.pth`-only hires source plan.

    Only the active pixel-neural strategy is executable. Historical disabled
    requests may still carry stale fields, but active execution never imports or
    falls back to retired hires implementations.
    """

    dimensions = dimensions or resolve_hires_dimensions(request)
    enabled = bool(getattr(request, "hires_enabled", False))
    legacy = str(getattr(request, "hires_upscaler", "") or "").strip()
    requested_strategy = str(
        getattr(request, "hires_strategy", "pixel_neural") or "pixel_neural"
    ).strip().casefold()
    requested_id = str(getattr(request, "hires_upscaler_id", "") or "").strip()
    selected_id = requested_id or legacy

    if requested_strategy not in _VALID_STRATEGIES:
        if not enabled:
            selected_id = ""
            requested_strategy = "pixel_neural"
        else:
            raise ValueError("hires_strategy must be 'pixel_neural'.")

    legacy_filter = str(
        getattr(request, "hires_exact_resize_filter", "bicubic") or "bicubic"
    ).strip().casefold()
    final_filter = str(
        getattr(request, "hires_final_size_correction_filter", "") or legacy_filter
    ).strip().casefold()
    aspect_policy = str(
        getattr(request, "hires_aspect_policy", "stretch") or "stretch"
    ).strip().casefold()
    padding_mode = str(
        getattr(request, "hires_padding_mode", "reflect") or "reflect"
    ).strip().casefold()
    if not bool(getattr(dimensions, "aspect_ratio_changed", False)):
        aspect_policy = "stretch"
    if final_filter not in SUPPORTED_FINAL_SIZE_CORRECTION_FILTERS:
        raise ValueError("hires_final_size_correction_filter must be auto, nearest, bilinear, bicubic, or area.")
    if aspect_policy not in SUPPORTED_ASPECT_POLICIES:
        raise ValueError("hires_aspect_policy must be stretch, crop_to_fill, or pad_to_fit.")
    if padding_mode not in SUPPORTED_PADDING_MODES:
        raise ValueError("hires_padding_mode must be reflect, replicate, blurred_edge, or black.")

    common = {
        "strategy": "pixel_neural",
        "legacy_value": selected_id,
        "latent_interpolation": None,
        "upscaler_id": selected_id or None,
        "target_width": int(dimensions.effective_width),
        "target_height": int(dimensions.effective_height),
        "exact_resize_filter": legacy_filter,
        "final_size_correction_filter": final_filter,
        "aspect_policy": aspect_policy,
        "padding_mode": padding_mode,
        "allow_tiling": False,
        "native_scale": 0,
        "predicted_native_width": 0,
        "predicted_native_height": 0,
        "requested_final_width": int(getattr(dimensions, "final_width", dimensions.effective_width)),
        "requested_final_height": int(getattr(dimensions, "final_height", dimensions.effective_height)),
        "tile_size": int(getattr(request, "hires_tile_size", 0) or 0),
        "tile_overlap": int(getattr(request, "hires_tile_overlap", 16) or 0),
        "tile_batch_size": int(getattr(request, "hires_tile_batch_size", 1) or 1),
    }

    if not selected_id:
        if enabled:
            raise ValueError(
                "Pixel-neural hires requires a discovered neural .pth upscaler ID."
            )
        return HiresUpscalePlan(descriptor=None, **common)

    if registry is None:
        registry = getattr(request, "_hires_upscaler_registry", None)
    if registry is None and not enabled:
        return HiresUpscalePlan(descriptor=None, **common)
    if registry is None:
        raise ValueError(
            f"Pixel-neural hires requested {selected_id!r}, but no upscaler registry is configured."
        )
    try:
        descriptor = registry.resolve_neural(selected_id)
    except Exception as exc:
        raise ValueError(
            f"Pixel-neural hires could not resolve stable upscaler ID {selected_id!r}: {exc}"
        ) from exc
    if not descriptor.selectable:
        raise ValueError(
            f"Pixel-neural upscaler {selected_id!r} is not supported: {descriptor.load_status}."
        )
    if not descriptor.sha256 or len(descriptor.sha256) != 64:
        raise ValueError(f"Pixel-neural upscaler {selected_id!r} lacks a complete SHA-256.")
    expected_sha256 = str(
        getattr(request, "hires_expected_upscaler_sha256", "") or ""
    ).strip().casefold()
    if expected_sha256:
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("Recorded neural upscaler SHA-256 is invalid.")
        if descriptor.sha256.casefold() != expected_sha256:
            raise ValueError(
                f"Recorded neural upscaler hash mismatch for {selected_id!r}: "
                f"expected {expected_sha256}, found {descriptor.sha256.casefold()}."
            )
    expected_native_scale = int(getattr(request, "hires_expected_native_scale", 0) or 0)
    if expected_native_scale and int(descriptor.native_scale) != expected_native_scale:
        raise ValueError(
            f"Recorded neural upscaler native scale mismatch for {selected_id!r}: "
            f"expected x{expected_native_scale}, found x{int(descriptor.native_scale)}."
        )
    native_width, native_height = compute_native_output_dimensions(
        source_width=int(getattr(dimensions, "base_width", getattr(request, "width", 512))),
        source_height=int(getattr(dimensions, "base_height", getattr(request, "height", 512))),
        native_scale=int(descriptor.native_scale),
    )
    resolved_common = dict(common)
    resolved_common.update(
        allow_tiling=bool(descriptor.tile_supported),
        native_scale=int(descriptor.native_scale),
        predicted_native_width=native_width,
        predicted_native_height=native_height,
    )
    return HiresUpscalePlan(descriptor=descriptor, **resolved_common)


@dataclass(frozen=True)
class HiresExecutionPlan:
    enabled: bool
    dimensions: HiresDimensionPlan
    steps: int
    internal_steps: int
    effective_steps: int
    denoising_strength: float
    safe_denoising_strength: float
    upscale_plan: HiresUpscalePlan
    sampler_name: str
    scheduler_name: str
    cfg_scale: float
    cfg_rescale: float
    step_policy: str = DEFAULT_HIRES_STEP_POLICY
    noise_policy: str = HIRES_NOISE_POLICY_ID

    @property
    def upscaler(self) -> str:
        return str(self.upscale_plan.upscaler_id or self.upscale_plan.legacy_value)

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
            "hires_strategy": self.upscale_plan.strategy,
            "upscale_plan": self.upscale_plan.to_dict(),
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
    upscale_plan = resolve_hires_upscale_plan(request, dimensions=dimensions)
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
        upscale_plan=upscale_plan,
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


def validate_recorded_hires_vae_identity(
    request: GenerationRequest,
    vae_provenance: Mapping[str, Any],
) -> None:
    """Reject pixel-neural replay when the executed VAE hash changed."""

    expected_sha256 = str(
        getattr(request, "hires_expected_vae_sha256", "") or ""
    ).strip().casefold()
    if not expected_sha256:
        return
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(
            "Recorded pixel-neural replay requires a complete expected VAE SHA-256."
        )
    actual_sha256 = str(vae_provenance.get("sha256") or "").strip().casefold()
    if actual_sha256 != expected_sha256:
        source_kind = str(
            getattr(request, "hires_expected_vae_source_kind", "") or "recorded VAE"
        ).strip()
        raise ValueError(
            f"Recorded VAE hash mismatch for {source_kind}: expected {expected_sha256}, "
            f"found {actual_sha256 or 'missing'}. Exact pixel-neural replay is blocked."
        )



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
    "HiresUpscalePlan",
    "PixelNeuralHiresSourceResult",
    "PROPORTIONAL_TAIL_V1",
    "SUPPORTED_HIRES_STEP_POLICIES",
    "add_hires_noise",
    "build_hires_request",
    "hires_schedule_baseline_metadata",
    "resolve_hires_execution_plan",
    "resolve_hires_upscale_plan",
    "slice_schedule_for_denoising",
    "validate_recorded_hires_vae_identity",
]
