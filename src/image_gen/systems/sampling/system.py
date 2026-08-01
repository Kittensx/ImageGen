from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from image_gen.contracts import (
    ConditioningOutput,
    GenerationRequest,
    GuidedModelFnProtocol,
    RawModelFnProtocol,
    SamplerAdapterProtocol,
    SamplerOutput,
    SchedulerOutput,
)
from modules.txt2img.seed_utils import create_torch_generator, resolve_seed_sequence


@dataclass(frozen=True)
class GenerationDimensionPlan:
    """Describe requested output dimensions and the model-compatible canvas.

    Stable Diffusion latent tensors operate on a fixed pixel grid, normally an
    8-pixel grid. User-facing output dimensions do not need to be constrained to
    that grid. The runtime rounds the generation canvas upward, then center-crops
    the decoded image back to the exact requested size.
    """

    requested_width: int
    requested_height: int
    generation_width: int
    generation_height: int
    latent_scale_factor: int
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int

    @property
    def crop_required(self) -> bool:
        return bool(
            self.requested_width != self.generation_width
            or self.requested_height != self.generation_height
        )

    def to_serializable_dict(self) -> dict[str, int | bool | str]:
        return {
            "requested_width": int(self.requested_width),
            "requested_height": int(self.requested_height),
            "generation_width": int(self.generation_width),
            "generation_height": int(self.generation_height),
            "latent_scale_factor": int(self.latent_scale_factor),
            "crop_required": bool(self.crop_required),
            "crop_mode": "center",
            "crop_left": int(self.crop_left),
            "crop_top": int(self.crop_top),
            "crop_right": int(self.crop_right),
            "crop_bottom": int(self.crop_bottom),
        }


class LatentPreparationSystem:
    """Own deterministic initial-noise creation in scheduler sigma space."""

    def __init__(
        self,
        *,
        latent_scale_factor: int = 8,
        device: torch.device,
        dtype: torch.dtype,
        latent_channels: int = 4,
    ) -> None:
        self.latent_scale_factor = int(latent_scale_factor)
        self.device = device
        self.dtype = dtype
        self.latent_channels = int(latent_channels)
        if self.latent_scale_factor < 1:
            raise ValueError("latent_scale_factor must be at least 1.")

    @staticmethod
    def _align_up(value: int, multiple: int) -> int:
        return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)

    def plan_dimensions(self, request: GenerationRequest) -> GenerationDimensionPlan:
        requested_width = int(request.width)
        requested_height = int(request.height)
        if requested_width <= 0 or requested_height <= 0:
            raise ValueError("Generation width and height must be positive.")

        generation_width = self._align_up(requested_width, self.latent_scale_factor)
        generation_height = self._align_up(requested_height, self.latent_scale_factor)
        width_delta = generation_width - requested_width
        height_delta = generation_height - requested_height
        crop_left = width_delta // 2
        crop_top = height_delta // 2
        plan = GenerationDimensionPlan(
            requested_width=requested_width,
            requested_height=requested_height,
            generation_width=generation_width,
            generation_height=generation_height,
            latent_scale_factor=self.latent_scale_factor,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=width_delta - crop_left,
            crop_bottom=height_delta - crop_top,
        )

        # These are runtime-only attributes. Keeping request.width/request.height
        # unchanged preserves the exact user request in UI state and metadata.
        request.generation_width = generation_width
        request.generation_height = generation_height
        request.dimension_plan = plan.to_serializable_dict()
        return plan

    def prepare(self, request: GenerationRequest, schedule: SchedulerOutput) -> torch.Tensor:
        plan = self.plan_dimensions(request)
        if request.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        resolved_seeds = resolve_seed_sequence(request.seed, request.batch_size)
        request.seed = resolved_seeds[0]
        request.resolved_seeds = list(resolved_seeds)
        single_shape = (
            1,
            self.latent_channels,
            plan.generation_height // self.latent_scale_factor,
            plan.generation_width // self.latent_scale_factor,
        )
        latent_items = [
            torch.randn(
                single_shape,
                generator=create_torch_generator(seed, device=self.device),
                device=self.device,
                dtype=self.dtype,
            )
            for seed in resolved_seeds
        ]
        latents = torch.cat(latent_items, dim=0)
        if schedule.initial_sigma <= 0:
            raise ValueError("The first scheduler sigma must be greater than zero.")
        return latents * float(schedule.initial_sigma)


class SamplingSystem:
    """Own sampler invocation and validate its latent output boundary.

    The sampler consumes an already selected and validated active schedule. It
    does not convert denoising strength into a transition count.
    """

    def __init__(self, adapter: SamplerAdapterProtocol) -> None:
        self.adapter = adapter

    def sample(
        self,
        *,
        raw_model_fn: RawModelFnProtocol,
        guided_model_fn: GuidedModelFnProtocol,
        latents: torch.Tensor,
        schedule: SchedulerOutput,
        conditioning: ConditioningOutput,
        request: GenerationRequest,
        state: Any | None = None,
    ) -> SamplerOutput:
        output = self.adapter.sample(
            raw_model_fn=raw_model_fn,
            guided_model_fn=guided_model_fn,
            latents=latents,
            schedule=schedule,
            conditioning=conditioning,
            request=request,
            state=state,
        )
        if not isinstance(output, SamplerOutput):
            raise TypeError("Sampler adapter must return SamplerOutput.")
        if output.latents.shape != latents.shape:
            raise ValueError("Sampler output latent shape changed unexpectedly.")
        if not torch.isfinite(output.latents).all():
            raise ValueError("Sampler returned non-finite latents.")
        return output
