from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from modules.contracts import SchedulerOutput
from modules.project_context import ProjectContext
from modules.sdxl_runtime_assets import SDXLRuntimeAssetResolver


def _training_sigmas(config: dict[str, Any], *, device: torch.device) -> torch.Tensor:
    n = int(config.get("num_train_timesteps", 1000))
    beta_start = float(config.get("beta_start", 0.00085))
    beta_end = float(config.get("beta_end", 0.012))
    beta_schedule = str(config.get("beta_schedule", "scaled_linear"))
    if n < 2:
        raise ValueError("num_train_timesteps must be at least 2")
    if beta_schedule == "scaled_linear":
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=torch.float64, device=device).square()
    elif beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, n, dtype=torch.float64, device=device)
    else:
        raise ValueError(f"Unsupported SDXL Euler beta_schedule: {beta_schedule!r}")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return torch.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)


def trailing_timesteps(num_train_timesteps: int, steps: int, *, device: torch.device) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    step_ratio = float(num_train_timesteps) / float(steps)
    values = torch.arange(float(num_train_timesteps), 0.0, -step_ratio, dtype=torch.float64, device=device)
    values = torch.round(values) - 1.0
    return values[:steps].to(dtype=torch.float32)


class SDXLEulerTrailingSchedulerAdapter:
    def __init__(self, state: Any = None) -> None:
        self.state = state

    @staticmethod
    def _device(request: Any, state: Any = None) -> torch.device:
        requested = getattr(request, "device", None)
        if requested is not None:
            return torch.device(requested)
        if state is not None and getattr(state, "d", None) is not None:
            candidate = getattr(state.d, "device", None)
            if candidate is not None:
                return torch.device(candidate)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _load_config(request: Any) -> tuple[dict[str, Any], str]:
        kwargs = dict(getattr(request, "scheduler_kwargs", {}) or {})
        explicit = str(kwargs.get("scheduler_config_path") or "").strip()
        if explicit:
            path = Path(explicit).expanduser().resolve()
        else:
            context = ProjectContext.load()
            path = SDXLRuntimeAssetResolver(context).resolve().scheduler_config
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"SDXL scheduler config must be an object: {path}")
        return payload, str(path)

    def build_schedule(self, request: Any, state: Any = None) -> SchedulerOutput:
        active_state = state if state is not None else self.state
        device = self._device(request, active_state)
        steps = int(getattr(request, "steps", 0) or 0)
        if steps < 1:
            raise ValueError("SDXL Euler trailing scheduler requires steps >= 1")
        config, config_path = self._load_config(request)
        n = int(config.get("num_train_timesteps", 1000))
        timesteps = trailing_timesteps(n, steps, device=device)
        training_sigmas = _training_sigmas(config, device=device)
        # Diffusers EulerDiscreteScheduler with trailing spacing linearly interpolates
        # the training sigma table at these timesteps. Trailing timesteps are integer
        # values for the supported Lightning step counts, but keep linear interpolation
        # for generic correctness.
        low = torch.floor(timesteps).long().clamp(0, n - 1)
        high = torch.ceil(timesteps).long().clamp(0, n - 1)
        weight = (timesteps.to(torch.float64) - low.to(torch.float64)).clamp(0.0, 1.0)
        sigmas = training_sigmas[low] * (1.0 - weight) + training_sigmas[high] * weight
        sigmas = torch.cat([sigmas.to(torch.float32), torch.zeros(1, device=device, dtype=torch.float32)])
        output = SchedulerOutput(
            sigmas=sigmas,
            timesteps=timesteps,
            requested_steps=steps,
            effective_steps=steps,
            scheduler_step_override_applied=False,
            compatibility_mode="fixed_steps",
            metadata={
                "scheduler_name": "sdxl_euler_trailing",
                "scheduler_family": "sdxl_euler",
                "schedule_domain": "vp_sigma",
                "timestep_spacing": "trailing",
                "prediction_type": str(config.get("prediction_type", "epsilon")),
                "scheduler_config_path": config_path,
                "num_train_timesteps": n,
                "beta_schedule": config.get("beta_schedule"),
                "beta_start": config.get("beta_start"),
                "beta_end": config.get("beta_end"),
            },
        )
        if active_state is not None and hasattr(active_state, "sched"):
            active_state.sched.sigmas = output.sigmas
            active_state.sched.timesteps = output.timesteps
            active_state.sched.scheduler_name = "sdxl_euler_trailing"
            active_state.sched.selected_scheduler_name = "sdxl_euler_trailing"
        return output


SCHEDULER_ADAPTER_CLASS = SDXLEulerTrailingSchedulerAdapter

PLUGIN_DESCRIPTOR = {
    "plugin_id": "scheduler.sdxl_euler_trailing",
    "kind": "scheduler",
    "name": "sdxl_euler_trailing",
    "label": "SDXL Euler Trailing",
    "description": "Euler sigma/timestep schedule using canonical SDXL training sigmas and trailing timestep spacing.",
    "version": "1",
    "module": __name__,
    "adapter_class": "SDXLEulerTrailingSchedulerAdapter",
    "aliases": ["sdxl euler trailing", "lightning euler", "sgm uniform"],
    "capabilities": {
        "pipeline_modes": ["fixed_steps", "compatible"],
        "supports_fixed_steps": True,
        "supports_step_expansion": False,
        "supports_tail_metadata": False,
        "supports_tail_steps": False,
        "supports_decay_tail": False,
        "supports_blended_tail": False,
        "supports_progressive_decay": False,
        "supports_hires_refinement": True,
        "scheduler_family": "sdxl_euler",
        "schedule_domain": "vp_sigma",
        # This scheduler is the canonical SDXL trailing-Euler schedule.  Keep
        # sampler compatibility explicit so WebUI capability filtering and
        # runtime validation share the same authoritative pair contract.
        "compatible_samplers": ["simple_euler"],
    },
    "config_schema": {
        "type": "object",
        "properties": {
            "scheduler_config_path": {"type": "string", "default": ""},
        },
        "required": [],
        "additionalProperties": False,
    },
}

__all__ = [
    "SDXLEulerTrailingSchedulerAdapter",
    "SCHEDULER_ADAPTER_CLASS",
    "PLUGIN_DESCRIPTOR",
    "trailing_timesteps",
]
