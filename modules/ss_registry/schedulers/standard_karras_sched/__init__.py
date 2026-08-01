from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from modules.contracts import SchedulerOutput
from modules.pipeline.schedule_compat import ensure_schedule_for_sampler


@dataclass(frozen=True)
class TrainingSigmaTable:
    sigmas: torch.Tensor
    log_sigmas: torch.Tensor
    sigma_min: float
    sigma_max: float


def _training_sigma_table(
    *,
    device: torch.device,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> TrainingSigmaTable:
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps must be at least 2.")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("beta_start and beta_end must satisfy 0 < start < end < 1.")

    betas = torch.linspace(
        beta_start**0.5,
        beta_end**0.5,
        num_train_timesteps,
        device=device,
        dtype=torch.float64,
    ).square()
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sigmas = torch.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)
    return TrainingSigmaTable(
        sigmas=sigmas,
        log_sigmas=sigmas.log(),
        sigma_min=float(sigmas[0].detach().cpu()),
        sigma_max=float(sigmas[-1].detach().cpu()),
    )


def _karras_sigmas(
    *,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if sigma_min <= 0.0:
        raise ValueError("sigma_min must be greater than zero.")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be greater than sigma_min.")
    if rho <= 0.0:
        raise ValueError("rho must be greater than zero.")

    ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float64)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).pow(rho)
    return torch.cat((sigmas, sigmas.new_zeros(1))).to(dtype=torch.float32)


def _sigmas_to_timesteps(
    sigmas: torch.Tensor,
    table: TrainingSigmaTable,
) -> tuple[torch.Tensor, dict[str, Any]]:
    requested = sigmas.to(device=sigmas.device, dtype=torch.float64).clamp(min=0.0)
    zero_mask = requested <= 0.0
    clipped_low = int(
        ((requested > 0.0) & (requested < table.sigma_min)).sum().item()
    )
    clipped_high = int((requested > table.sigma_max).sum().item())

    log_requested = requested.clamp(min=table.sigma_min).log()
    log_requested = log_requested.clamp(
        min=float(table.log_sigmas[0]),
        max=float(table.log_sigmas[-1]),
    )
    upper = torch.searchsorted(table.log_sigmas, log_requested)
    upper = upper.clamp(min=1, max=table.log_sigmas.numel() - 1)
    lower = upper - 1
    low_log = table.log_sigmas[lower]
    high_log = table.log_sigmas[upper]
    weight = (log_requested - low_log) / (high_log - low_log).clamp(min=1e-12)
    timesteps = lower.to(torch.float64) + weight
    timesteps = torch.where(zero_mask, torch.zeros_like(timesteps), timesteps)

    metadata = {
        "type": "sd_scaled_linear_beta_log_sigma_interpolation",
        "training_sigma_min": table.sigma_min,
        "training_sigma_max": table.sigma_max,
        "clipped_low_count": clipped_low,
        "clipped_high_count": clipped_high,
    }
    return timesteps.to(dtype=torch.float32), metadata


class StandardKarrasSchedulerAdapter:
    """Plain model-bounded Karras scheduler for fixed-step samplers.

    This adapter intentionally excludes Simple KES behavior: no blending,
    tails, randomization, prepass, caching, step expansion, noise scaling, or
    automatic stabilization. It exists as a conventional control scheduler for
    DPM++ and Euler comparisons.
    """

    def __init__(self, state: Any = None, default_name: str = "standard_karras") -> None:
        self.state = state
        self.default_name = default_name

    def build_schedule(self, request: Any, state: Any = None) -> SchedulerOutput:
        active_state = state or self.state
        kwargs = dict(getattr(request, "scheduler_kwargs", {}) or {})
        # Compatibility negotiation may clamp a feature-rich sampler to this
        # scheduler's fixed-step contract. Preserve the decision as metadata,
        # but do not treat it as a Karras schedule control.
        negotiated_pipeline_mode = str(kwargs.pop("pipeline_mode", "fixed_steps") or "fixed_steps")
        compatibility_clamp = dict(kwargs.pop("compatibility", {}) or {})

        device = getattr(request, "device", None)
        if device is None and active_state is not None:
            device = getattr(getattr(active_state, "d", None), "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        requested_sigma_min = kwargs.pop("sigma_min", None)
        requested_sigma_max = kwargs.pop("sigma_max", None)
        num_train_timesteps = int(kwargs.pop("num_train_timesteps", 1000))
        beta_start = float(kwargs.pop("beta_start", 0.00085))
        beta_end = float(kwargs.pop("beta_end", 0.012))
        rho = float(kwargs.pop("rho", 7.0))
        allow_out_of_range = bool(kwargs.pop("allow_out_of_range", False))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown standard_karras scheduler setting(s): {unknown}.")

        table = _training_sigma_table(
            device=device,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        sigma_min = table.sigma_min if requested_sigma_min is None else float(requested_sigma_min)
        sigma_max = table.sigma_max if requested_sigma_max is None else float(requested_sigma_max)

        if not allow_out_of_range:
            if sigma_min < table.sigma_min or sigma_max > table.sigma_max:
                raise ValueError(
                    "standard_karras defaults to the checkpoint training sigma range. "
                    f"Requested [{sigma_min}, {sigma_max}], model range is "
                    f"[{table.sigma_min}, {table.sigma_max}]. Set "
                    "allow_out_of_range=true only for an explicit experiment."
                )

        sigmas = _karras_sigmas(
            steps=int(request.steps),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=device,
        )
        timesteps, mapping = _sigmas_to_timesteps(sigmas, table)
        mapping.update(
            {
                "num_train_timesteps": num_train_timesteps,
                "beta_start": beta_start,
                "beta_end": beta_end,
            }
        )

        metadata = {
            "scheduler_name": "standard_karras",
            "scheduler_family": "standard",
            "schedule_mode": "karras",
            "schedule_control": "model_bounded_control",
            "rho": rho,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "uses_model_sigma_bounds": (
                requested_sigma_min is None and requested_sigma_max is None
            ),
            "allow_out_of_range": allow_out_of_range,
            "terminal_sigma_added": True,
            "active_blend_methods": [],
            "active_blend_weights": [],
            "tail_features_used": {
                "tail_steps_applied": False,
                "decay_tail_applied": False,
                "blended_tail_applied": False,
                "progressive_decay_applied": False,
                "auto_stabilization_applied": False,
                "step_expansion_applied": False,
            },
            "prepass_used": False,
            "predicted_stop_step": int(request.steps),
            "timestep_mapping": mapping,
            "compatibility_mode": negotiated_pipeline_mode,
            "sampler_capability_clamp": {
                "active": bool(
                    compatibility_clamp.get("step_expansion_clamped", False)
                    or compatibility_clamp.get("tail_metadata_clamped", False)
                ),
                **compatibility_clamp,
            },
        }
        output = SchedulerOutput(
            sigmas=sigmas,
            timesteps=timesteps,
            requested_steps=int(request.steps),
            effective_steps=int(request.steps),
            scheduler_step_override_applied=False,
            compatibility_mode=negotiated_pipeline_mode,
            metadata=metadata,
        )
        output = ensure_schedule_for_sampler(
            schedule=output,
            request=request,
            sampler_name=getattr(request, "sampler_name", None),
        )
        self._sync_state(active_state, request, output, device)
        return output

    @staticmethod
    def _sync_state(active_state: Any, request: Any, output: SchedulerOutput, device: torch.device) -> None:
        if active_state is None:
            return
        if hasattr(active_state, "d"):
            active_state.d.device = device
        if hasattr(active_state, "p"):
            for name in (
                "steps",
                "batch_size",
                "cfg_scale",
                "width",
                "height",
                "positive_prompt",
                "negative_prompt",
                "seed",
            ):
                if hasattr(active_state.p, name) and hasattr(request, name):
                    setattr(active_state.p, name, getattr(request, name))
        if hasattr(active_state, "sched"):
            active_state.sched.sigmas = output.sigmas
            active_state.sched.timesteps = output.timesteps
            active_state.sched.scheduler_name = "standard_karras"
            active_state.sched.selected_scheduler_name = "standard_karras"
            if hasattr(active_state.sched, "requested_steps"):
                active_state.sched.requested_steps = output.requested_steps
            if hasattr(active_state.sched, "effective_steps"):
                active_state.sched.effective_steps = output.effective_steps
            if hasattr(active_state.sched, "schedule_extra"):
                active_state.sched.schedule_extra = dict(output.extra)
            if hasattr(active_state.sched, "compatibility_mode"):
                active_state.sched.compatibility_mode = output.compatibility_mode
        if hasattr(active_state, "extra") and isinstance(active_state.extra, dict):
            active_state.extra["_scheduler_runtime_obj"] = None


SCHEDULER_ADAPTER_CLASS = StandardKarrasSchedulerAdapter

meta = {
    "name": "standard_karras",
    "label": "Standard Karras",
    "summary_text": (
        "Plain model-bounded Karras sigma schedule for DPM++/Euler control tests."
    ),
}

PLUGIN_DESCRIPTOR = {
    "plugin_id": "scheduler.standard_karras",
    "kind": "scheduler",
    "name": "standard_karras",
    "label": "Standard Karras",
    "description": meta["summary_text"],
    "version": "1",
    "module": __name__,
    "adapter_class": "StandardKarrasSchedulerAdapter",
    "aliases": ["standard karras", "karras", "dpm karras"],
    "capabilities": {
        "pipeline_modes": ["fixed_steps", "compatible"],
        "supports_fixed_steps": True,
        "supports_step_expansion": False,
        "supports_tail_metadata": False,
        "supports_tail_steps": False,
        "supports_decay_tail": False,
        "supports_blended_tail": False,
        "supports_progressive_decay": False,
        "scheduler_family": "standard",
    },
    "config_schema": {
        "type": "object",
        "properties": {
            "sigma_min": {"type": ["number", "null"], "default": None},
            "sigma_max": {"type": ["number", "null"], "default": None},
            "rho": {"type": "number", "default": 7.0},
            "num_train_timesteps": {"type": "integer", "default": 1000},
            "beta_start": {"type": "number", "default": 0.00085},
            "beta_end": {"type": "number", "default": 0.012},
            "allow_out_of_range": {"type": "boolean", "default": False},
        },
        "required": [],
        "additionalProperties": False,
    },
}

__all__ = [
    "StandardKarrasSchedulerAdapter",
    "SCHEDULER_ADAPTER_CLASS",
    "PLUGIN_DESCRIPTOR",
    "meta",
]
