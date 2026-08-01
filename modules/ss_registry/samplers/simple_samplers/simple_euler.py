from __future__ import annotations

import torch
from typing import Any, Optional

from modules.contracts import SamplerCapabilities, SamplerOutput

from modules.pipeline.conditioning_utils import resolve_step_conditioning
from modules.pipeline.sampler_trace_mixin import SamplerTraceMixin

# -------------------------
# Metadata (optional)
# -------------------------

meta = {
    "name": "simple_euler",
    "label": "Simple Euler",
    "description": "Minimal standard Euler sampler using pipeline-guided CFG.",
    "config_key": "shared",
}

required_args = {
    "guided_model_fn": "callable",
    "latents": "torch.Tensor",
    "schedule": "SchedulerOutput-like object with sigmas",
    "conditioning": "ConditioningOutput-like object",
    "request": "GenerationRequest-like object",
}
optional_args = {
    "state": "shared pipeline state",
}

# -------------------------
# Sampler
# -------------------------

class SimpleEulerSampler(SamplerTraceMixin):
    """
    Standard Euler sampler.

    Responsibilities:
    - iterate sigmas
    - resolve stepwise conditioning
    - call guided_model_fn
    - apply Euler update

    Does NOT:
    - implement CFG
    - interpret scheduler metadata
    - apply noise policies
    """

    SAMPLER_NAME = "simple_euler"

    SAMPLER_CAPABILITIES = SamplerCapabilities(
        sampler_name=SAMPLER_NAME,
        guidance_owner="pipeline",
        uses_raw_model_fn=False,
        uses_guided_model_fn=True,
        supports_step_expansion=False,
        supports_tail_metadata=False,
        requires_requested_step_schedule=True,
        strict_validation=True,
        forced_pipeline_mode="fixed_steps",
    )
    SAMPLER_SCHEDULE_CAPABILITIES = SAMPLER_CAPABILITIES

    def sample(
        self,
        raw_model_fn,      # unused
        guided_model_fn,
        latents,
        schedule,
        conditioning,
        request,
        state: Optional[Any] = None,
    ) -> SamplerOutput:

        sigmas = self._materialize_sigmas(schedule, latents)
        timesteps = self._materialize_timesteps(schedule, latents)

        x = latents
        cfg_scale = float(getattr(request, "cfg_scale", 1.0))
        progress = (
            getattr(state, "extra", {}).get("progress_reporter")
            if state is not None
            else None
        )

        stepwise_conditioning_used = False

        requested_steps, effective_steps = self._resolve_effective_steps(
            request=request,
            schedule=schedule,
            sigmas=sigmas,
        )

        for i in range(sigmas.numel() - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            timestep = timesteps[i]

            latent_before = x.detach()

            # 🔑 Stepwise conditioning
            cond, uncond = resolve_step_conditioning(
                conditioning=conditioning,
                step_index=i,
                latents=x,
                state=state,
            )

            if i == 0:
                resolver = getattr(conditioning, "extra", {}).get("resolver", None)
                stepwise_conditioning_used = resolver is not None

            # 🔑 Pipeline-owned CFG
            noise = guided_model_fn(
                x,
                sigma,
                timestep,
                cond,
                uncond,
                cfg_scale,
            )
            predicted_x0 = x - sigma * noise

            dt = sigma_next - sigma
            x = x + noise * dt
            
            self._trace_step(
                request,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                latent_before=latent_before,
                latent_after=x,
                guided_noise=noise,
                cfg_scale=cfg_scale,
                extra={
                    "integration_mode": self.SAMPLER_NAME,
                },
            )
            self._emit_live_preview(
                state,
                request=request,
                step_index=i,
                total_steps=effective_steps,
                latent=x,
                predicted_x0=predicted_x0,
                sigma=sigma,
                model_timestep=timestep,
                metadata={
                    "integration_mode": self.SAMPLER_NAME,
                },
            )
            if progress is not None:
                progress.update(1)
        self._trace_sampler_summary(
            request,
            requested_steps=requested_steps,
            effective_steps=effective_steps,
        )

       
        
        return SamplerOutput(
            latents=x,
            extra={
                
                "sampler_name": self.SAMPLER_NAME,
                
                "requested_steps": requested_steps,
                "effective_steps": effective_steps,
                "integration_mode_used": self.SAMPLER_NAME,
                "model_prediction_type": "epsilon",
                "integration_prediction_type": "epsilon",
                "schedule_transitions": int(sigmas.numel() - 1),
                "stepwise_conditioning_used": stepwise_conditioning_used,
                
            },
        )


# -------------------------
# Adapter
# -------------------------

class SimpleEulerSamplerAdapter:
    SAMPLER_CAPABILITIES = SimpleEulerSampler.SAMPLER_CAPABILITIES

    def __init__(self):
        self.sampler = SimpleEulerSampler()

    def sample(
        self,
        raw_model_fn,
        guided_model_fn,
        latents,
        schedule,
        conditioning,
        request,
        state=None,
    ):
        progress = (
            getattr(state, "extra", {}).get("progress_reporter")
            if state is not None
            else None
        )
        if progress is not None:
            effective_steps = getattr(schedule, "effective_steps", None)
            total = (
                int(effective_steps)
                if effective_steps is not None
                else int(schedule.sigmas.numel() - 1)
            )
            progress.start(total=total, desc="Simple Euler Sampling")
        try:
            return self.sampler.sample(
                raw_model_fn=raw_model_fn,
                guided_model_fn=guided_model_fn,
                latents=latents,
                schedule=schedule,
                conditioning=conditioning,
                request=request,
                state=state,
            )
        finally:
            if progress is not None:
                progress.close()


# Optional registry hook
SAMPLER_NAME = "simple euler"
SAMPLER_CLASS = SimpleEulerSampler
SAMPLER_ADAPTER_CLASS = SimpleEulerSamplerAdapter

PLUGIN_DESCRIPTOR = {
    "plugin_id": "sampler.simple_euler",
    "kind": "sampler",
    "name": "simple_euler",
    "label": "Simple Euler",
    "description": meta["description"],
    "version": "1",
    "module": __name__,
    "adapter_class": "SimpleEulerSamplerAdapter",
    "aliases": ['simple euler', 'euler'],
    "capabilities": SimpleEulerSamplerAdapter.SAMPLER_CAPABILITIES.to_serializable_dict(),
    "config_schema": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}
