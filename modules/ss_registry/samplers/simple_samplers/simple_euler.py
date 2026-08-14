from __future__ import annotations

import torch

from image_gen.systems.guidance import (
    EffectiveGuidanceController,
    EffectiveGuidanceProfile,
    prompt_cfg_payload_from_request,
    requested_cfg_scale_for_step,
)
from typing import Any, Optional

from modules.contracts import SamplerCapabilities, SamplerOutput

from modules.pipeline.conditioning_utils import (
    call_with_optional_model_conditioning,
    resolve_step_conditioning,
    resolve_step_model_conditioning,
)
from modules.pipeline.regional_conditioning import get_regional_conditioning_resolver
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
        cfg_effective_per_step = []
        regional_resolver = get_regional_conditioning_resolver(conditioning)
        regional_guidance_active_any = False

        requested_steps, effective_steps = self._resolve_effective_steps(
            request=request,
            schedule=schedule,
            sigmas=sigmas,
        )
        sampler_kwargs = dict(getattr(request, "sampler_kwargs", {}) or {})
        guidance_profile = EffectiveGuidanceProfile.from_settings(
            requested_cfg_scale=float(cfg_scale),
            get_setting=lambda name, default: sampler_kwargs.get(name, default),
            sampler_local_rescale_cfg=False,
            sampler_local_rescale_factor=1.0,
        )
        guidance_controller = EffectiveGuidanceController(guidance_profile)
        sigma_max = sigmas[0] if sigmas.numel() else torch.tensor(0.0, device=x.device)
        sigma_min = sigmas[-1] if sigmas.numel() else torch.tensor(0.0, device=x.device)
        guidance_shaping_active_any = False
        guidance_shaping_auto_applied_any = False
        cfg_early_floor_applied_any = False

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
            model_conditioning = resolve_step_model_conditioning(
                conditioning=conditioning,
                step_index=i,
                latents=x,
                request=request,
            )

            if i == 0:
                resolver = getattr(conditioning, "extra", {}).get("resolver", None)
                stepwise_conditioning_used = resolver is not None

            step_requested_cfg_scale, prompt_cfg_step = requested_cfg_scale_for_step(
                request, step_index=i, total_steps=effective_steps
            )
            step_effective_cfg_scale, guidance_step = guidance_controller.compute(
                step_index=i,
                total_steps=effective_steps,
                sigma=sigma,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                requested_cfg_scale=step_requested_cfg_scale,
            )
            guidance_step.update(prompt_cfg_step)
            guidance_shaping_active_any = guidance_shaping_active_any or bool(
                guidance_step.get("guidance_shaping_active")
            )
            guidance_shaping_auto_applied_any = guidance_shaping_auto_applied_any or bool(
                guidance_step.get("guidance_shaping_auto_applied")
            )
            cfg_early_floor_applied_any = cfg_early_floor_applied_any or bool(
                guidance_step.get("cfg_early_floor_applied")
            )
            cfg_effective_per_step.append({
                "step_index": int(i),
                "sigma": float(sigma.item()) if hasattr(sigma, "item") else float(sigma),
                "timestep": float(timestep.item()) if hasattr(timestep, "item") else float(timestep),
                "requested_cfg_scale": float(step_requested_cfg_scale),
                "effective_cfg_scale": float(step_effective_cfg_scale),
                "effective_cfg_scale_pre_rescale": float(
                    guidance_step.get("effective_cfg_scale_pre_rescale", step_effective_cfg_scale)
                ),
                "ui_cfg_scale": float(cfg_scale),
                **guidance_step,
            })

            active_regions = (
                regional_resolver.resolve_regions(step_index=i, latents=x)
                if regional_resolver is not None
                else []
            )

            regional_guidance_active_any = regional_guidance_active_any or bool(active_regions)

            # Pipeline-owned CFG remains canonical; REGION replaces only the
            # conditional model output before CFG is applied.
            if active_regions:
                regional_guided = getattr(guided_model_fn, "predict_regional_guided_noise", None)
                if not callable(regional_guided):
                    raise TypeError("The denoising system does not provide native regional guidance.")
                noise = call_with_optional_model_conditioning(
                    regional_guided,
                    x,
                    sigma,
                    timestep,
                    cond,
                    uncond,
                    step_effective_cfg_scale,
                    active_regions,
                    regional_resolver.overlap_policy,
                    model_conditioning=model_conditioning,
                )
            else:
                noise = call_with_optional_model_conditioning(
                    guided_model_fn,
                    x,
                    sigma,
                    timestep,
                    cond,
                    uncond,
                    step_effective_cfg_scale,
                    model_conditioning=model_conditioning,
                )
            predicted_x0 = x - sigma * noise

            dt = sigma_next - sigma
            x = x + noise * dt
            x = self._apply_latent_step_hook(
                state,
                request=request,
                latent=x,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                timestep=timestep,
            )
            
            self._trace_step(
                request,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                latent_before=latent_before,
                latent_after=x,
                guided_noise=noise,
                cfg_scale=step_effective_cfg_scale,
                extra={
                    "integration_mode": self.SAMPLER_NAME,
                    "regional_guidance_active": bool(active_regions),
                    "regional_active_count": len(active_regions),
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
                    "requested_cfg_scale": float(step_requested_cfg_scale),
                    "effective_cfg_scale": float(step_effective_cfg_scale),
                    **guidance_step,
                },
            )
            if progress is not None:
                progress.update(1)
        self._trace_sampler_summary(
            request,
            requested_steps=requested_steps,
            effective_steps=effective_steps,
        )

       
        
        regional_runtime = (
            regional_resolver.runtime_snapshot() if regional_resolver is not None else {}
        )
        model_prediction_type = "epsilon"
        prediction_contract = getattr(guided_model_fn, "model_prediction_contract", None)
        if callable(prediction_contract):
            try:
                model_prediction_type = str(
                    dict(prediction_contract() or {}).get("prediction_type") or "epsilon"
                )
            except Exception:
                model_prediction_type = "epsilon"
        if regional_runtime and isinstance(getattr(request, "diagnostics", None), dict):
            request.diagnostics["regional_runtime"] = regional_runtime
            passes = request.diagnostics.setdefault("regional_runtime_passes", {})
            passes[str(regional_runtime.get("pass") or "base")] = regional_runtime

        return SamplerOutput(
            latents=x,
            extra={
                
                "sampler_name": self.SAMPLER_NAME,
                
                "requested_steps": requested_steps,
                "effective_steps": effective_steps,
                "integration_mode_used": self.SAMPLER_NAME,
                "model_prediction_type": model_prediction_type,
                "integration_prediction_type": "epsilon",
                "schedule_transitions": int(sigmas.numel() - 1),
                "stepwise_conditioning_used": stepwise_conditioning_used,
                "prompt_cfg_schedule": prompt_cfg_payload_from_request(request),
                "prompt_cfg_applied": any(bool(item.get("prompt_cfg_applied")) for item in cfg_effective_per_step),
                "cfg_effective_per_step": cfg_effective_per_step,
                "cfg_step_series": {
                    "schema_version": 1,
                    "coordinate": "completed_denoising_step",
                    "source": "sampler_recorded",
                    "supports_future_step_overrides": True,
                    "points": [
                        {
                            "step_index": int(item.get("step_index", index)),
                            "requested_cfg_scale": float(item.get("requested_cfg_scale", cfg_scale)),
                            "effective_cfg_scale": float(item.get("effective_cfg_scale", cfg_scale)),
                            "sigma": item.get("sigma"),
                            "timestep": item.get("timestep"),
                            "guidance_mode": item.get("cfg_guidance_mode", guidance_profile.cfg_guidance_mode),
                            "cfg_rescale": float(getattr(request, "cfg_rescale", 0.0) or 0.0),
                            "cfg_rescale_applied": bool(float(getattr(request, "cfg_rescale", 0.0) or 0.0) > 0.0),
                            "override_source": (
                                item.get("cfg_source", "superhybrid_prompt")
                                if bool(item.get("prompt_cfg_applied", False))
                                else item.get("override_source", "base_request")
                            ),
                            "ui_cfg_scale": item.get("ui_cfg_scale", cfg_scale),
                            "prompt_cfg_applied": bool(item.get("prompt_cfg_applied", False)),
                            "transition_id": item.get("transition_id"),
                        }
                        for index, item in enumerate(cfg_effective_per_step)
                    ],
                },
                "cfg_effective_range": {
                    "min": min((float(item["effective_cfg_scale"]) for item in cfg_effective_per_step), default=float(cfg_scale)),
                    "max": max((float(item["effective_cfg_scale"]) for item in cfg_effective_per_step), default=float(cfg_scale)),
                },
                "guidance_mode": guidance_profile.cfg_guidance_mode,
                "guidance_shaping_active": guidance_shaping_active_any,
                "guidance_shaping_auto_applied": guidance_shaping_auto_applied_any,
                "cfg_early_floor_applied": cfg_early_floor_applied_any,
                "regional_guidance_used": bool(regional_guidance_active_any),
                "regional_backend": "image_gen_model_output" if regional_guidance_active_any else "none",
                "regional_runtime": regional_runtime,
                
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
