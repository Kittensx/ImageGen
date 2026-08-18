from __future__ import annotations

import copy
from typing import Any, Optional

import torch

from modules.contracts import SamplerCapabilities, SamplerOutput
from modules.pipeline.conditioning_utils import (
    resolve_step_conditioning,
    resolve_step_model_conditioning,
)
from modules.pipeline.regional_conditioning import get_regional_conditioning_resolver
from modules.pipeline.sampler_trace_mixin import SamplerTraceMixin
from modules.pipeline.schedule_domain import FLOW_MATCH_DOMAIN, normalize_schedule_domain
from modules.ss_registry.schedulers.flow_match_euler_sched import FlowMatchEulerSchedulerAdapter


meta = {
    "name": "flow_euler",
    "label": "Flow Euler",
    "description": "Reference SD3 Flow Euler sampler using the qualified Diffusers FlowMatch scheduler step.",
    "config_key": "shared",
}


class FlowEulerSampler(SamplerTraceMixin):
    """Reference flow-matching Euler integration for SD3.

    IMAGE_GEN owns conditioning/guidance resolution and delegates the actual
    x_t -> x_t-1 update to the exact FlowMatchEulerDiscreteScheduler runtime
    qualified by SD3-07. No VP/Karras epsilon/x0 conversion is performed.
    """

    SAMPLER_NAME = "flow_euler"
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
        schedule_domain=FLOW_MATCH_DOMAIN,
    )
    SAMPLER_SCHEDULE_CAPABILITIES = SAMPLER_CAPABILITIES

    @staticmethod
    def _schedule_domain(schedule: Any) -> str:
        metadata = dict(getattr(schedule, "metadata", {}) or {})
        return normalize_schedule_domain(metadata.get("schedule_domain"))

    @classmethod
    def _require_flow_schedule(cls, schedule: Any) -> None:
        actual = cls._schedule_domain(schedule)
        if actual != FLOW_MATCH_DOMAIN:
            raise ValueError(
                "Flow Euler scheduler-domain mismatch: "
                f"sampler requires {FLOW_MATCH_DOMAIN!r}, schedule provides {actual!r}."
            )

    @staticmethod
    def _scheduler_step_index(runtime_scheduler: Any) -> Any:
        return getattr(runtime_scheduler, "step_index", getattr(runtime_scheduler, "_step_index", None))

    @classmethod
    def _resolve_runtime_scheduler(
        cls,
        schedule: Any,
        request: Any,
        state: Any | None,
        *,
        latents: torch.Tensor,
    ) -> Any:
        runtime = getattr(schedule, "_runtime_scheduler", None)
        if runtime is None and state is not None:
            extra = getattr(state, "extra", None)
            if isinstance(extra, dict):
                runtime = extra.get("_scheduler_runtime_obj")

        # A scheduler already stepped belongs to a previous execution. Rebuild
        # from the same local config rather than mutating private Diffusers state.
        if runtime is not None and cls._scheduler_step_index(runtime) is None:
            return runtime

        metadata = dict(getattr(schedule, "metadata", {}) or {})
        config_path = str(metadata.get("scheduler_config_path") or "").strip()
        if not config_path:
            raise RuntimeError(
                "Flow Euler requires the qualified Flow Match scheduler runtime or its local scheduler_config_path."
            )

        rebuilt_request = copy.copy(request)
        rebuilt_request.steps = int(getattr(schedule, "effective_steps", 0) or 0)
        rebuilt_request.device = str(latents.device)
        scheduler_kwargs = dict(getattr(request, "scheduler_kwargs", {}) or {})
        scheduler_kwargs["scheduler_config_path"] = config_path
        if metadata.get("shift") is not None:
            scheduler_kwargs["shift"] = float(metadata["shift"])
        if metadata.get("num_train_timesteps") is not None:
            scheduler_kwargs["num_train_timesteps"] = int(metadata["num_train_timesteps"])
        rebuilt_request.scheduler_kwargs = scheduler_kwargs
        adapter = FlowMatchEulerSchedulerAdapter()
        rebuilt = adapter.build_schedule(rebuilt_request)

        expected_sigmas = torch.as_tensor(schedule.sigmas, dtype=torch.float32).cpu()
        expected_timesteps = torch.as_tensor(schedule.timesteps, dtype=torch.float32).cpu()
        if not torch.allclose(rebuilt.sigmas.cpu(), expected_sigmas, rtol=1e-6, atol=1e-6):
            raise RuntimeError("Reconstructed Flow Match sigmas do not match the active schedule.")
        if not torch.allclose(rebuilt.timesteps.cpu(), expected_timesteps, rtol=1e-6, atol=1e-6):
            raise RuntimeError("Reconstructed Flow Match timesteps do not match the active schedule.")
        return getattr(rebuilt, "_runtime_scheduler")

    def sample(
        self,
        raw_model_fn,
        guided_model_fn,
        latents,
        schedule,
        conditioning,
        request,
        state: Optional[Any] = None,
    ) -> SamplerOutput:
        self._require_flow_schedule(schedule)
        sigmas = self._materialize_sigmas(schedule, latents)
        timesteps = self._materialize_timesteps(schedule, latents)
        requested_steps, effective_steps = self._resolve_effective_steps(
            request=request,
            schedule=schedule,
            sigmas=sigmas,
        )
        if int(timesteps.numel()) != effective_steps:
            raise ValueError(
                "Flow Euler requires exactly one model timestep per denoising transition."
            )
        if int(sigmas.numel()) != effective_steps + 1:
            raise ValueError(
                "Flow Euler requires one terminal sigma in addition to the denoising transitions."
            )

        regional_resolver = get_regional_conditioning_resolver(conditioning)
        if regional_resolver is not None:
            raise ValueError("Regional conditioning is not qualified for the SD3 Flow Euler path in SD3-08.")

        flow_fn = getattr(guided_model_fn, "predict_guided_flow", None)
        if not callable(flow_fn):
            raise TypeError(
                "Flow Euler requires guided_model_fn.predict_guided_flow from the SD3 denoising system."
            )

        runtime_scheduler = self._resolve_runtime_scheduler(
            schedule, request, state, latents=latents
        )
        cfg_scale = float(getattr(request, "cfg_scale", 1.0))
        x = latents
        progress = (
            getattr(state, "extra", {}).get("progress_reporter")
            if state is not None
            else None
        )
        step_records: list[dict[str, Any]] = []

        for i in range(effective_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            timestep = timesteps[i]
            latent_before = x.detach()
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
            flow_prediction = flow_fn(
                x,
                sigma,
                timestep,
                cond,
                uncond,
                cfg_scale,
                model_conditioning,
            )
            if not torch.is_tensor(flow_prediction) or flow_prediction.shape != x.shape:
                raise ValueError("SD3 flow prediction must match the latent tensor shape.")
            if not bool(torch.isfinite(flow_prediction).all()):
                raise ValueError("SD3 flow prediction contains NaN or Inf values.")

            previous_dtype = x.dtype
            x = runtime_scheduler.step(
                flow_prediction,
                timestep,
                x,
                return_dict=False,
            )[0]
            if x.dtype != previous_dtype:
                x = x.to(dtype=previous_dtype)
            if not bool(torch.isfinite(x).all()):
                raise ValueError("FlowMatchEulerDiscreteScheduler.step returned non-finite latents.")
            x = self._apply_latent_step_hook(
                state,
                request=request,
                latent=x,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                timestep=timestep,
            )

            step_record = {
                "step_index": int(i),
                "sigma": float(sigma.detach().cpu().item()),
                "sigma_next": float(sigma_next.detach().cpu().item()),
                "timestep": float(timestep.detach().cpu().item()),
                "cfg_scale": cfg_scale,
                "cfg_active": bool(cfg_scale > 1.0),
            }
            step_records.append(step_record)
            self._trace_step(
                request,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                timestep=timestep,
                latent_before=latent_before,
                latent_after=x,
                guided_output=flow_prediction,
                cfg_scale=cfg_scale,
                extra={
                    "integration_mode": self.SAMPLER_NAME,
                    "schedule_domain": FLOW_MATCH_DOMAIN,
                    "model_output_semantics": "flow_match",
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
                "schedule_domain": FLOW_MATCH_DOMAIN,
                "integration_prediction_type": "flow_match",
                "model_prediction_type": "flow_match",
                "cfg_scale": cfg_scale,
                "cfg_active": bool(cfg_scale > 1.0),
                "scheduler_step_owner": "diffusers.FlowMatchEulerDiscreteScheduler",
                "step_records": step_records,
            },
        )


class FlowEulerSamplerAdapter:
    SAMPLER_CAPABILITIES = FlowEulerSampler.SAMPLER_CAPABILITIES

    def __init__(self) -> None:
        self.sampler = FlowEulerSampler()

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
            progress.start(
                total=int(getattr(schedule, "effective_steps", request.steps)),
                desc="SD3 Flow Euler Sampling",
            )
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


SAMPLER_NAME = "flow euler"
SAMPLER_CLASS = FlowEulerSampler
SAMPLER_ADAPTER_CLASS = FlowEulerSamplerAdapter

PLUGIN_DESCRIPTOR = {
    "plugin_id": "sampler.flow_euler",
    "kind": "sampler",
    "name": "flow_euler",
    "label": "Flow Euler",
    "description": meta["description"],
    "version": "1",
    "module": __name__,
    "adapter_class": "FlowEulerSamplerAdapter",
    "aliases": ["flow euler", "sd3 flow euler"],
    "capabilities": {
        **FlowEulerSamplerAdapter.SAMPLER_CAPABILITIES.to_serializable_dict(),
        "schedule_domain": FLOW_MATCH_DOMAIN,
    },
    "config_schema": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}


__all__ = [
    "FlowEulerSampler",
    "FlowEulerSamplerAdapter",
    "SAMPLER_ADAPTER_CLASS",
    "PLUGIN_DESCRIPTOR",
    "meta",
]
