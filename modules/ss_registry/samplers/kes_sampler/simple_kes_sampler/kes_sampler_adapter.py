from __future__ import annotations

from modules.contracts import SamplerCapabilities, SamplerOutput
from modules.ss_registry.samplers.kes_sampler.simple_kes_sampler.kes_sampler import KESSampler
        
class KESSamplerAdapter:
    """Pipeline adapter for the KES sampler.

    KES owns CFG internally, so it receives ``raw_model_fn`` rather than a
    pre-guided denoiser.
    """

    SAMPLER_CAPABILITIES = SamplerCapabilities(
        sampler_name="kes",
        guidance_owner="sampler",
        uses_raw_model_fn=True,
        uses_guided_model_fn=False,
        supports_step_expansion=True,
        supports_tail_metadata=True,
        requires_requested_step_schedule=False,
        strict_validation=True,
        forced_pipeline_mode="extended_steps",
    )

    def __init__(self, shared_state=None, default_name: str = "kes"):
        self.state = shared_state
        self.default_name = default_name

    def sample(
        self,
        raw_model_fn,
        guided_model_fn,
        latents,
        schedule,
        conditioning,
        request,
        state=None,
    ) -> SamplerOutput:
        state = state or self.state
        
        
        progress = None
        if state is not None:
            progress = getattr(state, "extra", {}).get("progress_reporter")

        effective_steps = getattr(schedule, "effective_steps", None)
        if progress is not None:
            total = int(effective_steps) if effective_steps is not None else int(schedule.sigmas.numel() - 1)
            progress.start(total=total, desc="KES Sampling")


        sampler = KESSampler(
            **(getattr(request, "sampler_kwargs", {}) or {})
        )
        try:
            output = sampler.sample(
                raw_model_fn=raw_model_fn,
                latents=latents,
                schedule=schedule,
                conditioning=conditioning,
                request=request,
                state=state,
            )
        finally: 
            if progress is not None:
                progress.close()

        if state is not None and hasattr(state, "samp"):
            state.samp.samples = output.latents
            state.samp.sampler_name = request.sampler_name or self.default_name
            state.samp.selected_sampler_name = request.sampler_name or self.default_name
            state.samp.sampler_fn = sampler

        return SamplerOutput(
            latents=output.latents,
            extra={
                "sampler_name": request.sampler_name or self.default_name,
                # store only serializable info instead of the full object
                "sampler_settings": getattr(sampler, "settings", {}),
                **(output.extra or {}),
            },
        )
        
SAMPLER_ADAPTER_CLASS = KESSamplerAdapter