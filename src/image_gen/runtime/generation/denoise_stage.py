from __future__ import annotations

from typing import Any

import torch

from modules.attention_runtime import get_execution_evidence
from image_gen.systems.outpainting import extract_outpaint_failure_stage, format_outpaint_failure

from .context import GenerationContext


class BaseDenoiseStageMixin:
    def _run_base_denoise_stage(self, ctx: GenerationContext):
        diagnostics = ctx.diagnostics
        session = ctx.session
        request = ctx.request
        performance = ctx.performance
        latents = ctx.latents
        schedule = ctx.schedule
        conditioning = ctx.conditioning
        base_preview_policy = ctx.base_preview_policy
        outpaint_enabled = ctx.outpaint_enabled

        raw_model_fn = diagnostics.wrap_callable(
            session,
            "denoising",
            "predict_raw_noise",
            self.systems.denoising.predict_raw_noise,
        )
        guided_model_fn = diagnostics.wrap_callable(
            session,
            "denoising",
            "predict_guided_noise",
            self.systems.denoising.predict_guided_noise,
        )
        regional_conditional_fn = diagnostics.wrap_callable(
            session,
            "denoising",
            "predict_regional_conditional_noise",
            self.systems.denoising.predict_regional_conditional_noise,
        )
        regional_guided_fn = diagnostics.wrap_callable(
            session,
            "denoising",
            "predict_regional_guided_noise",
            self.systems.denoising.predict_regional_guided_noise,
        )
        regional_denoised_fn = diagnostics.wrap_callable(
            session,
            "denoising",
            "predict_regional_denoised",
            self.systems.denoising.predict_regional_denoised,
        )
        setattr(
            raw_model_fn,
            "predict_regional_conditional_noise",
            regional_conditional_fn,
        )
        setattr(
            guided_model_fn,
            "predict_regional_guided_noise",
            regional_guided_fn,
        )
        setattr(
            guided_model_fn,
            "predict_regional_denoised",
            regional_denoised_fn,
        )
        composable_noise_fn = diagnostics.wrap_callable(
            session,
            "denoising",
            "predict_composable_guided_noise",
            self.systems.denoising.predict_composable_guided_noise,
        )
        composable_denoised_fn = diagnostics.wrap_callable(
            session,
            "denoising",
            "predict_composable_denoised",
            self.systems.denoising.predict_composable_denoised,
        )
        setattr(
            guided_model_fn,
            "predict_composable_guided_noise",
            composable_noise_fn,
        )
        setattr(
            guided_model_fn,
            "predict_composable_denoised",
            composable_denoised_fn,
        )
        if bool(getattr(self.systems.denoising, "is_flow_match", False)):
            guided_flow_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_guided_flow",
                self.systems.denoising.predict_guided_flow,
            )
            setattr(guided_model_fn, "predict_guided_flow", guided_flow_fn)
            composable_flow_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_composable_guided_flow",
                self.systems.denoising.predict_composable_guided_flow,
            )
            setattr(guided_model_fn, "predict_composable_guided_flow", composable_flow_fn)
            setattr(
                guided_model_fn,
                "denoising_contract",
                self.systems.denoising.flow_match_contract_metadata,
            )
        else:
            guided_denoiser_adapter = self.systems.denoising.build_guided_epsilon_denoiser(
                guided_model_fn,
            )
            denoised_model_fn = diagnostics.wrap_callable(
                session,
                "denoising",
                "predict_denoised",
                guided_denoiser_adapter,
            )
            setattr(guided_model_fn, "predict_denoised", denoised_model_fn)
            setattr(
                guided_model_fn,
                "denoising_contract",
                self.systems.denoising.guided_epsilon_contract_metadata,
            )
        setattr(
            guided_model_fn,
            "model_prediction_contract",
            self.systems.denoising.contract_metadata,
        )

        def _base_sampling_operation(attempt_latents: torch.Tensor) -> Any:
            return performance.run(
                "base_sampling",
                lambda: self.systems.sampling.sample(
                    raw_model_fn=raw_model_fn,
                    guided_model_fn=guided_model_fn,
                    latents=attempt_latents,
                    schedule=schedule,
                    conditioning=conditioning,
                    request=request,
                    state=self.state,
                ),
                operation_name="sampling.sample",
            )

        base_sampling_recovery = self._sampling_recovery_contract(
            stage="sampling",
            source_latents=latents,
            request=request,
            operation_builder=_base_sampling_operation,
        )
        try:
            sample_output = diagnostics.run_stage(
                session,
                "sampling",
                "sample",
                lambda: self.memory_manager.run_stage(
                    stage="sampling",
                    required=(
                        {"unet", "vae"}
                        if base_preview_policy.requires_vae
                        else {"unet"}
                    ),
                    optional=(
                        {"vae"}
                        if base_preview_policy.optional_vae
                        else set()
                    ),
                    preview_requires_vae=base_preview_policy.suspend_on_pressure,
                    operation=lambda: _base_sampling_operation(latents),
                    request=request,
                    recovery_contract=base_sampling_recovery,
                ),
            )
        except Exception as exc:
            if outpaint_enabled and not extract_outpaint_failure_stage(exc):
                raise RuntimeError(format_outpaint_failure(
                    "outpaint_sampling", f"Masked diffusion sampling failed: {exc}"
                )) from exc
            raise
        finally:
            state_extra_for_hook = getattr(self.state, "extra", None) if self.state is not None else None
            if outpaint_enabled and isinstance(state_extra_for_hook, dict):
                state_extra_for_hook.pop("sampling_latent_step_hook", None)
        diagnostics.update_sampler(session, sample_output)
        provider_execution_after_base = get_execution_evidence()
        provider_execution_after_hires = dict(provider_execution_after_base)

        ctx.raw_model_fn = raw_model_fn
        ctx.guided_model_fn = guided_model_fn
        ctx.sample_output = sample_output
        ctx.provider_execution_after_base = provider_execution_after_base
        ctx.provider_execution_after_hires = provider_execution_after_hires
