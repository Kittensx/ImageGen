# kes_sampler.py

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from image_gen.systems.guidance import (
    GuidanceSemantics,
    combine_guidance_outputs,
    prompt_cfg_payload_from_request,
    requested_cfg_scale_for_step,
)
from modules.ss_registry.samplers.kes_sampler.simple_kes_sampler.cfg_strategy import CFGStrategy
from modules.ss_registry.samplers.kes_sampler.simple_kes_sampler.effective_guidance import (
    EffectiveGuidanceController,
    EffectiveGuidanceProfile,
)
from modules.sampler_state import SamplerState
from modules.ss_registry.samplers.sampler_config_loader import prepare_sampler_config

from modules.pipeline.conditioning_utils import resolve_step_conditioning
from modules.pipeline.regional_conditioning import get_regional_conditioning_resolver
from modules.pipeline.sampler_trace_mixin import SamplerTraceMixin
from modules.txt2img.seed_utils import create_torch_generator, offset_seed

from modules.contracts import SamplerCapabilities, SamplerOutput


class KESSampler(SamplerTraceMixin):
    """
    Pipeline-facing sampler adapter.

    Expected call shape:
        sample(
            raw_model_fn,
            latents,
            schedule,
            conditioning,
            request,
            state=None,
        ) -> SamplerOutput

    Notes:
    - Consumes request/schedule/conditioning/state as inputs.
    - Returns final latents.
    - Returns only sampler-owned metadata in `extra`.
    """

    SAMPLER_NAME = "kes_sampler"
    SAMPLER_CAPABILITIES = SamplerCapabilities(
        sampler_name=SAMPLER_NAME,
        guidance_owner="sampler",
        uses_raw_model_fn=True,
        uses_guided_model_fn=False,
        supports_step_expansion=True,
        supports_tail_metadata=True,
        requires_requested_step_schedule=False,
        strict_validation=True,
        forced_pipeline_mode="extended_steps",
    )
    SAMPLER_SCHEDULE_CAPABILITIES = SAMPLER_CAPABILITIES

    def __init__(
        self,
        config_path: Optional[str] = None,
        preset_name: Optional[str] = None,
        sampler_state: Optional[SamplerState] = None,
        verbose: bool = False,
        **overrides: Any,
    ) -> None:
        self.verbose = verbose
        self.sampler_state = sampler_state or SamplerState()
        self._ensure_state_shape(self.sampler_state)

        self.settings = prepare_sampler_config(
            sampler_name="kes",
            sampler_state=self.sampler_state,
            config_path=config_path,
            preset_name=preset_name,
            overrides=overrides,
        )

    def _materialize_sigmas(self, schedule, latents: torch.Tensor) -> torch.Tensor:
        return super()._materialize_sigmas(schedule, latents)

    def _materialize_timesteps(
        self,
        schedule,
        sigmas: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        del sigmas
        return super()._materialize_timesteps(schedule, latents)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        raw_model_fn,        
        latents: torch.Tensor,
        schedule,
        conditioning,
        request,
        state: Optional[Any] = None,
    ) -> SamplerOutput:
        """
        Args:
            model_fn:
                Callable used like:
                    model_fn(x, sigma, cond)
                and expected to return noise prediction.

            latents:
                Starting latent tensor.

            schedule:
                Object with `sigmas`.

            conditioning:
                Object with `cond` and `uncond`.

            request:
                Object with at least:
                    steps
                    cfg_scale
                and optionally:
                    batch_size
                    shape
                    width / height
                    seed
                    positive_prompt / negative_prompt

            state:
                Optional shared pipeline state.

        Returns:
            SamplerOutput(latents=..., extra={sampler metadata only})
        """
        progress = None
        if state is not None:
            progress = getattr(state, "extra", {}).get("progress_reporter")
    
        if latents is None:
            raise ValueError("KESSampler.sample requires `latents`.")

        sigmas = self._materialize_sigmas(schedule, latents)
        timesteps = self._materialize_timesteps(schedule, sigmas, latents)
        if sigmas.numel() < 2:
            raise ValueError("KESSampler.sample requires at least 2 sigma values.")

        requested_steps, effective_steps = self._resolve_effective_steps(
            request=request,
            schedule=schedule,
            sigmas=sigmas,
        )

        #
        x = latents
        cond = getattr(conditioning, "cond", None)
        uncond = getattr(conditioning, "uncond", None)

        if cond is None or uncond is None:
            raise ValueError("conditioning must provide both `cond` and `uncond`.")

        resolver = None
        conditioning_extra = getattr(conditioning, "extra", None)
        if isinstance(conditioning_extra, dict):
            resolver = conditioning_extra.get("resolver")

        cfg_scale = float(getattr(request, "cfg_scale", 1.0))
        image_seeds = list(getattr(request, "resolved_seeds", []) or [])
        if len(image_seeds) != int(x.shape[0]):
            base_seed = int(getattr(request, "seed", 0) or 0)
            image_seeds = [offset_seed(base_seed, index) for index in range(int(x.shape[0]))]
        noise_generators = [
            create_torch_generator(offset_seed(seed, 1), device=x.device)
            for seed in image_seeds
        ]

        def next_batch_noise(reference: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    torch.randn(
                        (1, *reference.shape[1:]),
                        generator=generator,
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                    for generator in noise_generators
                ],
                dim=0,
            )
        #

        # Sync runtime data with sampler_state.
        self._prepare_runtime_state(
            x=x,
            sigmas=sigmas,
            request=request,
            steps=effective_steps,
            cfg_scale=cfg_scale,
            shared_state=state,
        )

        # Create strategy only after runtime state is prepared.
        strategy = CFGStrategy(
            shared_state=state,
            sampler_state=self.sampler_state,
            verbose=self.verbose,
        )

        # Resolve effective settings actually used by the sampler.
        sampler_type = str(self._setting("sampler_type", "euler")).lower()
        eta = float(self._setting("eta", 0.0))
        add_noise = bool(self._setting("add_noise", eta > 0.0))
        initial_noise_strength = float(self._setting("initial_noise_strength", 0.0))
        legacy_clamp_guidance = bool(
            self._setting("legacy_clamp_guidance", self._setting("rescale_cfg", False))
        )
        legacy_guidance_multiplier = float(
            self._setting(
                "legacy_guidance_multiplier",
                self._setting("rescale_cfg_factor", 1.0),
            )
        )
        canonical_cfg_rescale = float(getattr(request, "cfg_rescale", 0.0) or 0.0)
        noise_schedule_scaling = self._setting("noise_schedule_scaling", "none")
        eta_schedule_mode = self._setting("eta_schedule_mode", "none")
        use_adaptive_eta = bool(self._setting("use_adaptive_eta", False))
        clamp_range = self._normalize_clamp_range(self._setting("clamp_range", [-1.0, 1.0]))
        guidance_owner = "sampler"
        guidance_profile = EffectiveGuidanceProfile.from_settings(
            requested_cfg_scale=cfg_scale,
            get_setting=self._setting,
            sampler_local_rescale_cfg=False,
            sampler_local_rescale_factor=1.0,
        )
        guidance_controller = EffectiveGuidanceController(guidance_profile)
        effective_cfg_scale = cfg_scale * legacy_guidance_multiplier
        sigma_max = sigmas[0] if sigmas.numel() else torch.tensor(0.0, device=x.device, dtype=x.dtype)
        sigma_min = sigmas[-1] if sigmas.numel() else torch.tensor(0.0, device=x.device, dtype=x.dtype)
        guidance_semantics = GuidanceSemantics(
            guidance_owner=guidance_owner,
            guidance_mode=guidance_profile.cfg_guidance_mode,
            canonical_cfg_rescale=canonical_cfg_rescale,
            legacy_clamp_guidance=legacy_clamp_guidance,
            legacy_guidance_multiplier=legacy_guidance_multiplier,
            legacy_clamp_range=tuple(clamp_range),
            guidance_path="shared_guidance_helper",
        )

        # Track what actually happened.
        metadata: Dict[str, Any] = {
            "sampler_name": self.SAMPLER_NAME,
            "sampler_type_used": sampler_type,
            "sigma_transitions": int(sigmas.numel() - 1),
            "requested_steps": requested_steps,
            "effective_steps": effective_steps,
            "scheduler_step_override_applied": effective_steps != requested_steps,
            "stepwise_conditioning_used": resolver is not None,
            "model_prediction_type": "epsilon",
            "integration_prediction_type": "epsilon_to_denoised",
            "guidance_owner": guidance_owner,
            "guidance_path": "shared_guidance_helper",
            "guidance_math_version": guidance_semantics.guidance_math_version,
            "guidance_mode": guidance_profile.cfg_guidance_mode,
            "requested_cfg_scale": cfg_scale,
            "effective_cfg_scale": effective_cfg_scale,
            "prompt_cfg_schedule": dict(prompt_cfg_payload_from_request(request)),
            "cfg_rescale": canonical_cfg_rescale,
            "cfg_rescale_applied": bool(canonical_cfg_rescale > 0.0),
            "legacy_clamp_guidance": legacy_clamp_guidance,
            "legacy_guidance_multiplier": legacy_guidance_multiplier,
            "sampler_local_rescale_cfg": legacy_clamp_guidance,
            "cfg_guidance_mode": guidance_profile.cfg_guidance_mode,
            "cfg_curve_type": guidance_profile.cfg_curve_type,
            "cfg_effective_guidance_summary": {
                **guidance_controller.summary(),
                "guidance_math_version": guidance_semantics.guidance_math_version,
                "guidance_mode": guidance_profile.cfg_guidance_mode,
                "cfg_rescale": canonical_cfg_rescale,
                "cfg_rescale_applied": bool(canonical_cfg_rescale > 0.0),
                "legacy_clamp_guidance": legacy_clamp_guidance,
                "legacy_guidance_multiplier": legacy_guidance_multiplier,
            },
        }

        schedule_extra = getattr(schedule, "extra", None)
        if isinstance(schedule_extra, dict):
            metadata["schedule_extra"] = dict(schedule_extra)
            capability_clamp = dict(schedule_extra.get("sampler_capability_clamp") or {})
            clamp_active = bool(capability_clamp.get("active", False))
            metadata["sampler_capability_clamp"] = capability_clamp
            metadata["step_expansion_clamped"] = bool(
                capability_clamp.get("step_expansion_clamped", False)
            )
            metadata["tail_metadata_clamped"] = bool(
                capability_clamp.get("tail_metadata_clamped", False)
            )
            if clamp_active:
                if effective_steps != requested_steps or int(sigmas.numel() - 1) != requested_steps:
                    raise ValueError(
                        "KES fixed-step compatibility clamp requires exactly the requested "
                        "number of sigma transitions."
                    )
                metadata["compatibility_mode_used"] = "fixed_steps_clamped"
            else:
                metadata["compatibility_mode_used"] = str(
                    schedule_extra.get("compatibility_mode") or "native"
                )
            if "schedule_mode" in schedule_extra:
                metadata["schedule_mode"] = schedule_extra["schedule_mode"]
                metadata["schedule_mode_used"] = schedule_extra["schedule_mode"]

            if "requested_steps" in schedule_extra:
                metadata["scheduler_reported_requested_steps"] = schedule_extra["requested_steps"]

            if "effective_steps" in schedule_extra:
                metadata["scheduler_reported_effective_steps"] = schedule_extra["effective_steps"]

            if "scheduler_step_override_applied" in schedule_extra:
                metadata["scheduler_step_override_applied"] = bool(
                    schedule_extra["scheduler_step_override_applied"]
                )

            if "tail_features_used" in schedule_extra:
                metadata["tail_features_used"] = schedule_extra["tail_features_used"]

            if "active_blend_methods" in schedule_extra:
                metadata["active_blend_methods"] = schedule_extra["active_blend_methods"]

            if "active_blend_weights" in schedule_extra:
                metadata["active_blend_weights"] = schedule_extra["active_blend_weights"]

            if "prepass_used" in schedule_extra:
                metadata["scheduler_prepass_used"] = bool(schedule_extra["prepass_used"])

            if "predicted_stop_step" in schedule_extra:
                metadata["scheduler_predicted_stop_step"] = schedule_extra["predicted_stop_step"]

            if "compatibility_mode" in schedule_extra:
                metadata["scheduler_compatibility_mode"] = schedule_extra["compatibility_mode"]

        # Initial noise injection.
        if initial_noise_strength > 0.0:
            x = x + next_batch_noise(x) * initial_noise_strength
            metadata["initial_noise_applied"] = True
            metadata["initial_noise_strength_used"] = initial_noise_strength
        else:
            metadata["initial_noise_applied"] = False

        # Main sampling loop.
        stochastic_noise_applied = False
        cfg_rescale_applied = False
        heun_used = False
        effective_cfg_per_step: list[dict[str, Any]] = []
        cfg_early_floor_applied_any = False
        guidance_shaping_active_any = False
        guidance_shaping_auto_applied_any = False
        prompt_cfg_payload = prompt_cfg_payload_from_request(request)
        regional_resolver = get_regional_conditioning_resolver(conditioning)
        regional_guidance_active_any = False

        for i in range(sigmas.numel() - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            timestep = timesteps[i]
            timestep_next = timesteps[i + 1]
            
            latent_before = x.detach()

            step_requested_cfg_scale, prompt_cfg_step = requested_cfg_scale_for_step(
                request,
                step_index=i,
                total_steps=effective_steps,
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
            cfg_early_floor_applied_any = cfg_early_floor_applied_any or bool(
                guidance_step.get("cfg_early_floor_applied")
            )
            guidance_shaping_active_any = guidance_shaping_active_any or bool(
                guidance_step.get("guidance_shaping_active")
            )
            guidance_shaping_auto_applied_any = guidance_shaping_auto_applied_any or bool(
                guidance_step.get("guidance_shaping_auto_applied")
            )
            effective_cfg_per_step.append(
                {
                    "step_index": int(i),
                    "sigma": float(sigma.item()) if hasattr(sigma, "item") else float(sigma),
                    "timestep": float(timestep.item()) if hasattr(timestep, "item") else float(timestep),
                    "requested_cfg_scale": float(step_requested_cfg_scale),
                    "ui_cfg_scale": float(cfg_scale),
                    "cfg_source": prompt_cfg_step.get("cfg_source", "ui"),
                    "prompt_cfg_applied": bool(prompt_cfg_step.get("prompt_cfg_applied", False)),
                    "prompt_cfg_progress_fraction": prompt_cfg_step.get("prompt_cfg_progress_fraction"),
                    "effective_cfg_scale": float(step_effective_cfg_scale * legacy_guidance_multiplier),
                    "effective_cfg_scale_pre_rescale": float(
                        guidance_step.get("effective_cfg_scale_pre_rescale", step_effective_cfg_scale)
                    ),
                    "cfg_guidance_mode": guidance_step.get("cfg_guidance_mode"),
                    "guidance_shaping_active": bool(guidance_step.get("guidance_shaping_active")),
                    "guidance_shaping_auto_applied": bool(
                        guidance_step.get("guidance_shaping_auto_applied")
                    ),
                    "cfg_early_floor_applied": bool(guidance_step.get("cfg_early_floor_applied")),
                    "progress_fraction": guidance_step.get("progress_fraction"),
                    "sigma_fraction": guidance_step.get("sigma_fraction"),
                    "cfg_rescale": canonical_cfg_rescale,
                    "cfg_rescale_applied": bool(canonical_cfg_rescale > 0.0),
                    "legacy_clamp_guidance": legacy_clamp_guidance,
                    "legacy_guidance_multiplier": legacy_guidance_multiplier,
                }
            )

            cond, uncond = resolve_step_conditioning(
                conditioning=conditioning,
                step_index=i,
                latents=x,
                state=state,
            )

            active_regions = (
                regional_resolver.resolve_regions(step_index=i, latents=x)
                if regional_resolver is not None
                else []
            )
            regional_guidance_active_any = regional_guidance_active_any or bool(active_regions)
            noise_uncond = raw_model_fn(x, sigma, timestep, uncond)
            if active_regions:
                regional_conditional = getattr(raw_model_fn, "predict_regional_conditional_noise", None)
                if not callable(regional_conditional):
                    raise TypeError("The denoising system does not provide native regional conditional noise.")
                noise_cond = regional_conditional(
                    x,
                    sigma,
                    timestep,
                    cond,
                    active_regions,
                    regional_resolver.overlap_policy,
                    uncond_noise=noise_uncond,
                )
            else:
                noise_cond = raw_model_fn(x, sigma, timestep, cond)
            guidance_delta = noise_cond - noise_uncond

            guidance_result = combine_guidance_outputs(
                noise_uncond,
                noise_cond,
                effective_cfg_scale=guidance_step.get("effective_cfg_scale_pre_rescale", step_effective_cfg_scale),
                semantics=guidance_semantics,
            )
            noise = guidance_result.guided
            cfg_rescale_applied = cfg_rescale_applied or bool(
                guidance_result.metadata.get("cfg_rescale_applied")
            )

            denoised = x - sigma * noise
            dt = sigma_next - sigma

            if sampler_type == "heun" and i < (sigmas.numel() - 2):
                # Predictor
                x_pred = denoised + (sigma_next * noise)

                # Corrector uses conditioning for the next step when available
                cond_2, uncond_2 = resolve_step_conditioning(
                    conditioning=conditioning,
                    step_index=i + 1,
                    latents=x_pred,
                    state=state,
                )

                active_regions_2 = (
                    regional_resolver.resolve_regions(step_index=i + 1, latents=x_pred)
                    if regional_resolver is not None
                    else []
                )
                noise_uncond_2 = raw_model_fn(x_pred, sigma_next, timestep_next, uncond_2)
                if active_regions_2:
                    regional_conditional = getattr(raw_model_fn, "predict_regional_conditional_noise", None)
                    if not callable(regional_conditional):
                        raise TypeError("The denoising system does not provide native regional conditional noise.")
                    noise_cond_2 = regional_conditional(
                        x_pred,
                        sigma_next,
                        timestep_next,
                        cond_2,
                        active_regions_2,
                        regional_resolver.overlap_policy,
                        uncond_noise=noise_uncond_2,
                    )
                else:
                    noise_cond_2 = raw_model_fn(x_pred, sigma_next, timestep_next, cond_2)
                guidance_delta_2 = noise_cond_2 - noise_uncond_2

                guidance_result_2 = combine_guidance_outputs(
                    noise_uncond_2,
                    noise_cond_2,
                    effective_cfg_scale=guidance_step.get("effective_cfg_scale_pre_rescale", step_effective_cfg_scale),
                    semantics=guidance_semantics,
                )
                noise_2 = guidance_result_2.guided

                x = x + 0.5 * (noise + noise_2) * dt
                heun_used = True
            else:
                # Euler
                x = denoised + sigma_next * noise

            if use_adaptive_eta:
                gamma = strategy.get_eta_noise_gamma_adaptive(
                    step=i,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    total_steps=effective_steps,
                    x=x,
                    denoised=denoised,
                )
            else:
                gamma = strategy.get_eta_noise_gamma(
                    step=i,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    total_steps=effective_steps,
                    mode=noise_schedule_scaling,
                )

            if add_noise and gamma > 0:
                stochastic_noise = next_batch_noise(x)
                x = x + stochastic_noise * gamma
                stochastic_noise_applied = True

            # Optional legacy clamp path retained for backward compatibility.
            if legacy_clamp_guidance:
                x = torch.clamp(x, clamp_range[0], clamp_range[1])
            x = self._apply_latent_step_hook(
                state,
                request=request,
                latent=x,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                timestep=timestep,
            )
            
            # Traceback
            integration_mode = "heun" if (sampler_type == "heun" and i < (sigmas.numel() - 2)) else "euler"
            preview_metadata = {
                "sampler_name": self.SAMPLER_NAME,
                "integration_mode": integration_mode,
                "used_resolver": resolver is not None,
                "regional_guidance_active": bool(active_regions),
                "regional_active_count": len(active_regions),
                "cfg_rescale": canonical_cfg_rescale,
                "cfg_rescale_applied": cfg_rescale_applied,
                "legacy_clamp_guidance": legacy_clamp_guidance,
                "legacy_guidance_multiplier": legacy_guidance_multiplier,
                "add_noise": add_noise,
                "gamma": float(gamma) if gamma is not None else None,
                "used_stochastic_noise": bool(add_noise and gamma > 0),
                "guidance_owner": guidance_owner,
                "guidance_path": "shared_guidance_helper",
                "guidance_math_version": guidance_semantics.guidance_math_version,
                "guidance_mode": guidance_profile.cfg_guidance_mode,
                "requested_cfg_scale": step_requested_cfg_scale,
                "ui_cfg_scale": cfg_scale,
                "cfg_source": prompt_cfg_step.get("cfg_source", "ui"),
                "prompt_cfg_applied": bool(prompt_cfg_step.get("prompt_cfg_applied", False)),
                "effective_cfg_scale": step_effective_cfg_scale * legacy_guidance_multiplier,
                "cfg_guidance_mode": guidance_step.get("cfg_guidance_mode"),
                "guidance_shaping_active": bool(guidance_step.get("guidance_shaping_active")),
                "guidance_shaping_auto_applied": bool(guidance_step.get("guidance_shaping_auto_applied")),
                "cfg_early_floor_applied": bool(guidance_step.get("cfg_early_floor_applied")),
                "progress_fraction": guidance_step.get("progress_fraction"),
                "sigma_fraction": guidance_step.get("sigma_fraction"),
                "timestep": float(timestep.item()) if hasattr(timestep, "item") else float(timestep),
                "predicted_x0_available": True,
            }
            self._trace_step(
                request,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                timestep=timestep,
                latent_before=latent_before,
                latent_after=x,
                noise_pred=noise_cond,
                guided_noise=noise,
                cfg_scale=step_requested_cfg_scale,
                requested_cfg_scale=step_requested_cfg_scale,
                effective_cfg_scale=step_effective_cfg_scale * legacy_guidance_multiplier,
                guidance_owner=guidance_owner,
                unconditional_output=noise_uncond,
                conditional_output=noise_cond,
                guidance_delta=guidance_delta,
                guided_output=noise,
                predicted_x0=denoised,
                extra=preview_metadata,
                predicted_x0_snapshot=denoised,
            )
            self._emit_live_preview(
                state,
                request=request,
                step_index=i,
                total_steps=effective_steps,
                latent=x,
                predicted_x0=denoised,
                sigma=sigma,
                model_timestep=timestep,
                metadata=preview_metadata,
            )
            if progress is not None:
                progress.update(1)
            # Keep state fresh for downstream tools/debugging.
            self._update_tensor_state(x=x, sigma=sigma, denoised=denoised)

        metadata["cfg_rescale_applied"] = cfg_rescale_applied
        metadata["cfg_rescale"] = canonical_cfg_rescale
        metadata["legacy_clamp_guidance"] = legacy_clamp_guidance
        metadata["legacy_guidance_multiplier"] = legacy_guidance_multiplier
        metadata["cfg_rescale_factor_used"] = legacy_guidance_multiplier
        if legacy_clamp_guidance:
            metadata["clamp_applied"] = True
            metadata["clamp_range_used"] = list(clamp_range)
        else:
            metadata["clamp_applied"] = False

        metadata["stochastic_noise_applied"] = stochastic_noise_applied
        metadata["gamma_strategy_used"] = "adaptive" if use_adaptive_eta else "standard"
        metadata["eta_schedule_mode_used"] = eta_schedule_mode

        if stochastic_noise_applied:
            metadata["eta_used"] = eta
            metadata["noise_schedule_scaling_used"] = noise_schedule_scaling

        if heun_used:
            metadata["integration_mode_used"] = "heun"
        else:
            metadata["integration_mode_used"] = "euler"

        metadata["cfg_guidance_mode"] = guidance_profile.cfg_guidance_mode
        metadata["cfg_curve_type"] = guidance_profile.cfg_curve_type
        metadata["cfg_effective_guidance_summary"] = {
            **guidance_controller.summary(),
            "guidance_math_version": guidance_semantics.guidance_math_version,
            "guidance_mode": guidance_profile.cfg_guidance_mode,
            "cfg_rescale": canonical_cfg_rescale,
            "cfg_rescale_applied": bool(canonical_cfg_rescale > 0.0),
            "legacy_clamp_guidance": legacy_clamp_guidance,
            "legacy_guidance_multiplier": legacy_guidance_multiplier,
        }
        metadata["prompt_cfg_schedule"] = dict(prompt_cfg_payload)
        metadata["prompt_cfg_applied"] = any(
            bool(item.get("prompt_cfg_applied")) for item in effective_cfg_per_step
        )
        metadata["cfg_effective_per_step"] = effective_cfg_per_step
        metadata["cfg_step_series"] = {
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
                    "cfg_rescale": item.get("cfg_rescale", canonical_cfg_rescale),
                    "cfg_rescale_applied": bool(item.get("cfg_rescale_applied", False)),
                    "override_source": (
                        item.get("cfg_source", "superhybrid_prompt")
                        if bool(item.get("prompt_cfg_applied", False))
                        else item.get("override_source", "base_request")
                    ),
                    "ui_cfg_scale": item.get("ui_cfg_scale", cfg_scale),
                    "prompt_cfg_applied": bool(item.get("prompt_cfg_applied", False)),
                    "transition_id": item.get("transition_id"),
                }
                for index, item in enumerate(effective_cfg_per_step)
            ],
        }
        metadata["cfg_effective_range"] = {
            "min": min((float(item["effective_cfg_scale"]) for item in effective_cfg_per_step), default=float(effective_cfg_scale)),
            "max": max((float(item["effective_cfg_scale"]) for item in effective_cfg_per_step), default=float(effective_cfg_scale)),
        }
        metadata["guidance_shaping_active"] = guidance_shaping_active_any
        metadata["guidance_shaping_auto_applied"] = guidance_shaping_auto_applied_any
        metadata["cfg_early_floor_applied"] = cfg_early_floor_applied_any
        metadata["regional_guidance_used"] = bool(regional_guidance_active_any)
        metadata["regional_backend"] = (
            "image_gen_model_output" if regional_guidance_active_any else "none"
        )
        if regional_resolver is not None:
            regional_runtime = regional_resolver.runtime_snapshot()
            metadata["regional_runtime"] = regional_runtime
            if isinstance(getattr(request, "diagnostics", None), dict):
                request.diagnostics["regional_runtime"] = regional_runtime
                passes = request.diagnostics.setdefault("regional_runtime_passes", {})
                passes[str(regional_runtime.get("pass") or "base")] = regional_runtime

        stopping_index = None
        if isinstance(schedule_extra, dict):
            stopping_index = schedule_extra.get("predicted_stop_step")

        self._trace_sampler_summary(
            request,
            requested_steps=requested_steps,
            effective_steps=effective_steps,
            stopping_index=stopping_index,
        )
        

        return SamplerOutput(
            latents=x,
            extra=metadata,
            
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
            
    def _ensure_state_shape(self, sampler_state: SamplerState) -> None:
        """
        Smooth over naming drift while you refactor.
        Supports either:
          - sampler_state.tensor_data
          - sampler_state.sigdata
        """
        if hasattr(sampler_state, "tensor_data") and not hasattr(sampler_state, "sigdata"):
            sampler_state.sigdata = sampler_state.tensor_data

        if hasattr(sampler_state, "sigdata") and not hasattr(sampler_state, "tensor_data"):
            sampler_state.tensor_data = sampler_state.sigdata

    def _setting(self, key: str, default: Any = None) -> Any:
        """
        Read the effective sampler setting.
        Prefers sampler_state.cfg when present, then resolved settings dict.
        """
        cfg_obj = getattr(self.sampler_state, "cfg", None)
        if cfg_obj is not None and hasattr(cfg_obj, key):
            value = getattr(cfg_obj, key)
            if value is not None:
                return value
        return self.settings.get(key, default)

    

    def _prepare_runtime_state(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        request: Any,
        steps: int,
        cfg_scale: float,
        shared_state: Optional[Any],
    ) -> None:
        ss = self.sampler_state

        gen = getattr(ss, "gen", None)
        if gen is not None:
            if hasattr(gen, "steps"):
                gen.steps = steps
            if hasattr(gen, "cfg_scale"):
                gen.cfg_scale = cfg_scale
            if hasattr(gen, "batch_size"):
                gen.batch_size = getattr(request, "batch_size", x.shape[0] if x.ndim > 0 else 1)
            if hasattr(gen, "shape"):
                gen.shape = getattr(request, "shape", tuple(x.shape))
            if hasattr(gen, "device"):
                gen.device = str(x.device)

        tensor_data = getattr(ss, "tensor_data", None)
        if tensor_data is not None:
            if hasattr(tensor_data, "sigmas"):
                tensor_data.sigmas = sigmas
            if hasattr(tensor_data, "sample"):
                tensor_data.sample = x
            if hasattr(tensor_data, "model"):
                tensor_data.model = None

        if shared_state is not None:
            setattr(ss, "shared_state", shared_state)

    def _update_tensor_state(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoised: torch.Tensor,
    ) -> None:
        tensor_data = getattr(self.sampler_state, "tensor_data", None)
        if tensor_data is None:
            return

        if hasattr(tensor_data, "sample"):
            tensor_data.sample = x
        if hasattr(tensor_data, "sigma"):
            tensor_data.sigma = sigma
        if hasattr(tensor_data, "denoised"):
            tensor_data.denoised = denoised

    def _normalize_clamp_range(self, value: Any) -> tuple[float, float]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
        return -1.0, 1.0
        
SAMPLER_CLASS = KESSampler
