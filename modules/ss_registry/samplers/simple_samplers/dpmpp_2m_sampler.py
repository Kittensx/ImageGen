from __future__ import annotations

from typing import Any, Optional

import torch

from image_gen.systems.guidance import (
    EffectiveGuidanceController,
    EffectiveGuidanceProfile,
    prompt_cfg_payload_from_request,
    requested_cfg_scale_for_step,
)

from modules.pipeline.conditioning_utils import (
    call_with_optional_model_conditioning,
    resolve_step_composable_conditioning,
    resolve_step_conditioning,
    resolve_step_model_conditioning,
)
from modules.pipeline.regional_conditioning import get_regional_conditioning_resolver
from modules.pipeline.sampler_trace_mixin import SamplerTraceMixin

from modules.contracts import SamplerCapabilities, SamplerOutput


meta = {
    "name": "dpmpp_2m",
    "label": "DPM++ 2M",
    "description": "Pipeline-facing DPM++ 2M-style deterministic sampler for standard fixed-step schedules.",
    "config_key": "shared",
}

required_args = {
    "guided_model_fn": "callable",
    "latents": "torch.Tensor",
    "schedule": "SchedulerOutput-like object with sigmas",
    "conditioning": "ConditioningOutput-like object with cond/uncond",
    "request": "request object with steps/cfg_scale",
}

optional_args = {
    "state": "shared pipeline state",
}


class DPMPlusPlus2MSampler(SamplerTraceMixin):
    """Deterministic DPM++ 2M sampler with optional Phase 8B diagnostics."""

    SAMPLER_NAME = "dpmpp_2m"

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

    @staticmethod
    def _resolve_solver_dtype(request: Any, latent_dtype: torch.dtype) -> torch.dtype:
        sampler_kwargs = dict(getattr(request, "sampler_kwargs", {}) or {})
        selected = str(sampler_kwargs.get("solver_dtype", "float32")).strip().lower()
        aliases = {
            "latent": latent_dtype,
            "model": latent_dtype,
            "input": latent_dtype,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if selected not in aliases:
            raise ValueError(
                "DPM++ 2M sampler_kwargs.solver_dtype must be one of: "
                "latent, float16, or float32."
            )
        return aliases[selected]

    @staticmethod
    def _resolve_history_policy(request: Any) -> str:
        sampler_kwargs = dict(getattr(request, "sampler_kwargs", {}) or {})
        selected = str(sampler_kwargs.get("history_policy", "model_timestep_guarded")).strip().lower()
        aliases = {
            "default": "model_timestep_guarded",
            "dpmpp_2m": "model_timestep_guarded",
            "guarded": "model_timestep_guarded",
            "timestep_guarded": "model_timestep_guarded",
            "model-timestep-guarded": "model_timestep_guarded",
            "model_timestep_guarded": "model_timestep_guarded",
            "second_order": "multistep",
            "multistep": "multistep",
            "first_order": "first_order_only",
            "first-order-only": "first_order_only",
            "no_history": "first_order_only",
            "first_order_only": "first_order_only",
        }
        if selected not in aliases:
            raise ValueError(
                "DPM++ 2M sampler_kwargs.history_policy must be one of: "
                "model_timestep_guarded, multistep, or first_order_only."
            )
        return aliases[selected]

    @staticmethod
    def _resolve_prediction_mode(request: Any, history_policy: str) -> str:
        sampler_kwargs = dict(getattr(request, "sampler_kwargs", {}) or {})
        selected = str(
            sampler_kwargs.get("prediction_mode", "canonical_predicted_x0")
        ).strip().lower()
        aliases = {
            "canonical": "canonical_predicted_x0",
            "canonical_x0": "canonical_predicted_x0",
            "canonical_predicted_x0": "canonical_predicted_x0",
            "legacy": "legacy_sampler_local_epsilon",
            "legacy_epsilon": "legacy_sampler_local_epsilon",
            "legacy_sampler_local_epsilon": "legacy_sampler_local_epsilon",
        }
        if selected not in aliases:
            raise ValueError(
                "DPM++ 2M sampler_kwargs.prediction_mode must be one of: "
                "canonical_predicted_x0 or legacy_sampler_local_epsilon."
            )
        resolved = aliases[selected]
        if resolved == "legacy_sampler_local_epsilon":
            validation_mode = bool(sampler_kwargs.get("phase08d_validation_mode", False))
            if not validation_mode or history_policy != "first_order_only":
                raise ValueError(
                    "legacy_sampler_local_epsilon is a Phase 8D diagnostic-only mode and "
                    "requires phase08d_validation_mode=true with "
                    "history_policy=first_order_only."
                )
        return resolved

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "item"):
                value = value.item()
            result = float(value)
            return result if torch.isfinite(torch.tensor(result)).item() else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tensor_trace_stats(value: torch.Tensor) -> dict[str, Any]:
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
        finite_mask = torch.isfinite(tensor)
        finite_values = tensor[finite_mask]
        all_finite = bool(finite_mask.all().item())
        if finite_values.numel() == 0:
            minimum = maximum = mean = std = norm = None
        else:
            minimum = float(finite_values.min().item())
            maximum = float(finite_values.max().item())
            mean = float(finite_values.mean().item())
            std = float(finite_values.std(unbiased=False).item())
            norm = float(torch.linalg.vector_norm(finite_values).item())
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "norm": norm,
            "all_finite": all_finite,
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "pos_inf_count": int(torch.isposinf(tensor).sum().item()),
            "neg_inf_count": int(torch.isneginf(tensor).sum().item()),
            "finite_count": int(finite_values.numel()),
            "element_count": int(tensor.numel()),
        }

    @classmethod
    def _x0_comparison(
        cls,
        current: torch.Tensor,
        previous: Optional[torch.Tensor],
    ) -> dict[str, float | None]:
        if previous is None:
            return {
                "difference_norm": None,
                "difference_mean_absolute": None,
                "cosine_similarity": None,
            }
        current_flat = current.detach().to(dtype=torch.float32).reshape(-1)
        previous_flat = previous.detach().to(
            device=current.device, dtype=torch.float32
        ).reshape(-1)
        difference = current_flat - previous_flat
        current_norm = torch.linalg.vector_norm(current_flat)
        previous_norm = torch.linalg.vector_norm(previous_flat)
        denominator = current_norm * previous_norm
        cosine = None
        if bool(torch.isfinite(denominator).item()) and float(denominator.item()) > 0.0:
            value = torch.dot(current_flat, previous_flat) / denominator
            if bool(torch.isfinite(value).item()):
                cosine = float(value.item())
        return {
            "difference_norm": cls._scalar(torch.linalg.vector_norm(difference)),
            "difference_mean_absolute": cls._scalar(difference.abs().mean()),
            "cosine_similarity": cosine,
        }

    @classmethod
    def _reference_comparison(
        cls,
        current: torch.Tensor,
        reference: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if reference is None:
            return {
                "available": False,
                "difference_norm": None,
                "difference_mean_absolute": None,
                "cosine_similarity": None,
            }
        compared = cls._x0_comparison(current, reference)
        return {"available": True, **compared}

    def _coefficient_trace(
        self,
        *,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        previous_t: Optional[torch.Tensor],
    ) -> dict[str, float | None]:
        if float(sigma_next.detach().cpu()) <= 0.0:
            return {"t": None, "t_next": None, "h": None, "h_last": None, "r": None}
        t = self._sigma_to_t(sigma)
        t_next = self._sigma_to_t(sigma_next)
        h = t_next - t
        h_last = None
        r = None
        if previous_t is not None:
            h_last_tensor = t - previous_t.to(device=t.device, dtype=t.dtype)
            r_tensor = h_last_tensor / torch.clamp(h, min=torch.finfo(t.dtype).eps)
            r_tensor = torch.clamp(r_tensor, min=torch.finfo(t.dtype).eps)
            h_last = self._scalar(h_last_tensor)
            r = self._scalar(r_tensor)
        return {
            "t": self._scalar(t),
            "t_next": self._scalar(t_next),
            "h": self._scalar(h),
            "h_last": h_last,
            "r": r,
        }

    @staticmethod
    def _history_trace(
        *,
        old_denoised: Optional[torch.Tensor],
        previous_t: Optional[torch.Tensor],
        sigma_next: torch.Tensor,
        history_policy: str = "model_timestep_guarded",
        current_model_timestep: float | None = None,
        previous_model_timestep: float | None = None,
        history_blocked_until_distinct: bool = False,
    ) -> dict[str, Any]:
        if float(sigma_next.detach().cpu()) <= 0.0:
            return {
                "update_order": "terminal_denoised",
                "history_accepted": False,
                "history_rejection_reason": "terminal_sigma",
                "history_reset_required": False,
                "history_guard_rejected": False,
            }
        if history_policy == "first_order_only":
            return {
                "update_order": "first_order",
                "history_accepted": False,
                "history_rejection_reason": "history_policy_first_order_only",
                "history_reset_required": False,
                "history_guard_rejected": False,
            }
        if history_policy == "model_timestep_guarded":
            if (
                current_model_timestep is not None
                and previous_model_timestep is not None
            ):
                timestep_delta = current_model_timestep - previous_model_timestep
                if timestep_delta >= -1e-6:
                    reason = (
                        "repeated_model_timestep"
                        if abs(timestep_delta) <= 1e-6
                        else "non_decreasing_model_timestep"
                    )
                    return {
                        "update_order": "first_order",
                        "history_accepted": False,
                        "history_rejection_reason": reason,
                        "history_reset_required": True,
                        "history_guard_rejected": True,
                    }
            if history_blocked_until_distinct:
                return {
                    "update_order": "first_order",
                    "history_accepted": False,
                    "history_rejection_reason": "history_reset_after_timestep_plateau",
                    "history_reset_required": False,
                    "history_guard_rejected": True,
                }
        if old_denoised is None or previous_t is None:
            return {
                "update_order": "first_order",
                "history_accepted": False,
                "history_rejection_reason": "history_unavailable",
                "history_reset_required": False,
                "history_guard_rejected": False,
            }
        return {
            "update_order": "second_order",
            "history_accepted": True,
            "history_rejection_reason": None,
            "history_reset_required": False,
            "history_guard_rejected": False,
        }

    @staticmethod
    def _consume_guidance_trace(guided_model_fn: Any) -> dict[str, Any] | None:
        consumer = getattr(guided_model_fn, "consume_guidance_trace", None)
        if not callable(consumer):
            return None
        try:
            value = consumer()
        except Exception:
            return None
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _denoising_contract(guided_model_fn: Any) -> dict[str, Any]:
        provider = getattr(guided_model_fn, "denoising_contract", None)
        if not callable(provider):
            return {}
        try:
            value = provider()
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _actual_model_input(model_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_value = torch.as_tensor(
            sigma, device=model_x.device, dtype=model_x.dtype
        ).flatten()
        if sigma_value.numel() == 1 and model_x.shape[0] > 1:
            sigma_value = sigma_value.expand(model_x.shape[0])
        while sigma_value.ndim < model_x.ndim:
            sigma_value = sigma_value.unsqueeze(-1)
        return model_x / torch.sqrt(sigma_value.square() + 1.0)

    def sample(
        self,
        raw_model_fn,
        guided_model_fn,
        latents: torch.Tensor,
        schedule,
        conditioning,
        request,
        state: Optional[Any] = None,
    ) -> SamplerOutput:
        del raw_model_fn
        if latents is None:
            raise ValueError("DPMPlusPlus2MSampler.sample requires `latents`.")

        model_input_dtype = latents.dtype
        solver_dtype = self._resolve_solver_dtype(request, model_input_dtype)
        history_policy = self._resolve_history_policy(request)
        prediction_mode = self._resolve_prediction_mode(request, history_policy)
        x = latents.to(dtype=solver_dtype)
        sigmas = self._materialize_sigmas(schedule, x)
        timesteps = self._materialize_timesteps(schedule, x)
        if sigmas.numel() < 2:
            raise ValueError("DPMPlusPlus2MSampler.sample requires at least 2 sigma values.")

        cfg_scale = float(getattr(request, "cfg_scale", 1.0))
        history_enabled = history_policy != "first_order_only"
        old_denoised = None
        previous_t = None
        previous_predicted_x0 = None
        previous_model_timestep = None
        first_order_update_count = 0
        second_order_update_count = 0
        terminal_update_count = 0
        history_state_read_count = 0
        history_reset_count = 0
        history_guard_rejection_count = 0
        history_blocked_until_distinct = False
        euler_reference_latents = dict(
            getattr(request, "_phase08d_euler_reference_latents", {}) or {}
        )
        stepwise_conditioning_used = False
        cfg_effective_per_step: list[dict[str, Any]] = []
        regional_resolver = get_regional_conditioning_resolver(conditioning)
        regional_guidance_active_any = False
        trace_recorder = self._get_trace_recorder(request)
        trace_enabled = bool(
            trace_recorder is not None and getattr(trace_recorder, "enabled", False)
        )
        progress = (
            getattr(state, "extra", {}).get("progress_reporter")
            if state is not None
            else None
        )

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
            model_x = x.to(dtype=model_input_dtype)
            composable = resolve_step_composable_conditioning(
                conditioning,
                i,
                latents=model_x,
                request=request,
            )
            if composable is None:
                cond, uncond = resolve_step_conditioning(
                    conditioning=conditioning,
                    step_index=i,
                    latents=model_x,
                    state=state,
                )
                model_conditioning = resolve_step_model_conditioning(
                    conditioning=conditioning,
                    step_index=i,
                    latents=model_x,
                    request=request,
                )
            else:
                cond = composable["branches"][0]
                uncond = composable["uncond"]
                model_conditioning = None

            if i == 0:
                conditioning_extra = getattr(conditioning, "extra", {}) or {}
                resolver = conditioning_extra.get("resolver")
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
            if composable is not None and active_regions:
                raise ValueError(
                    "PPSR-09E composable AND and REGION guidance are not qualified together."
                )
            regional_guidance_active_any = regional_guidance_active_any or bool(active_regions)
            denoising_contract = self._denoising_contract(guided_model_fn)
            if prediction_mode == "canonical_predicted_x0":
                if composable is not None:
                    predict_denoised = getattr(
                        guided_model_fn, "predict_composable_denoised", None
                    )
                elif active_regions:
                    predict_denoised = getattr(guided_model_fn, "predict_regional_denoised", None)
                else:
                    predict_denoised = getattr(guided_model_fn, "predict_denoised", None)
                if not callable(predict_denoised):
                    raise TypeError(
                        "DPM++ 2M requires the Phase 8C canonical predict_denoised callback."
                    )
                if composable is not None:
                    denoised = predict_denoised(
                        x,
                        sigma,
                        timestep,
                        composable["branches"],
                        composable["weights"],
                        composable["uncond"],
                        step_effective_cfg_scale,
                        composable["branch_model_conditioning"],
                        composable["uncond_model_conditioning"],
                    )
                elif active_regions:
                    denoised = call_with_optional_model_conditioning(
                        predict_denoised,
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
                    denoised = call_with_optional_model_conditioning(
                        predict_denoised,
                        x,
                        sigma,
                        timestep,
                        cond,
                        uncond,
                        step_effective_cfg_scale,
                        model_conditioning=model_conditioning,
                    )
                if denoised.dtype != torch.float32:
                    raise TypeError(
                        "The canonical denoiser must return predicted x0 in torch.float32."
                    )
            else:
                if composable is not None:
                    composable_guided = getattr(
                        guided_model_fn, "predict_composable_guided_noise", None
                    )
                    if not callable(composable_guided):
                        raise TypeError(
                            "DPM++ 2M requires predict_composable_guided_noise for PPSR-09E AND."
                        )
                    guided_epsilon = composable_guided(
                        x,
                        sigma,
                        timestep,
                        composable["branches"],
                        composable["weights"],
                        composable["uncond"],
                        step_effective_cfg_scale,
                        composable["branch_model_conditioning"],
                        composable["uncond_model_conditioning"],
                    )
                elif active_regions:
                    regional_guided = getattr(guided_model_fn, "predict_regional_guided_noise", None)
                    if not callable(regional_guided):
                        raise TypeError("The denoising system does not provide native regional guidance.")
                    guided_epsilon = call_with_optional_model_conditioning(
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
                    guided_epsilon = call_with_optional_model_conditioning(
                        guided_model_fn,
                        x,
                        sigma,
                        timestep,
                        cond,
                        uncond,
                        step_effective_cfg_scale,
                        model_conditioning=model_conditioning,
                    )
                sigma_value = torch.as_tensor(
                    sigma, device=x.device, dtype=torch.float32
                )
                denoised = x.to(dtype=torch.float32) - sigma_value * guided_epsilon.to(
                    device=x.device, dtype=torch.float32
                )
                if not torch.isfinite(denoised).all():
                    raise ValueError(
                        "Legacy Phase 8D sampler-local predicted x0 contains non-finite values."
                    )
            guidance_trace = self._consume_guidance_trace(guided_model_fn)
            preview_predicted_x0 = denoised.detach()
            denoised = denoised.to(device=x.device, dtype=solver_dtype)

            current_model_timestep = self._scalar(timestep)
            history = self._history_trace(
                old_denoised=old_denoised,
                previous_t=previous_t,
                sigma_next=sigma_next,
                history_policy=history_policy,
                current_model_timestep=current_model_timestep,
                previous_model_timestep=previous_model_timestep,
                history_blocked_until_distinct=history_blocked_until_distinct,
            )
            if history["history_reset_required"]:
                history_reset_count += 1
            if history["history_guard_rejected"]:
                history_guard_rejection_count += 1
            effective_old_denoised = (
                old_denoised if history["history_accepted"] else None
            )
            effective_previous_t = (
                previous_t if history["history_accepted"] else None
            )
            if history["update_order"] == "first_order":
                first_order_update_count += 1
            elif history["update_order"] == "second_order":
                second_order_update_count += 1
                history_state_read_count += 1
            else:
                terminal_update_count += 1
            repeated_timestep = (
                previous_model_timestep is not None
                and current_model_timestep is not None
                and abs(current_model_timestep - previous_model_timestep) <= 1e-6
            )

            trace_extra: dict[str, Any] = {
                "integration_mode": self.SAMPLER_NAME,
                "regional_guidance_active": bool(active_regions),
                "regional_active_count": len(active_regions),
                "composition_mode": (
                    str(composable.get("mode") or "") if composable is not None else "standard"
                ),
                "composition_branch_count": (
                    len(composable.get("branches") or ()) if composable is not None else 1
                ),
                "logical_model_evaluations": (
                    len(composable.get("branches") or ()) + 1
                    if composable is not None
                    else 2
                ),
                "history_policy": history_policy,
                "prediction_mode": prediction_mode,
                "used_old_denoised": bool(history["history_accepted"]),
                "history_applied": bool(history["history_accepted"]),
                "model_input_dtype": str(model_input_dtype),
                "solver_dtype": str(solver_dtype),
                "timestep": current_model_timestep,
                "previous_timestep": previous_model_timestep,
                "repeated_model_timestep": repeated_timestep,
                "prediction_type": denoising_contract.get("prediction_type", "unknown"),
                "prediction_conversion": (
                    denoising_contract.get("prediction_conversion")
                    if prediction_mode == "canonical_predicted_x0"
                    else "legacy sampler-local x0 = solver_sample - sigma * guided_epsilon"
                ),
                "model_input_preconditioning": denoising_contract.get(
                    "model_input_preconditioning"
                ),
                "guidance_owner": denoising_contract.get("guidance_owner", "pipeline"),
                "guidance_mode": denoising_contract.get("guidance_mode", "flat"),
                "guidance_math_version": denoising_contract.get(
                    "guidance_math_version", "phase11g_shared_guidance_v1"
                ),
                "cfg_rescale": denoising_contract.get("cfg_rescale", 0.0),
                "requested_cfg_scale": float(step_requested_cfg_scale),
                "effective_cfg_scale": float(step_effective_cfg_scale),
                **guidance_step,
                "cfg_rescale_applied": bool(denoising_contract.get("cfg_rescale_applied", False)),
                "legacy_clamp_guidance": bool(denoising_contract.get("legacy_clamp_guidance", False)),
                **history,
            }

            if trace_enabled:
                if guidance_trace:
                    model_input_stats = dict(
                        guidance_trace.get("model_input") or {}
                    )
                    uncond_stats = dict(
                        guidance_trace.get("unconditional_model_output") or {}
                    )
                    cond_stats = dict(
                        guidance_trace.get("conditional_model_output") or {}
                    )
                    guided_stats = dict(
                        guidance_trace.get("guided_model_output")
                        or {"available": False}
                    )
                    trace_extra["prediction_type"] = guidance_trace.get(
                        "prediction_type", "epsilon"
                    )
                else:
                    model_input_stats = self._tensor_trace_stats(
                        self._actual_model_input(model_x, sigma)
                    )
                    guided_stats = {"available": False}
                    unavailable = {
                        "available": False,
                        "norm": None,
                        "all_finite": None,
                        "nan_count": None,
                        "pos_inf_count": None,
                        "neg_inf_count": None,
                    }
                    uncond_stats = dict(unavailable)
                    cond_stats = dict(unavailable)

                denoised_stats = self._tensor_trace_stats(denoised)
                solver_before_stats = self._tensor_trace_stats(latent_before)
                trace_extra.update(
                    {
                        "model_input": model_input_stats,
                        "solver_latent_before": solver_before_stats,
                        "unconditional_model_output": uncond_stats,
                        "conditional_model_output": cond_stats,
                        "guided_model_output": guided_stats,
                        "epsilon": dict(
                            guidance_trace.get("guided_epsilon") or guided_stats
                        ) if guidance_trace else guided_stats,
                        "predicted_x0": denoised_stats,
                        "denoised": denoised_stats,
                        "predicted_x0_comparison": self._x0_comparison(
                            denoised, previous_predicted_x0
                        ),
                        **self._coefficient_trace(
                            sigma=sigma,
                            sigma_next=sigma_next,
                            previous_t=effective_previous_t,
                        ),
                    }
                )

            x = self._dpmpp_2m_update(
                x=x,
                denoised=denoised,
                old_denoised=effective_old_denoised,
                previous_t=effective_previous_t,
                sigma=sigma,
                sigma_next=sigma_next,
                history_policy=history_policy,
            )
            x = self._apply_latent_step_hook(
                state,
                request=request,
                latent=x,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                timestep=timestep,
            )
            if trace_enabled:
                latent_after_stats = self._tensor_trace_stats(x)
                trace_extra["solver_latent_after"] = latent_after_stats
                trace_extra["difference_from_euler_reference"] = (
                    self._reference_comparison(x, euler_reference_latents.get(i))
                )
                trace_extra["finite_status"] = {
                    "model_input": trace_extra["model_input"].get("all_finite"),
                    "solver_latent_before": trace_extra["solver_latent_before"].get(
                        "all_finite"
                    ),
                    "unconditional_model_output": trace_extra[
                        "unconditional_model_output"
                    ].get("all_finite"),
                    "conditional_model_output": trace_extra[
                        "conditional_model_output"
                    ].get("all_finite"),
                    "guided_model_output": trace_extra["guided_model_output"].get(
                        "all_finite"
                    ),
                    "predicted_x0": trace_extra["predicted_x0"].get("all_finite"),
                    "solver_latent_after": latent_after_stats.get("all_finite"),
                }

            self._trace_step(
                request,
                step_index=i,
                sigma=sigma,
                sigma_next=sigma_next,
                latent_before=latent_before,
                latent_after=x,
                noise_pred=None,
                guided_noise=None,
                cfg_scale=step_effective_cfg_scale,
                extra=trace_extra,
                predicted_x0_snapshot=preview_predicted_x0.to(dtype=model_input_dtype),
            )
            self._emit_live_preview(
                state,
                request=request,
                step_index=i,
                total_steps=effective_steps,
                latent=x,
                predicted_x0=preview_predicted_x0,
                sigma=sigma,
                model_timestep=timestep,
                metadata=trace_extra,
            )
            if progress is not None:
                progress.update(1)

            previous_predicted_x0 = denoised
            if not history_enabled:
                old_denoised = None
                previous_t = None
            elif (
                history_policy == "model_timestep_guarded"
                and history["history_reset_required"]
            ):
                old_denoised = None
                previous_t = None
                history_blocked_until_distinct = True
            else:
                old_denoised = denoised
                previous_t = self._sigma_to_t(sigma)
                if (
                    history_policy == "model_timestep_guarded"
                    and history_blocked_until_distinct
                ):
                    history_blocked_until_distinct = False
            previous_model_timestep = current_model_timestep

        self._trace_sampler_summary(
            request,
            requested_steps=requested_steps,
            effective_steps=effective_steps,
        )

        regional_runtime = (
            regional_resolver.runtime_snapshot() if regional_resolver is not None else {}
        )
        if regional_runtime and isinstance(getattr(request, "diagnostics", None), dict):
            request.diagnostics["regional_runtime"] = regional_runtime
            passes = request.diagnostics.setdefault("regional_runtime_passes", {})
            passes[str(regional_runtime.get("pass") or "base")] = regional_runtime

        return SamplerOutput(
            latents=x.to(dtype=model_input_dtype),
            extra={
                "sampler_name": self.SAMPLER_NAME,
                "requested_steps": requested_steps,
                "effective_steps": effective_steps,
                "scheduler_step_override_applied": effective_steps != requested_steps,
                "integration_mode_used": self.SAMPLER_NAME,
                "model_prediction_type": denoising_contract.get(
                    "prediction_type", "unknown"
                ),
                "prediction_conversion": (
                    denoising_contract.get("prediction_conversion")
                    if prediction_mode == "canonical_predicted_x0"
                    else "legacy sampler-local x0 = solver_sample - sigma * guided_epsilon"
                ),
                "model_input_preconditioning": denoising_contract.get(
                    "model_input_preconditioning"
                ),
                "cfg_scale": cfg_scale,
                "cfg_rescale": denoising_contract.get("cfg_rescale", 0.0),
                "cfg_rescale_applied": bool(denoising_contract.get("cfg_rescale_applied", False)),
                "guidance_owner": denoising_contract.get("guidance_owner", "pipeline"),
                "guidance_mode": denoising_contract.get("guidance_mode", "flat"),
                "guidance_math_version": denoising_contract.get(
                    "guidance_math_version", "phase11g_shared_guidance_v1"
                ),
                "legacy_clamp_guidance": bool(denoising_contract.get("legacy_clamp_guidance", False)),
                "model_dtype": denoising_contract.get(
                    "model_dtype", str(model_input_dtype)
                ),
                "integration_prediction_type": prediction_mode,
                "history_policy": history_policy,
                "history_enabled": history_enabled,
                "history_state_read_count": history_state_read_count,
                "history_reset_count": history_reset_count,
                "history_guard_rejection_count": history_guard_rejection_count,
                "first_order_update_count": first_order_update_count,
                "second_order_update_count": second_order_update_count,
                "terminal_update_count": terminal_update_count,
                "schedule_transitions": int(sigmas.numel() - 1),
                "sigma_used_for_prediction": [
                    float(value) for value in sigmas[:-1].detach().cpu().flatten()
                ],
                "model_timestep": [
                    float(value) for value in timesteps[: sigmas.numel() - 1].detach().cpu().flatten()
                ],
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
                "model_input_dtype": str(model_input_dtype),
                "solver_dtype": str(solver_dtype),
                "solver_state_initial_dtype": str(model_input_dtype),
                "solver_state_internal_dtype": str(solver_dtype),
                "solver_state_return_dtype": str(model_input_dtype),
                "phase08b_trace_enabled": trace_enabled,
                "phase08d_first_order_validation": history_policy == "first_order_only",
            },
        )

    def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.clamp(sigma, min=1e-12)
        return sigma.log().neg()

    def _dpmpp_2m_update(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        old_denoised: Optional[torch.Tensor],
        previous_t: Optional[torch.Tensor],
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        history_policy: str = "model_timestep_guarded",
    ) -> torch.Tensor:
        """Apply the deterministic DPM++ 2M update in denoised space."""
        sigma_val = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        sigma_next_val = torch.as_tensor(sigma_next, device=x.device, dtype=x.dtype)

        if float(sigma_next_val.detach().cpu()) <= 0.0:
            return denoised

        t = self._sigma_to_t(sigma_val)
        t_next = self._sigma_to_t(sigma_next_val)
        h = t_next - t

        if history_policy == "first_order_only":
            denoised_d = denoised
        elif old_denoised is None or previous_t is None:
            denoised_d = denoised
        else:
            h_last = t - previous_t.to(device=x.device, dtype=x.dtype)
            r = h_last / torch.clamp(h, min=torch.finfo(x.dtype).eps)
            r = torch.clamp(r, min=torch.finfo(x.dtype).eps)
            denoised_d = (
                (1.0 + 1.0 / (2.0 * r)) * denoised
                - (1.0 / (2.0 * r)) * old_denoised
            )

        sigma_ratio = sigma_next_val / torch.clamp(
            sigma_val,
            min=torch.finfo(x.dtype).eps,
        )
        return sigma_ratio * x - torch.expm1(-h) * denoised_d


class DPMPlusPlus2MSamplerAdapter:
    SAMPLER_CAPABILITIES = DPMPlusPlus2MSampler.SAMPLER_CAPABILITIES

    def __init__(self):
        self.sampler = DPMPlusPlus2MSampler()

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
            progress.start(total=total, desc="DPM++ 2M Sampling")
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


SAMPLER_NAME = "DPM++ 2M"
SAMPLER_CLASS = DPMPlusPlus2MSampler
SAMPLER_ADAPTER_CLASS = DPMPlusPlus2MSamplerAdapter

PLUGIN_DESCRIPTOR = {
    "plugin_id": "sampler.dpmpp_2m",
    "kind": "sampler",
    "name": "dpmpp_2m",
    "label": "DPM++ 2M",
    "description": meta["description"],
    "version": "2",
    "module": __name__,
    "adapter_class": "DPMPlusPlus2MSamplerAdapter",
    "aliases": ["dpm++ 2m", "dpmpp2m", "dpm plus plus 2m"],
    "capabilities": DPMPlusPlus2MSamplerAdapter.SAMPLER_CAPABILITIES.to_serializable_dict(),
    "config_schema": {
        "type": "object",
        "properties": {
            "solver_dtype": {
                "type": "string",
                "default": "float32",
                "title": "Solver Dtype",
                "description": "Internal solver precision used during the DPM++ 2M integration loop.",
                "enum": ["latent", "model", "input", "float16", "fp16", "float32", "fp32"],
                "x_group": "Runtime Math",
            },
            "history_policy": {
                "type": "string",
                "default": "model_timestep_guarded",
                "title": "History Policy",
                "description": "Controls whether second-order history is guarded, fully multistep, or forced to first-order updates only.",
                "enum": ["model_timestep_guarded", "multistep", "first_order_only"],
                "x_group": "History Behavior",
            },
            "prediction_mode": {
                "type": "string",
                "default": "canonical_predicted_x0",
                "title": "Prediction Mode",
                "description": "Chooses the canonical x0 implementation or the legacy Phase 8D diagnostic prediction path.",
                "enum": [
                    "canonical_predicted_x0",
                    "legacy_sampler_local_epsilon",
                ],
                "x_group": "Runtime Math",
            },
            "phase08d_validation_mode": {
                "type": "boolean",
                "default": False,
                "title": "Phase 8D Validation Mode",
                "description": "Required only when testing the legacy diagnostic prediction path with first-order history.",
                "x_group": "Diagnostics",
            },
        },
        "required": [],
        "additionalProperties": True,
    },
}
