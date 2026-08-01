from __future__ import annotations

from typing import Any

import torch

from image_gen.systems.guidance import apply_canonical_cfg_rescale


class DenoisingSystem:
    """Canonical model-output interpretation boundary.

    The denoising system owns model-input preconditioning, UNet invocation,
    classifier-free guidance, optional guidance rescaling, prediction-type
    conversion, and the float32 predicted-clean-latent boundary required by
    denoised-space solvers such as DPM++ 2M.
    """

    SUPPORTED_PREDICTION_TYPES = {"epsilon", "v_prediction", "sample"}
    MODEL_INPUT_PRECONDITIONING = "sample / sqrt(sigma^2 + 1)"

    def __init__(
        self,
        unet: torch.nn.Module,
        *,
        prediction_type: str = "epsilon",
        prediction_type_source: str = "pipeline_components",
    ) -> None:
        self.unet = unet
        self.prediction_type = self.normalize_prediction_type(prediction_type)
        self.prediction_type_source = str(prediction_type_source or "unspecified")
        self._cfg_rescale = 0.0
        self._guidance_trace_enabled = False
        self._last_guidance_trace: dict[str, Any] | None = None

    @classmethod
    def normalize_prediction_type(cls, value: str | None) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "eps": "epsilon",
            "epsilon_prediction": "epsilon",
            "v": "v_prediction",
            "velocity": "v_prediction",
            "vprediction": "v_prediction",
            "x0": "sample",
            "x_0": "sample",
            "predicted_x0": "sample",
            "denoised": "sample",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.SUPPORTED_PREDICTION_TYPES:
            supported = ", ".join(sorted(cls.SUPPORTED_PREDICTION_TYPES))
            raise ValueError(
                f"Unsupported checkpoint prediction type {value!r}. Supported values: {supported}."
            )
        return normalized

    def configure_request(self, *, cfg_rescale: float | None = None) -> None:
        value = 0.0 if cfg_rescale is None else float(cfg_rescale)
        if not 0.0 <= value <= 1.0:
            raise ValueError("cfg_rescale must be between 0.0 and 1.0.")
        self._cfg_rescale = value

    def contract_metadata(self) -> dict[str, Any]:
        model_dtype = self._model_dtype(self.unet, torch.float32)
        return {
            "prediction_type": self.prediction_type,
            "prediction_type_source": self.prediction_type_source,
            "prediction_conversion": self._prediction_conversion_label(),
            "model_input_preconditioning": self.MODEL_INPUT_PRECONDITIONING,
            "cfg_rescale": float(self._cfg_rescale),
            "cfg_rescale_applied": bool(float(self._cfg_rescale) > 0.0),
            "guidance_owner": "pipeline",
            "guidance_mode": "flat",
            "guidance_math_version": "phase11g_shared_guidance_v1",
            "legacy_clamp_guidance": False,
            "model_dtype": str(model_dtype),
            "solver_dtype": "torch.float32",
        }


    @staticmethod
    def _parse_torch_dtype(value: Any) -> torch.dtype | None:
        if isinstance(value, torch.dtype):
            return value
        if value is None:
            return None
        lookup = {
            "torch.float16": torch.float16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "torch.float32": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "torch.bfloat16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return lookup.get(str(value).strip().lower())

    def guided_epsilon_contract_metadata(self) -> dict[str, Any]:
        model_dtype = self._model_dtype(self.unet, torch.float32)
        return {
            "prediction_type": "epsilon",
            "prediction_type_source": f"guided_epsilon_adapter_from_{self.prediction_type_source}",
            "prediction_conversion": "x0 = solver_sample - sigma * guided_epsilon",
            "model_input_preconditioning": self.MODEL_INPUT_PRECONDITIONING,
            "cfg_rescale": float(self._cfg_rescale),
            "cfg_rescale_applied": bool(float(self._cfg_rescale) > 0.0),
            "guidance_owner": "pipeline",
            "guidance_mode": "flat",
            "guidance_math_version": "phase11g_shared_guidance_v1",
            "legacy_clamp_guidance": False,
            "model_dtype": str(model_dtype),
            "solver_dtype": "torch.float32",
            "adapter_source": "guided_model_fn",
        }

    def build_guided_epsilon_denoiser(
        self,
        guided_model_fn: Any,
        *,
        model_call_dtype: torch.dtype | None = None,
    ):
        selected_model_dtype = model_call_dtype or self._model_dtype(self.unet, torch.float32)

        def predict_denoised(
            sample: torch.Tensor,
            sigma: torch.Tensor | float,
            timestep: torch.Tensor | float,
            cond: torch.Tensor,
            uncond: torch.Tensor,
            cfg_scale: float,
        ) -> torch.Tensor:
            solver_sample = sample.to(device=sample.device, dtype=torch.float32)
            model_sample = sample.to(device=sample.device, dtype=selected_model_dtype)
            guided_epsilon = guided_model_fn(
                model_sample,
                sigma,
                timestep,
                cond,
                uncond,
                cfg_scale,
            )
            sigma_values = self._normalize_sigma(
                solver_sample, sigma, dtype=torch.float32
            )
            sigma_b = self._broadcast_like(sigma_values, solver_sample)
            denoised = solver_sample - sigma_b * guided_epsilon.to(
                device=solver_sample.device, dtype=torch.float32
            )
            if not torch.isfinite(denoised).all():
                raise ValueError(
                    "Guided-epsilon denoiser adapter produced non-finite predicted x0 values."
                )
            return denoised

        return predict_denoised

    def _prediction_conversion_label(self) -> str:
        if self.prediction_type == "epsilon":
            return "x0 = solver_sample - sigma * guided_epsilon"
        if self.prediction_type == "v_prediction":
            return (
                "x0 = solver_sample / (sigma^2 + 1) "
                "- sigma / sqrt(sigma^2 + 1) * guided_v"
            )
        return "x0 = guided_model_output"

    @staticmethod
    def _model_dtype(unet: torch.nn.Module, fallback: torch.dtype) -> torch.dtype:
        try:
            return next(unet.parameters()).dtype
        except (StopIteration, AttributeError):
            return fallback

    @staticmethod
    def _normalize_sigma(
        latents: torch.Tensor,
        sigma: torch.Tensor | float,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            sigma,
            device=latents.device,
            dtype=dtype or latents.dtype,
        ).flatten()
        if value.numel() == 1 and latents.shape[0] > 1:
            value = value.expand(latents.shape[0])
        if value.numel() not in {1, latents.shape[0]}:
            raise ValueError("sigma must be scalar or provide one value per latent batch item.")
        return value

    @staticmethod
    def _broadcast_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        while value.ndim < reference.ndim:
            value = value.unsqueeze(-1)
        return value

    @staticmethod
    def _normalize_timestep(latents: torch.Tensor, timestep: torch.Tensor | float) -> torch.Tensor:
        value = torch.as_tensor(timestep, device=latents.device, dtype=torch.float32).flatten()
        if value.numel() == 1 and latents.shape[0] > 1:
            value = value.expand(latents.shape[0])
        if value.numel() not in {1, latents.shape[0]}:
            raise ValueError("timestep must be scalar or provide one value per latent batch item.")
        return value

    @staticmethod
    def extract_model_tensor(model_output: Any, *, owner: str) -> torch.Tensor:
        value = model_output
        if hasattr(value, "sample"):
            value = value.sample
        elif isinstance(value, (tuple, list)):
            if not value:
                raise TypeError(f"{owner} returned an empty sequence.")
            value = value[0]
        if not torch.is_tensor(value):
            raise TypeError(f"{owner} must return a tensor, tuple, or object with .sample.")
        return value

    @classmethod
    def scale_model_input(
        cls,
        latents: torch.Tensor,
        sigma_tensor: torch.Tensor,
    ) -> torch.Tensor:
        sigma_b = cls._broadcast_like(sigma_tensor, latents)
        return latents / torch.sqrt(sigma_b.square() + 1.0)

    @staticmethod
    def _trace_stats(value: torch.Tensor) -> dict[str, Any]:
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
        finite_mask = torch.isfinite(tensor)
        nan_count = int(torch.isnan(tensor).sum().item())
        pos_inf_count = int(torch.isposinf(tensor).sum().item())
        neg_inf_count = int(torch.isneginf(tensor).sum().item())
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
            "all_finite": all_finite,
            "nan_count": nan_count,
            "pos_inf_count": pos_inf_count,
            "neg_inf_count": neg_inf_count,
            "finite_count": int(finite_values.numel()),
            "element_count": int(tensor.numel()),
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "norm": norm,
        }

    def _prepare_model_input(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_dtype = self._model_dtype(self.unet, sample.dtype)
        model_sample = sample.to(dtype=model_dtype)
        sigma_model = self._normalize_sigma(model_sample, sigma, dtype=model_dtype)
        return self.scale_model_input(model_sample, sigma_model), sigma_model

    def predict_model_output(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        conditioning: torch.Tensor,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        model_input, _ = self._prepare_model_input(sample, sigma)
        timestep_tensor = self._normalize_timestep(model_input, timestep)
        model_out = self.unet(
            sample=model_input,
            timestep=timestep_tensor,
            encoder_hidden_states=conditioning,
            **(extra_cond_kwargs or {}),
        )
        prediction = self.extract_model_tensor(model_out, owner="UNet")
        if prediction.shape != sample.shape:
            raise ValueError(
                f"UNet prediction shape {tuple(prediction.shape)} does not match "
                f"sample shape {tuple(sample.shape)}."
            )
        if not torch.isfinite(prediction).all():
            raise ValueError("UNet returned non-finite values.")
        return prediction

    @classmethod
    def convert_model_output_to_denoised(
        cls,
        *,
        solver_sample: torch.Tensor,
        sigma: torch.Tensor | float,
        model_output: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        normalized = cls.normalize_prediction_type(prediction_type)
        sample32 = solver_sample.to(dtype=torch.float32)
        output32 = model_output.to(device=sample32.device, dtype=torch.float32)
        sigma32 = cls._normalize_sigma(sample32, sigma, dtype=torch.float32)
        sigma_b = cls._broadcast_like(sigma32, sample32)

        if normalized == "epsilon":
            denoised = sample32 - sigma_b * output32
        elif normalized == "v_prediction":
            sigma_sq_plus_one = sigma_b.square() + 1.0
            denoised = (
                sample32 / sigma_sq_plus_one
                - sigma_b / torch.sqrt(sigma_sq_plus_one) * output32
            )
        else:
            denoised = output32

        if not torch.isfinite(denoised).all():
            raise ValueError("Canonical predicted x0 contains non-finite values.")
        return denoised

    @classmethod
    def convert_model_output_to_epsilon(
        cls,
        *,
        solver_sample: torch.Tensor,
        sigma: torch.Tensor | float,
        model_output: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        normalized = cls.normalize_prediction_type(prediction_type)
        if normalized == "epsilon":
            return model_output

        sample32 = solver_sample.to(dtype=torch.float32)
        denoised = cls.convert_model_output_to_denoised(
            solver_sample=sample32,
            sigma=sigma,
            model_output=model_output,
            prediction_type=normalized,
        )
        sigma32 = cls._normalize_sigma(sample32, sigma, dtype=torch.float32)
        sigma_b = cls._broadcast_like(sigma32, sample32)
        if bool((sigma_b == 0).any().item()):
            raise ValueError("Cannot convert x0 or v-prediction to epsilon at sigma zero.")
        epsilon = (sample32 - denoised) / sigma_b
        return epsilon.to(dtype=model_output.dtype)

    def _guided_model_output(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Branch order is deliberate and covered by Phase 8C tests.
        output_uncond = self.predict_model_output(
            sample, sigma, timestep, uncond, extra_cond_kwargs
        )
        output_cond = self.predict_model_output(
            sample, sigma, timestep, cond, extra_cond_kwargs
        )
        guided = output_uncond + float(cfg_scale) * (output_cond - output_uncond)
        guided = apply_canonical_cfg_rescale(guided, output_cond, self._cfg_rescale)
        model_input, _ = self._prepare_model_input(sample, sigma)
        return guided, output_uncond, output_cond, model_input

    def predict_epsilon(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        conditioning: torch.Tensor,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        output = self.predict_model_output(
            sample, sigma, timestep, conditioning, extra_cond_kwargs
        )
        return self.convert_model_output_to_epsilon(
            solver_sample=sample,
            sigma=sigma,
            model_output=output,
            prediction_type=self.prediction_type,
        )

    def predict_raw_noise(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Compatibility callback returning epsilon for Euler/KES paths."""

        return self.predict_epsilon(
            latents, sigma, timestep, cond, extra_cond_kwargs
        )

    def predict_guided_noise(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        guided_output, output_uncond, output_cond, model_input = self._guided_model_output(
            latents,
            sigma,
            timestep,
            cond,
            uncond,
            cfg_scale,
            extra_cond_kwargs,
        )
        epsilon = self.convert_model_output_to_epsilon(
            solver_sample=latents,
            sigma=sigma,
            model_output=guided_output,
            prediction_type=self.prediction_type,
        )
        denoised = self.convert_model_output_to_denoised(
            solver_sample=latents,
            sigma=sigma,
            model_output=guided_output,
            prediction_type=self.prediction_type,
        )
        self._record_guidance_trace(
            sample=latents,
            sigma=sigma,
            timestep=timestep,
            cfg_scale=cfg_scale,
            model_input=model_input,
            output_uncond=output_uncond,
            output_cond=output_cond,
            guided_output=guided_output,
            epsilon=epsilon,
            denoised=denoised,
        )
        return epsilon

    def predict_denoised(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        guided_output, output_uncond, output_cond, model_input = self._guided_model_output(
            sample,
            sigma,
            timestep,
            cond,
            uncond,
            cfg_scale,
            extra_cond_kwargs,
        )
        denoised = self.convert_model_output_to_denoised(
            solver_sample=sample,
            sigma=sigma,
            model_output=guided_output,
            prediction_type=self.prediction_type,
        )
        epsilon = self.convert_model_output_to_epsilon(
            solver_sample=sample,
            sigma=sigma,
            model_output=guided_output,
            prediction_type=self.prediction_type,
        )
        self._record_guidance_trace(
            sample=sample,
            sigma=sigma,
            timestep=timestep,
            cfg_scale=cfg_scale,
            model_input=model_input,
            output_uncond=output_uncond,
            output_cond=output_cond,
            guided_output=guided_output,
            epsilon=epsilon,
            denoised=denoised,
        )
        if denoised.dtype != torch.float32:
            raise AssertionError("predict_denoised must return float32 solver output.")
        return denoised

    def _record_guidance_trace(
        self,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cfg_scale: float,
        model_input: torch.Tensor,
        output_uncond: torch.Tensor,
        output_cond: torch.Tensor,
        guided_output: torch.Tensor,
        epsilon: torch.Tensor,
        denoised: torch.Tensor,
    ) -> None:
        if not self._guidance_trace_enabled:
            self._last_guidance_trace = None
            return
        sigma_values = self._normalize_sigma(
            sample.to(dtype=torch.float32), sigma, dtype=torch.float32
        )
        timestep_values = self._normalize_timestep(sample, timestep)
        self._last_guidance_trace = {
            "solver_sample": self._trace_stats(sample),
            "model_input": self._trace_stats(model_input),
            "unconditional_model_output": self._trace_stats(output_uncond),
            "conditional_model_output": self._trace_stats(output_cond),
            "guided_model_output": self._trace_stats(guided_output),
            "guided_epsilon": self._trace_stats(epsilon),
            "predicted_x0": self._trace_stats(denoised),
            "prediction_type": self.prediction_type,
            "prediction_type_source": self.prediction_type_source,
            "prediction_conversion": self._prediction_conversion_label(),
            "model_input_preconditioning": self.MODEL_INPUT_PRECONDITIONING,
            "guidance_owner": "pipeline",
            "guidance_mode": "flat",
            "guidance_math_version": "phase11g_shared_guidance_v1",
            "cfg_scale": float(cfg_scale),
            "cfg_rescale": float(self._cfg_rescale),
            "cfg_rescale_applied": bool(float(self._cfg_rescale) > 0.0),
            "legacy_clamp_guidance": False,
            "cfg_branch_order": ["unconditional", "conditional", "guided"],
            "model_dtype": str(model_input.dtype),
            "solver_dtype": str(denoised.dtype),
            "sigma_used_for_prediction": [
                float(value) for value in sigma_values.detach().cpu().flatten()
            ],
            "model_timestep": [
                float(value) for value in timestep_values.detach().cpu().flatten()
            ],
        }

    def set_guidance_trace_enabled(self, enabled: bool) -> None:
        self._guidance_trace_enabled = bool(enabled)
        if not self._guidance_trace_enabled:
            self._last_guidance_trace = None

    def consume_guidance_trace(self) -> dict[str, Any] | None:
        value = self._last_guidance_trace
        self._last_guidance_trace = None
        return value
