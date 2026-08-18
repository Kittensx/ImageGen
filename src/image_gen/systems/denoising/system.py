from __future__ import annotations

from typing import Any

import math
import time

import torch

from image_gen.contracts.model_conditioning import (
    ModelConditioningKwargs,
    select_model_conditioning_branch,
)
from image_gen.systems.guidance import apply_canonical_cfg_rescale


class DenoiserAdapter:
    """Adapter contract for denoiser-family model invocation."""

    denoiser_kind = "generic"
    owner_label = "Denoiser"
    model_input_preconditioning = "adapter_defined"

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    def prepare_model_input(
        self,
        system: "DenoisingSystem",
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_dtype = system._model_dtype(self.module, sample.dtype)
        model_sample = sample.to(dtype=model_dtype)
        sigma_model = system._normalize_sigma(model_sample, sigma, dtype=model_dtype)
        return model_sample, sigma_model

    def invoke_model(
        self,
        *,
        system: "DenoisingSystem",
        model_input: torch.Tensor,
        sigma_model: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
        extra_cond_kwargs: dict[str, Any],
    ) -> Any:
        raise NotImplementedError("Denoiser adapters must implement invoke_model().")


class SD3TransformerDenoiserAdapter(DenoiserAdapter):
    """SD3 MMDiT invocation contract for flow-matching inference.

    Unlike the UNet adapter, SD3 receives the current latent tensor directly.
    No VP/Karras ``sample / sqrt(sigma^2 + 1)`` preconditioning is applied.
    """

    denoiser_kind = "transformer"
    owner_label = "SD3 Transformer"
    model_input_preconditioning = "identity (flow_match)"

    @staticmethod
    def _config_value(module: torch.nn.Module, key: str, default: Any = None) -> Any:
        config = getattr(module, "config", None)
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def prepare_model_input(
        self,
        system: "DenoisingSystem",
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_sample, sigma_model = DenoiserAdapter.prepare_model_input(
            self, system, sample, sigma
        )
        if model_sample.ndim != 4:
            raise ValueError(
                "SD3 transformer latents must be rank-4 [batch, channels, height, width]."
            )
        expected_channels = int(self._config_value(self.module, "in_channels", 16) or 16)
        actual_channels = int(model_sample.shape[1])
        if actual_channels != expected_channels:
            raise ValueError(
                "SD3 transformer latent-channel mismatch: "
                f"expected {expected_channels}, got {actual_channels}."
            )
        return model_sample, sigma_model

    def invoke_model(
        self,
        *,
        system: "DenoisingSystem",
        model_input: torch.Tensor,
        sigma_model: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
        extra_cond_kwargs: dict[str, Any],
    ) -> Any:
        if conditioning.ndim != 3:
            raise ValueError(
                "SD3 encoder_hidden_states must be rank-3 [batch, sequence, width]."
            )
        expected_joint_width = int(
            self._config_value(self.module, "joint_attention_dim", conditioning.shape[-1])
        )
        actual_joint_width = int(conditioning.shape[-1])
        if actual_joint_width != expected_joint_width:
            raise ValueError(
                "SD3 conditioning-width mismatch: "
                f"expected {expected_joint_width}, got {actual_joint_width}."
            )

        pooled = extra_cond_kwargs.get("pooled_projections")
        if pooled is None:
            raise ValueError(
                "SD3 transformer conditioning is missing pooled_projections."
            )
        if not torch.is_tensor(pooled) or pooled.ndim != 2:
            raise ValueError(
                "SD3 pooled_projections must be a rank-2 tensor [batch, width]."
            )
        expected_pooled_width = int(
            self._config_value(self.module, "pooled_projection_dim", pooled.shape[-1])
        )
        actual_pooled_width = int(pooled.shape[-1])
        if actual_pooled_width != expected_pooled_width:
            raise ValueError(
                "SD3 pooled-projection width mismatch: "
                f"expected {expected_pooled_width}, got {actual_pooled_width}."
            )
        if int(conditioning.shape[0]) != int(model_input.shape[0]):
            raise ValueError(
                "SD3 conditioning batch does not match latent batch."
            )
        if int(pooled.shape[0]) != int(model_input.shape[0]):
            raise ValueError(
                "SD3 pooled-projection batch does not match latent batch."
            )

        pooled = pooled.to(device=model_input.device, dtype=model_input.dtype)
        encoder_hidden_states = conditioning.to(
            device=model_input.device, dtype=model_input.dtype
        )
        joint_attention_kwargs = extra_cond_kwargs.get("joint_attention_kwargs")
        unknown = set(extra_cond_kwargs) - {"pooled_projections", "joint_attention_kwargs"}
        if unknown:
            raise ValueError(
                "Unknown SD3 transformer conditioning kwarg(s): " + ", ".join(sorted(unknown))
            )
        return self.module(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )


class UNetDenoiserAdapter(DenoiserAdapter):
    denoiser_kind = "unet"
    owner_label = "UNet"
    model_input_preconditioning = "sample / sqrt(sigma^2 + 1)"

    def prepare_model_input(
        self,
        system: "DenoisingSystem",
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_sample, sigma_model = super().prepare_model_input(system, sample, sigma)
        return system.scale_model_input(model_sample, sigma_model), sigma_model

    def invoke_model(
        self,
        *,
        system: "DenoisingSystem",
        model_input: torch.Tensor,
        sigma_model: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
        extra_cond_kwargs: dict[str, Any],
    ) -> Any:
        return self.module(
            sample=model_input,
            timestep=timestep,
            encoder_hidden_states=conditioning.to(
                device=model_input.device, dtype=model_input.dtype
            ),
            **extra_cond_kwargs,
        )


class DenoisingSystem:
    """Canonical model-output interpretation boundary.

    The denoising system owns model invocation through a denoiser-role adapter,
    classifier-free guidance, optional guidance rescaling, prediction-type
    conversion, and the float32 predicted-clean-latent boundary required by
    denoised-space solvers such as DPM++ 2M.
    """

    SUPPORTED_PREDICTION_TYPES = {"epsilon", "v_prediction", "sample"}
    MODEL_INPUT_PRECONDITIONING = UNetDenoiserAdapter.model_input_preconditioning

    def __init__(
        self,
        denoiser: torch.nn.Module,
        *,
        denoiser_kind: str = "unet",
        adapter: DenoiserAdapter | None = None,
        prediction_type: str = "epsilon",
        prediction_type_source: str = "pipeline_components",
    ) -> None:
        self.denoiser = denoiser
        self.denoiser_kind = str(denoiser_kind or "unet").strip().lower() or "unet"
        self.adapter = adapter or self._build_adapter(denoiser, self.denoiser_kind)
        self.unet = denoiser if self.denoiser_kind == "unet" else None
        prediction_token = str(prediction_type or "").strip().lower().replace("-", "_")
        self.is_flow_match = bool(
            self.denoiser_kind == "transformer"
            and prediction_token in {"flow_match", "not_applicable_flow_match", "flow"}
        )
        self.prediction_type = (
            "flow_match" if self.is_flow_match else self.normalize_prediction_type(prediction_type)
        )
        self.prediction_type_source = str(prediction_type_source or "unspecified")
        self._cfg_rescale = 0.0
        self._guidance_trace_enabled = False
        self._last_guidance_trace: dict[str, Any] | None = None

    @staticmethod
    def _build_adapter(denoiser: torch.nn.Module, denoiser_kind: str) -> DenoiserAdapter:
        if denoiser_kind == "unet":
            return UNetDenoiserAdapter(denoiser)
        if denoiser_kind == "transformer":
            return SD3TransformerDenoiserAdapter(denoiser)
        raise NotImplementedError(
            f"No built-in adapter is registered for denoiser kind {denoiser_kind!r}."
        )

    def _model_input_preconditioning_label(self) -> str:
        return str(getattr(self.adapter, "model_input_preconditioning", self.MODEL_INPUT_PRECONDITIONING))

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
        if self.is_flow_match and value != 0.0:
            raise ValueError(
                "cfg_rescale is not qualified for the SD3 flow-matching path in SD3-08."
            )
        self._cfg_rescale = value

    def contract_metadata(self) -> dict[str, Any]:
        model_dtype = self._model_dtype(self.denoiser, torch.float32)
        return {
            "denoiser_kind": self.denoiser_kind,
            "denoiser_owner": getattr(self.adapter, "owner_label", "Denoiser"),
            "prediction_type": self.prediction_type,
            "prediction_type_source": self.prediction_type_source,
            "prediction_conversion": self._prediction_conversion_label(),
            "model_input_preconditioning": self._model_input_preconditioning_label(),
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

    def flow_match_contract_metadata(self) -> dict[str, Any]:
        if not self.is_flow_match:
            raise RuntimeError("Flow-match metadata is only available for flow-matching denoisers.")
        model_dtype = self._model_dtype(self.denoiser, torch.float32)
        return {
            "denoiser_kind": self.denoiser_kind,
            "denoiser_owner": getattr(self.adapter, "owner_label", "Denoiser"),
            "prediction_type": "flow_match",
            "prediction_type_source": self.prediction_type_source,
            "prediction_conversion": "none; scheduler consumes transformer output directly",
            "model_input_preconditioning": self._model_input_preconditioning_label(),
            "guidance_owner": "pipeline",
            "guidance_mode": "ordinary_cfg",
            "cfg_rescale": 0.0,
            "model_dtype": str(model_dtype),
            "scheduler_domain": "flow_match",
        }

    def guided_epsilon_contract_metadata(self) -> dict[str, Any]:
        if self.is_flow_match:
            raise RuntimeError(
                "Guided-epsilon conversion is not defined for the SD3 flow-matching denoiser."
            )
        model_dtype = self._model_dtype(self.denoiser, torch.float32)
        return {
            "denoiser_kind": self.denoiser_kind,
            "denoiser_owner": getattr(self.adapter, "owner_label", "Denoiser"),
            "prediction_type": "epsilon",
            "prediction_type_source": f"guided_epsilon_adapter_from_{self.prediction_type_source}",
            "prediction_conversion": "x0 = solver_sample - sigma * guided_epsilon",
            "model_input_preconditioning": self._model_input_preconditioning_label(),
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
        if self.is_flow_match:
            raise RuntimeError(
                "Guided-epsilon denoiser construction is not valid for SD3 flow matching."
            )
        selected_model_dtype = model_call_dtype or self._model_dtype(self.denoiser, torch.float32)

        def predict_denoised(
            sample: torch.Tensor,
            sigma: torch.Tensor | float,
            timestep: torch.Tensor | float,
            cond: torch.Tensor,
            uncond: torch.Tensor,
            cfg_scale: float,
            extra_cond_kwargs: ModelConditioningKwargs = None,
        ) -> torch.Tensor:
            solver_sample = sample.to(device=sample.device, dtype=torch.float32)
            model_sample = sample.to(device=sample.device, dtype=selected_model_dtype)
            if extra_cond_kwargs is None:
                guided_epsilon = guided_model_fn(
                    model_sample,
                    sigma,
                    timestep,
                    cond,
                    uncond,
                    cfg_scale,
                )
            else:
                guided_epsilon = guided_model_fn(
                    model_sample,
                    sigma,
                    timestep,
                    cond,
                    uncond,
                    cfg_scale,
                    extra_cond_kwargs,
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
        if self.is_flow_match:
            return "flow model output is consumed directly by FlowMatchEulerDiscreteScheduler.step"
        if self.prediction_type == "epsilon":
            return "x0 = solver_sample - sigma * guided_epsilon"
        if self.prediction_type == "v_prediction":
            return (
                "x0 = solver_sample / (sigma^2 + 1) "
                "- sigma / sqrt(sigma^2 + 1) * guided_v"
            )
        return "x0 = guided_model_output"

    @staticmethod
    def _model_dtype(module: torch.nn.Module, fallback: torch.dtype) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
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
        return self.adapter.prepare_model_input(self, sample, sigma)

    @staticmethod
    def _prepare_extra_cond_kwargs(
        model_input: torch.Tensor,
        extra_cond_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not extra_cond_kwargs:
            return {}
        prepared = dict(extra_cond_kwargs)
        added = prepared.get("added_cond_kwargs")
        if isinstance(added, dict):
            normalized_added = dict(added)
            for key in ("text_embeds", "time_ids"):
                value = normalized_added.get(key)
                if torch.is_tensor(value):
                    normalized_added[key] = value.to(
                        device=model_input.device,
                        dtype=model_input.dtype,
                    )
            prepared["added_cond_kwargs"] = normalized_added
        return prepared

    def predict_model_output(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        conditioning: torch.Tensor,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        model_input, sigma_model = self._prepare_model_input(sample, sigma)
        timestep_tensor = self._normalize_timestep(model_input, timestep)
        prepared_extra_cond_kwargs = self._prepare_extra_cond_kwargs(
            model_input, extra_cond_kwargs
        )
        model_out = self.adapter.invoke_model(
            system=self,
            model_input=model_input,
            sigma_model=sigma_model,
            timestep=timestep_tensor,
            conditioning=conditioning,
            extra_cond_kwargs=prepared_extra_cond_kwargs,
        )
        owner = getattr(self.adapter, "owner_label", "Denoiser")
        prediction = self.extract_model_tensor(model_out, owner=owner)
        if prediction.shape != sample.shape:
            raise ValueError(
                f"{owner} prediction shape {tuple(prediction.shape)} does not match "
                f"sample shape {tuple(sample.shape)}."
            )
        if not torch.isfinite(prediction).all():
            raise ValueError(f"{owner} returned non-finite values.")
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
        if str(prediction_type or "").strip().lower().replace("-", "_") in {
            "flow_match", "not_applicable_flow_match", "flow"
        }:
            raise ValueError(
                "Flow-matching transformer output must not be converted through VP epsilon/x0 semantics."
            )
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
        if str(prediction_type or "").strip().lower().replace("-", "_") in {
            "flow_match", "not_applicable_flow_match", "flow"
        }:
            raise ValueError(
                "Flow-matching transformer output must not be converted through VP epsilon/x0 semantics."
            )
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

    @staticmethod
    def _slice_regional_batch_value(
        value: torch.Tensor | float,
        *,
        slot: int,
        batch_size: int,
    ) -> torch.Tensor | float:
        """Slice a batch-shaped sigma/timestep for one regional latent slot."""
        if not torch.is_tensor(value):
            return value
        tensor = value
        if tensor.ndim == 0 or tensor.numel() == 1:
            return value
        if int(tensor.shape[0]) == int(batch_size):
            return tensor[slot:slot + 1]
        return value

    @staticmethod
    def _regional_branch_parameters(region: Any) -> tuple[float, float]:
        metadata = dict(getattr(region, "metadata", {}) or {})
        weight = float(metadata.get("weight", 1.0))
        base_ratio = float(metadata.get("base_ratio", 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Regional branch weight must be finite and non-negative.")
        if not math.isfinite(base_ratio) or not 0.0 <= base_ratio <= 1.0:
            raise ValueError("Regional base ratio must be between 0 and 1.")
        return weight, base_ratio

    @classmethod
    def _blend_regional_outputs(
        cls,
        *,
        base_output: torch.Tensor,
        reference_output: torch.Tensor,
        regions: list[Any],
        region_outputs: list[torch.Tensor],
        overlap_policy: str,
    ) -> torch.Tensor:
        """Blend region branches in one output space before canonical CFG.

        The target branch follows SuperHybrid's latent backend semantics:
        ``base_ratio * base + (1-base_ratio) * weighted_region`` where the
        weighted region is measured relative to the unconditional/reference
        branch. Linear REGION curves are already represented as temporal
        strength 1.0 by the resolver.
        """
        if len(regions) != len(region_outputs):
            raise ValueError("Regional branch/output counts do not match.")
        policy = str(overlap_policy or "additive").strip().lower()
        if policy not in {"normalize", "additive", "priority"}:
            raise ValueError(f"Unsupported regional overlap policy: {policy!r}.")
        blended = base_output.clone()
        by_slot: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for region, region_output in zip(regions, region_outputs):
            slot = int(region.slot_index)
            if slot < 0 or slot >= int(base_output.shape[0]):
                raise ValueError("Regional conditioning slot is outside the latent batch.")
            base_slot = base_output[slot:slot + 1]
            reference_slot = reference_output[slot:slot + 1]
            weight, base_ratio = cls._regional_branch_parameters(region)
            weighted_region = reference_slot + weight * (region_output - reference_slot)
            target = base_ratio * base_slot + (1.0 - base_ratio) * weighted_region
            temporal_mask = (region.mask * float(region.strength)).to(
                device=base_output.device, dtype=base_output.dtype
            ).clamp_min(0.0)
            by_slot.setdefault(slot, []).append((target, temporal_mask))

        for slot, values in by_slot.items():
            base_slot = base_output[slot:slot + 1]
            if policy == "priority":
                current = base_slot
                for target, temporal_mask in values:
                    alpha = temporal_mask.clamp(0.0, 1.0)
                    current = current * (1.0 - alpha) + target * alpha
                blended[slot:slot + 1] = current
                continue

            total = torch.zeros_like(values[0][1])
            for _target, temporal_mask in values:
                total = total + temporal_mask
            scale = torch.ones_like(total)
            if policy == "normalize":
                scale = torch.where(
                    total > 1.0, total.clamp_min(1e-8).reciprocal(), scale
                )
            delta = torch.zeros_like(base_slot)
            for target, temporal_mask in values:
                delta = delta + (temporal_mask * scale) * (target - base_slot)
            blended[slot:slot + 1] = base_slot + delta

        if not torch.isfinite(blended).all():
            raise ValueError("Regional conditional output contains non-finite values.")
        return blended

    def _predict_regional_model_outputs(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        regions: list[Any],
        extra_cond_kwargs: dict[str, Any] | None,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        batch_size = int(sample.shape[0])
        for region in regions:
            slot = int(region.slot_index)
            if slot < 0 or slot >= batch_size:
                raise ValueError("Regional conditioning slot is outside the latent batch.")
            started = time.perf_counter()
            try:
                output = self.predict_model_output(
                    sample[slot:slot + 1],
                    self._slice_regional_batch_value(
                        sigma, slot=slot, batch_size=batch_size
                    ),
                    self._slice_regional_batch_value(
                        timestep, slot=slot, batch_size=batch_size
                    ),
                    region.conditioning,
                    extra_cond_kwargs,
                )
            finally:
                telemetry = None
                metadata = getattr(region, "metadata", None)
                if isinstance(metadata, dict):
                    telemetry = metadata.get("_regional_telemetry")
                if telemetry is not None and hasattr(telemetry, "record_denoiser_call"):
                    telemetry.record_denoiser_call(
                        slot_index=slot,
                        region_index=int(region.region_index),
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
                elif telemetry is not None and hasattr(telemetry, "record_unet_call"):
                    telemetry.record_unet_call(
                        slot_index=slot,
                        region_index=int(region.region_index),
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
            outputs.append(output)
        return outputs

    def _regional_conditional_model_output(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        base_cond: torch.Tensor,
        regions: list[Any],
        *,
        reference_output: torch.Tensor | None = None,
        overlap_policy: str = "additive",
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        base_output = self.predict_model_output(
            sample, sigma, timestep, base_cond, extra_cond_kwargs
        )
        if not regions:
            return base_output
        reference = (
            reference_output.to(device=base_output.device, dtype=base_output.dtype)
            if reference_output is not None
            else torch.zeros_like(base_output)
        )
        region_outputs = self._predict_regional_model_outputs(
            sample, sigma, timestep, regions, extra_cond_kwargs
        )
        return self._blend_regional_outputs(
            base_output=base_output,
            reference_output=reference,
            regions=regions,
            region_outputs=region_outputs,
            overlap_policy=overlap_policy,
        )

    def predict_regional_conditional_noise(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        base_cond: torch.Tensor,
        regions: list[Any],
        overlap_policy: str = "additive",
        extra_cond_kwargs: dict[str, Any] | None = None,
        *,
        uncond_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_model_output = self.predict_model_output(
            latents, sigma, timestep, base_cond, extra_cond_kwargs
        )
        base_noise = self.convert_model_output_to_epsilon(
            solver_sample=latents,
            sigma=sigma,
            model_output=base_model_output,
            prediction_type=self.prediction_type,
        )
        if not regions:
            return base_noise
        region_model_outputs = self._predict_regional_model_outputs(
            latents, sigma, timestep, regions, extra_cond_kwargs
        )
        region_noises: list[torch.Tensor] = []
        batch_size = int(latents.shape[0])
        for region, model_output in zip(regions, region_model_outputs):
            slot = int(region.slot_index)
            region_noises.append(
                self.convert_model_output_to_epsilon(
                    solver_sample=latents[slot:slot + 1],
                    sigma=self._slice_regional_batch_value(
                        sigma, slot=slot, batch_size=batch_size
                    ),
                    model_output=model_output,
                    prediction_type=self.prediction_type,
                )
            )
        reference = (
            uncond_noise.to(device=base_noise.device, dtype=base_noise.dtype)
            if uncond_noise is not None
            else torch.zeros_like(base_noise)
        )
        return self._blend_regional_outputs(
            base_output=base_noise,
            reference_output=reference,
            regions=regions,
            region_outputs=region_noises,
            overlap_policy=overlap_policy,
        )

    def _regional_guided_model_output(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        base_cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        regions: list[Any],
        overlap_policy: str = "additive",
        extra_cond_kwargs: ModelConditioningKwargs = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uncond_kwargs = select_model_conditioning_branch(
            extra_cond_kwargs, "unconditional"
        )
        cond_kwargs = select_model_conditioning_branch(
            extra_cond_kwargs, "conditional"
        )
        output_uncond = self.predict_model_output(
            sample, sigma, timestep, uncond, uncond_kwargs
        )
        output_cond = self._regional_conditional_model_output(
            sample,
            sigma,
            timestep,
            base_cond,
            regions,
            reference_output=output_uncond,
            overlap_policy=overlap_policy,
            extra_cond_kwargs=cond_kwargs,
        )
        guided = output_uncond + float(cfg_scale) * (output_cond - output_uncond)
        guided = apply_canonical_cfg_rescale(guided, output_cond, self._cfg_rescale)
        model_input, _ = self._prepare_model_input(sample, sigma)
        return guided, output_uncond, output_cond, model_input

    def predict_regional_guided_noise(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        base_cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        regions: list[Any],
        overlap_policy: str = "additive",
        extra_cond_kwargs: ModelConditioningKwargs = None,
    ) -> torch.Tensor:
        guided_output, output_uncond, output_cond, model_input = self._regional_guided_model_output(
            latents, sigma, timestep, base_cond, uncond, cfg_scale, regions,
            overlap_policy=overlap_policy,
            extra_cond_kwargs=extra_cond_kwargs,
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
            sample=latents, sigma=sigma, timestep=timestep, cfg_scale=cfg_scale,
            model_input=model_input, output_uncond=output_uncond,
            output_cond=output_cond, guided_output=guided_output,
            epsilon=epsilon, denoised=denoised,
        )
        return epsilon

    def predict_regional_denoised(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        base_cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        regions: list[Any],
        overlap_policy: str = "additive",
        extra_cond_kwargs: ModelConditioningKwargs = None,
    ) -> torch.Tensor:
        guided_output, output_uncond, output_cond, model_input = self._regional_guided_model_output(
            sample, sigma, timestep, base_cond, uncond, cfg_scale, regions,
            overlap_policy=overlap_policy,
            extra_cond_kwargs=extra_cond_kwargs,
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
            sample=sample, sigma=sigma, timestep=timestep, cfg_scale=cfg_scale,
            model_input=model_input, output_uncond=output_uncond,
            output_cond=output_cond, guided_output=guided_output,
            epsilon=epsilon, denoised=denoised,
        )
        return denoised.to(dtype=torch.float32)

    @staticmethod
    def _repeat_flow_batch_value(
        value: torch.Tensor | float,
        *,
        source_batch: int,
        repeats: int,
    ) -> torch.Tensor | float:
        if not torch.is_tensor(value):
            return value
        if value.ndim == 0 or value.numel() == 1:
            return value
        if int(value.shape[0]) == source_batch:
            return torch.cat([value] * repeats, dim=0)
        return value

    def predict_flow_model_output(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        conditioning: torch.Tensor,
        extra_cond_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if not self.is_flow_match:
            raise RuntimeError(
                "predict_flow_model_output is only valid for an SD3 flow-matching denoiser."
            )
        return self.predict_model_output(
            sample, sigma, timestep, conditioning, extra_cond_kwargs
        )

    def predict_guided_flow(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        extra_cond_kwargs: ModelConditioningKwargs = None,
    ) -> torch.Tensor:
        """Return the SD3 flow-model prediction with Diffusers-compatible CFG.

        Diffusers treats CFG as active only when guidance_scale > 1. When active,
        negative and positive branches are concatenated and evaluated in one
        transformer call. At CFG <= 1 only the conditional branch is evaluated.
        """
        if not self.is_flow_match:
            raise RuntimeError(
                "predict_guided_flow is only valid for an SD3 flow-matching denoiser."
            )
        scale = float(cfg_scale)
        cond_kwargs = select_model_conditioning_branch(extra_cond_kwargs, "conditional") or {}
        if scale <= 1.0:
            return self.predict_flow_model_output(
                sample, sigma, timestep, cond, cond_kwargs
            )

        uncond_kwargs = select_model_conditioning_branch(extra_cond_kwargs, "unconditional") or {}
        pooled_cond = cond_kwargs.get("pooled_projections")
        pooled_uncond = uncond_kwargs.get("pooled_projections")
        if pooled_cond is None or pooled_uncond is None:
            raise ValueError(
                "SD3 classifier-free guidance requires conditional and unconditional pooled_projections."
            )
        cond_joint = cond_kwargs.get("joint_attention_kwargs")
        uncond_joint = uncond_kwargs.get("joint_attention_kwargs")
        if cond_joint != uncond_joint:
            raise ValueError(
                "SD3 CFG requires shared joint_attention_kwargs for conditional and unconditional branches."
            )

        batch = int(sample.shape[0])
        model_sample = torch.cat([sample, sample], dim=0)
        model_cond = torch.cat([uncond, cond], dim=0)
        model_pooled = torch.cat([pooled_uncond, pooled_cond], dim=0)
        model_sigma = self._repeat_flow_batch_value(
            sigma, source_batch=batch, repeats=2
        )
        model_timestep = self._repeat_flow_batch_value(
            timestep, source_batch=batch, repeats=2
        )
        combined = self.predict_flow_model_output(
            model_sample,
            model_sigma,
            model_timestep,
            model_cond,
            {
                "pooled_projections": model_pooled,
                "joint_attention_kwargs": cond_joint,
            },
        )
        output_uncond, output_cond = combined.chunk(2, dim=0)
        guided = output_uncond + scale * (output_cond - output_uncond)
        if not torch.isfinite(guided).all():
            raise ValueError("SD3 classifier-free guidance produced non-finite flow output.")
        return guided

    def _guided_model_output(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor | float,
        timestep: torch.Tensor | float,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg_scale: float,
        extra_cond_kwargs: ModelConditioningKwargs = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Branch order is deliberate and covered by Phase 8C tests.
        # SDXL supplies different pooled text projections to the negative and
        # positive branches while sharing the same geometry time IDs.
        uncond_kwargs = select_model_conditioning_branch(
            extra_cond_kwargs, "unconditional"
        )
        cond_kwargs = select_model_conditioning_branch(
            extra_cond_kwargs, "conditional"
        )
        output_uncond = self.predict_model_output(
            sample, sigma, timestep, uncond, uncond_kwargs
        )
        output_cond = self.predict_model_output(
            sample, sigma, timestep, cond, cond_kwargs
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
        if self.is_flow_match:
            raise RuntimeError(
                "SD3 flow-matching output is not epsilon and cannot use the VP/Karras noise path."
            )
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
        extra_cond_kwargs: ModelConditioningKwargs = None,
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
        extra_cond_kwargs: ModelConditioningKwargs = None,
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
            "model_input_preconditioning": self._model_input_preconditioning_label(),
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
