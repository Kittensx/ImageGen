from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class SD3ConversionReport:
    component: str
    backend: str
    source_key_count: int
    converted_key_count: int
    synthesized_keys: tuple[str, ...] = ()
    unconsumed_source_keys: tuple[str, ...] = ()
    source_key_samples: tuple[str, ...] = ()
    converted_key_samples: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "backend": self.backend,
            "source_key_count": self.source_key_count,
            "converted_key_count": self.converted_key_count,
            "synthesized_key_count": len(self.synthesized_keys),
            "synthesized_keys": list(self.synthesized_keys),
            "unconsumed_source_key_count": len(self.unconsumed_source_keys),
            "unconsumed_source_keys": list(self.unconsumed_source_keys[:50]),
            "source_key_samples": list(self.source_key_samples),
            "converted_key_samples": list(self.converted_key_samples),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SD3ConvertedState:
    state_dict: dict[str, Any]
    report: SD3ConversionReport


class SD3StateDictConverter:
    """Convert Phase-01-proven SD3 single-file component namespaces.

    IMAGE_GEN owns routing and validation. For the MMDiT and LDM VAE key
    translation, this adapter deliberately reuses the already-installed
    Diffusers conversion implementation instead of maintaining a second large,
    drift-prone mapping table.
    """

    @staticmethod
    def _single_file_utils() -> Any:
        try:
            from diffusers.loaders import single_file_utils
        except Exception as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError(
                "SD3 component conversion requires the installed Diffusers single-file utilities."
            ) from exc
        return single_file_utils

    @staticmethod
    def _samples(keys: Any, limit: int = 16) -> tuple[str, ...]:
        return tuple(sorted(str(key) for key in keys)[:limit])

    def convert_transformer(
        self,
        state_dict: Mapping[str, Any],
        transformer_config: Mapping[str, Any],
    ) -> SD3ConvertedState:
        if not state_dict:
            raise ValueError("SD3 transformer conversion received an empty state dict.")

        utils = self._single_file_utils()
        converter = getattr(utils, "convert_sd3_transformer_checkpoint_to_diffusers", None)
        if converter is None:
            raise RuntimeError(
                "Installed Diffusers does not expose convert_sd3_transformer_checkpoint_to_diffusers."
            )

        signature = inspect.signature(converter)
        parameters = list(signature.parameters.values())
        if not parameters:
            raise RuntimeError("Unexpected Diffusers SD3 converter signature with no checkpoint parameter.")

        available = {
            "num_layers": int(transformer_config.get("num_layers", 0)),
            "caption_projection_dim": int(transformer_config.get("caption_projection_dim", 0)),
            "dual_attention_layers": tuple(transformer_config.get("dual_attention_layers") or ()),
            "has_qk_norm": bool(transformer_config.get("qk_norm")),
        }
        kwargs: dict[str, Any] = {}
        for parameter in parameters[1:]:
            if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            if parameter.name in available:
                kwargs[parameter.name] = available[parameter.name]
            elif parameter.default is inspect.Parameter.empty:
                raise RuntimeError(
                    "Unsupported required parameter in the installed Diffusers SD3 converter: "
                    f"{parameter.name!r}; signature={signature}"
                )

        source = dict(state_dict)
        converted = converter(source, **kwargs)
        if not isinstance(converted, Mapping) or not converted:
            raise RuntimeError("Diffusers SD3 transformer conversion returned no state dict.")
        converted_dict = dict(converted)
        return SD3ConvertedState(
            state_dict=converted_dict,
            report=SD3ConversionReport(
                component="transformer",
                backend="diffusers.convert_sd3_transformer_checkpoint_to_diffusers",
                source_key_count=len(state_dict),
                converted_key_count=len(converted_dict),
                unconsumed_source_keys=tuple(sorted(str(key) for key in source.keys())),
                source_key_samples=self._samples(state_dict),
                converted_key_samples=self._samples(converted_dict),
                notes=(
                    "Source keys are already stripped of model.diffusion_model by StateDictMapper.",
                ),
            ),
        )

    def convert_vae(
        self,
        state_dict: Mapping[str, Any],
        vae_config: Mapping[str, Any] | Any,
    ) -> SD3ConvertedState:
        if not state_dict:
            raise ValueError("SD3 VAE conversion received an empty state dict.")
        utils = self._single_file_utils()
        converter = getattr(utils, "convert_ldm_vae_checkpoint", None)
        if converter is None:
            raise RuntimeError("Installed Diffusers does not expose convert_ldm_vae_checkpoint.")

        # Diffusers' LDM VAE converter consumes the original first_stage_model
        # namespace, while IMAGE_GEN routes that prefix away before conversion.
        original_namespace = {f"first_stage_model.{key}": value for key, value in state_dict.items()}
        converted = converter(dict(original_namespace), vae_config)
        if not isinstance(converted, Mapping) or not converted:
            raise RuntimeError("Diffusers LDM VAE conversion returned no state dict.")
        converted_dict = dict(converted)
        return SD3ConvertedState(
            state_dict=converted_dict,
            report=SD3ConversionReport(
                component="vae",
                backend="diffusers.convert_ldm_vae_checkpoint",
                source_key_count=len(state_dict),
                converted_key_count=len(converted_dict),
                source_key_samples=self._samples(state_dict),
                converted_key_samples=self._samples(converted_dict),
                notes=("first_stage_model prefix restored only for the Diffusers conversion call.",),
            ),
        )

    def convert_clip_l(
        self,
        state_dict: Mapping[str, Any],
        text_config: Mapping[str, Any],
    ) -> SD3ConvertedState:
        if not state_dict:
            raise ValueError("SD3 CLIP-L conversion received an empty state dict.")
        converted = dict(state_dict)
        synthesized: list[str] = []

        # The original SD3 CLIP-L package contains the CLIP text-transformer
        # weights but not a learned projection. Diffusers' SD3 contract uses
        # CLIPTextModelWithProjection and initializes the added projection as a
        # diagonal identity matrix. Make that step explicit and auditable.
        if "text_projection.weight" not in converted:
            hidden_size = int(text_config.get("hidden_size", 0))
            projection_dim = int(text_config.get("projection_dim", hidden_size))
            if hidden_size <= 0 or projection_dim <= 0:
                raise ValueError("SD3 CLIP-L config is missing hidden_size/projection_dim.")
            if hidden_size != projection_dim:
                raise ValueError(
                    "SD3 CLIP-L identity projection requires hidden_size == projection_dim; "
                    f"got hidden_size={hidden_size}, projection_dim={projection_dim}."
                )
            reference = next(
                (value for value in converted.values() if isinstance(value, torch.Tensor) and value.is_floating_point()),
                None,
            )
            dtype = reference.dtype if reference is not None else torch.float32
            device = reference.device if reference is not None else torch.device("cpu")
            converted["text_projection.weight"] = torch.eye(
                projection_dim,
                hidden_size,
                dtype=dtype,
                device=device,
            )
            synthesized.append("text_projection.weight")

        return SD3ConvertedState(
            state_dict=converted,
            report=SD3ConversionReport(
                component="clip_l",
                backend="image_gen.sd3_clip_l_identity_projection",
                source_key_count=len(state_dict),
                converted_key_count=len(converted),
                synthesized_keys=tuple(synthesized),
                source_key_samples=self._samples(state_dict),
                converted_key_samples=self._samples(converted),
                notes=(
                    "Embedded SD3 CLIP-L keys are already Transformers-layout keys after routing.",
                    "Missing CLIP-L text_projection is explicitly synthesized as identity for the SD3 Diffusers contract.",
                ),
            ),
        )

    def convert_clip_g(self, state_dict: Mapping[str, Any]) -> SD3ConvertedState:
        if not state_dict:
            raise ValueError("SD3 CLIP-G conversion received an empty state dict.")
        converted = dict(state_dict)
        return SD3ConvertedState(
            state_dict=converted,
            report=SD3ConversionReport(
                component="clip_g",
                backend="image_gen.sd3_clip_g_identity_mapping",
                source_key_count=len(state_dict),
                converted_key_count=len(converted),
                source_key_samples=self._samples(state_dict),
                converted_key_samples=self._samples(converted),
                notes=("Embedded SD3 CLIP-G keys already match the Transformers target layout after routing.",),
            ),
        )
