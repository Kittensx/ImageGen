from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class SDXLConditioningBatch:
    prompts: tuple[str, ...]
    cross_attention: torch.Tensor
    pooled: torch.Tensor
    text_encoder_1_hidden: torch.Tensor
    text_encoder_2_hidden: torch.Tensor

    def summary(self) -> dict[str, Any]:
        return {
            "batch_size": len(self.prompts),
            "cross_attention_shape": list(self.cross_attention.shape),
            "pooled_shape": list(self.pooled.shape),
            "text_encoder_1_hidden_shape": list(self.text_encoder_1_hidden.shape),
            "text_encoder_2_hidden_shape": list(self.text_encoder_2_hidden.shape),
            "dtype": str(self.cross_attention.dtype),
            "device": str(self.cross_attention.device),
        }


class SDXLConditioningRuntime:
    """Dual-tokenizer / dual-text-encoder SDXL conditioning runtime.

    This runtime intentionally owns prompt encoding only. SDXL time IDs and
    branch-aware UNet ``added_cond_kwargs`` are owned by the pipeline model-
    conditioning layer. The returned cross-attention tensor follows the SDXL Base contract:
    768-wide CLIP-L penultimate hidden states concatenated with 1280-wide
    OpenCLIP-G penultimate hidden states, for a 2048-wide context. The pooled
    1280-wide projection is taken from Text Encoder 2.
    """

    CONTEXT_LENGTH = 77
    TE1_HIDDEN_SIZE = 768
    TE2_HIDDEN_SIZE = 1280
    CROSS_ATTENTION_DIM = 2048
    POOLED_DIM = 1280

    def __init__(
        self,
        *,
        tokenizer: Any,
        tokenizer_2: Any,
        text_encoder: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
    ) -> None:
        if tokenizer is None or tokenizer_2 is None:
            raise ValueError("SDXL conditioning requires tokenizer and tokenizer_2.")
        if text_encoder is None or text_encoder_2 is None:
            raise ValueError("SDXL conditioning requires text_encoder and text_encoder_2.")

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self._validate_component_contract()

    @staticmethod
    def _config_int(component: Any, field: str) -> int:
        config = getattr(component, "config", None)
        return int(getattr(config, field, 0) or 0)

    def _validate_component_contract(self) -> None:
        te1_hidden = self._config_int(self.text_encoder, "hidden_size")
        te2_hidden = self._config_int(self.text_encoder_2, "hidden_size")
        projection_dim = self._config_int(self.text_encoder_2, "projection_dim")

        if te1_hidden != self.TE1_HIDDEN_SIZE:
            raise ValueError(
                "SDXL Text Encoder 1 must use hidden_size=768; "
                f"got {te1_hidden}."
            )
        if te2_hidden != self.TE2_HIDDEN_SIZE:
            raise ValueError(
                "SDXL Text Encoder 2 must use hidden_size=1280; "
                f"got {te2_hidden}."
            )
        if projection_dim != self.POOLED_DIM:
            raise ValueError(
                "SDXL Text Encoder 2 must use projection_dim=1280; "
                f"got {projection_dim}."
            )

        for label, tokenizer in (("tokenizer", self.tokenizer), ("tokenizer_2", self.tokenizer_2)):
            max_length = int(getattr(tokenizer, "model_max_length", self.CONTEXT_LENGTH) or self.CONTEXT_LENGTH)
            if max_length != self.CONTEXT_LENGTH:
                raise ValueError(
                    f"SDXL {label} must expose model_max_length=77; got {max_length}."
                )

    @staticmethod
    def _device(component: torch.nn.Module) -> torch.device:
        try:
            return next(component.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _tokenizer_batch(tokenizer: Any, prompts: list[str]) -> Any:
        return tokenizer(
            prompts,
            padding="max_length",
            max_length=SDXLConditioningRuntime.CONTEXT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

    @staticmethod
    def _model_kwargs(component: torch.nn.Module, encoded: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "input_ids": encoded.input_ids.to(SDXLConditioningRuntime._device(component)),
            "output_hidden_states": True,
        }
        config = getattr(component, "config", None)
        attention_mask = getattr(encoded, "attention_mask", None)
        if bool(getattr(config, "use_attention_mask", False)) and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(SDXLConditioningRuntime._device(component))
        return kwargs

    @staticmethod
    def _penultimate_hidden(outputs: Any, *, label: str) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) < 2:
            raise RuntimeError(f"{label} did not return hidden_states required for SDXL conditioning.")
        value = hidden_states[-2]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{label} penultimate hidden state is not a tensor.")
        return value

    @staticmethod
    def _pooled_projection(outputs: Any) -> torch.Tensor:
        pooled = getattr(outputs, "text_embeds", None)
        if pooled is None:
            try:
                candidate = outputs[0]
            except (TypeError, IndexError, KeyError):
                candidate = None
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
                pooled = candidate
        if not isinstance(pooled, torch.Tensor):
            raise RuntimeError(
                "SDXL Text Encoder 2 did not return the pooled text projection required by SDXL."
            )
        return pooled

    @staticmethod
    def _validate_shape(tensor: torch.Tensor, expected: tuple[int | None, ...], *, label: str) -> None:
        if tensor.ndim != len(expected):
            raise RuntimeError(f"{label} has shape {tuple(tensor.shape)}; expected rank {len(expected)}.")
        for index, expected_dim in enumerate(expected):
            if expected_dim is not None and int(tensor.shape[index]) != int(expected_dim):
                raise RuntimeError(
                    f"{label} has shape {tuple(tensor.shape)}; expected dimension {index}={expected_dim}."
                )

    def encode_batch_result(self, prompts: Iterable[str]) -> SDXLConditioningBatch:
        texts = [str(value or "") for value in prompts]
        if not texts:
            raise ValueError("SDXL conditioning requires at least one prompt.")

        tokens_1 = self._tokenizer_batch(self.tokenizer, texts)
        tokens_2 = self._tokenizer_batch(self.tokenizer_2, texts)

        with torch.inference_mode():
            outputs_1 = self.text_encoder(**self._model_kwargs(self.text_encoder, tokens_1))
            outputs_2 = self.text_encoder_2(**self._model_kwargs(self.text_encoder_2, tokens_2))

        hidden_1 = self._penultimate_hidden(outputs_1, label="SDXL Text Encoder 1")
        hidden_2 = self._penultimate_hidden(outputs_2, label="SDXL Text Encoder 2")
        pooled = self._pooled_projection(outputs_2)

        batch_size = len(texts)
        self._validate_shape(
            hidden_1,
            (batch_size, self.CONTEXT_LENGTH, self.TE1_HIDDEN_SIZE),
            label="SDXL Text Encoder 1 hidden states",
        )
        self._validate_shape(
            hidden_2,
            (batch_size, self.CONTEXT_LENGTH, self.TE2_HIDDEN_SIZE),
            label="SDXL Text Encoder 2 hidden states",
        )
        self._validate_shape(
            pooled,
            (batch_size, self.POOLED_DIM),
            label="SDXL pooled projection",
        )

        # Text Encoder 2 is the canonical dtype/device owner for SDXL prompt
        # conditioning. This mirrors the downstream UNet context contract and
        # avoids a mixed-device concat if component placement differs.
        hidden_1 = hidden_1.to(device=hidden_2.device, dtype=hidden_2.dtype)
        pooled = pooled.to(device=hidden_2.device, dtype=hidden_2.dtype)
        cross_attention = torch.cat([hidden_1, hidden_2], dim=-1)
        self._validate_shape(
            cross_attention,
            (batch_size, self.CONTEXT_LENGTH, self.CROSS_ATTENTION_DIM),
            label="SDXL combined cross-attention conditioning",
        )

        if not torch.isfinite(cross_attention).all():
            raise RuntimeError("SDXL combined cross-attention conditioning contains NaN or Inf values.")
        if not torch.isfinite(pooled).all():
            raise RuntimeError("SDXL pooled conditioning contains NaN or Inf values.")

        return SDXLConditioningBatch(
            prompts=tuple(texts),
            cross_attention=cross_attention,
            pooled=pooled,
            text_encoder_1_hidden=hidden_1,
            text_encoder_2_hidden=hidden_2,
        )

    def encode_batch(self, prompts: Iterable[str]) -> dict[str, torch.Tensor]:
        result = self.encode_batch_result(prompts)
        # Dict output is intentional: PromptParserClass already preserves dict
        # values per scheduled prompt. StepConditioningResolver consumes the two
        # fields independently so prompt schedules/composable weights apply to
        # pooled and token conditioning together.
        return {
            "cross_attention": result.cross_attention,
            "pooled": result.pooled,
        }

    def get_learned_conditioning(self, texts: Iterable[str]) -> dict[str, torch.Tensor]:
        return self.encode_batch(texts)

    def encode(self, texts: Iterable[str]) -> dict[str, torch.Tensor]:
        return self.encode_batch(texts)
