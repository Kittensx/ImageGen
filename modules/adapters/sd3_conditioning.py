from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import warnings

import torch

from image_gen.contracts.model_conditioning import SemanticConditioningCapabilities
from modules.adapters.a1111_clip_conditioning import A1111PromptCapabilities, encode_a1111_clip_batch
from modules.prompt_parsers.a1111_semantics import a1111_plain_text
import torch.nn.functional as F


@dataclass(frozen=True)
class SD3ConditioningBatch:
    prompts: tuple[str, ...]
    prompts_2: tuple[str, ...]
    encoder_hidden_states: torch.Tensor
    pooled_projections: torch.Tensor
    clip_l_hidden: torch.Tensor
    clip_g_hidden: torch.Tensor
    clip_l_pooled: torch.Tensor
    clip_g_pooled: torch.Tensor
    zero_t5_hidden: torch.Tensor
    t5_hidden: torch.Tensor | None = None
    t5_enabled: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "batch_size": len(self.prompts),
            "encoder_hidden_states_shape": list(self.encoder_hidden_states.shape),
            "pooled_projections_shape": list(self.pooled_projections.shape),
            "clip_l_hidden_shape": list(self.clip_l_hidden.shape),
            "clip_g_hidden_shape": list(self.clip_g_hidden.shape),
            "clip_l_pooled_shape": list(self.clip_l_pooled.shape),
            "clip_g_pooled_shape": list(self.clip_g_pooled.shape),
            "zero_t5_hidden_shape": list(self.zero_t5_hidden.shape),
            "t5_hidden_shape": list(self.t5_hidden.shape) if self.t5_hidden is not None else None,
            "t5_enabled": bool(self.t5_enabled),
            "dtype": str(self.encoder_hidden_states.dtype),
            "device": str(self.encoder_hidden_states.device),
        }


class SD3ConditioningRuntime:
    """CLIP-L + CLIP-G conditioning for the initial SD3 no-T5 runtime.

    This follows the Diffusers StableDiffusion3Pipeline conditioning contract:

    1. Encode CLIP-L and CLIP-G with 77-token CLIP tokenizers.
    2. Use the penultimate hidden state from each encoder.
    3. Concatenate 768 + 1280 -> 2048 sequence features.
    4. Zero-pad those sequence features to the transformer's 4096-wide joint
       attention dimension.
    5. When T5 is disabled, append a deterministic zero sequence with the same
       width. The current SD3 reference length is 256 positions, producing
       77 + 256 = 333 sequence positions.
    6. Concatenate CLIP-L and CLIP-G pooled projections -> 2048.

    The no-T5 zeros are a contract-preserving absence representation, not a
    learned or fabricated T5 embedding.
    """

    CONTEXT_LENGTH = 77
    CLIP_L_HIDDEN_SIZE = 768
    CLIP_G_HIDDEN_SIZE = 1280
    CLIP_COMBINED_WIDTH = 2048
    JOINT_ATTENTION_DIM = 4096
    POOLED_DIM = 2048
    DEFAULT_T5_SEQUENCE_LENGTH = 256

    def __init__(
        self,
        *,
        tokenizer: Any,
        tokenizer_2: Any,
        text_encoder: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        text_encoder_3: torch.nn.Module | None = None,
        tokenizer_3: Any | None = None,
        joint_attention_dim: int = JOINT_ATTENTION_DIM,
        t5_sequence_length: int = DEFAULT_T5_SEQUENCE_LENGTH,
    ) -> None:
        if tokenizer is None or tokenizer_2 is None:
            raise ValueError("SD3 CLIP-only conditioning requires tokenizer and tokenizer_2.")
        if text_encoder is None or text_encoder_2 is None:
            raise ValueError("SD3 CLIP-only conditioning requires CLIP-L and CLIP-G text encoders.")

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer_3 = tokenizer_3
        if (self.text_encoder_3 is None) != (self.tokenizer_3 is None):
            raise ValueError("SD3 T5 conditioning requires text_encoder_3 and tokenizer_3 together.")
        self.joint_attention_dim = int(joint_attention_dim)
        self.t5_sequence_length = int(t5_sequence_length)
        self._validate_component_contract()

    @staticmethod
    def _config_int(component: Any, field: str) -> int:
        config = getattr(component, "config", None)
        return int(getattr(config, field, 0) or 0)

    def _validate_component_contract(self) -> None:
        clip_l_hidden = self._config_int(self.text_encoder, "hidden_size")
        clip_g_hidden = self._config_int(self.text_encoder_2, "hidden_size")
        clip_l_projection = self._config_int(self.text_encoder, "projection_dim")
        clip_g_projection = self._config_int(self.text_encoder_2, "projection_dim")

        if clip_l_hidden != self.CLIP_L_HIDDEN_SIZE:
            raise ValueError(
                "SD3 CLIP-L must use hidden_size=768; "
                f"got {clip_l_hidden}."
            )
        if clip_g_hidden != self.CLIP_G_HIDDEN_SIZE:
            raise ValueError(
                "SD3 CLIP-G must use hidden_size=1280; "
                f"got {clip_g_hidden}."
            )
        if clip_l_projection != self.CLIP_L_HIDDEN_SIZE:
            raise ValueError(
                "SD3 CLIP-L must use projection_dim=768; "
                f"got {clip_l_projection}."
            )
        if clip_g_projection != self.CLIP_G_HIDDEN_SIZE:
            raise ValueError(
                "SD3 CLIP-G must use projection_dim=1280; "
                f"got {clip_g_projection}."
            )
        if self.joint_attention_dim < self.CLIP_COMBINED_WIDTH:
            raise ValueError(
                "SD3 joint_attention_dim must be at least 2048 for CLIP-L/G conditioning; "
                f"got {self.joint_attention_dim}."
            )
        if self.t5_sequence_length < 0:
            raise ValueError("SD3 T5 replacement sequence length cannot be negative.")

        for label, tokenizer in (("tokenizer", self.tokenizer), ("tokenizer_2", self.tokenizer_2)):
            max_length = int(getattr(tokenizer, "model_max_length", self.CONTEXT_LENGTH) or self.CONTEXT_LENGTH)
            if max_length != self.CONTEXT_LENGTH:
                raise ValueError(
                    f"SD3 {label} must expose model_max_length=77; got {max_length}."
                )

    @staticmethod
    def _device(component: torch.nn.Module) -> torch.device:
        try:
            return next(component.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _dtype(component: torch.nn.Module) -> torch.dtype:
        try:
            return next(component.parameters()).dtype
        except StopIteration:
            return torch.float32

    @staticmethod
    def _normalize_prompts(prompts: Iterable[str]) -> list[str]:
        values = [str(value or "") for value in prompts]
        if not values:
            raise ValueError("SD3 conditioning requires at least one prompt.")
        return values

    def _tokenize(self, tokenizer: Any, prompts: list[str], *, label: str) -> Any:
        encoded = tokenizer(
            prompts,
            padding="max_length",
            max_length=self.CONTEXT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

        # Match the reference behavior by checking the untruncated tokenization
        # separately. This is diagnostic only and never changes conditioning.
        try:
            untruncated = tokenizer(prompts, padding="longest", return_tensors="pt")
            truncated_ids = getattr(encoded, "input_ids", None)
            untruncated_ids = getattr(untruncated, "input_ids", None)
            if (
                isinstance(truncated_ids, torch.Tensor)
                and isinstance(untruncated_ids, torch.Tensor)
                and untruncated_ids.shape[-1] >= truncated_ids.shape[-1]
                and not torch.equal(truncated_ids, untruncated_ids)
            ):
                removed_text = None
                batch_decode = getattr(tokenizer, "batch_decode", None)
                if callable(batch_decode):
                    try:
                        removed_text = batch_decode(
                            untruncated_ids[:, self.CONTEXT_LENGTH - 1 : -1]
                        )
                    except Exception:
                        removed_text = None
                suffix = f" Removed text: {removed_text}" if removed_text else ""
                warnings.warn(
                    f"SD3 {label} truncated prompt input because CLIP supports at most "
                    f"{self.CONTEXT_LENGTH} positions.{suffix}",
                    RuntimeWarning,
                    stacklevel=3,
                )
        except Exception:
            # Tokenizers with a reduced testing/fake interface may not support
            # the diagnostic untruncated call. Encoding remains authoritative.
            pass
        return encoded

    def _model_kwargs(self, component: torch.nn.Module, encoded: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "input_ids": encoded.input_ids.to(self._device(component)),
            "output_hidden_states": True,
        }
        config = getattr(component, "config", None)
        attention_mask = getattr(encoded, "attention_mask", None)
        if bool(getattr(config, "use_attention_mask", False)) and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self._device(component))
        return kwargs

    @staticmethod
    def _penultimate_hidden(outputs: Any, *, label: str) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) < 2:
            raise RuntimeError(f"{label} did not return hidden_states required for SD3 conditioning.")
        value = hidden_states[-2]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{label} penultimate hidden state is not a tensor.")
        return value

    @staticmethod
    def _pooled_projection(outputs: Any, *, label: str) -> torch.Tensor:
        pooled = getattr(outputs, "text_embeds", None)
        if pooled is None:
            try:
                candidate = outputs[0]
            except (TypeError, IndexError, KeyError):
                candidate = None
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
                pooled = candidate
        if not isinstance(pooled, torch.Tensor):
            raise RuntimeError(f"{label} did not return the pooled text projection required by SD3.")
        return pooled

    @staticmethod
    def _validate_shape(tensor: torch.Tensor, expected: Sequence[int | None], *, label: str) -> None:
        if tensor.ndim != len(expected):
            raise RuntimeError(f"{label} has shape {tuple(tensor.shape)}; expected rank {len(expected)}.")
        for index, expected_dim in enumerate(expected):
            if expected_dim is not None and int(tensor.shape[index]) != int(expected_dim):
                raise RuntimeError(
                    f"{label} has shape {tuple(tensor.shape)}; expected dimension {index}={expected_dim}."
                )

    def encode_batch_result(
        self,
        prompts: Iterable[str],
        *,
        prompts_2: Iterable[str] | None = None,
    ) -> SD3ConditioningBatch:
        texts = self._normalize_prompts(prompts)
        texts_2 = self._normalize_prompts(prompts_2 if prompts_2 is not None else texts)
        if len(texts_2) != len(texts):
            raise ValueError(
                "SD3 CLIP-L and CLIP-G prompt batches must have equal size; "
                f"got {len(texts)} and {len(texts_2)}."
            )

        tokens_l = self._tokenize(self.tokenizer, texts, label="CLIP-L")
        tokens_g = self._tokenize(self.tokenizer_2, texts_2, label="CLIP-G")

        with torch.inference_mode():
            outputs_l = self.text_encoder(**self._model_kwargs(self.text_encoder, tokens_l))
            outputs_g = self.text_encoder_2(**self._model_kwargs(self.text_encoder_2, tokens_g))

        hidden_l = self._penultimate_hidden(outputs_l, label="SD3 CLIP-L")
        hidden_g = self._penultimate_hidden(outputs_g, label="SD3 CLIP-G")
        pooled_l = self._pooled_projection(outputs_l, label="SD3 CLIP-L")
        pooled_g = self._pooled_projection(outputs_g, label="SD3 CLIP-G")

        batch_size = len(texts)
        self._validate_shape(
            hidden_l,
            (batch_size, self.CONTEXT_LENGTH, self.CLIP_L_HIDDEN_SIZE),
            label="SD3 CLIP-L hidden states",
        )
        self._validate_shape(
            hidden_g,
            (batch_size, self.CONTEXT_LENGTH, self.CLIP_G_HIDDEN_SIZE),
            label="SD3 CLIP-G hidden states",
        )
        self._validate_shape(
            pooled_l,
            (batch_size, self.CLIP_L_HIDDEN_SIZE),
            label="SD3 CLIP-L pooled projection",
        )
        self._validate_shape(
            pooled_g,
            (batch_size, self.CLIP_G_HIDDEN_SIZE),
            label="SD3 CLIP-G pooled projection",
        )

        # Diffusers' SD3 reference uses CLIP-L's dtype for the combined CLIP
        # sequence. Components are expected to share an execution device in this
        # phase; later memory-residency work can stage them sequentially.
        target_device = hidden_g.device
        target_dtype = self._dtype(self.text_encoder)
        hidden_l = hidden_l.to(device=target_device, dtype=target_dtype)
        hidden_g = hidden_g.to(device=target_device, dtype=target_dtype)
        pooled_l = pooled_l.to(device=target_device, dtype=target_dtype)
        pooled_g = pooled_g.to(device=target_device, dtype=target_dtype)

        clip_hidden = torch.cat([hidden_l, hidden_g], dim=-1)
        self._validate_shape(
            clip_hidden,
            (batch_size, self.CONTEXT_LENGTH, self.CLIP_COMBINED_WIDTH),
            label="SD3 combined CLIP sequence conditioning",
        )
        clip_hidden = F.pad(
            clip_hidden,
            (0, self.joint_attention_dim - self.CLIP_COMBINED_WIDTH),
        )

        zero_t5_hidden = torch.zeros(
            (batch_size, self.t5_sequence_length, self.joint_attention_dim),
            device=target_device,
            dtype=target_dtype,
        )
        t5_hidden = None
        if self.text_encoder_3 is not None and self.tokenizer_3 is not None:
            t5_tokens = self.tokenizer_3(
                texts,
                padding="max_length",
                max_length=self.t5_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            t5_device = self._device(self.text_encoder_3)
            t5_kwargs: dict[str, Any] = {
                "input_ids": t5_tokens.input_ids.to(t5_device),
                "return_dict": False,
            }
            t5_attention_mask = getattr(t5_tokens, "attention_mask", None)
            if t5_attention_mask is not None:
                t5_kwargs["attention_mask"] = t5_attention_mask.to(t5_device)
            with torch.inference_mode():
                t5_output = self.text_encoder_3(**t5_kwargs)[0]
            self._validate_shape(
                t5_output,
                (batch_size, self.t5_sequence_length, self.joint_attention_dim),
                label="SD3 T5 hidden states",
            )
            if not torch.isfinite(t5_output).all():
                raise RuntimeError("SD3 T5 hidden states contain NaN or Inf values.")
            t5_hidden = t5_output.to(device=target_device, dtype=target_dtype)
            t5_sequence = t5_hidden
        else:
            t5_sequence = zero_t5_hidden
        encoder_hidden_states = torch.cat([clip_hidden, t5_sequence], dim=-2)
        pooled = torch.cat([pooled_l, pooled_g], dim=-1)

        self._validate_shape(
            encoder_hidden_states,
            (batch_size, self.CONTEXT_LENGTH + self.t5_sequence_length, self.joint_attention_dim),
            label="SD3 no-T5 encoder hidden states",
        )
        self._validate_shape(
            pooled,
            (batch_size, self.POOLED_DIM),
            label="SD3 pooled projections",
        )

        if not torch.isfinite(encoder_hidden_states).all():
            raise RuntimeError("SD3 encoder hidden states contain NaN or Inf values.")
        if not torch.isfinite(pooled).all():
            raise RuntimeError("SD3 pooled projections contain NaN or Inf values.")

        return SD3ConditioningBatch(
            prompts=tuple(texts),
            prompts_2=tuple(texts_2),
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            clip_l_hidden=hidden_l,
            clip_g_hidden=hidden_g,
            clip_l_pooled=pooled_l,
            clip_g_pooled=pooled_g,
            zero_t5_hidden=zero_t5_hidden,
            t5_hidden=t5_hidden,
            t5_enabled=t5_hidden is not None,
        )

    def encode_batch(
        self,
        prompts: Iterable[str],
        *,
        prompts_2: Iterable[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        result = self.encode_batch_result(prompts, prompts_2=prompts_2)
        # Keep the structured keys aligned with the existing prompt parser and
        # StepConditioningResolver contract. For SD3, cross_attention is the
        # transformer's encoder_hidden_states and pooled is pooled_projections.
        return {
            "cross_attention": result.encoder_hidden_states,
            "pooled": result.pooled_projections,
        }

    def encode_a1111_conditioning(self, texts: Iterable[str], *, forced_segments_by_prompt=None) -> dict[str, torch.Tensor]:
        prompts = self._normalize_prompts(texts)
        forced = list(forced_segments_by_prompt or [None] * len(prompts))
        if len(forced) != len(prompts):
            raise ValueError("forced_segments_by_prompt must match the SD3 prompt batch length.")
        hidden_l = encode_a1111_clip_batch(
            tokenizer=self.tokenizer, text_encoder=self.text_encoder, prompts=prompts,
            hidden_state_index=-2, forced_segments_by_prompt=forced,
        )
        hidden_g = encode_a1111_clip_batch(
            tokenizer=self.tokenizer_2, text_encoder=self.text_encoder_2, prompts=prompts,
            hidden_state_index=-2, forced_segments_by_prompt=forced,
        )
        target_tokens = max(int(hidden_l.shape[1]), int(hidden_g.shape[1]))

        def pad_stream(value: torch.Tensor, *, tokenizer, text_encoder) -> torch.Tensor:
            missing = target_tokens - int(value.shape[1])
            if missing <= 0:
                return value
            if missing % self.CONTEXT_LENGTH:
                raise RuntimeError("SD3 A1111 CLIP streams are not aligned to 77-token chunks.")
            empty = encode_a1111_clip_batch(
                tokenizer=tokenizer, text_encoder=text_encoder, prompts=[""] * len(prompts),
                hidden_state_index=-2,
            )
            return torch.cat([value, empty.repeat(1, missing // self.CONTEXT_LENGTH, 1)], dim=1)

        hidden_l = pad_stream(hidden_l, tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        hidden_g = pad_stream(hidden_g, tokenizer=self.tokenizer_2, text_encoder=self.text_encoder_2)
        target_device = hidden_g.device
        target_dtype = self._dtype(self.text_encoder)
        hidden_l = hidden_l.to(device=target_device, dtype=target_dtype)
        hidden_g = hidden_g.to(device=target_device, dtype=target_dtype)
        clip_hidden = torch.cat([hidden_l, hidden_g], dim=-1)
        clip_hidden = F.pad(clip_hidden, (0, self.joint_attention_dim - self.CLIP_COMBINED_WIDTH))

        whole_prompts = []
        for prompt, segments in zip(prompts, forced):
            source = " ".join(str(item or "") for item in segments) if segments is not None else prompt
            whole_prompts.append(a1111_plain_text(source))
        whole = self.encode_batch_result(whole_prompts)
        t5_sequence = (
            whole.t5_hidden if whole.t5_hidden is not None else whole.zero_t5_hidden
        ).to(device=target_device, dtype=target_dtype)
        cross_attention = torch.cat([clip_hidden, t5_sequence], dim=1)
        pooled = whole.pooled_projections.to(device=target_device, dtype=target_dtype)
        if not torch.isfinite(cross_attention).all() or not torch.isfinite(pooled).all():
            raise RuntimeError("SD3 A1111 conditioning contains NaN or Inf values.")
        return {"cross_attention": cross_attention, "pooled": pooled}

    def a1111_prompt_capabilities(self) -> A1111PromptCapabilities:
        t5_enabled = self.text_encoder_3 is not None and self.tokenizer_3 is not None
        return A1111PromptCapabilities(
            architecture="sd3.x", attention=True, composable_and=True, schedules=True, alternation=True,
            chunk_break=True, long_clip_chunking=True, clip_streams=("clip_l", "clip_g"),
            non_clip_policy=("t5_whole_lowered_prompt_once" if t5_enabled else "zero_t5_sequence_once"),
            pooled_policy="whole_lowered_prompt_once",
        )

    def encode_chunk_break_conditioning(
        self,
        segments: Iterable[str],
        *,
        full_prompt: str,
    ) -> dict[str, torch.Tensor]:
        """Encode a forced BREAK without projecting CLIP chunk rules onto T5.

        CLIP-L and CLIP-G are the fixed-context streams, so each BREAK segment
        receives its own native 77-position CLIP encoding and the resulting
        joint-width CLIP chunks are concatenated. T5, when enabled, is encoded
        once from the complete lowered branch; when disabled, exactly one native
        zero-T5 replacement sequence is appended. The pooled CLIP-L/G projection
        likewise comes from one complete lowered-branch encode rather than from
        averaging segment pooled vectors.
        """

        texts = self._normalize_prompts(segments)
        if len(texts) < 2:
            raise ValueError("SD3 BREAK conditioning requires at least two non-empty segments.")
        whole_text = str(full_prompt or "").strip()
        if not whole_text:
            raise ValueError("SD3 BREAK conditioning requires non-empty full_prompt text.")

        segment_result = self.encode_batch_result(texts)
        whole_result = self.encode_batch_result([whole_text])

        clip_hidden = torch.cat(
            [segment_result.clip_l_hidden, segment_result.clip_g_hidden],
            dim=-1,
        )
        clip_hidden = F.pad(
            clip_hidden,
            (0, self.joint_attention_dim - self.CLIP_COMBINED_WIDTH),
        )
        forced_clip = torch.cat(
            [clip_hidden[index] for index in range(len(texts))],
            dim=0,
        )
        t5_sequence = (
            whole_result.t5_hidden[0]
            if whole_result.t5_hidden is not None
            else whole_result.zero_t5_hidden[0]
        )
        cross_attention = torch.cat([forced_clip, t5_sequence], dim=0)
        pooled = whole_result.pooled_projections[0]

        expected_sequence = len(texts) * self.CONTEXT_LENGTH + self.t5_sequence_length
        self._validate_shape(
            cross_attention,
            (expected_sequence, self.joint_attention_dim),
            label="SD3 BREAK encoder hidden states",
        )
        self._validate_shape(
            pooled,
            (self.POOLED_DIM,),
            label="SD3 BREAK pooled projections",
        )
        if not torch.isfinite(cross_attention).all():
            raise RuntimeError("SD3 BREAK encoder hidden states contain NaN or Inf values.")
        if not torch.isfinite(pooled).all():
            raise RuntimeError("SD3 BREAK pooled projections contain NaN or Inf values.")
        return {"cross_attention": cross_attention, "pooled": pooled}

    def get_learned_conditioning(self, texts: Iterable[str]) -> dict[str, torch.Tensor]:
        return self.encode_batch(texts)

    def encode(self, texts: Iterable[str]) -> dict[str, torch.Tensor]:
        return self.encode_batch(texts)

    def semantic_conditioning_capabilities(self) -> SemanticConditioningCapabilities:
        t5_enabled = self.text_encoder_3 is not None and self.tokenizer_3 is not None
        return SemanticConditioningCapabilities(
            architecture="sd3.x",
            runtime_name=type(self).__name__,
            output_kind="structured",
            composable_fields=("cross_attention", "pooled"),
            required_fields=("cross_attention", "pooled"),
            supports_pooled_conditioning=True,
            t5_policy=("enabled_same_branch_text" if t5_enabled else "disabled_zero_sequence"),
        )

    def contract_metadata(self) -> dict[str, Any]:
        t5_enabled = self.text_encoder_3 is not None and self.tokenizer_3 is not None
        return {
            "architecture": "sd3.x",
            "encoder_mode": "clip_l_clip_g_t5xxl" if t5_enabled else "clip_l_clip_g_no_t5",
            "a1111_prompt_capabilities": self.a1111_prompt_capabilities().to_dict(),
            "clip_context_length": self.CONTEXT_LENGTH,
            "clip_l_hidden_size": self.CLIP_L_HIDDEN_SIZE,
            "clip_g_hidden_size": self.CLIP_G_HIDDEN_SIZE,
            "clip_combined_width": self.CLIP_COMBINED_WIDTH,
            "joint_attention_dim": self.joint_attention_dim,
            "t5_enabled": bool(t5_enabled),
            "t5_replacement": None if t5_enabled else "zero_sequence",
            "t5_sequence_length": self.t5_sequence_length,
            "final_sequence_length": self.CONTEXT_LENGTH + self.t5_sequence_length,
            "pooled_projection_dim": self.POOLED_DIM,
            "chunk_break_policy": {
                "algorithm": "encoder_chunk_break_v1",
                "clip_streams": "forced_77_position_segments",
                "t5_stream": "whole_lowered_prompt_once",
                "pooled": "whole_lowered_prompt_once",
                "cfg_unequal_context": "sequential_branch_evaluation",
            },
            "semantic_conditioning_capabilities": self.semantic_conditioning_capabilities().to_dict(),
        }
