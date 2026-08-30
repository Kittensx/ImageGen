from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from image_gen.contracts.model_conditioning import SemanticConditioningCapabilities
from safetensors import safe_open

from modules.sd2_openclip_reference_converter import SD2OpenCLIPReferenceConverter
from modules.adapters.a1111_clip_conditioning import A1111PromptCapabilities, encode_a1111_clip_batch


_KNOWN_REFERENCE_BUFFER_KEYS = (
    "text_model.embeddings.position_ids",
)


def _drop_known_reference_buffers(state: dict[str, torch.Tensor]) -> tuple[str, ...]:
    """Remove only non-parameter HF buffers that modern Transformers recreates."""
    removed: list[str] = []
    for key in _KNOWN_REFERENCE_BUFFER_KEYS:
        if key in state:
            state.pop(key)
            removed.append(key)
    return tuple(removed)


@dataclass(frozen=True)
class SD2ConditioningResult:
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    embeddings: torch.Tensor

    def summary(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "input_ids_shape": list(self.input_ids.shape),
            "attention_mask_shape": list(self.attention_mask.shape) if self.attention_mask is not None else None,
            "embeddings_shape": list(self.embeddings.shape),
            "embeddings_dtype": str(self.embeddings.dtype),
            "embeddings_device": str(self.embeddings.device),
        }


class SD2OpenCLIPConditioningRuntime:
    """Standalone SD2/OpenCLIP text-conditioning runtime.

    The class intentionally does not own UNet or sampler behavior. It exists so
    tokenizer/text-encoder execution can be qualified independently before SD2
    generation is enabled in the main pipeline.
    """

    def __init__(self, *, tokenizer: Any, text_encoder: Any) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    @staticmethod
    def load_checkpoint_text_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        state: dict[str, torch.Tensor] = {}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for full_key in handle.keys():
                if not full_key.startswith("cond_stage_model."):
                    continue
                state[full_key[len("cond_stage_model."):]] = handle.get_tensor(full_key)
        if not state:
            raise ValueError(f"Checkpoint contains no cond_stage_model text-encoder tensors: {checkpoint}")
        return state

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        tokenizer_dir: str | Path,
        text_encoder_config: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "SD2OpenCLIPConditioningRuntime":
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

        tokenizer_root = Path(tokenizer_dir).expanduser().resolve()
        config_path = Path(text_encoder_config).expanduser().resolve()

        tokenizer = CLIPTokenizer.from_pretrained(
            str(tokenizer_root),
            local_files_only=True,
        )
        config = CLIPTextConfig.from_json_file(str(config_path))
        if int(config.hidden_size) != 1024:
            raise ValueError(f"SD2 text encoder hidden_size must be 1024, got {config.hidden_size}")
        if int(config.num_hidden_layers) != 23:
            raise ValueError(
                "SD2 Hugging Face runtime text encoder must expose the 23-layer penultimate contract; "
                f"got num_hidden_layers={config.num_hidden_layers}"
            )

        model = CLIPTextModel(config)
        source = cls.load_checkpoint_text_state(checkpoint_path)
        converted = SD2OpenCLIPReferenceConverter().convert(source)
        load_result = model.load_state_dict(converted, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "SD2 checkpoint-derived text encoder did not load strictly: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        model.eval()
        model.to(device=device, dtype=dtype)
        return cls(tokenizer=tokenizer, text_encoder=model)

    @classmethod
    def from_reference(
        cls,
        *,
        tokenizer_dir: str | Path,
        text_encoder_config: str | Path,
        text_encoder_weights: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "SD2OpenCLIPConditioningRuntime":
        """Build the retained HF reference encoder from split config + safetensors.

        Runtime assets intentionally contain only lightweight configuration and
        tokenizer files; heavyweight reference weights live under model_tooling.
        This loader keeps those responsibilities separate instead of requiring a
        Hugging Face-style directory with config and weights side by side.
        """
        from safetensors.torch import load_file
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            str(Path(tokenizer_dir).expanduser().resolve()),
            local_files_only=True,
        )
        config_path = Path(text_encoder_config).expanduser().resolve()
        weights_path = Path(text_encoder_weights).expanduser().resolve()
        config = CLIPTextConfig.from_json_file(str(config_path))
        model = CLIPTextModel(config)
        state = load_file(str(weights_path), device="cpu")
        # Older Diffusers/Transformers exports may persist CLIP position_ids as a
        # buffer.  Current Transformers reconstructs that buffer internally and
        # does not expose it as a loadable state-dict key.  Ignore exactly this
        # known non-parameter buffer; all learned parameters still load strictly.
        _drop_known_reference_buffers(state)
        load_result = model.load_state_dict(state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "SD2 reference text encoder did not load strictly: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        model.eval()
        model.to(device=device, dtype=dtype)
        return cls(tokenizer=tokenizer, text_encoder=model)

    def encode_batch(self, prompts: list[str] | tuple[str, ...]) -> torch.Tensor:
        texts = [str(value or "") for value in prompts]
        max_length = int(getattr(self.tokenizer, "model_max_length", 77) or 77)
        if max_length <= 0 or max_length > 4096:
            max_length = 77
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        device = next(self.text_encoder.parameters()).device
        input_ids = tokens.input_ids.to(device)
        attention_mask = getattr(tokens, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        kwargs: dict[str, Any] = {}
        if bool(getattr(self.text_encoder.config, "use_attention_mask", False)) and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            outputs = self.text_encoder(input_ids=input_ids, **kwargs)
        embeddings = outputs.last_hidden_state
        if embeddings.ndim != 3 or int(embeddings.shape[-1]) != 1024:
            raise RuntimeError(
                "SD2 conditioning must produce [batch, tokens, 1024] hidden states; "
                f"got {tuple(embeddings.shape)}"
            )
        if int(embeddings.shape[1]) != 77:
            raise RuntimeError(
                "SD2 conditioning must preserve the 77-token OpenCLIP context; "
                f"got {tuple(embeddings.shape)}"
            )
        return embeddings

    def get_learned_conditioning(self, texts):
        return self.encode_batch(list(texts))

    def encode_a1111_conditioning(self, texts, *, forced_segments_by_prompt=None):
        return encode_a1111_clip_batch(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            prompts=list(texts),
            hidden_state_index=-2,
            forced_segments_by_prompt=forced_segments_by_prompt,
        )

    def a1111_prompt_capabilities(self) -> A1111PromptCapabilities:
        return A1111PromptCapabilities(
            architecture="sd2.x",
            attention=True,
            composable_and=True,
            schedules=True,
            alternation=True,
            chunk_break=True,
            long_clip_chunking=True,
            clip_streams=("openclip_h",),
        )

    def encode(self, prompt: str) -> SD2ConditioningResult:
        max_length = int(getattr(self.tokenizer, "model_max_length", 77) or 77)
        if max_length <= 0 or max_length > 4096:
            max_length = 77
        tokens = self.tokenizer(
            str(prompt or ""),
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        device = next(self.text_encoder.parameters()).device
        input_ids = tokens.input_ids.to(device)
        attention_mask = getattr(tokens, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        kwargs: dict[str, Any] = {}
        if bool(getattr(self.text_encoder.config, "use_attention_mask", False)) and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            outputs = self.text_encoder(input_ids=input_ids, **kwargs)
        embeddings = outputs.last_hidden_state
        if embeddings.ndim != 3 or int(embeddings.shape[-1]) != 1024:
            raise RuntimeError(
                "SD2 conditioning must produce [batch, tokens, 1024] hidden states; "
                f"got {tuple(embeddings.shape)}"
            )
        return SD2ConditioningResult(
            prompt=str(prompt or ""),
            input_ids=input_ids,
            attention_mask=attention_mask,
            embeddings=embeddings,
        )
    def semantic_conditioning_capabilities(self) -> SemanticConditioningCapabilities:
        return SemanticConditioningCapabilities(
            architecture="sd2.x",
            runtime_name=type(self).__name__,
            output_kind="tensor",
            composable_fields=("cross_attention",),
            required_fields=("cross_attention",),
        )

    def contract_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "sd2.x",
            "encoder_mode": "openclip_h_14",
            "a1111_prompt_capabilities": self.a1111_prompt_capabilities().to_dict(),
            "semantic_conditioning_capabilities": self.semantic_conditioning_capabilities().to_dict(),
        }

