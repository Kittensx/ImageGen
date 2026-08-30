from __future__ import annotations

import torch

from image_gen.contracts.model_conditioning import SemanticConditioningCapabilities
from modules.adapters.a1111_clip_conditioning import A1111PromptCapabilities, encode_a1111_clip_batch


class LocalCLIPConditioningWrapper:
    def __init__(self, text_encoder, tokenizer, device, max_length: int = 77):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def encode(self, texts):
        prompts = list(texts)

        batch_encoding = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = batch_encoding["input_ids"].to(self.device)

        # keep attention_mask if tokenizer returns it
        model_kwargs = {"input_ids": input_ids}
        if "attention_mask" in batch_encoding:
            model_kwargs["attention_mask"] = batch_encoding["attention_mask"].to(self.device)

        outputs = self.text_encoder(**model_kwargs)
        return outputs.last_hidden_state
    def get_learned_conditioning(self, texts):
        return self.encode(texts)

    def encode_a1111_conditioning(self, texts, *, forced_segments_by_prompt=None):
        return encode_a1111_clip_batch(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            prompts=list(texts),
            hidden_state_index=None,
            forced_segments_by_prompt=forced_segments_by_prompt,
        )

    def a1111_prompt_capabilities(self) -> A1111PromptCapabilities:
        return A1111PromptCapabilities(
            architecture="sd1.x",
            attention=True,
            composable_and=True,
            schedules=True,
            alternation=True,
            chunk_break=True,
            long_clip_chunking=True,
            clip_streams=("clip",),
        )

    def semantic_conditioning_capabilities(self) -> SemanticConditioningCapabilities:
        return SemanticConditioningCapabilities(
            architecture="sd1.x",
            runtime_name=type(self).__name__,
            output_kind="tensor",
            composable_fields=("cross_attention",),
            required_fields=("cross_attention",),
        )

    def contract_metadata(self) -> dict:
        return {
            "architecture": "sd1.x",
            "a1111_prompt_capabilities": self.a1111_prompt_capabilities().to_dict(),
            "semantic_conditioning_capabilities": self.semantic_conditioning_capabilities().to_dict(),
        }
