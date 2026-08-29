from __future__ import annotations

import torch

from image_gen.contracts.model_conditioning import SemanticConditioningCapabilities


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
            "semantic_conditioning_capabilities": self.semantic_conditioning_capabilities().to_dict(),
        }
