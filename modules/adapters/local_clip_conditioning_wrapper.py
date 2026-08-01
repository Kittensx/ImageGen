from __future__ import annotations

import torch


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
