import torch

class CLIPConditioningWrapper:
    def __init__(self, text_encoder, tokenizer, device):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device

    def get_learned_conditioning(self, texts):
        # texts is SdConditioning (list-like)
        prompts = list(texts)

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        outputs = self.text_encoder(**tokens)

        # A1111 expects per-prompt outputs
        return outputs.last_hidden_state