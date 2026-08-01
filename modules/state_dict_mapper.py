from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MappedStateDict:
    unet: Dict[str, object] = field(default_factory=dict)
    vae: Dict[str, object] = field(default_factory=dict)
    text_encoder: Dict[str, object] = field(default_factory=dict)
    extras: Dict[str, object] = field(default_factory=dict)


class StateDictMapper:
    """
    Splits a monolithic A1111-style checkpoint into component-specific state dicts.
    """

    UNET_PREFIX = "model.diffusion_model."
    VAE_PREFIX = "first_stage_model."
    TEXT_PREFIX = "cond_stage_model."

    def split_checkpoint(self, state_dict: Dict[str, object]) -> MappedStateDict:
        mapped = MappedStateDict()

        for key, value in state_dict.items():
            if key.startswith(self.UNET_PREFIX):
                new_key = key[len(self.UNET_PREFIX):]
                mapped.unet[new_key] = value
            elif key.startswith(self.VAE_PREFIX):
                new_key = key[len(self.VAE_PREFIX):]
                mapped.vae[new_key] = value
            elif key.startswith(self.TEXT_PREFIX):
                new_key = key[len(self.TEXT_PREFIX):]
                mapped.text_encoder[new_key] = value
            else:
                mapped.extras[key] = value

        return mapped