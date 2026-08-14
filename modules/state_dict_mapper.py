from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MappedStateDict:
    unet: Dict[str, object] = field(default_factory=dict)
    vae: Dict[str, object] = field(default_factory=dict)
    text_encoder: Dict[str, object] = field(default_factory=dict)
    text_encoder_2: Dict[str, object] = field(default_factory=dict)
    extras: Dict[str, object] = field(default_factory=dict)


class StateDictMapper:
    """
    Splits a monolithic A1111/LDM-style checkpoint into component-specific state dicts.

    SDXL monolithic checkpoints use two conditioner branches:
    - conditioner.embedders.0.* -> text_encoder
    - conditioner.embedders.1.* -> text_encoder_2

    Phase SDXL-03 intentionally stops at component separation. Text-encoder key conversion
    into Transformers-compatible layouts belongs to the following SDXL conversion/build phase.
    """

    UNET_PREFIX = "model.diffusion_model."
    VAE_PREFIX = "first_stage_model."
    TEXT_PREFIX = "cond_stage_model."
    SDXL_TEXT_ENCODER_1_PREFIX = "conditioner.embedders.0."
    SDXL_TEXT_ENCODER_2_PREFIX = "conditioner.embedders.1."

    def route_key(self, key: str) -> tuple[str, str]:
        """Return ``(component_name, stripped_key)`` for one checkpoint key.

        Unknown keys remain in ``extras`` with their original key preserved. Keeping routing
        logic in one place allows header-only validators to exercise the exact same mapping
        rules as full checkpoint loading without materializing tensor payloads.
        """
        if key.startswith(self.UNET_PREFIX):
            return "unet", key[len(self.UNET_PREFIX):]
        if key.startswith(self.VAE_PREFIX):
            return "vae", key[len(self.VAE_PREFIX):]
        if key.startswith(self.SDXL_TEXT_ENCODER_1_PREFIX):
            return "text_encoder", key[len(self.SDXL_TEXT_ENCODER_1_PREFIX):]
        if key.startswith(self.SDXL_TEXT_ENCODER_2_PREFIX):
            return "text_encoder_2", key[len(self.SDXL_TEXT_ENCODER_2_PREFIX):]
        if key.startswith(self.TEXT_PREFIX):
            return "text_encoder", key[len(self.TEXT_PREFIX):]
        return "extras", key

    def split_checkpoint(self, state_dict: Dict[str, object]) -> MappedStateDict:
        mapped = MappedStateDict()

        for key, value in state_dict.items():
            component_name, new_key = self.route_key(key)
            target = getattr(mapped, component_name)
            if new_key in target:
                raise ValueError(
                    "State-dict mapping collision for "
                    f"component={component_name!r}, mapped_key={new_key!r}, source_key={key!r}."
                )
            target[new_key] = value

        return mapped
