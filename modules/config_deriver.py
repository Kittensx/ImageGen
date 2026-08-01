from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DerivedVAEConfig:
    block_out_channels: list[int]
    layers_per_block: int = 2
    latent_channels: int = 4
    in_channels: int = 3
    out_channels: int = 3
    sample_size: int = 512
    scaling_factor: float = 0.18215
    norm_num_groups: int = 32
    act_fn: str = "silu"
    down_block_types: tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )

    def to_diffusers_config(self) -> dict[str, Any]:
        return {
            "_class_name": "AutoencoderKL",
            "act_fn": self.act_fn,
            "block_out_channels": self.block_out_channels,
            "down_block_types": list(self.down_block_types),
            "up_block_types": list(self.up_block_types),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "latent_channels": self.latent_channels,
            "layers_per_block": self.layers_per_block,
            "norm_num_groups": self.norm_num_groups,
            "sample_size": self.sample_size,
            "scaling_factor": self.scaling_factor,
        }


class ConfigDeriver:
    """
    Derives component config details directly from mapped checkpoint weights.
    """

    def derive_vae_config(self, vae_state: dict[str, Any]) -> DerivedVAEConfig:
        encoder_channels = []
        decoder_channels = []

        # encoder.down.{i}.block.0.conv1.weight shape: [out_c, in_c, 3, 3]
        for i in range(4):
            key = f"encoder.down.{i}.block.0.conv1.weight"
            if key in vae_state:
                out_c = int(vae_state[key].shape[0])
                encoder_channels.append(out_c)

        # decoder.up.{i}.block.0.conv1.weight shape: [out_c, in_c, 3, 3]
        for i in range(4):
            key = f"decoder.up.{i}.block.0.conv1.weight"
            if key in vae_state:
                out_c = int(vae_state[key].shape[0])
                decoder_channels.append(out_c)

        if len(encoder_channels) != 4:
            raise ValueError(
                f"Could not derive VAE encoder block_out_channels cleanly. "
                f"Found {encoder_channels}"
            )

        # encoder is correct
        encoder_channels = encoder_channels

        # IMPORTANT: derive decoder separately
        if len(decoder_channels) == 4:
            # Always use encoder channels for diffusers config
            block_out_channels = encoder_channels
        else:
            block_out_channels = encoder_channels
        print("Encoder channels:", encoder_channels)
        print("Decoder channels:", decoder_channels)
        print("Final block_out_channels:", block_out_channels)

        return DerivedVAEConfig(block_out_channels=block_out_channels)

    def derive_unet_cross_attention_dim(self, unet_state: dict[str, Any]) -> int | None:
        candidates = [
            "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight",
            "middle_block.1.transformer_blocks.0.attn2.to_k.weight",
        ]

        for key in candidates:
            if key in unet_state:
                # shape usually [inner_dim, cross_attention_dim]
                return int(unet_state[key].shape[1])

        return None