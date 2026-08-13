from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

from modules.text_encoder_strategy import text_encoder_strategy_for
from modules.sd2_openclip_reference_converter import SD2OpenCLIPReferenceConverter


@dataclass
class ConvertedStateDict:
    unet: Dict[str, Any] = field(default_factory=dict)
    vae: Dict[str, Any] = field(default_factory=dict)
    text_encoder: Dict[str, Any] = field(default_factory=dict)


class StateDictConverter:
    """
    Converts A1111 / original LDM-style checkpoint keys into diffusers-compatible keys.
    """

    VALID_PARAM_SUFFIXES = {"weight", "bias"}

    def convert_all(
        self,
        unet_state: Dict[str, Any],
        vae_state: Dict[str, Any],
        text_state: Dict[str, Any],
        architecture: str | None = None,
    ) -> ConvertedStateDict:
        return ConvertedStateDict(
            unet=self.convert_unet_state_dict(unet_state),
            vae=self.convert_vae_state_dict(vae_state),
            text_encoder=self.convert_text_encoder_state_dict(text_state, architecture=architecture),
        )

    def convert_unet_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        new_state: Dict[str, Any] = {}

        for key, value in state_dict.items():
            new_key = key

            # time embedding
            new_key = new_key.replace("time_embed.0.", "time_embedding.linear_1.")
            new_key = new_key.replace("time_embed.2.", "time_embedding.linear_2.")

            # input/output conv
            new_key = new_key.replace("input_blocks.0.0.", "conv_in.")
            new_key = new_key.replace("out.0.", "conv_norm_out.")
            new_key = new_key.replace("out.2.", "conv_out.")

            # structural mapping
            new_key = self._convert_unet_input_blocks(new_key)
            new_key = self._convert_unet_middle_block(new_key)
            new_key = self._convert_unet_output_blocks(new_key)

            # common subkeys
            new_key = self._convert_resnet_subkeys(new_key)
            new_key = self._convert_attention_subkeys(new_key)

            # targeted cleanup for remaining mismatches
            new_key = self._cleanup_unet_key(new_key)

            if new_key in new_state:
                print(f"COLLISION: {key} -> {new_key}")
            new_state[new_key] = value

        return new_state

    def convert_vae_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        new_state: Dict[str, Any] = {}

        for key, value in state_dict.items():
            new_key = self.convert_vae_key(key)
            new_value = self.convert_vae_tensor(new_key, value)

            if new_key in new_state:
                print(f"COLLISION: {key} -> {new_key}")
            new_state[new_key] = new_value

        return new_state

    def convert_vae_key(self, key: str) -> str:
        """Convert one original-LDM VAE key to the Diffusers AutoencoderKL layout."""
        new_key = key
        new_key = self._convert_vae_encoder_blocks(new_key)
        new_key = self._convert_vae_decoder_blocks(new_key)
        new_key = self._convert_vae_mid_blocks(new_key)
        new_key = new_key.replace("norm_out.", "conv_norm_out.")
        new_key = self._convert_resnet_subkeys(new_key)
        new_key = new_key.replace(".nin_shortcut.", ".conv_shortcut.")
        new_key = self._convert_vae_attention_subkeys(new_key)
        return new_key

    @staticmethod
    def convert_vae_tensor(converted_key: str, value: Any) -> Any:
        """Transform LDM 1x1-convolution attention weights into Linear weights."""
        attention_weight_suffixes = (
            ".attentions.0.query.weight",
            ".attentions.0.key.weight",
            ".attentions.0.value.weight",
            ".attentions.0.proj_attn.weight",
        )
        if str(converted_key).endswith(attention_weight_suffixes):
            shape = tuple(getattr(value, "shape", ()) or ())
            if len(shape) == 4 and shape[-2:] == (1, 1):
                return value[:, :, 0, 0]
        return value

    def convert_text_encoder_state_dict(
        self,
        state_dict: Dict[str, Any],
        architecture: str | None = None,
    ) -> Dict[str, Any]:
        normalized_architecture = str(architecture or "").strip().lower()
        if normalized_architecture in {"sd2", "sd2.1", "sd2.x", "stable-diffusion-2.x"}:
            return SD2OpenCLIPReferenceConverter().convert(state_dict)
        strategy = text_encoder_strategy_for(architecture)
        return strategy.convert_state_dict(state_dict)

    def _cleanup_unet_key(self, key: str) -> str:
        """
        Final cleanup pass for leftover mismatches seen in debug output.
        """

        # normalize leftover conv.op leafs
        key = key.replace(".conv.op.weight", ".conv.weight")
        key = key.replace(".conv.op.bias", ".conv.bias")

        # normalize any accidental bad attention leaf naming
        key = key.replace("to_conv_norm_out.weight", "to_out.0.weight")
        key = key.replace("to_conv_norm_out.bias", "to_out.0.bias")
        key = key.replace("to_conv_norm_out.", "to_out.0.")

        # leftover output-block upsamplers that were not structurally rewritten
        key = key.replace("output_blocks.5.upsamplers.0.", "up_blocks.1.upsamplers.0.")
        key = key.replace("output_blocks.8.upsamplers.0.", "up_blocks.2.upsamplers.0.")

        # some checkpoints leave these as raw conv leafs after the block translation
        key = key.replace("output_blocks.5.2.conv.", "up_blocks.1.upsamplers.0.conv.")
        key = key.replace("output_blocks.8.2.conv.", "up_blocks.2.upsamplers.0.conv.")

        # suffix-aware normalization
        key = self._normalize_leaf_suffix(key)

        return key

    def _normalize_leaf_suffix(self, key: str) -> str:
        """
        Normalize leaf module names only when the parameter suffix makes it safe.
        """
        if not self._has_valid_param_suffix(key):
            return key

        # conv.op -> conv for leaf params only
        if ".conv.op." in key:
            key = key.replace(".conv.op.", ".conv.")

        return key

    def _has_valid_param_suffix(self, key: str) -> bool:
        parts = key.split(".")
        return len(parts) >= 2 and parts[-1] in self.VALID_PARAM_SUFFIXES

    def _convert_unet_input_blocks(self, key: str) -> str:
        mappings = {
            "input_blocks.1.0.": "down_blocks.0.resnets.0.",
            "input_blocks.2.0.": "down_blocks.0.resnets.1.",
            "input_blocks.3.0.": "down_blocks.0.downsamplers.0.conv.",

            "input_blocks.4.0.": "down_blocks.1.resnets.0.",
            "input_blocks.5.0.": "down_blocks.1.resnets.1.",
            "input_blocks.6.0.": "down_blocks.1.downsamplers.0.conv.",

            "input_blocks.7.0.": "down_blocks.2.resnets.0.",
            "input_blocks.8.0.": "down_blocks.2.resnets.1.",
            "input_blocks.9.0.": "down_blocks.2.downsamplers.0.conv.",

            "input_blocks.10.0.": "down_blocks.3.resnets.0.",
            "input_blocks.11.0.": "down_blocks.3.resnets.1.",
        }

        attn_mappings = {
            "input_blocks.1.1.": "down_blocks.0.attentions.0.",
            "input_blocks.2.1.": "down_blocks.0.attentions.1.",

            "input_blocks.4.1.": "down_blocks.1.attentions.0.",
            "input_blocks.5.1.": "down_blocks.1.attentions.1.",

            "input_blocks.7.1.": "down_blocks.2.attentions.0.",
            "input_blocks.8.1.": "down_blocks.2.attentions.1.",
        }

        for old, new in mappings.items():
            if key.startswith(old):
                return key.replace(old, new, 1)

        for old, new in attn_mappings.items():
            if key.startswith(old):
                return key.replace(old, new, 1)

        return key

    def _convert_unet_middle_block(self, key: str) -> str:
        mappings = {
            "middle_block.0.": "mid_block.resnets.0.",
            "middle_block.1.": "mid_block.attentions.0.",
            "middle_block.2.": "mid_block.resnets.1.",
        }

        for old, new in mappings.items():
            if key.startswith(old):
                return key.replace(old, new, 1)

        return key

    def _convert_unet_output_blocks(self, key: str) -> str:
        """
        Single authoritative output-block mapper.
        """

        # up_blocks.0
        if key.startswith("output_blocks.0.0."):
            return key.replace("output_blocks.0.0.", "up_blocks.0.resnets.0.", 1)
        if key.startswith("output_blocks.1.0."):
            return key.replace("output_blocks.1.0.", "up_blocks.0.resnets.1.", 1)
        if key.startswith("output_blocks.2.0."):
            return key.replace("output_blocks.2.0.", "up_blocks.0.resnets.2.", 1)
        if key.startswith("output_blocks.2.1."):
            return key.replace("output_blocks.2.1.", "up_blocks.0.upsamplers.0.", 1)

        # up_blocks.1
        if key.startswith("output_blocks.3.0."):
            return key.replace("output_blocks.3.0.", "up_blocks.1.resnets.0.", 1)
        if key.startswith("output_blocks.3.1."):
            return key.replace("output_blocks.3.1.", "up_blocks.1.attentions.0.", 1)

        if key.startswith("output_blocks.4.0."):
            return key.replace("output_blocks.4.0.", "up_blocks.1.resnets.1.", 1)
        if key.startswith("output_blocks.4.1."):
            return key.replace("output_blocks.4.1.", "up_blocks.1.attentions.1.", 1)

        if key.startswith("output_blocks.5.0."):
            return key.replace("output_blocks.5.0.", "up_blocks.1.resnets.2.", 1)
        if key.startswith("output_blocks.5.1."):
            return key.replace("output_blocks.5.1.", "up_blocks.1.attentions.2.", 1)
        if key.startswith("output_blocks.5.2."):
            return key.replace("output_blocks.5.2.", "up_blocks.1.upsamplers.0.", 1)

        # up_blocks.2
        if key.startswith("output_blocks.6.0."):
            return key.replace("output_blocks.6.0.", "up_blocks.2.resnets.0.", 1)
        if key.startswith("output_blocks.6.1."):
            return key.replace("output_blocks.6.1.", "up_blocks.2.attentions.0.", 1)

        if key.startswith("output_blocks.7.0."):
            return key.replace("output_blocks.7.0.", "up_blocks.2.resnets.1.", 1)
        if key.startswith("output_blocks.7.1."):
            return key.replace("output_blocks.7.1.", "up_blocks.2.attentions.1.", 1)

        if key.startswith("output_blocks.8.0."):
            return key.replace("output_blocks.8.0.", "up_blocks.2.resnets.2.", 1)
        if key.startswith("output_blocks.8.1."):
            return key.replace("output_blocks.8.1.", "up_blocks.2.attentions.2.", 1)
        if key.startswith("output_blocks.8.2."):
            return key.replace("output_blocks.8.2.", "up_blocks.2.upsamplers.0.", 1)

        # up_blocks.3
        if key.startswith("output_blocks.9.0."):
            return key.replace("output_blocks.9.0.", "up_blocks.3.resnets.0.", 1)
        if key.startswith("output_blocks.9.1."):
            return key.replace("output_blocks.9.1.", "up_blocks.3.attentions.0.", 1)

        if key.startswith("output_blocks.10.0."):
            return key.replace("output_blocks.10.0.", "up_blocks.3.resnets.1.", 1)
        if key.startswith("output_blocks.10.1."):
            return key.replace("output_blocks.10.1.", "up_blocks.3.attentions.1.", 1)

        if key.startswith("output_blocks.11.0."):
            return key.replace("output_blocks.11.0.", "up_blocks.3.resnets.2.", 1)
        if key.startswith("output_blocks.11.1."):
            return key.replace("output_blocks.11.1.", "up_blocks.3.attentions.2.", 1)

        return key

    def _convert_vae_encoder_blocks(self, key: str) -> str:
        mappings = {
            "encoder.down.0.block.0.": "encoder.down_blocks.0.resnets.0.",
            "encoder.down.0.block.1.": "encoder.down_blocks.0.resnets.1.",
            "encoder.down.0.downsample.": "encoder.down_blocks.0.downsamplers.0.",
            "encoder.down.1.block.0.": "encoder.down_blocks.1.resnets.0.",
            "encoder.down.1.block.1.": "encoder.down_blocks.1.resnets.1.",
            "encoder.down.1.downsample.": "encoder.down_blocks.1.downsamplers.0.",
            "encoder.down.2.block.0.": "encoder.down_blocks.2.resnets.0.",
            "encoder.down.2.block.1.": "encoder.down_blocks.2.resnets.1.",
            "encoder.down.2.downsample.": "encoder.down_blocks.2.downsamplers.0.",
            "encoder.down.3.block.0.": "encoder.down_blocks.3.resnets.0.",
            "encoder.down.3.block.1.": "encoder.down_blocks.3.resnets.1.",
        }
        for old, new in mappings.items():
            if key.startswith(old):
                return key.replace(old, new, 1)
        return key

    def _convert_vae_decoder_blocks(self, key: str) -> str:
        # LDM decoder levels are numbered opposite to Diffusers execution order.
        mappings = {
            "decoder.up.3.block.0.": "decoder.up_blocks.0.resnets.0.",
            "decoder.up.3.block.1.": "decoder.up_blocks.0.resnets.1.",
            "decoder.up.3.block.2.": "decoder.up_blocks.0.resnets.2.",
            "decoder.up.3.upsample.": "decoder.up_blocks.0.upsamplers.0.",

            "decoder.up.2.block.0.": "decoder.up_blocks.1.resnets.0.",
            "decoder.up.2.block.1.": "decoder.up_blocks.1.resnets.1.",
            "decoder.up.2.block.2.": "decoder.up_blocks.1.resnets.2.",
            "decoder.up.2.upsample.": "decoder.up_blocks.1.upsamplers.0.",

            "decoder.up.1.block.0.": "decoder.up_blocks.2.resnets.0.",
            "decoder.up.1.block.1.": "decoder.up_blocks.2.resnets.1.",
            "decoder.up.1.block.2.": "decoder.up_blocks.2.resnets.2.",
            "decoder.up.1.upsample.": "decoder.up_blocks.2.upsamplers.0.",

            "decoder.up.0.block.0.": "decoder.up_blocks.3.resnets.0.",
            "decoder.up.0.block.1.": "decoder.up_blocks.3.resnets.1.",
            "decoder.up.0.block.2.": "decoder.up_blocks.3.resnets.2.",
        }
        for old, new in mappings.items():
            if key.startswith(old):
                return key.replace(old, new, 1)
        return key

    def _convert_vae_mid_blocks(self, key: str) -> str:
        mappings = {
            "encoder.mid.block_1.": "encoder.mid_block.resnets.0.",
            "encoder.mid.attn_1.": "encoder.mid_block.attentions.0.",
            "encoder.mid.block_2.": "encoder.mid_block.resnets.1.",

            "decoder.mid.block_1.": "decoder.mid_block.resnets.0.",
            "decoder.mid.attn_1.": "decoder.mid_block.attentions.0.",
            "decoder.mid.block_2.": "decoder.mid_block.resnets.1.",
        }
        for old, new in mappings.items():
            if key.startswith(old):
                return key.replace(old, new, 1)
        return key

    def _convert_resnet_subkeys(self, key: str) -> str:
        key = key.replace("in_layers.0.", "norm1.")
        key = key.replace("in_layers.2.", "conv1.")
        key = key.replace("out_layers.0.", "norm2.")
        key = key.replace("out_layers.3.", "conv2.")
        key = key.replace("emb_layers.1.", "time_emb_proj.")
        key = key.replace("skip_connection.", "conv_shortcut.")
        return key

    def _convert_attention_subkeys(self, key: str) -> str:
        # keep only meaningful conversions
        key = key.replace("proj_out.weight", "proj_out.weight")
        key = key.replace("proj_out.bias", "proj_out.bias")
        return key

    def _convert_vae_attention_subkeys(self, key: str) -> str:
        # Rewrite only attention leaf names. Global replacements such as
        # ``k. -> key.`` corrupt ``mid_block.`` into ``mid_blockey.``.
        replacements = {
            ".attentions.0.norm.": ".attentions.0.group_norm.",
            ".attentions.0.q.": ".attentions.0.query.",
            ".attentions.0.k.": ".attentions.0.key.",
            ".attentions.0.v.": ".attentions.0.value.",
            ".attentions.0.proj_out.": ".attentions.0.proj_attn.",
        }
        for old, new in replacements.items():
            key = key.replace(old, new)
        return key
