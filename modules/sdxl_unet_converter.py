from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SDXLUNetConversionReport:
    source_keys: int
    converted_keys: int
    down_block_count: int
    up_block_count: int
    layers_per_block: int
    collisions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_keys": int(self.source_keys),
            "converted_keys": int(self.converted_keys),
            "down_block_count": int(self.down_block_count),
            "up_block_count": int(self.up_block_count),
            "layers_per_block": int(self.layers_per_block),
            "collisions": list(self.collisions),
        }


class SDXLUNetStateDictConverter:
    """Convert original-LDM SDXL UNet keys into Diffusers UNet2DConditionModel keys.

    SDXL Base does not use the four-level SD1/SD2 UNet topology.  The block
    mapping is therefore derived from the canonical SDXL UNet config instead of
    sharing the legacy fixed input_blocks/output_blocks table.

    The mapping follows the same LDM-to-Diffusers structural contract used by
    Diffusers' single-file loader: input/output block indices are grouped by
    ``layers_per_block + 1`` and SDXL's ``label_emb`` MLP is the UNet
    ``add_embedding`` MLP used for text-time conditioning.
    """

    _DIRECT_PREFIXES = (
        ("time_embed.0.", "time_embedding.linear_1."),
        ("time_embed.2.", "time_embedding.linear_2."),
        ("input_blocks.0.0.", "conv_in."),
        ("out.0.", "conv_norm_out."),
        ("out.2.", "conv_out."),
        ("label_emb.0.0.", "add_embedding.linear_1."),
        ("label_emb.0.2.", "add_embedding.linear_2."),
    )

    def __init__(self, unet_config: Mapping[str, Any]) -> None:
        config = dict(unet_config or {})
        self.down_block_types = tuple(config.get("down_block_types") or ())
        self.up_block_types = tuple(config.get("up_block_types") or ())
        self.block_out_channels = tuple(config.get("block_out_channels") or ())
        raw_layers = config.get("layers_per_block", 2)
        if isinstance(raw_layers, (list, tuple)):
            values = tuple(int(value) for value in raw_layers)
            if not values or len(set(values)) != 1:
                raise ValueError(
                    "SDXL UNet conversion currently requires a uniform layers_per_block value."
                )
            self.layers_per_block = values[0]
        else:
            self.layers_per_block = int(raw_layers)

        self.down_block_count = len(self.down_block_types or self.block_out_channels)
        self.up_block_count = len(self.up_block_types or self.block_out_channels)
        if self.down_block_count <= 0 or self.up_block_count <= 0:
            raise ValueError("SDXL UNet config must declare down/up block topology.")
        if self.down_block_count != self.up_block_count:
            raise ValueError(
                "SDXL UNet down/up block counts must match for original-LDM conversion."
            )
        if self.layers_per_block <= 0:
            raise ValueError("SDXL UNet layers_per_block must be positive.")

    @staticmethod
    def _convert_resnet_subkeys(key: str) -> str:
        key = key.replace("in_layers.0.", "norm1.")
        key = key.replace("in_layers.2.", "conv1.")
        key = key.replace("out_layers.0.", "norm2.")
        key = key.replace("out_layers.3.", "conv2.")
        key = key.replace("emb_layers.1.", "time_emb_proj.")
        key = key.replace("skip_connection.", "conv_shortcut.")
        return key

    @staticmethod
    def _direct_key(key: str) -> str | None:
        for source, target in SDXLUNetStateDictConverter._DIRECT_PREFIXES:
            if key.startswith(source):
                return target + key[len(source):]
        return None

    def _input_key(self, key: str) -> str | None:
        if not key.startswith("input_blocks."):
            return None
        parts = key.split(".")
        if len(parts) < 4:
            return None
        try:
            source_block = int(parts[1])
            source_module = int(parts[2])
        except ValueError:
            return None
        if source_block == 0:
            return None

        group = self.layers_per_block + 1
        block_id = (source_block - 1) // group
        layer_in_group = (source_block - 1) % group
        if block_id >= self.down_block_count:
            return None
        suffix = ".".join(parts[3:])

        # The final entry in each non-final LDM input group is a downsampler
        # whose module 0 owns ``op.weight`` / ``op.bias``.
        if (
            block_id < self.down_block_count - 1
            and layer_in_group == self.layers_per_block
            and source_module == 0
            and suffix.startswith("op.")
        ):
            suffix = suffix[len("op."):]
            return f"down_blocks.{block_id}.downsamplers.0.conv.{suffix}"

        if layer_in_group >= self.layers_per_block:
            return None
        if source_module == 0:
            return self._convert_resnet_subkeys(
                f"down_blocks.{block_id}.resnets.{layer_in_group}.{suffix}"
            )
        if source_module == 1:
            return f"down_blocks.{block_id}.attentions.{layer_in_group}.{suffix}"
        return None

    @staticmethod
    def _middle_key(key: str) -> str | None:
        mappings = {
            "middle_block.0.": "mid_block.resnets.0.",
            "middle_block.1.": "mid_block.attentions.0.",
            "middle_block.2.": "mid_block.resnets.1.",
        }
        for source, target in mappings.items():
            if key.startswith(source):
                mapped = target + key[len(source):]
                return SDXLUNetStateDictConverter._convert_resnet_subkeys(mapped)
        return None

    def _output_component_roles(self, state_dict: Mapping[str, Any]) -> dict[tuple[int, int], str]:
        """Identify non-resnet LDM output submodules as attention or upsampler."""
        roles: dict[tuple[int, int], str] = {}
        for key in state_dict:
            if not key.startswith("output_blocks."):
                continue
            parts = key.split(".")
            if len(parts) < 5:
                continue
            try:
                source_block = int(parts[1])
                source_module = int(parts[2])
            except ValueError:
                continue
            if source_module == 0:
                continue
            suffix = ".".join(parts[3:])
            token = (source_block, source_module)
            role = "upsampler" if suffix.startswith("conv.") else "attention"
            previous = roles.get(token)
            if previous is not None and previous != role:
                raise ValueError(
                    f"Ambiguous SDXL output block role for output_blocks.{source_block}.{source_module}: "
                    f"observed both {previous!r} and {role!r}."
                )
            roles[token] = role
        return roles

    def _output_key(self, key: str, roles: Mapping[tuple[int, int], str]) -> str | None:
        if not key.startswith("output_blocks."):
            return None
        parts = key.split(".")
        if len(parts) < 4:
            return None
        try:
            source_block = int(parts[1])
            source_module = int(parts[2])
        except ValueError:
            return None

        group = self.layers_per_block + 1
        block_id = source_block // group
        layer_in_group = source_block % group
        if block_id >= self.up_block_count:
            return None
        suffix = ".".join(parts[3:])

        if source_module == 0:
            return self._convert_resnet_subkeys(
                f"up_blocks.{block_id}.resnets.{layer_in_group}.{suffix}"
            )

        role = roles.get((source_block, source_module))
        if role == "upsampler":
            return f"up_blocks.{block_id}.upsamplers.0.{suffix}"
        if role == "attention":
            return f"up_blocks.{block_id}.attentions.{layer_in_group}.{suffix}"
        return None

    def convert(self, state_dict: Mapping[str, Any]) -> tuple[dict[str, Any], SDXLUNetConversionReport]:
        roles = self._output_component_roles(state_dict)
        converted: dict[str, Any] = {}
        collisions: list[str] = []

        for key, value in state_dict.items():
            new_key = self._direct_key(key)
            if new_key is None:
                new_key = self._input_key(key)
            if new_key is None:
                new_key = self._middle_key(key)
            if new_key is None:
                new_key = self._output_key(key, roles)
            if new_key is None:
                # Preserve an unknown key rather than silently dropping it. The
                # component-load coverage gate will surface it as unexpected.
                new_key = key

            if new_key in converted:
                collisions.append(f"{key} -> {new_key}")
            converted[new_key] = value

        report = SDXLUNetConversionReport(
            source_keys=len(state_dict),
            converted_keys=len(converted),
            down_block_count=self.down_block_count,
            up_block_count=self.up_block_count,
            layers_per_block=self.layers_per_block,
            collisions=tuple(collisions),
        )
        if collisions:
            raise ValueError(
                "SDXL UNet conversion produced mapped-key collisions: " + "; ".join(collisions[:8])
            )
        return converted, report
