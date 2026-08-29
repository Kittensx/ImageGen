from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class StandardAdapterArchitectureContract:
    architecture_adapter_id: str
    family: str
    denoiser_target: str
    component_targets: frozenset[str]
    pipeline_prefixes: tuple[tuple[str, str], ...]
    kohya_prefixes: tuple[tuple[str, str], ...]
    native_prefixes: Mapping[str, tuple[str, ...]]


_CONTRACTS = {
    "sd1": StandardAdapterArchitectureContract(
        "image_gen.adapter_architecture.sd1.v1", "sd1", "unet",
        frozenset({"unet", "text_encoder"}),
        (("text_encoder.", "text_encoder"), ("unet.", "unet")),
        (("lora_te1_", "text_encoder"), ("lora_te_", "text_encoder"), ("lora_unet_", "unet")),
        {"unet": ("down_blocks.", "up_blocks.", "mid_block.", "conv_in.", "conv_out.", "time_embedding.", "time_embed.", "add_embedding.", "transformer_in.", "proj_in.", "proj_out."),
         "text_encoder": ("text_model.", "encoder.layers.", "embeddings.", "final_layer_norm.")},
    ),
    "sd2": StandardAdapterArchitectureContract(
        "image_gen.adapter_architecture.sd2.v1", "sd2", "unet",
        frozenset({"unet", "text_encoder"}),
        (("text_encoder.", "text_encoder"), ("unet.", "unet")),
        (("lora_te1_", "text_encoder"), ("lora_te_", "text_encoder"), ("lora_unet_", "unet")),
        {"unet": ("down_blocks.", "up_blocks.", "mid_block.", "conv_in.", "conv_out.", "time_embedding.", "time_embed.", "add_embedding.", "transformer_in.", "proj_in.", "proj_out."),
         "text_encoder": ("text_model.", "encoder.layers.", "embeddings.", "final_layer_norm.")},
    ),
    "sdxl": StandardAdapterArchitectureContract(
        "image_gen.adapter_architecture.sdxl.v1", "sdxl", "unet",
        frozenset({"unet", "text_encoder", "text_encoder_2"}),
        (("text_encoder_2.", "text_encoder_2"), ("text_encoder.", "text_encoder"), ("unet.", "unet")),
        (("lora_te2_", "text_encoder_2"), ("lora_te1_", "text_encoder"), ("lora_te_", "text_encoder"), ("lora_unet_", "unet")),
        {"unet": ("down_blocks.", "up_blocks.", "mid_block.", "conv_in.", "conv_out.", "time_embedding.", "time_embed.", "add_embedding.", "transformer_in.", "proj_in.", "proj_out."),
         "text_encoder": ("text_model.", "encoder.layers.", "embeddings.", "final_layer_norm."),
         "text_encoder_2": ("text_model.", "encoder.layers.", "embeddings.", "final_layer_norm.")},
    ),
    "sd3": StandardAdapterArchitectureContract(
        "image_gen.adapter_architecture.sd3.v1", "sd3", "transformer",
        frozenset({"transformer", "text_encoder", "text_encoder_2", "text_encoder_3"}),
        (("text_encoder_3.", "text_encoder_3"), ("text_encoder_2.", "text_encoder_2"), ("text_encoder.", "text_encoder"), ("transformer.", "transformer")),
        (("lora_te3_", "text_encoder_3"), ("lora_te2_", "text_encoder_2"), ("lora_te1_", "text_encoder"), ("lora_te_", "text_encoder"), ("lora_transformer_", "transformer")),
        {"transformer": ("transformer_blocks.", "single_transformer_blocks.", "pos_embed.", "time_text_embed.", "context_embedder.", "x_embedder.", "proj_out."),
         "text_encoder": ("text_model.", "encoder.layers.", "embeddings.", "final_layer_norm."),
         "text_encoder_2": ("text_model.", "encoder.layers.", "embeddings.", "final_layer_norm."),
         "text_encoder_3": ("encoder.block.", "shared.", "final_layer_norm.", "text_model.")},
    ),
}


def architecture_contract(family: str) -> StandardAdapterArchitectureContract:
    token = str(family or "").strip().lower()
    return _CONTRACTS.get(token) or _CONTRACTS["sd1"]


__all__ = ["StandardAdapterArchitectureContract", "architecture_contract"]
