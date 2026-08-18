from __future__ import annotations

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt"}
_TEXTUAL_INVERSION_EXTENSIONS = {".safetensors", ".pt", ".bin"}
ASSET_CATALOG_CONTRACT_VERSION = "image-gen-asset-catalog-v1"
_ASSET_TYPES = ("checkpoint", "lora", "vae", "textual_inversion")
_ASSET_PLURAL_KEYS = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "vae": "vaes",
    "textual_inversion": "textual_inversions",
}

__all__ = ["ASSET_CATALOG_CONTRACT_VERSION"]
