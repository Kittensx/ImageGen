from __future__ import annotations

from .models import ArchitectureCapability


_CAPABILITIES = {
    "sd1.x": ArchitectureCapability(
        architecture="sd1.x",
        status="validation_target",
        generation_supported=True,
        validation_supported=True,
        reason="Single CLIP text encoder, 768-wide conditioning, four-channel latent pipeline.",
        requirements=(
            "full monolithic safetensors checkpoint",
            "local CLIP ViT-L/14 tokenizer",
            "SD1 local UNet/text/VAE configuration",
        ),
    ),
    "sd2.x": ArchitectureCapability(
        architecture="sd2.x",
        status="blocked",
        generation_supported=False,
        validation_supported=False,
        reason="The active conditioning path uses the SD1 CLIP tokenizer/text-encoder contract, not the required SD2 OpenCLIP contract.",
        requirements=(
            "explicit OpenCLIP tokenizer and text encoder",
            "1024-wide conditioning tests",
            "separate SD2 checkpoint mapping validation",
        ),
    ),
    "sdxl": ArchitectureCapability(
        architecture="sdxl",
        status="blocked",
        generation_supported=False,
        validation_supported=False,
        reason="The active runtime does not yet implement dual text encoders, pooled conditioning, or SDXL time IDs.",
        requirements=(
            "two text encoders and tokenizers",
            "pooled prompt embeddings",
            "added conditioning time IDs",
            "SDXL-specific UNet call contract",
        ),
    ),
    "sd1.x_or_sd2.x": ArchitectureCapability(
        architecture="sd1.x_or_sd2.x",
        status="unresolved",
        generation_supported=False,
        validation_supported=False,
        reason="Checkpoint header evidence did not distinguish SD1 from SD2 safely.",
        requirements=("identify text or cross-attention conditioning dimension",),
    ),
    "unknown": ArchitectureCapability(
        architecture="unknown",
        status="unsupported",
        generation_supported=False,
        validation_supported=False,
        reason="Checkpoint architecture could not be identified.",
        requirements=("add an explicit architecture detector and capability declaration",),
    ),
}


def capability_for(architecture: str) -> ArchitectureCapability:
    normalized = str(architecture or "unknown").strip().lower()
    aliases = {
        "sd1": "sd1.x",
        "sd1.5": "sd1.x",
        "stable-diffusion-1.x": "sd1.x",
        "sd2": "sd2.x",
        "sd2.1": "sd2.x",
        "stable-diffusion-2.x": "sd2.x",
    }
    return _CAPABILITIES.get(aliases.get(normalized, normalized), _CAPABILITIES["unknown"])


def capability_matrix() -> dict[str, dict]:
    return {name: capability.to_dict() for name, capability in _CAPABILITIES.items()}
