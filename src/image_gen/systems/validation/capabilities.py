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
        status="supported",
        generation_supported=True,
        validation_supported=True,
        reason="Qualified SD2.x checkpoints are enabled for normal generation using the local OpenCLIP text-conditioning runtime and SD2 runtime-profile contract.",
        requirements=(
            "full monolithic safetensors checkpoint",
            "local OpenCLIP tokenizer and text encoder runtime",
            "qualified SD2 runtime profile or explicit profile override when needed",
        ),
    ),
    "sdxl": ArchitectureCapability(
        architecture="sdxl",
        status="supported",
        generation_supported=True,
        validation_supported=True,
        reason=(
            "Generic SDXL Base-compatible checkpoints and qualified SDXL-Lightning runtime profiles "
            "have passed end-to-end txt2img qualification, including staged low-VRAM residency and "
            "FP32 SDXL VAE decode. Profile-level gates still block unqualified SDXL variants such as "
            "the Refiner and SDXL-Turbo from normal txt2img."
        ),
        requirements=(
            "full monolithic SDXL Base-compatible safetensors checkpoint",
            "canonical SDXL Base runtime assets",
            "generation-qualified SDXL runtime profile",
        ),
    ),
    "sd3.x": ArchitectureCapability(
        architecture="sd3.x",
        status="supported",
        generation_supported=True,
        validation_supported=True,
        reason=(
            "SD3 Medium and SD3.5 Medium txt2img are generation-qualified through the real 20-step "
            "SD3-11 image matrix, including embedded/external CLIP sourcing, Flow Match/Flow Euler, "
            "16-channel VAE decode, and staged low-VRAM component residency."
        ),
        requirements=(
            "full SD3 Medium or SD3.5 Medium safetensors checkpoint",
            "matching local SD3 runtime assets",
            "CLIP-L and CLIP-G either embedded or available in the shared TextEncoders library",
            "sampler/scheduler combination compatible with the model's flow-match mathematical domain",
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
