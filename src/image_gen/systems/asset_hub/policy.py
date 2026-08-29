from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from image_gen.systems.validation.capabilities import capability_matrix


_PROVIDER_TYPE_TO_KIND = {
    "checkpoint": "checkpoint",
    "model": "checkpoint",
    "lora": "lora",
    "locon": "lora",
    "dora": "lora",
    "vae": "vae",
    "textualinversion": "textual_inversion",
    "textual inversion": "textual_inversion",
    "embedding": "textual_inversion",
    "upscaler": "upscaler",
    "controlnet": "controlnet",
    "workflow": "workflow",
    "workflows": "workflow",
    "other": "other",
}

_BASE_MODEL_ALIASES = {
    "sd 1.4": "sd1.x",
    "sd 1.5": "sd1.x",
    "sd1": "sd1.x",
    "sd1.x": "sd1.x",
    "stable diffusion 1": "sd1.x",
    "stable diffusion 1.4": "sd1.x",
    "stable diffusion 1.5": "sd1.x",
    "sd 2.0": "sd2.x",
    "sd 2.1": "sd2.x",
    "sd2": "sd2.x",
    "sd2.x": "sd2.x",
    "stable diffusion 2": "sd2.x",
    "sdxl": "sdxl",
    "sd xl": "sdxl",
    "stable diffusion xl": "sdxl",
    "pony": "sdxl",
    "illustrious": "sdxl",
    "sd 3": "sd3.x",
    "sd3": "sd3.x",
    "sd 3.5": "sd3.x",
    "flux": "flux",
    "flux.1 d": "flux",
    "flux.1 s": "flux",
}

_PROVIDER_BASE_MODELS = {
    "sd1.x": ("SD 1.4", "SD 1.5"),
    "sd2.x": ("SD 2.0", "SD 2.1"),
    "sdxl": ("SDXL 1.0",),
    "sd3.x": ("SD 3", "SD 3.5"),
    "flux": ("Flux.1 D", "Flux.1 S"),
}

_BROWSABLE_KINDS = frozenset({"checkpoint", "lora", "vae", "textual_inversion", "upscaler"})
_DIFFUSION_COUPLED_KINDS = frozenset({"checkpoint", "lora", "vae", "textual_inversion"})


def normalize_asset_kind(value: Any) -> str:
    token = str(value or "").strip().casefold().replace("-", "_")
    if token in {"textualinversion", "textual_inversions", "embedding", "embeddings"}:
        return "textual_inversion"
    if token in {"loras", "checkpoint", "checkpoints", "vae", "vaes", "upscalers", "upscaler"}:
        return token.rstrip("s") if token != "vaes" else "vae"
    return token or "unknown"


def provider_type_to_asset_kind(value: Any) -> str:
    token = " ".join(str(value or "").strip().casefold().replace("_", " ").replace("-", " ").split())
    compact = token.replace(" ", "")
    return _PROVIDER_TYPE_TO_KIND.get(token, _PROVIDER_TYPE_TO_KIND.get(compact, "unknown"))


def normalize_architecture(value: Any) -> str:
    token = " ".join(str(value or "").strip().casefold().replace("_", " ").replace("-", " ").split())
    if not token:
        return ""
    direct = _BASE_MODEL_ALIASES.get(token)
    if direct:
        return direct
    if "flux" in token:
        return "flux"
    if "sdxl" in token or "stable diffusion xl" in token or token.startswith("xl"):
        return "sdxl"
    if "sd 3" in token or "sd3" in token or "stable diffusion 3" in token:
        return "sd3.x"
    if "sd 2" in token or "sd2" in token or "stable diffusion 2" in token:
        return "sd2.x"
    if "sd 1" in token or "sd1" in token or "stable diffusion 1" in token:
        return "sd1.x"
    return ""


@dataclass(frozen=True)
class ArchitectureCompatibilityPolicy:
    """Server-owned architecture and asset-kind policy for remote discovery."""

    def supported_architectures(self, asset_kind: str) -> tuple[str, ...]:
        kind = normalize_asset_kind(asset_kind)
        if kind == "upscaler":
            return ("independent",)
        if kind not in _DIFFUSION_COUPLED_KINDS:
            return ()
        supported = [
            name
            for name, payload in capability_matrix().items()
            if bool(payload.get("generation_supported")) and name not in {"unknown", "sd1.x_or_sd2.x"}
        ]
        return tuple(sorted(dict.fromkeys(supported)))

    def provider_base_model_filters(self, asset_kind: str) -> tuple[str, ...]:
        output: list[str] = []
        for architecture in self.supported_architectures(asset_kind):
            output.extend(_PROVIDER_BASE_MODELS.get(architecture, ()))
        return tuple(dict.fromkeys(output))

    def is_browsable_kind(self, asset_kind: str) -> bool:
        return normalize_asset_kind(asset_kind) in _BROWSABLE_KINDS

    def normalize_requested_architectures(self, asset_kind: str, requested: tuple[str, ...]) -> tuple[str, ...]:
        supported = self.supported_architectures(asset_kind)
        if not requested:
            return supported
        normalized = tuple(dict.fromkeys(filter(None, (normalize_architecture(item) for item in requested))))
        return tuple(item for item in normalized if item in supported)

    def is_compatible(self, asset_kind: str, base_model: str) -> bool:
        kind = normalize_asset_kind(asset_kind)
        supported = self.supported_architectures(kind)
        if kind == "upscaler":
            return True
        architecture = normalize_architecture(base_model)
        return bool(architecture and architecture in supported)

    def compatibility_payload(self, asset_kind: str) -> dict[str, Any]:
        kind = normalize_asset_kind(asset_kind)
        return {
            "assetKind": kind,
            "supported": self.is_browsable_kind(kind),
            "architectures": list(self.supported_architectures(kind)),
            "providerBaseModels": list(self.provider_base_model_filters(kind)),
            "unknownArchitecturePolicy": "exclude",
        }
