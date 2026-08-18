from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class LatentVAEContract:
    """Normalized sampling-latent and VAE transform contract.

    ``sampling_latent`` is the representation consumed by IMAGE_GEN's sampler.
    ``vae_latent`` is the representation consumed/produced by the VAE itself.

    The transforms are intentionally explicit because SD3-family VAEs use a
    non-zero latent shift while SD1/SD2/SDXL commonly use zero shift::

        sampling_latent = (vae_latent - shift_factor) * scaling_factor
        vae_latent      = sampling_latent / scaling_factor + shift_factor
    """

    latent_channels: int = 4
    latent_scale_factor: int = 8
    scaling_factor: float = 0.18215
    shift_factor: float = 0.0
    force_upcast: bool = False
    use_quant_conv: bool | None = None
    use_post_quant_conv: bool | None = None
    source: str = "runtime_defaults"

    def __post_init__(self) -> None:
        if int(self.latent_channels) <= 0:
            raise ValueError("latent_channels must be positive.")
        if int(self.latent_scale_factor) <= 0:
            raise ValueError("latent_scale_factor must be positive.")
        if not math.isfinite(float(self.scaling_factor)) or float(self.scaling_factor) <= 0.0:
            raise ValueError("scaling_factor must be a positive finite value.")
        if not math.isfinite(float(self.shift_factor)):
            raise ValueError("shift_factor must be finite.")

    def encode_sampling_latent(self, vae_latent: torch.Tensor) -> torch.Tensor:
        """Map an unscaled VAE latent into sampler/model latent space."""

        return (vae_latent - float(self.shift_factor)) * float(self.scaling_factor)

    def decode_sampling_latent(self, sampling_latent: torch.Tensor) -> torch.Tensor:
        """Map a sampler/model latent back into the VAE's native latent space."""

        return sampling_latent / float(self.scaling_factor) + float(self.shift_factor)

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "latent_channels": int(self.latent_channels),
            "latent_scale_factor": int(self.latent_scale_factor),
            "scaling_factor": float(self.scaling_factor),
            "shift_factor": float(self.shift_factor),
            "force_upcast": bool(self.force_upcast),
            "use_quant_conv": self.use_quant_conv,
            "use_post_quant_conv": self.use_post_quant_conv,
            "source": str(self.source),
            "encode_formula": "(vae_latent - shift_factor) * scaling_factor",
            "decode_formula": "sampling_latent / scaling_factor + shift_factor",
        }


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _derive_scale_factor(vae: Any, fallback: int) -> tuple[int, str]:
    config = getattr(vae, "config", None)
    block_out_channels = _config_value(config, "block_out_channels", None)
    if block_out_channels:
        try:
            count = len(tuple(block_out_channels))
        except TypeError:
            count = 0
        if count > 0:
            return 2 ** (count - 1), "vae.config.block_out_channels"
    return int(fallback), "pipeline_components"


def resolve_latent_vae_contract(
    components: Any,
    *,
    latent_scale_factor: int | None = None,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
) -> LatentVAEContract:
    """Resolve the effective latent/VAE contract from the active VAE.

    The effective VAE configuration is preferred because it describes the
    component that will actually execute, including an external VAE override.
    Pipeline component fields provide stable fallbacks for legacy/custom VAEs
    that do not expose Diffusers-style configuration attributes.
    """

    vae = getattr(components, "vae", None)
    config = getattr(vae, "config", None)

    component_channels = int(getattr(components, "latent_channels", 4) or 4)
    config_channels = _config_value(config, "latent_channels", None)
    channels = int(config_channels if config_channels is not None else component_channels)

    component_scale = int(getattr(components, "latent_scale_factor", 8) or 8)
    if latent_scale_factor is None:
        resolved_scale, scale_source = _derive_scale_factor(vae, component_scale)
    else:
        resolved_scale, scale_source = int(latent_scale_factor), "explicit_runtime_override"

    component_scaling = float(getattr(components, "vae_scaling_factor", 0.18215) or 0.18215)
    config_scaling = _config_value(config, "scaling_factor", None)
    resolved_scaling = float(
        scaling_factor
        if scaling_factor is not None
        else (config_scaling if config_scaling is not None else component_scaling)
    )

    component_shift = float(getattr(components, "vae_shift_factor", 0.0) or 0.0)
    config_shift = _config_value(config, "shift_factor", None)
    resolved_shift = float(
        shift_factor
        if shift_factor is not None
        else (config_shift if config_shift is not None else component_shift)
    )

    config_force_upcast = _config_value(config, "force_upcast", None)
    force_upcast = bool(
        config_force_upcast
        if config_force_upcast is not None
        else getattr(components, "vae_force_upcast", False)
    )

    use_quant_conv = _optional_bool(
        _config_value(
            config,
            "use_quant_conv",
            getattr(components, "vae_use_quant_conv", None),
        )
    )
    use_post_quant_conv = _optional_bool(
        _config_value(
            config,
            "use_post_quant_conv",
            getattr(components, "vae_use_post_quant_conv", None),
        )
    )

    source_parts = ["active_vae_config" if config is not None else "pipeline_components", scale_source]
    return LatentVAEContract(
        latent_channels=channels,
        latent_scale_factor=resolved_scale,
        scaling_factor=resolved_scaling,
        shift_factor=resolved_shift,
        force_upcast=force_upcast,
        use_quant_conv=use_quant_conv,
        use_post_quant_conv=use_post_quant_conv,
        source="+".join(source_parts),
    )


__all__ = ["LatentVAEContract", "resolve_latent_vae_contract"]
