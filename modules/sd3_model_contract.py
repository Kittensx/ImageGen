from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.project_context import ProjectContext
from modules.sd3_runtime_assets import SD3RuntimeAssetResolver, SD3RuntimeAssets
from modules.sd3_runtime_profile import (
    SD3RuntimeProfile,
    profile_from_checkpoint_variant,
    profile_from_id,
)


@dataclass(frozen=True)
class SD3ResolvedModelContract:
    profile: SD3RuntimeProfile
    assets: SD3RuntimeAssets
    profile_source: str
    checkpoint_variant: str = ""

    architecture: str = "sd3.x"
    denoiser_type: str = "transformer"
    denoising_domain: str = "flow_match"
    model_dimension: int = 4096
    latent_channels: int = 16
    has_text_encoder_3: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "profile_source": self.profile_source,
            "checkpoint_variant": self.checkpoint_variant,
            "architecture": self.architecture,
            "denoiser_type": self.denoiser_type,
            "denoising_domain": self.denoising_domain,
            "model_dimension": self.model_dimension,
            "latent_channels": self.latent_channels,
            "has_text_encoder_3": self.has_text_encoder_3,
            "runtime_assets": self.assets.to_dict(),
        }


def resolve_sd3_model_contract(
    context: ProjectContext,
    *,
    checkpoint_variant: str | None = None,
    explicit_profile_id: str | None = None,
    runtime_assets_root: str | Path | None = None,
) -> SD3ResolvedModelContract:
    explicit_profile = profile_from_id(explicit_profile_id) if explicit_profile_id else None
    inferred_profile = profile_from_checkpoint_variant(checkpoint_variant) if checkpoint_variant else None

    if explicit_profile_id and explicit_profile is None:
        raise ValueError(f"Unknown SD3 runtime profile: {explicit_profile_id!r}")
    if checkpoint_variant and inferred_profile is None:
        raise ValueError(
            "Unknown SD3 checkpoint variant. Expected a qualified phase-01 variant such as "
            "'sd3_medium' or 'sd3_5_medium', got: "
            f"{checkpoint_variant!r}"
        )
    if explicit_profile is not None and inferred_profile is not None:
        if explicit_profile.profile_id != inferred_profile.profile_id:
            raise ValueError(
                "Explicit SD3 runtime profile conflicts with the qualified checkpoint variant: "
                f"explicit={explicit_profile.profile_id!r}, checkpoint_variant={checkpoint_variant!r}"
            )

    profile = explicit_profile or inferred_profile
    if profile is None:
        raise ValueError(
            "SD3 runtime profile is unresolved. The SD3 family alone is not enough to select between "
            "SD3 Medium and SD3.5 Medium. Provide a qualified checkpoint_variant or an explicit SD3 runtime profile."
        )

    if explicit_profile is not None:
        profile_source = "explicit_runtime_profile"
    else:
        profile_source = "checkpoint_variant"

    resolver = SD3RuntimeAssetResolver(context)
    assets = (
        resolver.resolve_from_root(runtime_assets_root, profile)
        if runtime_assets_root is not None
        else resolver.resolve(profile)
    )
    signature = assets.validate_contract_signature()

    return SD3ResolvedModelContract(
        profile=profile,
        assets=assets,
        profile_source=profile_source,
        checkpoint_variant=str(checkpoint_variant or ""),
        architecture="sd3.x",
        denoiser_type="transformer",
        denoising_domain="flow_match",
        model_dimension=int(signature["transformer_joint_attention_dim"]),
        latent_channels=int(signature["vae_latent_channels"]),
        has_text_encoder_3=True,
    )
