from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.project_context import ProjectContext
from modules.sdxl_runtime_assets import SDXLRuntimeAssetResolver, SDXLRuntimeAssets
from modules.sdxl_runtime_profile import (
    SDXLRuntimeProfile,
    get_sdxl_runtime_profile,
    profile_for_sdxl_filename,
)


@dataclass(frozen=True)
class SDXLResolvedModelContract:
    profile: SDXLRuntimeProfile
    assets: SDXLRuntimeAssets
    profile_source: str

    @property
    def prediction_type(self) -> str:
        return self.profile.prediction_type

    @property
    def prediction_type_source(self) -> str:
        return f"sdxl_runtime_profile:{self.profile.profile_id}"

    @property
    def vae_scaling_factor(self) -> float:
        return float(self.profile.vae_scaling_factor)

    @property
    def vae_force_upcast(self) -> bool:
        return bool(self.assets.validate_architecture_signature().get("vae_force_upcast", False))

    @property
    def vae_execution_dtype(self) -> str:
        return "torch.float32" if self.vae_force_upcast else "model_dtype"

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "sdxl",
            "profile_source": self.profile_source,
            "profile": self.profile.to_dict(),
            "assets": self.assets.to_dict(),
        }


def resolve_sdxl_model_contract(
    context: ProjectContext,
    *,
    checkpoint_filename: str,
    explicit_profile_id: str | None = None,
) -> SDXLResolvedModelContract:
    if explicit_profile_id:
        profile = get_sdxl_runtime_profile(explicit_profile_id)
        source = "explicit_runtime_profile"
    else:
        profile = profile_for_sdxl_filename(checkpoint_filename)
        source = "checkpoint_filename" if profile.family in {"lightning", "turbo"} else "generic_sdxl_default"
    if profile.family == "refiner":
        raise ValueError(
            "SDXL Refiner is a distinct second-stage architecture and cannot be loaded through "
            "the normal SDXL Base/Lightning txt2img contract. Use the Refiner qualification path."
        )
    assets = SDXLRuntimeAssetResolver(context).resolve()
    signature = assets.validate_architecture_signature()
    scaling = float(signature.get("vae_scaling_factor"))
    if abs(scaling - float(profile.vae_scaling_factor)) > 1e-8:
        raise ValueError(
            "SDXL runtime profile VAE scaling conflicts with canonical SDXL Base assets: "
            f"profile={profile.vae_scaling_factor}, assets={scaling}"
        )
    return SDXLResolvedModelContract(profile=profile, assets=assets, profile_source=source)
