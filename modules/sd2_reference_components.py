from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.asset_discovery import resolve_nested_asset
from modules.project_context import ProjectContext
from modules.sd2_runtime_profile import SD2RuntimeProfile


@dataclass(frozen=True)
class SD2ReferenceComponents:
    root: Path
    text_encoder_weights: Path
    unet_weights: Path
    vae_weights: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "text_encoder_weights": str(self.text_encoder_weights),
            "unet_weights": str(self.unet_weights),
            "vae_weights": str(self.vae_weights),
        }


class SD2ReferenceComponentResolver:
    """Resolve optional heavyweight reference component weights for tooling/tests."""

    DEFAULT_BASE = Path("reference_components") / "stable_diffusion"

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def profile_root(self, profile: SD2RuntimeProfile) -> Path:
        profile_dir = {
            "sd2.1-base-512": "sd2_1_base",
            "sd2.1-768-v": "sd2_1_768",
            "sd2.0-base-512": "sd2_0_base",
            "sd2.0-768-v": "sd2_0_768",
        }.get(profile.profile_id, profile.profile_id.replace(".", "_").replace("-", "_"))
        return (self.context.model_tooling_root / self.DEFAULT_BASE / profile_dir).resolve()

    @staticmethod
    def _required(root: Path, relative_hint: str) -> Path:
        found = resolve_nested_asset(root, relative_hint, extensions={".safetensors"}, allow_stem_match=False)
        if found is None:
            raise FileNotFoundError(
                f"Required SD2 reference component {relative_hint!r} was not found anywhere under: {root}"
            )
        return found

    def resolve(self, profile: SD2RuntimeProfile) -> SD2ReferenceComponents:
        return self.resolve_from_root(self.profile_root(profile))

    def resolve_from_root(self, root: str | Path) -> SD2ReferenceComponents:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"SD2 reference component root does not exist: {root}")
        return SD2ReferenceComponents(
            root=root,
            text_encoder_weights=self._required(root, "text_encoder/model.safetensors"),
            unet_weights=self._required(root, "unet/diffusion_pytorch_model.safetensors"),
            vae_weights=self._required(root, "vae/diffusion_pytorch_model.safetensors"),
        )
