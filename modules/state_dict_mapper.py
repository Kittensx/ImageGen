from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from safetensors import safe_open


@dataclass
class MappedStateDict:
    unet: Dict[str, object] = field(default_factory=dict)
    transformer: Dict[str, object] = field(default_factory=dict)
    vae: Dict[str, object] = field(default_factory=dict)
    text_encoder: Dict[str, object] = field(default_factory=dict)
    text_encoder_2: Dict[str, object] = field(default_factory=dict)
    text_encoder_3: Dict[str, object] = field(default_factory=dict)
    extras: Dict[str, object] = field(default_factory=dict)


class StateDictMapper:
    """Split monolithic checkpoints into architecture-specific component state dicts.

    ``model.diffusion_model`` is an overloaded namespace: it contains a UNet for
    SD1/SD2/SDXL and the MMDiT transformer for SD3. Callers that know the
    architecture must pass it so the same source prefix is not misclassified.
    """

    DENOISER_PREFIX = "model.diffusion_model."
    VAE_PREFIX = "first_stage_model."
    TEXT_PREFIX = "cond_stage_model."
    SDXL_TEXT_ENCODER_1_PREFIX = "conditioner.embedders.0."
    SDXL_TEXT_ENCODER_2_PREFIX = "conditioner.embedders.1."
    SD3_CLIP_L_PREFIX = "text_encoders.clip_l.transformer."
    SD3_CLIP_G_PREFIX = "text_encoders.clip_g.transformer."
    SD3_T5_PREFIX = "text_encoders.t5xxl."

    @staticmethod
    def _is_sd3(architecture: str | None) -> bool:
        return str(architecture or "").strip().lower() in {
            "sd3",
            "sd3.x",
            "sd3.5",
            "stable-diffusion-3",
            "stable-diffusion-3.x",
        }

    def route_key(self, key: str, *, architecture: str | None = None) -> tuple[str, str]:
        """Return ``(component_name, stripped_key)`` for one checkpoint key."""
        if self._is_sd3(architecture):
            if key.startswith(self.DENOISER_PREFIX):
                return "transformer", key[len(self.DENOISER_PREFIX):]
            if key.startswith(self.SD3_CLIP_L_PREFIX):
                return "text_encoder", key[len(self.SD3_CLIP_L_PREFIX):]
            if key.startswith(self.SD3_CLIP_G_PREFIX):
                return "text_encoder_2", key[len(self.SD3_CLIP_G_PREFIX):]
            if key.startswith(self.SD3_T5_PREFIX):
                return "text_encoder_3", key[len(self.SD3_T5_PREFIX):]

        if key.startswith(self.DENOISER_PREFIX):
            return "unet", key[len(self.DENOISER_PREFIX):]
        if key.startswith(self.VAE_PREFIX):
            return "vae", key[len(self.VAE_PREFIX):]
        if key.startswith(self.SDXL_TEXT_ENCODER_1_PREFIX):
            return "text_encoder", key[len(self.SDXL_TEXT_ENCODER_1_PREFIX):]
        if key.startswith(self.SDXL_TEXT_ENCODER_2_PREFIX):
            return "text_encoder_2", key[len(self.SDXL_TEXT_ENCODER_2_PREFIX):]
        if key.startswith(self.TEXT_PREFIX):
            return "text_encoder", key[len(self.TEXT_PREFIX):]
        return "extras", key

    def split_checkpoint(
        self,
        state_dict: Dict[str, object],
        *,
        architecture: str | None = None,
    ) -> MappedStateDict:
        mapped = MappedStateDict()

        for key, value in state_dict.items():
            component_name, new_key = self.route_key(key, architecture=architecture)
            target = getattr(mapped, component_name)
            if new_key in target:
                raise ValueError(
                    "State-dict mapping collision for "
                    f"component={component_name!r}, mapped_key={new_key!r}, source_key={key!r}."
                )
            target[new_key] = value

        return mapped

    def load_selected_checkpoint_components(
        self,
        path: str,
        *,
        architecture: str | None = None,
        roles: Iterable[str],
    ) -> MappedStateDict:
        """Materialize only requested component tensors from a Safetensors donor.

        Advanced Models uses checkpoint files as component containers. Reading an
        unrelated UNet/VAE/text encoder merely because another role was selected
        defeats low-memory composition, so this path routes header keys first and
        calls ``get_tensor`` only for requested component roles.
        """
        requested = {str(role or "").strip() for role in roles if str(role or "").strip()}
        mapped = MappedStateDict()
        if not requested:
            return mapped

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                component_name, new_key = self.route_key(key, architecture=architecture)
                if component_name not in requested:
                    continue
                target = getattr(mapped, component_name)
                if new_key in target:
                    raise ValueError(
                        "State-dict mapping collision for "
                        f"component={component_name!r}, mapped_key={new_key!r}, source_key={key!r}."
                    )
                target[new_key] = handle.get_tensor(key)
        return mapped
