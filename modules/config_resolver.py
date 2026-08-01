from __future__ import annotations

from dataclasses import dataclass
import json
import os


@dataclass
class ResolvedConfigs:
    architecture: str
    root_dir: str
    manifest_path: str
    manifest: dict
    unet_config_path: str
    vae_config_path: str
    text_encoder_config_path: str


class ConfigResolver:
    """
    Resolves local-only config files for a detected model family.
    No remote fallback is allowed.
    """

    def __init__(self, local_config_root: str):
        self.local_config_root = local_config_root

    def resolve(self, architecture: str) -> ResolvedConfigs:
        arch_dir = self._map_architecture_to_dir(architecture)
        root_dir = os.path.join(self.local_config_root, arch_dir)

        manifest_path = os.path.join(root_dir, "model_manifest.json")
        unet_config_path = os.path.join(root_dir, "unet_config.json")
        vae_config_path = os.path.join(root_dir, "vae_config.json")
        text_encoder_config_path = os.path.join(root_dir, "text_encoder_config.json")

        missing = [
            path for path in [
                manifest_path,
                unet_config_path,
                vae_config_path,
                text_encoder_config_path,
            ]
            if not os.path.exists(path)
        ]
        if missing:
            joined = "\n".join(missing)
            raise FileNotFoundError(
                f"Missing required local config files for architecture '{architecture}':\n{joined}"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        return ResolvedConfigs(
            architecture=architecture,
            root_dir=root_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            unet_config_path=unet_config_path,
            vae_config_path=vae_config_path,
            text_encoder_config_path=text_encoder_config_path,
        )

    def _map_architecture_to_dir(self, architecture: str) -> str:
        if architecture == "sdxl":
            return "sdxl"

        if architecture in {"sd1.x", "sd1.5"}:
            # temporary: until we separate sd1 vs sd2 more strongly
            return "sd1"

        if architecture in {"sd2.x", "sd2.1"}:
            return "sd2"

        raise ValueError(
            f"Unsupported or unresolved architecture '{architecture}'. "
            "Add stronger detection before continuing."
        )