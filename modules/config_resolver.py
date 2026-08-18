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
    text_encoder_2_config_path: str = ""
    tokenizer_dir: str = ""
    tokenizer_2_dir: str = ""
    scheduler_config_path: str = ""
    transformer_config_path: str = ""
    text_encoder_3_config_path: str = ""
    tokenizer_3_dir: str = ""


class ConfigResolver:
    """
    Resolves local-only config files for a detected model family.
    No remote fallback is allowed.

    SDXL is intentionally excluded from the legacy local-config lookup. Its
    canonical architecture files are setup-acquired runtime assets under
    runtime_assets/stable_diffusion/SDXL_Base and must enter through
    SDXLRuntimeAssetResolver + resolve_explicit().
    """

    def __init__(self, local_config_root: str):
        self.local_config_root = local_config_root

    def resolve(self, architecture: str) -> ResolvedConfigs:
        if str(architecture or "").strip().lower() == "sdxl":
            raise ValueError(
                "SDXL configs must be resolved from runtime_assets/stable_diffusion/SDXL_Base "
                "through SDXLRuntimeAssetResolver; modules/local_configs/sdxl is not an "
                "authoritative SDXL runtime source."
            )

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

    def resolve_explicit(
        self,
        *,
        architecture: str,
        root_dir: str,
        unet_config_path: str,
        vae_config_path: str,
        text_encoder_config_path: str,
        manifest_path: str | None = None,
        text_encoder_2_config_path: str | None = None,
        tokenizer_dir: str | None = None,
        tokenizer_2_dir: str | None = None,
        scheduler_config_path: str | None = None,
        transformer_config_path: str | None = None,
        text_encoder_3_config_path: str | None = None,
        tokenizer_3_dir: str | None = None,
    ) -> ResolvedConfigs:
        architecture_key = str(architecture or "").strip().lower()
        required_files = [unet_config_path, vae_config_path, text_encoder_config_path]
        required_dirs: list[str] = []

        if architecture_key == "sdxl":
            required_files.extend(
                [
                    str(manifest_path or ""),
                    str(text_encoder_2_config_path or ""),
                    str(scheduler_config_path or ""),
                ]
            )
            required_dirs.extend([str(tokenizer_dir or ""), str(tokenizer_2_dir or "")])
        elif architecture_key in {"sd3", "sd3.x", "stable-diffusion-3.x"}:
            # SD3 has a transformer denoiser rather than a UNet. Keep the legacy
            # unet_config_path field available for older callers but do not require it.
            required_files = [
                str(transformer_config_path or ""),
                str(vae_config_path or ""),
                str(text_encoder_config_path or ""),
                str(text_encoder_2_config_path or ""),
                str(text_encoder_3_config_path or ""),
                str(scheduler_config_path or ""),
            ]
            required_dirs.extend(
                [str(tokenizer_dir or ""), str(tokenizer_2_dir or ""), str(tokenizer_3_dir or "")]
            )

        missing_files = [path for path in required_files if not path or not os.path.isfile(path)]
        missing_dirs = [path for path in required_dirs if not path or not os.path.isdir(path)]
        if missing_files or missing_dirs:
            lines = []
            lines.extend(f"file: {path or '(not supplied)'}" for path in missing_files)
            lines.extend(f"dir:  {path or '(not supplied)'}" for path in missing_dirs)
            joined = "\n".join(lines)
            raise FileNotFoundError(
                f"Missing required explicit config assets for architecture '{architecture}':\n{joined}"
            )

        manifest: dict = {}
        resolved_manifest_path = str(manifest_path or "")
        if resolved_manifest_path and os.path.exists(resolved_manifest_path):
            with open(resolved_manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            manifest = payload if isinstance(payload, dict) else {}

        return ResolvedConfigs(
            architecture=architecture,
            root_dir=str(root_dir),
            manifest_path=resolved_manifest_path,
            manifest=manifest,
            unet_config_path=str(unet_config_path),
            vae_config_path=str(vae_config_path),
            text_encoder_config_path=str(text_encoder_config_path),
            text_encoder_2_config_path=str(text_encoder_2_config_path or ""),
            tokenizer_dir=str(tokenizer_dir or ""),
            tokenizer_2_dir=str(tokenizer_2_dir or ""),
            scheduler_config_path=str(scheduler_config_path or ""),
            transformer_config_path=str(transformer_config_path or ""),
            text_encoder_3_config_path=str(text_encoder_3_config_path or ""),
            tokenizer_3_dir=str(tokenizer_3_dir or ""),
        )

    def _map_architecture_to_dir(self, architecture: str) -> str:
        if architecture in {"sd1.x", "sd1.5"}:
            return "sd1"

        if architecture in {"sd2.x", "sd2.1"}:
            return "sd2"

        raise ValueError(
            f"Unsupported or unresolved architecture '{architecture}'. "
            "Add stronger detection before continuing."
        )
