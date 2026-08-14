from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from modules.asset_discovery import resolve_nested_asset
from modules.config_resolver import ConfigResolver, ResolvedConfigs
from modules.project_context import ProjectContext


SDXL_BASE_RUNTIME_SUBDIR = Path("stable_diffusion") / "SDXL_Base"
SDXL_REFINER_RUNTIME_SUBDIR = Path("stable_diffusion") / "SDXL_Base_Refiner"


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


@dataclass(frozen=True)
class SDXLRuntimeAssets:
    """Canonical lightweight support assets for the generic SDXL architecture.

    These assets define the architecture/tokenization contract only. Heavy model
    weights continue to come from the selected monolithic checkpoint (for example
    SDXL-Lightning). The canonical source for this asset set is SDXL Base 1.0.
    """

    root: Path
    model_index: Path
    scheduler_config: Path

    tokenizer_dir: Path
    tokenizer_config: Path
    tokenizer_vocab: Path
    tokenizer_merges: Path
    tokenizer_special_tokens_map: Path

    tokenizer_2_dir: Path
    tokenizer_2_config: Path
    tokenizer_2_vocab: Path
    tokenizer_2_merges: Path
    tokenizer_2_special_tokens_map: Path

    text_encoder_config: Path
    text_encoder_2_config: Path
    unet_config: Path
    vae_config: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "model_index": str(self.model_index),
            "scheduler_config": str(self.scheduler_config),
            "tokenizer_dir": str(self.tokenizer_dir),
            "tokenizer_config": str(self.tokenizer_config),
            "tokenizer_vocab": str(self.tokenizer_vocab),
            "tokenizer_merges": str(self.tokenizer_merges),
            "tokenizer_special_tokens_map": str(self.tokenizer_special_tokens_map),
            "tokenizer_2_dir": str(self.tokenizer_2_dir),
            "tokenizer_2_config": str(self.tokenizer_2_config),
            "tokenizer_2_vocab": str(self.tokenizer_2_vocab),
            "tokenizer_2_merges": str(self.tokenizer_2_merges),
            "tokenizer_2_special_tokens_map": str(self.tokenizer_2_special_tokens_map),
            "text_encoder_config": str(self.text_encoder_config),
            "text_encoder_2_config": str(self.text_encoder_2_config),
            "unet_config": str(self.unet_config),
            "vae_config": str(self.vae_config),
        }

    def architecture_signature(self) -> dict[str, Any]:
        model_index = _load_json_object(self.model_index, label="SDXL model index")
        text_encoder = _load_json_object(self.text_encoder_config, label="SDXL text encoder 1 config")
        text_encoder_2 = _load_json_object(self.text_encoder_2_config, label="SDXL text encoder 2 config")
        unet = _load_json_object(self.unet_config, label="SDXL UNet config")
        vae = _load_json_object(self.vae_config, label="SDXL VAE config")
        scheduler = _load_json_object(self.scheduler_config, label="SDXL scheduler config")

        return {
            "pipeline_class": model_index.get("_class_name"),
            "model_index_text_encoder": model_index.get("text_encoder"),
            "model_index_text_encoder_2": model_index.get("text_encoder_2"),
            "model_index_tokenizer": model_index.get("tokenizer"),
            "model_index_tokenizer_2": model_index.get("tokenizer_2"),
            "text_encoder_hidden_size": text_encoder.get("hidden_size"),
            "text_encoder_max_positions": text_encoder.get("max_position_embeddings"),
            "text_encoder_2_hidden_size": text_encoder_2.get("hidden_size"),
            "text_encoder_2_projection_dim": text_encoder_2.get("projection_dim"),
            "text_encoder_2_max_positions": text_encoder_2.get("max_position_embeddings"),
            "unet_cross_attention_dim": unet.get("cross_attention_dim"),
            "unet_addition_embed_type": unet.get("addition_embed_type"),
            "unet_addition_time_embed_dim": unet.get("addition_time_embed_dim"),
            "unet_projection_class_embeddings_input_dim": unet.get(
                "projection_class_embeddings_input_dim"
            ),
            "unet_sample_size": unet.get("sample_size"),
            "vae_scaling_factor": vae.get("scaling_factor"),
            "vae_force_upcast": bool(vae.get("force_upcast", False)),
            "scheduler_class": scheduler.get("_class_name"),
            "scheduler_prediction_type": scheduler.get("prediction_type"),
        }

    def validate_architecture_signature(self) -> dict[str, Any]:
        signature = self.architecture_signature()
        errors: list[str] = []

        if signature["pipeline_class"] != "StableDiffusionXLPipeline":
            errors.append(
                "model_index.json must declare _class_name='StableDiffusionXLPipeline'"
            )

        expected_components = {
            "model_index_text_encoder": "CLIPTextModel",
            "model_index_text_encoder_2": "CLIPTextModelWithProjection",
            "model_index_tokenizer": "CLIPTokenizer",
            "model_index_tokenizer_2": "CLIPTokenizer",
        }
        for field, class_name in expected_components.items():
            value = signature[field]
            if not isinstance(value, list) or len(value) < 2 or value[1] != class_name:
                errors.append(f"model_index.json {field} must reference {class_name}")

        expected_exact = {
            "text_encoder_hidden_size": 768,
            "text_encoder_max_positions": 77,
            "text_encoder_2_hidden_size": 1280,
            "text_encoder_2_projection_dim": 1280,
            "text_encoder_2_max_positions": 77,
            "unet_cross_attention_dim": 2048,
            "unet_addition_embed_type": "text_time",
            "unet_addition_time_embed_dim": 256,
            "unet_projection_class_embeddings_input_dim": 2816,
            "unet_sample_size": 128,
        }
        for field, expected in expected_exact.items():
            if signature[field] != expected:
                errors.append(f"{field} must be {expected!r}, got {signature[field]!r}")

        scaling_factor = signature["vae_scaling_factor"]
        try:
            scaling_value = float(scaling_factor)
        except (TypeError, ValueError):
            scaling_value = float("nan")
        if abs(scaling_value - 0.13025) > 1e-8:
            errors.append(
                f"vae_scaling_factor must be 0.13025, got {scaling_factor!r}"
            )
        if signature.get("vae_force_upcast") is not True:
            errors.append(
                f"vae_force_upcast must be True for the canonical SDXL VAE, got {signature.get('vae_force_upcast')!r}"
            )

        if errors:
            details = "\n  - ".join(errors)
            raise ValueError(
                "SDXL runtime assets do not match the canonical SDXL Base architecture contract:\n"
                f"  - {details}\nRuntime asset root: {self.root}"
            )
        return signature

    def to_resolved_configs(self, config_resolver: ConfigResolver) -> ResolvedConfigs:
        """Bridge canonical SDXL assets into the existing local config contract."""
        return config_resolver.resolve_explicit(
            architecture="sdxl",
            root_dir=str(self.root),
            manifest_path=str(self.model_index),
            unet_config_path=str(self.unet_config),
            vae_config_path=str(self.vae_config),
            text_encoder_config_path=str(self.text_encoder_config),
            text_encoder_2_config_path=str(self.text_encoder_2_config),
            tokenizer_dir=str(self.tokenizer_dir),
            tokenizer_2_dir=str(self.tokenizer_2_dir),
            scheduler_config_path=str(self.scheduler_config),
        )


class SDXLRuntimeAssetResolver:
    """Resolve the canonical SDXL Base support tree from runtime_assets."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def canonical_root(self) -> Path:
        root = (Path(self.context.runtime_assets_root) / SDXL_BASE_RUNTIME_SUBDIR).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                "SDXL Base runtime asset root does not exist. Expected setup-acquired assets at: "
                f"{root}"
            )
        return root

    @staticmethod
    def _required(root: Path, relative_hint: str, *, extensions: set[str] | None = None) -> Path:
        resolved = resolve_nested_asset(
            root,
            relative_hint,
            extensions=extensions,
            allow_stem_match=False,
        )
        if resolved is not None:
            try:
                relative = resolved.relative_to(root).as_posix().casefold()
            except ValueError:
                relative = ""
            wanted = Path(relative_hint.replace("\\", "/")).as_posix().lstrip("./").casefold()
            if relative == wanted or relative.endswith("/" + wanted):
                return resolved
        raise FileNotFoundError(
            f"Required SDXL runtime asset {relative_hint!r} was not found under: {root}"
        )

    def resolve(self) -> SDXLRuntimeAssets:
        return self.resolve_from_root(self.canonical_root())

    def resolve_from_root(self, root: str | Path) -> SDXLRuntimeAssets:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"SDXL runtime asset root does not exist: {root}")

        tokenizer_config = self._required(root, "tokenizer/tokenizer_config.json", extensions={".json"})
        tokenizer_2_config = self._required(root, "tokenizer_2/tokenizer_config.json", extensions={".json"})

        assets = SDXLRuntimeAssets(
            root=root,
            model_index=self._required(root, "model_index.json", extensions={".json"}),
            scheduler_config=self._required(root, "scheduler/scheduler_config.json", extensions={".json"}),
            tokenizer_dir=tokenizer_config.parent,
            tokenizer_config=tokenizer_config,
            tokenizer_vocab=self._required(root, "tokenizer/vocab.json", extensions={".json"}),
            tokenizer_merges=self._required(root, "tokenizer/merges.txt", extensions={".txt"}),
            tokenizer_special_tokens_map=self._required(
                root, "tokenizer/special_tokens_map.json", extensions={".json"}
            ),
            tokenizer_2_dir=tokenizer_2_config.parent,
            tokenizer_2_config=tokenizer_2_config,
            tokenizer_2_vocab=self._required(root, "tokenizer_2/vocab.json", extensions={".json"}),
            tokenizer_2_merges=self._required(root, "tokenizer_2/merges.txt", extensions={".txt"}),
            tokenizer_2_special_tokens_map=self._required(
                root, "tokenizer_2/special_tokens_map.json", extensions={".json"}
            ),
            text_encoder_config=self._required(root, "text_encoder/config.json", extensions={".json"}),
            text_encoder_2_config=self._required(root, "text_encoder_2/config.json", extensions={".json"}),
            unet_config=self._required(root, "unet/config.json", extensions={".json"}),
            vae_config=self._required(root, "vae/config.json", extensions={".json"}),
        )
        assets.validate_architecture_signature()
        return assets

    def resolve_configs(
        self,
        config_resolver: ConfigResolver,
        *,
        root: str | Path | None = None,
    ) -> ResolvedConfigs:
        assets = self.resolve_from_root(root) if root is not None else self.resolve()
        return assets.to_resolved_configs(config_resolver)


@dataclass(frozen=True)
class SDXLRefinerRuntimeAssets:
    """Lightweight architecture assets for the separate SDXL Refiner stage.

    The Refiner is deliberately not converted into ``ResolvedConfigs`` for the
    normal SDXL txt2img loader. Its UNet and conditioning contract differs from
    SDXL Base and is qualified as a future second-stage runtime.
    """

    root: Path
    scheduler_config: Path
    tokenizer_2_dir: Path
    tokenizer_2_config: Path
    tokenizer_2_vocab: Path
    tokenizer_2_merges: Path
    tokenizer_2_special_tokens_map: Path
    text_encoder_2_config: Path
    unet_config: Path
    vae_config: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "scheduler_config": str(self.scheduler_config),
            "tokenizer_2_dir": str(self.tokenizer_2_dir),
            "tokenizer_2_config": str(self.tokenizer_2_config),
            "tokenizer_2_vocab": str(self.tokenizer_2_vocab),
            "tokenizer_2_merges": str(self.tokenizer_2_merges),
            "tokenizer_2_special_tokens_map": str(self.tokenizer_2_special_tokens_map),
            "text_encoder_2_config": str(self.text_encoder_2_config),
            "unet_config": str(self.unet_config),
            "vae_config": str(self.vae_config),
        }

    def architecture_signature(self) -> dict[str, Any]:
        text_encoder_2 = _load_json_object(self.text_encoder_2_config, label="SDXL Refiner text encoder 2 config")
        unet = _load_json_object(self.unet_config, label="SDXL Refiner UNet config")
        vae = _load_json_object(self.vae_config, label="SDXL Refiner VAE config")
        scheduler = _load_json_object(self.scheduler_config, label="SDXL Refiner scheduler config")
        return {
            "text_encoder_2_hidden_size": text_encoder_2.get("hidden_size"),
            "text_encoder_2_projection_dim": text_encoder_2.get("projection_dim"),
            "text_encoder_2_max_positions": text_encoder_2.get("max_position_embeddings"),
            "unet_cross_attention_dim": unet.get("cross_attention_dim"),
            "unet_addition_embed_type": unet.get("addition_embed_type"),
            "unet_addition_time_embed_dim": unet.get("addition_time_embed_dim"),
            "unet_projection_class_embeddings_input_dim": unet.get("projection_class_embeddings_input_dim"),
            "unet_sample_size": unet.get("sample_size"),
            "unet_block_out_channels": list(unet.get("block_out_channels") or []),
            "vae_scaling_factor": vae.get("scaling_factor"),
            "vae_force_upcast": bool(vae.get("force_upcast", False)),
            "scheduler_class": scheduler.get("_class_name"),
            "scheduler_prediction_type": scheduler.get("prediction_type"),
            "scheduler_timestep_spacing": scheduler.get("timestep_spacing"),
        }

    def validate_architecture_signature(self) -> dict[str, Any]:
        signature = self.architecture_signature()
        expected = {
            "text_encoder_2_hidden_size": 1280,
            "text_encoder_2_projection_dim": 1280,
            "text_encoder_2_max_positions": 77,
            "unet_cross_attention_dim": 1280,
            "unet_addition_embed_type": "text_time",
            "unet_addition_time_embed_dim": 256,
            "unet_projection_class_embeddings_input_dim": 2560,
            "unet_sample_size": 128,
            "unet_block_out_channels": [384, 768, 1536, 1536],
            "scheduler_class": "EulerDiscreteScheduler",
            "scheduler_prediction_type": "epsilon",
        }
        errors = [
            f"{field} must be {value!r}, got {signature.get(field)!r}"
            for field, value in expected.items()
            if signature.get(field) != value
        ]
        try:
            scaling = float(signature.get("vae_scaling_factor"))
        except (TypeError, ValueError):
            scaling = float("nan")
        if abs(scaling - 0.13025) > 1e-8:
            errors.append(
                f"vae_scaling_factor must be 0.13025, got {signature.get('vae_scaling_factor')!r}"
            )
        if signature.get("vae_force_upcast") is not True:
            errors.append(
                f"vae_force_upcast must be True for the qualified Refiner VAE, got {signature.get('vae_force_upcast')!r}"
            )
        if errors:
            raise ValueError(
                "SDXL Refiner runtime assets do not match the qualified refiner architecture contract:\n  - "
                + "\n  - ".join(errors)
                + f"\nRuntime asset root: {self.root}"
            )
        return signature


class SDXLRefinerRuntimeAssetResolver:
    """Resolve/validate the separate SDXL Refiner support tree.

    This resolver intentionally exposes no normal txt2img ``ResolvedConfigs``
    bridge. Refiner execution remains a distinct future pipeline stage.
    """

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def canonical_root(self) -> Path:
        root = (Path(self.context.runtime_assets_root) / SDXL_REFINER_RUNTIME_SUBDIR).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                "SDXL Refiner runtime asset root does not exist. Expected setup-acquired assets at: "
                f"{root}"
            )
        return root

    def resolve(self) -> SDXLRefinerRuntimeAssets:
        return self.resolve_from_root(self.canonical_root())

    def resolve_from_root(self, root: str | Path) -> SDXLRefinerRuntimeAssets:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"SDXL Refiner runtime asset root does not exist: {root}")
        required = SDXLRuntimeAssetResolver._required
        tokenizer_2_config = required(root, "tokenizer_2/tokenizer_config.json", extensions={".json"})
        assets = SDXLRefinerRuntimeAssets(
            root=root,
            scheduler_config=required(root, "scheduler/scheduler_config.json", extensions={".json"}),
            tokenizer_2_dir=tokenizer_2_config.parent,
            tokenizer_2_config=tokenizer_2_config,
            tokenizer_2_vocab=required(root, "tokenizer_2/vocab.json", extensions={".json"}),
            tokenizer_2_merges=required(root, "tokenizer_2/merges.txt", extensions={".txt"}),
            tokenizer_2_special_tokens_map=required(root, "tokenizer_2/special_tokens_map.json", extensions={".json"}),
            text_encoder_2_config=required(root, "text_encoder_2/config.json", extensions={".json"}),
            unet_config=required(root, "unet/config.json", extensions={".json"}),
            vae_config=required(root, "vae/config.json", extensions={".json"}),
        )
        assets.validate_architecture_signature()
        return assets
