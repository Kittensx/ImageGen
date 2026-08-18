from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from modules.asset_discovery import resolve_nested_asset
from modules.config_resolver import ConfigResolver, ResolvedConfigs
from modules.project_context import ProjectContext
from modules.sd3_runtime_profile import SD3RuntimeProfile, profile_from_id


LEGACY_SD3_RUNTIME_ASSET_SUBDIRS: dict[str, tuple[str, ...]] = {
    "sd3-medium": (str(Path("stable_diffusion") / "SD3_Medium"),),
    "sd3.5-medium": (str(Path("stable_diffusion") / "SD3_5_Medium"),),
}


REQUIRED_SD3_RELATIVE_FILES: tuple[str, ...] = (
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "text_encoder_3/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_3/tokenizer_config.json",
    "tokenizer_3/tokenizer.json",
    "tokenizer_3/spiece.model",
    "tokenizer_3/special_tokens_map.json",
)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


@dataclass(frozen=True)
class SD3RuntimeAssets:
    profile: SD3RuntimeProfile
    root: Path

    scheduler_config: Path
    transformer_config: Path
    vae_config: Path

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

    tokenizer_3_dir: Path
    tokenizer_3_config: Path
    tokenizer_3_json: Path
    tokenizer_3_spiece_model: Path
    tokenizer_3_special_tokens_map: Path

    text_encoder_config: Path
    text_encoder_2_config: Path
    text_encoder_3_config: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "root": str(self.root),
            "scheduler_config": str(self.scheduler_config),
            "transformer_config": str(self.transformer_config),
            "vae_config": str(self.vae_config),
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
            "tokenizer_3_dir": str(self.tokenizer_3_dir),
            "tokenizer_3_config": str(self.tokenizer_3_config),
            "tokenizer_3_json": str(self.tokenizer_3_json),
            "tokenizer_3_spiece_model": str(self.tokenizer_3_spiece_model),
            "tokenizer_3_special_tokens_map": str(self.tokenizer_3_special_tokens_map),
            "text_encoder_config": str(self.text_encoder_config),
            "text_encoder_2_config": str(self.text_encoder_2_config),
            "text_encoder_3_config": str(self.text_encoder_3_config),
        }


    def to_resolved_configs(self, config_resolver: ConfigResolver) -> ResolvedConfigs:
        """Bridge local SD3 runtime assets into the shared component-builder contract."""
        return config_resolver.resolve_explicit(
            architecture="sd3.x",
            root_dir=str(self.root),
            manifest_path="",
            unet_config_path="",
            transformer_config_path=str(self.transformer_config),
            vae_config_path=str(self.vae_config),
            text_encoder_config_path=str(self.text_encoder_config),
            text_encoder_2_config_path=str(self.text_encoder_2_config),
            text_encoder_3_config_path=str(self.text_encoder_3_config),
            tokenizer_dir=str(self.tokenizer_dir),
            tokenizer_2_dir=str(self.tokenizer_2_dir),
            tokenizer_3_dir=str(self.tokenizer_3_dir),
            scheduler_config_path=str(self.scheduler_config),
        )

    def scheduler_payload(self) -> dict[str, Any]:
        return _load_json_object(self.scheduler_config, label="SD3 scheduler config")

    def transformer_payload(self) -> dict[str, Any]:
        return _load_json_object(self.transformer_config, label="SD3 transformer config")

    def vae_payload(self) -> dict[str, Any]:
        return _load_json_object(self.vae_config, label="SD3 VAE config")

    def text_encoder_payload(self) -> dict[str, Any]:
        return _load_json_object(self.text_encoder_config, label="SD3 text encoder config")

    def text_encoder_2_payload(self) -> dict[str, Any]:
        return _load_json_object(self.text_encoder_2_config, label="SD3 text encoder 2 config")

    def text_encoder_3_payload(self) -> dict[str, Any]:
        return _load_json_object(self.text_encoder_3_config, label="SD3 text encoder 3 config")

    @staticmethod
    def preferred_t5_tokenizer_class_name() -> str:
        try:
            from transformers import T5TokenizerFast  # type: ignore

            if T5TokenizerFast is not None:
                return "T5TokenizerFast"
        except Exception:
            pass
        return "T5Tokenizer"

    def contract_signature(self) -> dict[str, Any]:
        scheduler = self.scheduler_payload()
        transformer = self.transformer_payload()
        vae = self.vae_payload()
        text_encoder = self.text_encoder_payload()
        text_encoder_2 = self.text_encoder_2_payload()
        text_encoder_3 = self.text_encoder_3_payload()
        tokenizer = _load_json_object(self.tokenizer_config, label="SD3 tokenizer config")
        tokenizer_2 = _load_json_object(self.tokenizer_2_config, label="SD3 tokenizer 2 config")
        tokenizer_3 = _load_json_object(self.tokenizer_3_config, label="SD3 tokenizer 3 config")

        return {
            "profile_id": self.profile.profile_id,
            "scheduler_class": scheduler.get("_class_name"),
            "scheduler_num_train_timesteps": scheduler.get("num_train_timesteps"),
            "scheduler_shift": scheduler.get("shift"),
            "transformer_class": transformer.get("_class_name"),
            "transformer_num_layers": transformer.get("num_layers"),
            "transformer_num_attention_heads": transformer.get("num_attention_heads"),
            "transformer_attention_head_dim": transformer.get("attention_head_dim"),
            "transformer_in_channels": transformer.get("in_channels"),
            "transformer_out_channels": transformer.get("out_channels"),
            "transformer_joint_attention_dim": transformer.get("joint_attention_dim"),
            "transformer_pooled_projection_dim": transformer.get("pooled_projection_dim"),
            "transformer_caption_projection_dim": transformer.get("caption_projection_dim"),
            "transformer_patch_size": transformer.get("patch_size"),
            "transformer_pos_embed_max_size": transformer.get("pos_embed_max_size"),
            "transformer_sample_size": transformer.get("sample_size"),
            "transformer_qk_norm": transformer.get("qk_norm"),
            "transformer_dual_attention_layers": list(transformer.get("dual_attention_layers") or []),
            "vae_class": vae.get("_class_name"),
            "vae_latent_channels": vae.get("latent_channels"),
            "vae_scaling_factor": vae.get("scaling_factor"),
            "vae_shift_factor": vae.get("shift_factor"),
            "vae_sample_size": vae.get("sample_size"),
            "vae_force_upcast": bool(vae.get("force_upcast", False)),
            "vae_mid_block_add_attention": vae.get("mid_block_add_attention"),
            "text_encoder_architecture": list(text_encoder.get("architectures") or []),
            "text_encoder_hidden_size": text_encoder.get("hidden_size"),
            "text_encoder_projection_dim": text_encoder.get("projection_dim"),
            "text_encoder_max_positions": text_encoder.get("max_position_embeddings"),
            "text_encoder_2_architecture": list(text_encoder_2.get("architectures") or []),
            "text_encoder_2_hidden_size": text_encoder_2.get("hidden_size"),
            "text_encoder_2_projection_dim": text_encoder_2.get("projection_dim"),
            "text_encoder_2_max_positions": text_encoder_2.get("max_position_embeddings"),
            "text_encoder_3_architecture": list(text_encoder_3.get("architectures") or []),
            "text_encoder_3_d_model": text_encoder_3.get("d_model"),
            "text_encoder_3_num_layers": text_encoder_3.get("num_layers"),
            "text_encoder_3_num_heads": text_encoder_3.get("num_heads"),
            "tokenizer_class": tokenizer.get("tokenizer_class"),
            "tokenizer_model_max_length": tokenizer.get("model_max_length"),
            "tokenizer_2_class": tokenizer_2.get("tokenizer_class"),
            "tokenizer_2_model_max_length": tokenizer_2.get("model_max_length"),
            "tokenizer_3_class": tokenizer_3.get("tokenizer_class"),
            "tokenizer_3_model_max_length": tokenizer_3.get("model_max_length"),
            "preferred_t5_tokenizer_class": self.preferred_t5_tokenizer_class_name(),
        }

    def validate_contract_signature(self) -> dict[str, Any]:
        signature = self.contract_signature()
        profile = self.profile
        errors: list[str] = []

        def check_exact(field: str, expected: Any) -> None:
            if signature.get(field) != expected:
                errors.append(f"{field} must be {expected!r}, got {signature.get(field)!r}")

        def check_float(field: str, expected: float, *, tol: float = 1e-8) -> None:
            value = signature.get(field)
            try:
                actual = float(value)
            except (TypeError, ValueError):
                actual = math.nan
            if math.isnan(actual) or abs(actual - expected) > tol:
                errors.append(f"{field} must be {expected!r}, got {value!r}")

        check_exact("scheduler_class", profile.scheduler_class)
        check_exact("scheduler_num_train_timesteps", profile.scheduler_num_train_timesteps)
        check_float("scheduler_shift", profile.scheduler_shift)

        check_exact("transformer_class", profile.transformer_class)
        check_exact("transformer_num_layers", profile.transformer_num_layers)
        check_exact("transformer_num_attention_heads", profile.transformer_num_attention_heads)
        check_exact("transformer_attention_head_dim", profile.transformer_attention_head_dim)
        check_exact("transformer_in_channels", profile.transformer_in_channels)
        check_exact("transformer_out_channels", profile.transformer_out_channels)
        check_exact("transformer_joint_attention_dim", profile.transformer_joint_attention_dim)
        check_exact("transformer_pooled_projection_dim", profile.transformer_pooled_projection_dim)
        check_exact("transformer_caption_projection_dim", profile.transformer_caption_projection_dim)
        check_exact("transformer_patch_size", profile.transformer_patch_size)
        check_exact("transformer_sample_size", profile.transformer_sample_size)
        check_exact("transformer_pos_embed_max_size", profile.transformer_pos_embed_max_size)

        expected_qk_norm = profile.transformer_qk_norm or None
        actual_qk_norm = signature.get("transformer_qk_norm")
        if actual_qk_norm == "":
            actual_qk_norm = None
        if actual_qk_norm != expected_qk_norm:
            errors.append(
                f"transformer_qk_norm must be {expected_qk_norm!r}, got {signature.get('transformer_qk_norm')!r}"
            )

        expected_dual_attention = list(profile.transformer_dual_attention_layers)
        if list(signature.get("transformer_dual_attention_layers") or []) != expected_dual_attention:
            errors.append(
                "transformer_dual_attention_layers must be "
                f"{expected_dual_attention!r}, got {signature.get('transformer_dual_attention_layers')!r}"
            )

        check_exact("vae_class", profile.vae_class)
        check_exact("vae_latent_channels", profile.vae_latent_channels)
        check_exact("vae_sample_size", profile.vae_sample_size)
        check_float("vae_scaling_factor", profile.vae_scaling_factor)
        check_float("vae_shift_factor", profile.vae_shift_factor)
        check_exact("vae_force_upcast", profile.vae_force_upcast)
        if profile.vae_mid_block_add_attention is not None:
            check_exact("vae_mid_block_add_attention", profile.vae_mid_block_add_attention)

        if "CLIPTextModelWithProjection" not in set(signature.get("text_encoder_architecture") or []):
            errors.append("text_encoder_architecture must include 'CLIPTextModelWithProjection'")
        check_exact("text_encoder_hidden_size", 768)
        check_exact("text_encoder_projection_dim", 768)
        check_exact("text_encoder_max_positions", 77)

        if "CLIPTextModelWithProjection" not in set(signature.get("text_encoder_2_architecture") or []):
            errors.append("text_encoder_2_architecture must include 'CLIPTextModelWithProjection'")
        check_exact("text_encoder_2_hidden_size", 1280)
        check_exact("text_encoder_2_projection_dim", 1280)
        check_exact("text_encoder_2_max_positions", 77)

        if "T5EncoderModel" not in set(signature.get("text_encoder_3_architecture") or []):
            errors.append("text_encoder_3_architecture must include 'T5EncoderModel'")
        check_exact("text_encoder_3_d_model", 4096)
        check_exact("text_encoder_3_num_layers", 24)
        check_exact("text_encoder_3_num_heads", 64)

        check_exact("tokenizer_class", "CLIPTokenizer")
        check_exact("tokenizer_model_max_length", 77)
        check_exact("tokenizer_2_class", "CLIPTokenizer")
        check_exact("tokenizer_2_model_max_length", 77)
        tokenizer_3_class = str(signature.get("tokenizer_3_class") or "")
        if tokenizer_3_class not in {"T5Tokenizer", "T5TokenizerFast"}:
            errors.append(
                f"tokenizer_3_class must be 'T5Tokenizer' or 'T5TokenizerFast', got {tokenizer_3_class!r}"
            )
        check_exact("tokenizer_3_model_max_length", 512)

        if errors:
            details = "\n  - ".join(errors)
            raise ValueError(
                "SD3 runtime assets do not match the required contract for "
                f"{profile.display_name}:\n  - {details}\nRuntime asset root: {self.root}"
            )
        return signature


class SD3RuntimeAssetResolver:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    @staticmethod
    def _required(root: Path, relative_hint: str, *, extensions: set[str] | None = None) -> Path:
        resolved = resolve_nested_asset(
            root,
            relative_hint,
            extensions=extensions,
            allow_stem_match=False,
        )
        if resolved is None:
            raise FileNotFoundError(
                f"Required SD3 runtime asset {relative_hint!r} was not found under: {root}"
            )
        return resolved

    @staticmethod
    def _candidate_root_from_relative(resolved_file: Path, relative_hint: str) -> Path:
        relative_parts = Path(relative_hint.replace("\\", "/")).parts
        candidate = resolved_file
        for _ in relative_parts:
            candidate = candidate.parent
        return candidate.resolve()

    def canonical_root(self, profile: SD3RuntimeProfile) -> Path:
        preferred = (Path(self.context.runtime_assets_root) / profile.runtime_assets_subdir).resolve()
        candidates = [preferred]
        for relative in LEGACY_SD3_RUNTIME_ASSET_SUBDIRS.get(profile.profile_id, ()):
            candidate = (Path(self.context.runtime_assets_root) / relative).resolve()
            if candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        checked = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "SD3 runtime asset root does not exist for the selected profile. Checked:\n  - "
            f"{checked}"
        )

    def resolve(self, profile: SD3RuntimeProfile | str) -> SD3RuntimeAssets:
        resolved_profile = profile_from_id(profile) if isinstance(profile, str) else profile
        if resolved_profile is None:
            raise ValueError(f"Unknown SD3 runtime profile: {profile!r}")
        return self.resolve_from_root(self.canonical_root(resolved_profile), resolved_profile)

    def resolve_from_root(self, root: str | Path, profile: SD3RuntimeProfile | str) -> SD3RuntimeAssets:
        resolved_profile = profile_from_id(profile) if isinstance(profile, str) else profile
        if resolved_profile is None:
            raise ValueError(f"Unknown SD3 runtime profile: {profile!r}")

        search_root = Path(root).expanduser().resolve()
        if not search_root.exists():
            raise FileNotFoundError(f"SD3 runtime asset root does not exist: {search_root}")

        scheduler_config = self._required(search_root, "scheduler/scheduler_config.json", extensions={".json"})
        contract_root = self._candidate_root_from_relative(scheduler_config, "scheduler/scheduler_config.json")
        if not contract_root.is_dir():
            raise FileNotFoundError(f"Derived SD3 runtime contract root does not exist: {contract_root}")

        def required(relative_hint: str, *, extensions: set[str] | None = None) -> Path:
            candidate = (contract_root / relative_hint).resolve()
            if candidate.is_file() and (extensions is None or candidate.suffix.lower() in extensions):
                return candidate
            resolved = self._required(contract_root, relative_hint, extensions=extensions)
            derived = self._candidate_root_from_relative(resolved, relative_hint)
            if derived != contract_root:
                raise FileNotFoundError(
                    f"Required SD3 runtime asset {relative_hint!r} resolved outside the selected contract root: {resolved}"
                )
            return resolved

        tokenizer_config = required("tokenizer/tokenizer_config.json", extensions={".json"})
        tokenizer_vocab = required("tokenizer/vocab.json", extensions={".json"})
        tokenizer_merges = required("tokenizer/merges.txt", extensions={".txt"})
        tokenizer_special_tokens_map = required("tokenizer/special_tokens_map.json", extensions={".json"})

        tokenizer_2_config = required("tokenizer_2/tokenizer_config.json", extensions={".json"})
        tokenizer_2_vocab = required("tokenizer_2/vocab.json", extensions={".json"})
        tokenizer_2_merges = required("tokenizer_2/merges.txt", extensions={".txt"})
        tokenizer_2_special_tokens_map = required("tokenizer_2/special_tokens_map.json", extensions={".json"})

        tokenizer_3_config = required("tokenizer_3/tokenizer_config.json", extensions={".json"})
        tokenizer_3_json = required("tokenizer_3/tokenizer.json", extensions={".json"})
        tokenizer_3_spiece_model = required("tokenizer_3/spiece.model")
        tokenizer_3_special_tokens_map = required("tokenizer_3/special_tokens_map.json", extensions={".json"})

        assets = SD3RuntimeAssets(
            profile=resolved_profile,
            root=contract_root,
            scheduler_config=scheduler_config,
            transformer_config=required("transformer/config.json", extensions={".json"}),
            vae_config=required("vae/config.json", extensions={".json"}),
            tokenizer_dir=tokenizer_config.parent,
            tokenizer_config=tokenizer_config,
            tokenizer_vocab=tokenizer_vocab,
            tokenizer_merges=tokenizer_merges,
            tokenizer_special_tokens_map=tokenizer_special_tokens_map,
            tokenizer_2_dir=tokenizer_2_config.parent,
            tokenizer_2_config=tokenizer_2_config,
            tokenizer_2_vocab=tokenizer_2_vocab,
            tokenizer_2_merges=tokenizer_2_merges,
            tokenizer_2_special_tokens_map=tokenizer_2_special_tokens_map,
            tokenizer_3_dir=tokenizer_3_config.parent,
            tokenizer_3_config=tokenizer_3_config,
            tokenizer_3_json=tokenizer_3_json,
            tokenizer_3_spiece_model=tokenizer_3_spiece_model,
            tokenizer_3_special_tokens_map=tokenizer_3_special_tokens_map,
            text_encoder_config=required("text_encoder/config.json", extensions={".json"}),
            text_encoder_2_config=required("text_encoder_2/config.json", extensions={".json"}),
            text_encoder_3_config=required("text_encoder_3/config.json", extensions={".json"}),
        )
        assets.validate_contract_signature()
        return assets
