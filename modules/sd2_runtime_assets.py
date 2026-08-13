from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from modules.asset_discovery import resolve_nested_asset
from modules.project_context import ProjectContext
from modules.sd2_runtime_profile import SD2RuntimeProfile


@dataclass(frozen=True)
class SD2RuntimeAssets:
    """Lightweight assets required to run an SD2-family checkpoint.

    Heavy component weights intentionally do not belong here. Runtime model
    weights come from the selected monolithic checkpoint. Hugging Face component
    weights, when retained for qualification/model-tooling work, are resolved by
    ``SD2ReferenceComponentResolver`` instead.
    """

    root: Path
    tokenizer_dir: Path
    tokenizer_config: Path
    tokenizer_vocab: Path
    tokenizer_merges: Path
    special_tokens_map: Path | None
    text_encoder_config: Path
    scheduler_config: Path
    unet_config: Path
    vae_config: Path
    model_index: Path | None = None
    feature_extractor_config: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "tokenizer_dir": str(self.tokenizer_dir),
            "tokenizer_config": str(self.tokenizer_config),
            "tokenizer_vocab": str(self.tokenizer_vocab),
            "tokenizer_merges": str(self.tokenizer_merges),
            "special_tokens_map": str(self.special_tokens_map) if self.special_tokens_map else None,
            "text_encoder_config": str(self.text_encoder_config),
            "scheduler_config": str(self.scheduler_config),
            "unet_config": str(self.unet_config),
            "vae_config": str(self.vae_config),
            "model_index": str(self.model_index) if self.model_index else None,
            "feature_extractor_config": str(self.feature_extractor_config) if self.feature_extractor_config else None,
        }

    def scheduler_payload(self) -> dict[str, Any]:
        payload = json.loads(self.scheduler_config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Scheduler config must contain a JSON object: {self.scheduler_config}")
        return payload

    def text_encoder_payload(self) -> dict[str, Any]:
        payload = json.loads(self.text_encoder_config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Text encoder config must contain a JSON object: {self.text_encoder_config}")
        return payload


class SD2RuntimeAssetResolver:
    """Resolve lightweight SD2 runtime assets independently from model storage."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def profile_root(self, profile: SD2RuntimeProfile) -> Path:
        configured = self.context.runtime_assets_root
        subdir = str(profile.runtime_assets_subdir or "").strip()
        root = (configured / subdir).resolve() if subdir else configured.resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"SD2 runtime asset root does not exist for profile {profile.profile_id!r}: {root}"
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
        if resolved is None:
            raise FileNotFoundError(
                f"Required SD2 runtime asset {relative_hint!r} was not found anywhere under: {root}"
            )
        return resolved

    @staticmethod
    def _optional(root: Path, relative_hint: str, *, extensions: set[str] | None = None) -> Path | None:
        return resolve_nested_asset(
            root,
            relative_hint,
            extensions=extensions,
            allow_stem_match=False,
        )

    def resolve(self, profile: SD2RuntimeProfile) -> SD2RuntimeAssets:
        return self.resolve_from_root(self.profile_root(profile))

    def resolve_from_root(self, root: str | Path) -> SD2RuntimeAssets:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"SD2 runtime asset root does not exist: {root}")

        tokenizer_config = self._required(root, "tokenizer/tokenizer_config.json", extensions={".json"})
        tokenizer_vocab = self._required(root, "tokenizer/vocab.json", extensions={".json"})
        tokenizer_merges = self._required(root, "tokenizer/merges.txt", extensions={".txt"})
        tokenizer_dir = tokenizer_config.parent

        return SD2RuntimeAssets(
            root=root,
            tokenizer_dir=tokenizer_dir,
            tokenizer_config=tokenizer_config,
            tokenizer_vocab=tokenizer_vocab,
            tokenizer_merges=tokenizer_merges,
            special_tokens_map=self._optional(root, "tokenizer/special_tokens_map.json", extensions={".json"}),
            text_encoder_config=self._required(root, "text_encoder/config.json", extensions={".json"}),
            scheduler_config=self._required(root, "scheduler/scheduler_config.json", extensions={".json"}),
            unet_config=self._required(root, "unet/config.json", extensions={".json"}),
            vae_config=self._required(root, "vae/config.json", extensions={".json"}),
            model_index=self._optional(root, "model_index.json", extensions={".json"}),
            feature_extractor_config=self._optional(
                root,
                "feature_extractor/preprocessor_config.json",
                extensions={".json"},
            ),
        )
