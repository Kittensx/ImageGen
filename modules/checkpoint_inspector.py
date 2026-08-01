from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

from safetensors import safe_open
from safetensors.torch import load_file


def _display_architecture(value: str | None) -> str:
    architecture = str(value or "").strip().lower()
    mapping = {
        "sd1.x": "SD 1.x",
        "sd2.x": "SD 2.x",
        "sdxl": "SDXL",
        "sd1.x_or_sd2.x": "SD 1.x / SD 2.x",
        "unknown": "Unknown",
        "": "Unknown",
    }
    return mapping.get(architecture, str(value or "Unknown"))


def _normalize_prediction_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    mapping = {
        "eps": "epsilon",
        "epsilon": "epsilon",
        "epsilon_prediction": "epsilon",
        "v": "v-prediction",
        "v_prediction": "v-prediction",
        "vpred": "v-prediction",
        "v_pred": "v-prediction",
        "sample": "sample",
        "sample_prediction": "sample",
    }
    return mapping.get(text, "")


def _default_prediction_type_for_architecture(architecture: str | None) -> str:
    normalized = str(architecture or "").strip().lower()
    defaults = {
        "sd1.x": "epsilon",
        "sd2.x": "v-prediction",
        "sdxl": "epsilon",
    }
    return defaults.get(normalized, "")


def _default_conditioning_dimension_for_architecture(architecture: str | None) -> int | None:
    normalized = str(architecture or "").strip().lower()
    defaults = {
        "sd1.x": 768,
        "sd2.x": 1024,
        "sdxl": 2048,
    }
    return defaults.get(normalized)


_METADATA_PREDICTION_KEYS = (
    "modelspec.prediction_type",
    "prediction_type",
    "parameterization",
    "model.parameterization",
    "ss_prediction_type",
    "ss_parameterization",
)


def detect_prediction_type(metadata: Mapping[str, Any] | None, architecture: str | None = None) -> tuple[str, str]:
    source = dict(metadata or {})
    for key in _METADATA_PREDICTION_KEYS:
        if key in source:
            normalized = _normalize_prediction_type(source.get(key))
            if normalized:
                return normalized, "metadata"
    inferred = _default_prediction_type_for_architecture(architecture)
    if inferred:
        return inferred, "architecture_default"
    return "", ""


@dataclass
class ArchitectureContract:
    family: str = ""
    prediction_type: str = ""
    conditioning_dimension: int | None = None
    summary: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "prediction_type": self.prediction_type,
            "conditioning_dimension": self.conditioning_dimension,
            "summary": self.summary,
            "source": self.source,
        }


def build_architecture_contract(
    architecture: str | None = None,
    prediction_type: str | None = None,
    conditioning_dimension: Any = None,
    *,
    summary: str | None = None,
    source: str | None = None,
) -> ArchitectureContract:
    family_display = _display_architecture(architecture)
    normalized_prediction = _normalize_prediction_type(prediction_type)
    dimension: int | None
    try:
        dimension = int(conditioning_dimension) if conditioning_dimension not in (None, "") else None
    except (TypeError, ValueError):
        dimension = None
    if dimension is None:
        dimension = _default_conditioning_dimension_for_architecture(architecture)

    pieces = []
    if family_display and family_display != "Unknown":
        pieces.append(family_display)
    if normalized_prediction:
        pieces.append(normalized_prediction)
    if dimension is not None:
        pieces.append(str(dimension))
    resolved_summary = str(summary or "").strip() or " / ".join(pieces)
    resolved_source = str(source or "").strip()
    return ArchitectureContract(
        family=family_display,
        prediction_type=normalized_prediction,
        conditioning_dimension=dimension,
        summary=resolved_summary,
        source=resolved_source,
    )


@dataclass
class CheckpointReport:
    model_path: str
    file_name: str
    total_keys: int
    architecture: str
    checkpoint_kind: str
    key_prefixes: List[str] = field(default_factory=list)
    example_keys: List[str] = field(default_factory=list)
    has_unet: bool = False
    has_vae: bool = False
    has_text_encoder: bool = False
    has_sdxl_text_encoder_2: bool = False
    file_size_bytes: int = 0
    sha256: str = ""
    dtype_summary: dict[str, int] = field(default_factory=dict)
    tensor_shape_summary: dict[str, list[int]] = field(default_factory=dict)
    architecture_evidence: list[str] = field(default_factory=list)
    safetensors_metadata: dict[str, str] = field(default_factory=dict)
    model_dimension: int | None = None
    prediction_type: str = ""
    prediction_type_source: str = ""
    architecture_summary: str = ""
    architecture_source: str = ""

    @property
    def architecture_contract(self) -> ArchitectureContract:
        return build_architecture_contract(
            self.architecture,
            self.prediction_type,
            self.model_dimension,
            summary=self.architecture_summary,
            source=self.architecture_source or self.prediction_type_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "total_keys": self.total_keys,
            "architecture": self.architecture,
            "architecture_evidence": list(self.architecture_evidence),
            "checkpoint_kind": self.checkpoint_kind,
            "key_prefixes": list(self.key_prefixes),
            "example_keys": list(self.example_keys),
            "has_unet": self.has_unet,
            "has_vae": self.has_vae,
            "has_text_encoder": self.has_text_encoder,
            "has_sdxl_text_encoder_2": self.has_sdxl_text_encoder_2,
            "dtype_summary": dict(self.dtype_summary),
            "tensor_shape_summary": dict(self.tensor_shape_summary),
            "safetensors_metadata": dict(self.safetensors_metadata),
            "model_dimension": self.model_dimension,
            "prediction_type": self.prediction_type,
            "prediction_type_source": self.prediction_type_source,
            "architecture_summary": self.architecture_summary,
            "architecture_source": self.architecture_source,
            "architecture_contract": self.architecture_contract.to_dict(),
        }


class CheckpointInspector:
    """Inspect monolithic safetensors checkpoints without materializing tensors.

    ``inspect`` reads the safetensors header, shapes, dtypes, and file checksum.
    The full tensor state is loaded only by ``load_state_dict`` when component
    construction actually begins.
    """

    _TEXT_EMBEDDING_SUFFIXES = (
        "text_model.embeddings.token_embedding.weight",
        "token_embedding.weight",
    )

    def load_state_dict(self, model_path: str) -> Dict[str, object]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return load_file(model_path)

    @staticmethod
    def sha256_file(model_path: str | os.PathLike[str]) -> str:
        digest = hashlib.sha256()
        with Path(model_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def inspect(self, model_path: str) -> CheckpointReport:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        if path.suffix.lower() != ".safetensors":
            raise ValueError(
                f"Phase 07 validation requires a .safetensors checkpoint, got: {path.name}"
            )

        keys: list[str] = []
        shapes: dict[str, tuple[int, ...]] = {}
        dtype_counter: Counter[str] = Counter()
        metadata: Mapping[str, str] | None = None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
            for key in keys:
                tensor_slice = handle.get_slice(key)
                shape = tuple(int(value) for value in tensor_slice.get_shape())
                shapes[key] = shape
                dtype_counter[str(tensor_slice.get_dtype())] += 1

        prefixes = self._collect_prefixes(keys)
        has_unet = any(k.startswith("model.diffusion_model.") for k in keys)
        has_vae = any(k.startswith("first_stage_model.") for k in keys)
        has_text_encoder = any(k.startswith("cond_stage_model.") for k in keys)
        has_te2 = any(
            "conditioner.embedders.1" in k or "text_encoder_2" in k for k in keys
        )

        checkpoint_kind = self._detect_checkpoint_kind(
            keys, has_unet, has_vae, has_text_encoder
        )
        architecture, evidence, model_dimension = self._detect_architecture(keys, shapes, has_te2)
        prediction_type, prediction_source = detect_prediction_type(metadata, architecture)
        contract = build_architecture_contract(
            architecture,
            prediction_type,
            model_dimension,
            source=prediction_source,
        )

        selected_shapes = self._select_shape_summary(keys, shapes)
        return CheckpointReport(
            model_path=str(path),
            file_name=path.name,
            total_keys=len(keys),
            architecture=architecture,
            checkpoint_kind=checkpoint_kind,
            key_prefixes=prefixes[:20],
            example_keys=keys[:20],
            has_unet=has_unet,
            has_vae=has_vae,
            has_text_encoder=has_text_encoder,
            has_sdxl_text_encoder_2=has_te2,
            file_size_bytes=path.stat().st_size,
            sha256=self.sha256_file(path),
            dtype_summary=dict(sorted(dtype_counter.items())),
            tensor_shape_summary={
                key: list(shape) for key, shape in selected_shapes.items()
            },
            architecture_evidence=evidence,
            safetensors_metadata=dict(metadata or {}),
            model_dimension=model_dimension,
            prediction_type=prediction_type,
            prediction_type_source=prediction_source,
            architecture_summary=contract.summary,
            architecture_source=contract.source,
        )

    def _collect_prefixes(self, keys: List[str]) -> List[str]:
        seen: Set[str] = set()
        prefixes: List[str] = []
        for key in keys:
            parts = key.split(".")
            prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            if prefix not in seen:
                seen.add(prefix)
                prefixes.append(prefix)
        return sorted(prefixes)

    def _detect_checkpoint_kind(
        self,
        keys: List[str],
        has_unet: bool,
        has_vae: bool,
        has_text_encoder: bool,
    ) -> str:
        lora_signals = any(
            ".lora_" in k or "lora_up." in k or "lora_down." in k for k in keys
        )
        if lora_signals and not has_unet and not has_vae:
            return "lora"
        if has_unet and has_vae and has_text_encoder:
            return "full"
        if has_unet or has_vae or has_text_encoder:
            return "partial"
        return "unknown"

    def _detect_architecture(
        self,
        keys: List[str],
        shapes: Mapping[str, tuple[int, ...]],
        has_te2: bool,
    ) -> tuple[str, list[str], int | None]:
        evidence: list[str] = []
        if has_te2 or any(k.startswith("conditioner.embedders.") for k in keys):
            evidence.append("dual/conditioner text-encoder key detected")
            return "sdxl", evidence, 2048

        dimensions: set[int] = set()
        for key, shape in shapes.items():
            if key.startswith("cond_stage_model.") and key.endswith(
                self._TEXT_EMBEDDING_SUFFIXES
            ):
                if len(shape) >= 2:
                    dimensions.add(int(shape[-1]))
                    evidence.append(f"{key} hidden_size={shape[-1]}")
            if (
                key.startswith("model.diffusion_model.")
                and (".attn2.to_k.weight" in key or ".attn2.to_v.weight" in key)
                and len(shape) == 2
            ):
                dimensions.add(int(shape[1]))
                if len(evidence) < 8:
                    evidence.append(f"{key} cross_attention_dim={shape[1]}")

        if 2048 in dimensions:
            evidence.append("cross-attention dimension 2048 is SDXL-style")
            return "sdxl", evidence, 2048
        if 1024 in dimensions and 768 not in dimensions:
            evidence.append("text/cross-attention dimension 1024 is SD2-style")
            return "sd2.x", evidence, 1024
        if 768 in dimensions and 1024 not in dimensions:
            evidence.append("text/cross-attention dimension 768 is SD1-style")
            return "sd1.x", evidence, 768
        if 768 in dimensions and 1024 in dimensions:
            evidence.append("conflicting 768 and 1024 conditioning dimensions")
            return "unknown", evidence, None
        if any(k.startswith("model.diffusion_model.") for k in keys):
            evidence.append("UNet keys found but conditioning dimension was not identifiable")
            return "sd1.x_or_sd2.x", evidence, None
        return "unknown", evidence, None

    @staticmethod
    def _select_shape_summary(
        keys: list[str],
        shapes: Mapping[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        selected: dict[str, tuple[int, ...]] = {}
        preferred_fragments = (
            "token_embedding.weight",
            "input_blocks.0.0.weight",
            "attn2.to_k.weight",
            "first_stage_model.encoder.conv_in.weight",
        )
        for key in keys:
            if any(fragment in key for fragment in preferred_fragments):
                selected[key] = shapes[key]
                if len(selected) >= 12:
                    break
        return selected
