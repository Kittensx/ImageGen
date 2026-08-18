from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

from safetensors import safe_open
from safetensors.torch import load_file

from modules.model_qualification_registry import qualification_for_sha256
from modules.sd2_runtime_profile import normalize_prediction_type


def _display_architecture(value: str | None) -> str:
    architecture = str(value or "").strip().lower()
    mapping = {
        "sd1.x": "SD 1.x",
        "sd2.x": "SD 2.x",
        "sdxl": "SDXL",
        "sd3.x": "SD3.x",
        "sd1.x_or_sd2.x": "SD 1.x / SD 2.x",
        "unknown": "Unknown",
        "": "Unknown",
    }
    return mapping.get(architecture, str(value or "Unknown"))


def _normalize_prediction_type(value: Any) -> str:
    return normalize_prediction_type(value)


def _default_prediction_type_for_architecture(architecture: str | None) -> str:
    normalized = str(architecture or "").strip().lower()
    defaults = {
        "sd1.x": "epsilon",
        "sdxl": "epsilon",
    }
    return defaults.get(normalized, "")


def _default_conditioning_dimension_for_architecture(architecture: str | None) -> int | None:
    normalized = str(architecture or "").strip().lower()
    defaults = {
        "sd1.x": 768,
        "sd2.x": 1024,
        "sdxl": 2048,
        "sd3.x": 4096,
    }
    return defaults.get(normalized)


_METADATA_MODEL_NAME_KEYS = (
    "modelspec.title",
    "modelspec.name",
)


def _looks_like_version_only(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    compact = text.replace("version", "").strip()
    if compact.startswith("v"):
        compact = compact[1:].strip()
    return bool(compact) and all(ch.isdigit() or ch in ".-_ " for ch in compact)


def detect_model_name(
    metadata: Mapping[str, Any] | None,
    fallback: str | None = None,
) -> tuple[str, str]:
    """Resolve a human checkpoint name from safetensors metadata safely.

    Only ModelSpec title/name fields are considered authoritative enough to
    replace the local filename stem. Generic metadata such as ``name`` or
    provider/version labels is intentionally ignored. Obvious version-only
    values (for example ``v1`` or ``1.0``) also fall back to the filename.
    """
    source = dict(metadata or {})
    for key in _METADATA_MODEL_NAME_KEYS:
        value = str(source.get(key) or "").strip()
        if value and not _looks_like_version_only(value):
            return value, key
    fallback_text = str(fallback or "").strip()
    return fallback_text, "filename" if fallback_text else ""

_METADATA_PREDICTION_KEYS = (
    "modelspec.prediction_type",
    "prediction_type",
    "parameterization",
    "model.parameterization",
    "ss_prediction_type",
    "ss_parameterization",
)


def detect_prediction_type(
    metadata: Mapping[str, Any] | None,
    architecture: str | None = None,
    *,
    filename: str | None = None,
) -> tuple[str, str]:
    normalized_architecture = str(architecture or "").strip().lower()
    source = dict(metadata or {})
    if normalized_architecture == "sd3.x":
        return "", "not_applicable_flow_match"
    if normalized_architecture == "sd2.x":
        for key in _METADATA_PREDICTION_KEYS:
            if key not in source:
                continue
            normalized = _normalize_prediction_type(source.get(key))
            if normalized:
                return normalized, "metadata"
        return "", ""

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
    denoising_domain: str = ""
    denoiser_type: str = ""
    latent_channels: int | None = None
    summary: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "prediction_type": self.prediction_type,
            "conditioning_dimension": self.conditioning_dimension,
            "denoising_domain": self.denoising_domain,
            "denoiser_type": self.denoiser_type,
            "latent_channels": self.latent_channels,
            "summary": self.summary,
            "source": self.source,
        }


def build_architecture_contract(
    architecture: str | None = None,
    prediction_type: str | None = None,
    conditioning_dimension: Any = None,
    *,
    denoising_domain: str | None = None,
    denoiser_type: str | None = None,
    latent_channels: Any = None,
    summary: str | None = None,
    source: str | None = None,
) -> ArchitectureContract:
    normalized_architecture = str(architecture or "").strip().lower()
    family_display = _display_architecture(architecture)
    normalized_prediction = _normalize_prediction_type(prediction_type)
    dimension: int | None
    try:
        dimension = int(conditioning_dimension) if conditioning_dimension not in (None, "") else None
    except (TypeError, ValueError):
        dimension = None
    if dimension is None:
        dimension = _default_conditioning_dimension_for_architecture(architecture)

    default_domain = "flow_match" if normalized_architecture == "sd3.x" else (
        "vp_sigma" if normalized_architecture in {"sd1.x", "sd2.x", "sdxl", "sd1.x_or_sd2.x"} else ""
    )
    default_denoiser = "transformer" if normalized_architecture == "sd3.x" else (
        "unet" if normalized_architecture in {"sd1.x", "sd2.x", "sdxl", "sd1.x_or_sd2.x"} else ""
    )
    resolved_domain = str(denoising_domain or default_domain).strip()
    resolved_denoiser = str(denoiser_type or default_denoiser).strip()
    try:
        resolved_latent_channels = int(latent_channels) if latent_channels not in (None, "") else None
    except (TypeError, ValueError):
        resolved_latent_channels = None
    if resolved_latent_channels is None:
        if normalized_architecture == "sd3.x":
            resolved_latent_channels = 16
        elif normalized_architecture in {"sd1.x", "sd2.x", "sdxl"}:
            resolved_latent_channels = 4

    pieces = []
    if family_display and family_display != "Unknown":
        pieces.append(family_display)
    if normalized_prediction:
        pieces.append(normalized_prediction)
    if normalized_architecture == "sd3.x":
        if resolved_domain:
            pieces.append(resolved_domain)
        if resolved_denoiser:
            pieces.append(resolved_denoiser)
    if dimension is not None:
        pieces.append(str(dimension))
    resolved_summary = str(summary or "").strip() or " / ".join(pieces)
    resolved_source = str(source or "").strip()
    return ArchitectureContract(
        family=family_display,
        prediction_type=normalized_prediction,
        conditioning_dimension=dimension,
        denoising_domain=resolved_domain,
        denoiser_type=resolved_denoiser,
        latent_channels=resolved_latent_channels,
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
    has_sdxl_text_encoder_1: bool = False
    has_sdxl_text_encoder_2: bool = False
    architecture_variant: str = ""
    denoiser_type: str = ""
    has_transformer: bool = False
    has_text_encoder_3: bool = False
    has_clip_l: bool = False
    has_clip_g: bool = False
    has_t5: bool = False
    text_encoder_packaging: str = ""
    checkpoint_packaging: str = ""
    latent_channels: int | None = None
    flow_matching: bool = False
    denoising_domain: str = ""
    key_prefix_counts: dict[str, int] = field(default_factory=dict)
    representative_keys_by_prefix: dict[str, list[str]] = field(default_factory=dict)
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
    model_name: str = ""
    model_name_source: str = ""

    @property
    def architecture_contract(self) -> ArchitectureContract:
        return build_architecture_contract(
            self.architecture,
            self.prediction_type,
            self.model_dimension,
            denoising_domain=self.denoising_domain,
            denoiser_type=self.denoiser_type,
            latent_channels=self.latent_channels,
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
            "has_sdxl_text_encoder_1": self.has_sdxl_text_encoder_1,
            "has_sdxl_text_encoder_2": self.has_sdxl_text_encoder_2,
            "architecture_variant": self.architecture_variant,
            "denoiser_type": self.denoiser_type,
            "has_transformer": self.has_transformer,
            "has_text_encoder_3": self.has_text_encoder_3,
            "has_clip_l": self.has_clip_l,
            "has_clip_g": self.has_clip_g,
            "has_t5": self.has_t5,
            "text_encoder_packaging": self.text_encoder_packaging,
            "checkpoint_packaging": self.checkpoint_packaging,
            "latent_channels": self.latent_channels,
            "flow_matching": self.flow_matching,
            "denoising_domain": self.denoising_domain,
            "key_prefix_counts": dict(self.key_prefix_counts),
            "representative_keys_by_prefix": {
                key: list(value) for key, value in self.representative_keys_by_prefix.items()
            },
            "dtype_summary": dict(self.dtype_summary),
            "tensor_shape_summary": dict(self.tensor_shape_summary),
            "safetensors_metadata": dict(self.safetensors_metadata),
            "model_dimension": self.model_dimension,
            "prediction_type": self.prediction_type,
            "prediction_type_source": self.prediction_type_source,
            "architecture_summary": self.architecture_summary,
            "architecture_source": self.architecture_source,
            "model_name": self.model_name,
            "model_name_source": self.model_name_source,
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

    @staticmethod
    def _is_sdxl_text_encoder_1_key(key: str) -> bool:
        return key.startswith("conditioner.embedders.0.") or key.startswith("text_encoder.")

    @staticmethod
    def _is_sdxl_text_encoder_2_key(key: str) -> bool:
        return key.startswith("conditioner.embedders.1.") or key.startswith("text_encoder_2.")

    @classmethod
    def _is_primary_text_encoder_key(cls, key: str) -> bool:
        return key.startswith("cond_stage_model.") or cls._is_sdxl_text_encoder_1_key(key)

    @staticmethod
    def _is_sd3_clip_l_key(key: str) -> bool:
        return key.startswith("text_encoders.clip_l.") or key.startswith("clip_l.")

    @staticmethod
    def _is_sd3_clip_g_key(key: str) -> bool:
        return key.startswith("text_encoders.clip_g.") or key.startswith("clip_g.")

    @staticmethod
    def _is_sd3_t5_key(key: str) -> bool:
        return (
            key.startswith("text_encoders.t5xxl.")
            or key.startswith("t5xxl.")
            or key.startswith("text_encoder_3.")
        )

    @staticmethod
    def _has_sd3_transformer_signature(keys: List[str]) -> bool:
        normalized = set(keys)
        signature_keys = {
            "joint_blocks.0.context_block.adaLN_modulation.1.bias",
            "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias",
        }
        if normalized.intersection(signature_keys):
            return True
        structural_fragments = (
            ".joint_blocks.0.context_block.",
            ".joint_blocks.0.x_block.",
            ".x_embedder.",
            ".context_embedder.",
        )
        matches = {fragment for fragment in structural_fragments if any(fragment in f".{key}" for key in keys)}
        return len(matches) >= 3

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

    def inspect_architecture_contract(self, model_path: str) -> ArchitectureContract:
        """Inspect checkpoint architecture from the Safetensors header without hashing payload bytes."""

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        if path.suffix.lower() != ".safetensors":
            raise ValueError(f"Architecture preflight requires a .safetensors checkpoint, got: {path.name}")

        keys: list[str] = []
        shapes: dict[str, tuple[int, ...]] = {}
        metadata: Mapping[str, str] | None = None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
            for key in keys:
                if not (
                    key.startswith("cond_stage_model.")
                    or key.startswith("model.diffusion_model.")
                    or key.startswith("conditioner.embedders.")
                    or key.startswith("text_encoder.")
                    or "text_encoder_2" in key
                    or key.startswith("text_encoders.")
                    or key.startswith("clip_l.")
                    or key.startswith("clip_g.")
                    or key.startswith("t5xxl.")
                    or "joint_blocks." in key
                    or key.endswith("pos_embed")
                    or ".x_embedder." in key
                    or ".context_embedder." in key
                ):
                    continue
                try:
                    shapes[key] = tuple(int(value) for value in handle.get_slice(key).get_shape())
                except Exception:
                    continue

        has_te2 = any(self._is_sdxl_text_encoder_2_key(key) for key in keys)
        architecture, _, model_dimension = self._detect_architecture(keys, shapes, has_te2)
        prediction_type, prediction_source = detect_prediction_type(metadata, architecture, filename=path.name)
        return build_architecture_contract(
            architecture,
            prediction_type,
            model_dimension,
            source=prediction_source or "checkpoint_header_preflight",
        )

    def inspect(self, model_path: str, *, compute_sha256: bool = True) -> CheckpointReport:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        if path.suffix.lower() != ".safetensors":
            raise ValueError(
                f"Checkpoint inspection requires a .safetensors checkpoint, got: {path.name}"
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
        prefix_counts, representative_keys = self._summarize_component_prefixes(keys)
        has_sd3_transformer = self._has_sd3_transformer_signature(keys)
        has_unet = any(k.startswith("model.diffusion_model.") for k in keys) and not has_sd3_transformer
        has_vae = any(k.startswith("first_stage_model.") or k.startswith("vae.") for k in keys)
        has_sdxl_te1 = any(self._is_sdxl_text_encoder_1_key(k) for k in keys)
        has_te2 = any(self._is_sdxl_text_encoder_2_key(k) for k in keys)
        has_clip_l = any(self._is_sd3_clip_l_key(k) for k in keys)
        has_clip_g = any(self._is_sd3_clip_g_key(k) for k in keys)
        has_t5 = any(self._is_sd3_t5_key(k) for k in keys)
        has_text_encoder = any(self._is_primary_text_encoder_key(k) for k in keys) or has_clip_l or has_clip_g or has_t5

        architecture, evidence, model_dimension = self._detect_architecture(keys, shapes, has_te2)
        architecture_variant = self._detect_architecture_variant(architecture, keys, shapes, path.name, evidence)
        latent_channels = self._detect_latent_channels(architecture, keys, shapes)
        denoiser_type = "transformer" if architecture == "sd3.x" else ("unet" if has_unet else "")
        flow_matching = architecture == "sd3.x"
        denoising_domain = "flow_match" if flow_matching else (
            "vp_sigma" if architecture in {"sd1.x", "sd2.x", "sdxl", "sd1.x_or_sd2.x"} else ""
        )
        text_encoder_packaging = self._describe_text_encoder_packaging(has_clip_l, has_clip_g, has_t5)
        checkpoint_packaging = self._describe_checkpoint_packaging(
            architecture, has_clip_l, has_clip_g, has_t5
        )
        checkpoint_kind = self._detect_checkpoint_kind(
            keys,
            has_unet,
            has_vae,
            has_text_encoder,
            architecture=architecture,
            has_text_encoder_2=has_te2,
            has_transformer=has_sd3_transformer,
        )
        prediction_type, prediction_source = detect_prediction_type(metadata, architecture, filename=path.name)
        file_sha256 = self.sha256_file(path) if compute_sha256 else ""
        if architecture == "sd2.x" and not prediction_type:
            qualification = qualification_for_sha256(file_sha256)
            if qualification is not None and qualification.architecture == "sd2.x":
                prediction_type = _normalize_prediction_type(qualification.prediction_type)
                prediction_source = qualification.source
        model_name, model_name_source = detect_model_name(metadata, path.stem)
        contract = build_architecture_contract(
            architecture,
            prediction_type,
            model_dimension,
            denoising_domain=denoising_domain,
            denoiser_type=denoiser_type,
            latent_channels=latent_channels,
            source=prediction_source or ("checkpoint_header_structural" if architecture == "sd3.x" else ""),
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
            has_sdxl_text_encoder_1=has_sdxl_te1,
            has_sdxl_text_encoder_2=has_te2,
            architecture_variant=architecture_variant,
            denoiser_type=denoiser_type,
            has_transformer=has_sd3_transformer,
            has_text_encoder_3=has_t5,
            has_clip_l=has_clip_l,
            has_clip_g=has_clip_g,
            has_t5=has_t5,
            text_encoder_packaging=text_encoder_packaging,
            checkpoint_packaging=checkpoint_packaging,
            latent_channels=latent_channels,
            flow_matching=flow_matching,
            denoising_domain=denoising_domain,
            key_prefix_counts=prefix_counts,
            representative_keys_by_prefix=representative_keys,
            file_size_bytes=path.stat().st_size,
            sha256=file_sha256,
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
            model_name=model_name,
            model_name_source=model_name_source,
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
        *,
        architecture: str = "",
        has_text_encoder_2: bool = False,
        has_transformer: bool = False,
    ) -> str:
        lora_signals = any(
            ".lora_" in k or "lora_up." in k or "lora_down." in k for k in keys
        )
        if lora_signals and not has_unet and not has_vae:
            return "lora"
        if architecture == "sdxl":
            if has_unet and has_vae and has_text_encoder and has_text_encoder_2:
                return "full"
            if has_unet or has_vae or has_text_encoder or has_text_encoder_2:
                return "partial"
            return "unknown"
        if architecture == "sd3.x":
            if has_transformer and has_vae:
                return "full"
            if has_transformer or has_vae or has_text_encoder:
                return "partial"
            return "unknown"
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
        sd3_signature_keys = (
            "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias",
            "joint_blocks.0.context_block.adaLN_modulation.1.bias",
        )
        for signature_key in sd3_signature_keys:
            shape = shapes.get(signature_key)
            if signature_key in keys and shape:
                if int(shape[-1]) == 9216:
                    evidence.append(f"{signature_key} shape[-1]=9216 SD3 MMDiT signature")
                    return "sd3.x", evidence, 4096
                evidence.append(f"{signature_key} found with unexpected shape={shape}")

        structural_markers = (
            any("joint_blocks.0.context_block." in key for key in keys),
            any("joint_blocks.0.x_block." in key for key in keys),
            any("x_embedder." in key for key in keys),
            any("context_embedder." in key for key in keys),
        )
        if sum(1 for marker in structural_markers if marker) >= 3:
            evidence.append("SD3 MMDiT joint/x/context transformer structure detected")
            return "sd3.x", evidence, 4096

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
    def _component_prefix(key: str) -> str:
        if key.startswith("model.diffusion_model."):
            return "model.diffusion_model"
        if key.startswith("first_stage_model."):
            return "first_stage_model"
        if key.startswith("text_encoders."):
            parts = key.split(".")
            return ".".join(parts[:2]) if len(parts) >= 2 else key
        if key.startswith("conditioner.embedders."):
            parts = key.split(".")
            return ".".join(parts[:3]) if len(parts) >= 3 else key
        if key.startswith("cond_stage_model."):
            return "cond_stage_model"
        if key.startswith("vae."):
            return "vae"
        parts = key.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else key

    @classmethod
    def _summarize_component_prefixes(
        cls, keys: List[str], *, representative_limit: int = 4
    ) -> tuple[dict[str, int], dict[str, list[str]]]:
        counts: Counter[str] = Counter()
        representatives: dict[str, list[str]] = {}
        for key in sorted(keys):
            prefix = cls._component_prefix(key)
            counts[prefix] += 1
            bucket = representatives.setdefault(prefix, [])
            if len(bucket) < representative_limit:
                bucket.append(key)
        ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        ordered_representatives = {prefix: representatives[prefix] for prefix in ordered}
        return ordered, ordered_representatives

    @staticmethod
    def _sd3_pos_embed_shape(
        keys: List[str], shapes: Mapping[str, tuple[int, ...]]
    ) -> tuple[str, tuple[int, ...]] | tuple[None, None]:
        for key in ("model.diffusion_model.pos_embed", "pos_embed"):
            shape = shapes.get(key)
            if key in keys and shape:
                return key, shape
        return None, None

    def _detect_architecture_variant(
        self,
        architecture: str,
        keys: List[str],
        shapes: Mapping[str, tuple[int, ...]],
        file_name: str,
        evidence: list[str],
    ) -> str:
        if architecture != "sd3.x":
            return ""
        if any(
            key.endswith("joint_blocks.37.x_block.mlp.fc1.weight")
            or ".joint_blocks.37.x_block.mlp.fc1.weight" in key
            for key in keys
        ):
            evidence.append("joint_blocks.37 indicates an SD3.5 large-depth transformer")
            return "sd3_5_large"

        pos_key, pos_shape = self._sd3_pos_embed_shape(keys, shapes)
        if pos_key and pos_shape and len(pos_shape) >= 2:
            positions = int(pos_shape[1])
            if positions == 36864:
                evidence.append(f"{pos_key} shape[1]=36864 identifies SD3 Medium")
                return "sd3_medium"
            if positions == 147456:
                evidence.append(f"{pos_key} shape[1]=147456 identifies SD3.5 Medium")
                return "sd3_5_medium"
            evidence.append(f"{pos_key} shape[1]={positions} is not a known Medium position grid")

        # Filename is secondary evidence only after a structural SD3 signature exists.
        normalized_name = file_name.lower().replace("-", "_")
        if "sd3.5" in normalized_name or "sd35" in normalized_name or "sd3_5" in normalized_name:
            evidence.append("filename secondary hint selects SD3.5 variant after structural SD3 detection")
            return "sd3_5_medium"
        if "sd3" in normalized_name and "medium" in normalized_name:
            evidence.append("filename secondary hint selects SD3 Medium after structural SD3 detection")
            return "sd3_medium"
        return "sd3_unknown"

    @staticmethod
    def _detect_latent_channels(
        architecture: str,
        keys: List[str],
        shapes: Mapping[str, tuple[int, ...]],
    ) -> int | None:
        if architecture != "sd3.x":
            return 4 if architecture in {"sd1.x", "sd2.x", "sdxl"} else None
        for key in (
            "model.diffusion_model.x_embedder.proj.weight",
            "x_embedder.proj.weight",
            "model.diffusion_model.x_embedder.weight",
            "x_embedder.weight",
        ):
            shape = shapes.get(key)
            if key in keys and shape and len(shape) >= 2:
                return int(shape[1])
        return 16

    @staticmethod
    def _describe_text_encoder_packaging(has_clip_l: bool, has_clip_g: bool, has_t5: bool) -> str:
        roles = []
        if has_clip_l:
            roles.append("clip_l")
        if has_clip_g:
            roles.append("clip_g")
        if has_t5:
            roles.append("t5xxl")
        if not roles:
            return "external"
        return "embedded:" + ",".join(roles)

    @staticmethod
    def _describe_checkpoint_packaging(
        architecture: str, has_clip_l: bool, has_clip_g: bool, has_t5: bool
    ) -> str:
        if architecture != "sd3.x":
            return ""
        if has_clip_l and has_clip_g and not has_t5:
            return "incl_clips"
        if has_clip_l or has_clip_g or has_t5:
            return "embedded_text_encoders"
        return "base"

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
            "joint_blocks.0.context_block.adaLN_modulation.1.bias",
            "pos_embed",
            "x_embedder.proj.weight",
            "context_embedder.weight",
            "text_encoders.clip_l",
            "text_encoders.clip_g",
            "text_encoders.t5xxl",
        )
        for key in keys:
            if any(fragment in key for fragment in preferred_fragments):
                selected[key] = shapes[key]
                if len(selected) >= 20:
                    break
        return selected
