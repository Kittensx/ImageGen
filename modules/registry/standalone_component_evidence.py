from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open


@dataclass(frozen=True)
class StandaloneComponentEvidence:
    component_role: str
    component_subtype: str
    provider_family_evidence: tuple[str, ...]
    evidence: Mapping[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "component_role": self.component_role,
            "component_subtype": self.component_subtype,
            "provider_family_evidence": list(self.provider_family_evidence),
            "evidence": dict(self.evidence),
            "classification_basis": "standalone_safetensors_structure",
        }


def _tensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            try:
                shapes[str(key)] = tuple(int(value) for value in handle.get_slice(key).get_shape())
            except Exception:
                continue
    return shapes


def _shape_for_suffix(shapes: Mapping[str, tuple[int, ...]], *suffixes: str) -> tuple[str, tuple[int, ...]] | None:
    normalized = tuple(str(item) for item in suffixes)
    for key in sorted(shapes):
        if any(key == suffix or key.endswith("." + suffix) for suffix in normalized):
            return key, shapes[key]
    return None


def classify_standalone_text_encoder(path: str | Path) -> StandaloneComponentEvidence:
    resolved = Path(path).expanduser().resolve()
    shapes = _tensor_shapes(resolved)
    lowered_keys = tuple(key.casefold() for key in shapes)

    has_t5_blocks = any("encoder.block." in key for key in lowered_keys)
    has_t5_attention = any(
        token in key
        for key in lowered_keys
        for token in (
            "selfattention.q.weight",
            "selfattention.k.weight",
            "selfattention.v.weight",
        )
    )
    has_t5_shared_embedding = _shape_for_suffix(shapes, "shared.weight", "encoder.embed_tokens.weight") is not None
    t5_evidence = has_t5_blocks and (has_t5_attention or has_t5_shared_embedding)
    if t5_evidence:
        shared = _shape_for_suffix(shapes, "shared.weight", "encoder.embed_tokens.weight")
        return StandaloneComponentEvidence(
            component_role="text_encoder_3",
            component_subtype="t5_or_t5xxl",
            provider_family_evidence=("sd3.x",),
            evidence={
                "encoder_block_structure": has_t5_blocks,
                "self_attention_structure": has_t5_attention,
                "shared_embedding_tensor": shared[0] if shared else None,
                "shared_embedding_shape": list(shared[1]) if shared else None,
            },
        )

    token_embedding = _shape_for_suffix(
        shapes,
        "text_model.embeddings.token_embedding.weight",
        "token_embedding.weight",
    )
    hidden_size = int(token_embedding[1][-1]) if token_embedding and len(token_embedding[1]) >= 2 else 0

    if hidden_size == 1280:
        return StandaloneComponentEvidence(
            component_role="text_encoder_2",
            component_subtype="clip_g_or_openclip_bigg",
            provider_family_evidence=("sdxl", "sd3.x"),
            evidence={
                "token_embedding_tensor": token_embedding[0] if token_embedding else None,
                "token_embedding_shape": list(token_embedding[1]) if token_embedding else None,
                "hidden_size": hidden_size,
            },
        )

    if hidden_size == 1024:
        return StandaloneComponentEvidence(
            component_role="text_encoder",
            component_subtype="openclip_h",
            provider_family_evidence=("sd2.x",),
            evidence={
                "token_embedding_tensor": token_embedding[0] if token_embedding else None,
                "token_embedding_shape": list(token_embedding[1]) if token_embedding else None,
                "hidden_size": hidden_size,
            },
        )

    if hidden_size == 768:
        return StandaloneComponentEvidence(
            component_role="text_encoder",
            component_subtype="clip_l",
            provider_family_evidence=("sd1.x", "sdxl", "sd3.x"),
            evidence={
                "token_embedding_tensor": token_embedding[0] if token_embedding else None,
                "token_embedding_shape": list(token_embedding[1]) if token_embedding else None,
                "hidden_size": hidden_size,
            },
        )

    return StandaloneComponentEvidence(
        component_role="standalone_text_encoder",
        component_subtype="unclassified_text_encoder",
        provider_family_evidence=(),
        evidence={
            "token_embedding_tensor": token_embedding[0] if token_embedding else None,
            "token_embedding_shape": list(token_embedding[1]) if token_embedding else None,
            "hidden_size": hidden_size or None,
        },
    )


def classify_standalone_vae(path: str | Path) -> StandaloneComponentEvidence:
    resolved = Path(path).expanduser().resolve()
    shapes = _tensor_shapes(resolved)
    latent_evidence = _shape_for_suffix(
        shapes,
        "decoder.conv_in.weight",
        "post_quant_conv.weight",
        "quant_conv.weight",
    )
    latent_channels = 0
    if latent_evidence:
        key, shape = latent_evidence
        if key.endswith("decoder.conv_in.weight") and len(shape) >= 2:
            latent_channels = int(shape[1])
        elif len(shape) >= 2 and int(shape[0]) == int(shape[1]):
            latent_channels = int(shape[0])

    if latent_channels == 16:
        subtype = "vae_latent16"
    elif latent_channels == 4:
        subtype = "vae_latent4"
    else:
        subtype = "vae_unclassified_latent_width"

    return StandaloneComponentEvidence(
        component_role="vae",
        component_subtype=subtype,
        # Latent width alone is not enough to claim cross-family VAE compatibility.
        # Family eligibility is inherited from exact/structural component evidence
        # already registered from provider-supported checkpoint occurrences.
        provider_family_evidence=(),
        evidence={
            "latent_tensor": latent_evidence[0] if latent_evidence else None,
            "latent_tensor_shape": list(latent_evidence[1]) if latent_evidence else None,
            "latent_channels": latent_channels or None,
        },
    )
