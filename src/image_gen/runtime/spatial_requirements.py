from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping


DEFAULT_LATENT_SCALE_FACTOR = 8

_VAE_DOWNSAMPLE_PATTERNS = (
    re.compile(r"(?:^|\.)encoder\.down\.(\d+)\.downsample\.conv\.weight$"),
    re.compile(r"(?:^|\.)encoder\.down_blocks\.(\d+)\.downsamplers\.\d+\.conv\.weight$"),
)
_TRANSFORMER_PATCH_WEIGHT_SUFFIXES = (
    "x_embedder.proj.weight",
    "pos_embed.proj.weight",
)


def _positive_int(value: Any, *, fallback: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(fallback)
    return parsed if parsed > 0 else int(fallback)



def infer_vae_latent_scale_factor(keys: Iterable[str]) -> int:
    """Infer pixel-to-latent compression from VAE encoder downsample structure.

    Each encoder downsample stage halves spatial resolution. Counting distinct
    structural downsample blocks therefore yields the latent compression factor
    without relying on checkpoint filenames or model-family conditionals.
    """

    stages: set[int] = set()
    for raw_key in keys:
        key = str(raw_key or "")
        for pattern in _VAE_DOWNSAMPLE_PATTERNS:
            match = pattern.search(key)
            if match:
                stages.add(int(match.group(1)))
                break
    return 2 ** len(stages) if stages else 0


def infer_transformer_latent_patch_multiple(
    keys: Iterable[str],
    shapes: Mapping[str, Iterable[int]] | None = None,
) -> int:
    """Infer transformer patchification from patch-embedding kernel shape.

    Conv2d patch embedders expose the patch size in the final two tensor
    dimensions. Only square, positive kernels are accepted as authoritative
    evidence; ambiguous/non-convolutional embeddings fail closed.
    """

    shape_map = dict(shapes or {})
    for raw_key in keys:
        key = str(raw_key or "")
        if not key.endswith(_TRANSFORMER_PATCH_WEIGHT_SUFFIXES):
            continue
        shape = tuple(int(value) for value in (shape_map.get(key) or ()))
        if len(shape) < 4:
            continue
        patch_h, patch_w = int(shape[-2]), int(shape[-1])
        if patch_h > 0 and patch_h == patch_w:
            return patch_h
    return 0


def checkpoint_spatial_evidence(
    *,
    keys: Iterable[str],
    shapes: Mapping[str, Iterable[int]] | None = None,
    denoiser_kind: str = "",
) -> dict[str, Any]:
    """Return tensor-header spatial evidence without hydrating checkpoint data."""

    key_list = [str(key) for key in keys]
    latent_scale = infer_vae_latent_scale_factor(key_list)
    kind = str(denoiser_kind or "").strip().casefold()
    if kind and kind != "transformer":
        latent_patch = 1
        patch_source = "unet_latent_grid"
    elif kind == "transformer":
        latent_patch = infer_transformer_latent_patch_multiple(key_list, shapes)
        patch_source = "checkpoint_transformer_patch" if latent_patch else "transformer_patch_unresolved"
    else:
        latent_patch = 0
        patch_source = "denoiser_kind_unresolved"
    pixel_alignment = latent_scale * latent_patch if latent_scale > 0 and latent_patch > 0 else 0
    source_parts = []
    source_parts.append("checkpoint_vae_downsample_structure" if latent_scale else "vae_scale_unresolved")
    source_parts.append(patch_source)
    return {
        "latent_scale_factor": latent_scale,
        "latent_patch_multiple": latent_patch,
        "pixel_alignment_multiple": pixel_alignment,
        "source": "+".join(source_parts),
    }

def resolve_latent_patch_multiple(
    *,
    denoiser_kind: str,
    runtime_profile: Mapping[str, Any] | None = None,
    denoiser_config: Any = None,
    fail_closed: bool = False,
) -> int:
    """Resolve the denoiser's latent-grid patch requirement from runtime evidence.

    UNet denoisers operate directly on the latent grid and therefore require a
    latent patch multiple of one. Transformer denoisers may patchify the latent
    grid again. Runtime-profile evidence is authoritative before a live module
    exists; a live module config is the runtime fallback once components are
    hydrated.

    ``fail_closed`` is used by zero-GPU capability resolution. If a transformer
    patch size has not been qualified yet, returning zero prevents the WebUI
    from inventing a permissive spatial contract. Runtime construction keeps
    the historical fallback of one when no patch metadata is available.
    """

    kind = str(denoiser_kind or "").strip().casefold()
    if kind and kind != "transformer":
        return 1
    if not kind:
        return 0 if fail_closed else 1

    profile = dict(runtime_profile or {})
    raw = profile.get("transformer_patch_size")
    if raw in (None, "") and denoiser_config is not None:
        raw = getattr(denoiser_config, "patch_size", None)
    value = _positive_int(raw)
    if value:
        return value
    return 0 if fail_closed else 1


@dataclass(frozen=True)
class RuntimeSpatialRequirements:
    latent_scale_factor: int
    latent_patch_multiple: int
    pixel_alignment_multiple: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_scale_factor": int(self.latent_scale_factor),
            "latent_patch_multiple": int(self.latent_patch_multiple),
            "pixel_alignment_multiple": int(self.pixel_alignment_multiple),
            "source": self.source,
        }


def resolve_runtime_spatial_requirements(
    *,
    denoiser_kind: str,
    runtime_profile: Mapping[str, Any] | None = None,
    latent_scale_factor: Any = None,
    latent_patch_multiple: Any = None,
    denoiser_config: Any = None,
    fail_closed: bool = False,
) -> RuntimeSpatialRequirements:
    """Return the pixel-grid alignment contract shared by runtime and WebUI.

    The generation runtime's dimensional invariant is:

        pixel alignment = VAE latent scale * denoiser latent patch multiple

    The helper is deliberately tensor-free. It consumes already-resolved
    profile/config metadata and therefore can be used by GFP capability
    resolution without loading checkpoint weights.
    """

    profile = dict(runtime_profile or {})
    explicit_scale = _positive_int(latent_scale_factor)
    profile_scale = _positive_int(profile.get("latent_scale_factor"))
    scale = explicit_scale or profile_scale
    if not scale and not fail_closed:
        scale = DEFAULT_LATENT_SCALE_FACTOR

    explicit_patch = _positive_int(latent_patch_multiple)
    patch = explicit_patch or resolve_latent_patch_multiple(
        denoiser_kind=denoiser_kind,
        runtime_profile=profile,
        denoiser_config=denoiser_config,
        fail_closed=fail_closed,
    )
    pixel_multiple = scale * patch if scale > 0 and patch > 0 else 0

    source_parts = []
    if explicit_scale:
        source_parts.append("resolved_latent_scale")
    elif profile_scale:
        source_parts.append("runtime_profile_latent_scale")
    elif fail_closed:
        source_parts.append("latent_scale_unresolved")
    else:
        source_parts.append("platform_default_latent_scale")
    if explicit_patch:
        source_parts.append("resolved_latent_patch")
    elif str(denoiser_kind or "").strip().casefold() == "transformer":
        if _positive_int(profile.get("transformer_patch_size")):
            source_parts.append("runtime_profile_transformer_patch")
        elif denoiser_config is not None and _positive_int(getattr(denoiser_config, "patch_size", None)):
            source_parts.append("denoiser_config_patch")
        elif fail_closed:
            source_parts.append("transformer_patch_unresolved")
        else:
            source_parts.append("runtime_patch_fallback")
    else:
        source_parts.append("unet_latent_grid")

    return RuntimeSpatialRequirements(
        latent_scale_factor=scale,
        latent_patch_multiple=patch,
        pixel_alignment_multiple=pixel_multiple,
        source="+".join(source_parts),
    )


__all__ = [
    "DEFAULT_LATENT_SCALE_FACTOR",
    "RuntimeSpatialRequirements",
    "checkpoint_spatial_evidence",
    "infer_transformer_latent_patch_multiple",
    "infer_vae_latent_scale_factor",
    "resolve_latent_patch_multiple",
    "resolve_runtime_spatial_requirements",
]
