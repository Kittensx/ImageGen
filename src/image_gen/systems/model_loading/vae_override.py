from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch

if TYPE_CHECKING:
    from torch.nn import Module

from image_gen.contracts.vae_provenance import attach_vae_provenance
from modules.asset_discovery import resolve_nested_asset
from modules.state_dict_converter import StateDictConverter
from modules.state_dict_mapper import StateDictMapper

_SUPPORTED_VAE_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
_SD2_NEGATIVE_HINTS = ("sdxl", "xl", "sd3", "flux", "pony")
_VAE_AUXILIARY_STATE_KEYS = frozenset({
    "model_ema.decay",
    "model_ema.num_updates",
})


@dataclass(frozen=True)
class ExternalVAELoadResult:
    vae: "Module"
    provenance: dict[str, Any]
    source_format: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_state_dict(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(path), device="cpu")
    else:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Standalone VAE file did not load as a mapping: {path}")
    nested = payload.get("state_dict")
    if isinstance(nested, dict) and nested:
        return nested
    nested = payload.get("model")
    if isinstance(nested, dict) and nested:
        return nested
    return payload


def _looks_like_ldm_vae(keys: Iterable[str]) -> bool:
    structural_markers = (
        "encoder.down.",
        "decoder.up.",
        "encoder.mid.block_",
        "encoder.mid.attn_",
        "decoder.mid.block_",
        "decoder.mid.attn_",
    )
    return any(str(key).startswith(structural_markers) for key in keys)


def _looks_like_diffusers_vae(keys: Iterable[str]) -> bool:
    structural_markers = (
        "encoder.down_blocks.",
        "encoder.mid_block.",
        "decoder.up_blocks.",
        "decoder.mid_block.",
    )
    return any(str(key).startswith(structural_markers) for key in keys)


def normalize_vae_state_dict(raw_state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Normalize an external VAE state dict for a Diffusers AutoencoderKL target.

    This public compatibility helper retains the historical Diffusers-targeted
    behavior. The active IMAGE_GEN LDMVAE runtime uses a separate boundary
    adapter below so the embedded checkpoint path does not need to change.
    """

    keys = tuple(str(key) for key in raw_state.keys())
    mapper = StateDictMapper()
    converter = StateDictConverter()
    if any(key.startswith(mapper.VAE_PREFIX) for key in keys):
        split = mapper.split_checkpoint(raw_state)
        if not split.vae:
            raise ValueError("VAE override checkpoint exposed first_stage_model keys but no VAE weights were extracted.")
        return converter.convert_vae_state_dict(split.vae), "ldm_embedded_vae"
    if _looks_like_ldm_vae(keys):
        return converter.convert_vae_state_dict(raw_state), "ldm_standalone_vae"
    return dict(raw_state), "diffusers_standalone_vae"


def _base_component_device_dtype(base_vae: "Module") -> tuple[torch.device, torch.dtype]:
    first_param = next(base_vae.parameters(), None)
    if first_param is None:
        return torch.device("cpu"), torch.float32
    return first_param.device, first_param.dtype


def _is_ldm_runtime_vae(base_vae: "Module") -> bool:
    from modules.ldm_vae_builder import LDMVAE

    return isinstance(base_vae, LDMVAE)


def _clone_diffusers_vae_shell(base_vae: "Module") -> "Module":
    config = getattr(base_vae, "config", None)
    if config is None:
        raise RuntimeError(
            "Loaded base VAE is neither IMAGE_GEN LDMVAE nor a configured Diffusers AutoencoderKL; "
            "cannot apply external VAE override."
        )
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_config(config)


def _diffusers_to_ldm_key_map(base_vae: "Module") -> dict[str, str]:
    converter = StateDictConverter()
    reverse: dict[str, str] = {}
    for ldm_key in base_vae.state_dict().keys():
        diffusers_key = converter.convert_vae_key(str(ldm_key))
        existing = reverse.get(diffusers_key)
        if existing is not None and existing != ldm_key:
            raise RuntimeError(
                "Cannot build an unambiguous Diffusers-to-LDM VAE key map: "
                f"{diffusers_key!r} maps to both {existing!r} and {ldm_key!r}."
            )
        reverse[diffusers_key] = str(ldm_key)
    return reverse


def _adapt_diffusers_tensor_for_ldm_target(
    value: Any,
    *,
    target_value: Any,
) -> Any:
    source_shape = tuple(getattr(value, "shape", ()) or ())
    target_shape = tuple(getattr(target_value, "shape", ()) or ())
    if source_shape == target_shape:
        return value
    if (
        len(source_shape) == 2
        and len(target_shape) == 4
        and target_shape[-2:] == (1, 1)
        and source_shape == target_shape[:2]
        and isinstance(value, torch.Tensor)
    ):
        return value[:, :, None, None]
    return value


def _convert_diffusers_state_for_ldm_runtime(
    base_vae: "Module",
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    reverse_keys = _diffusers_to_ldm_key_map(base_vae)
    base_state = base_vae.state_dict()
    converted: dict[str, Any] = {}
    for source_key, value in state_dict.items():
        source_key = str(source_key)
        target_key = reverse_keys.get(source_key, source_key)
        target_value = base_state.get(target_key)
        if target_value is not None:
            value = _adapt_diffusers_tensor_for_ldm_target(value, target_value=target_value)
        converted[target_key] = value
    return converted


def _normalize_vae_state_for_ldm_runtime(
    base_vae: "Module",
    raw_state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    keys = tuple(str(key) for key in raw_state.keys())
    mapper = StateDictMapper()

    if any(key.startswith(mapper.VAE_PREFIX) for key in keys):
        split = mapper.split_checkpoint(raw_state)
        if not split.vae:
            raise ValueError("VAE override checkpoint exposed first_stage_model keys but no VAE weights were extracted.")
        return dict(split.vae), "ldm_embedded_vae"

    if _looks_like_ldm_vae(keys):
        return dict(raw_state), "ldm_standalone_vae"

    if _looks_like_diffusers_vae(keys):
        return _convert_diffusers_state_for_ldm_runtime(base_vae, raw_state), "diffusers_standalone_vae"

    # Some compact or partially-pruned VAE files contain only keys shared by
    # both layouts. If every key is already a valid LDM runtime key, preserve it
    # directly rather than guessing a Diffusers conversion.
    base_keys = set(str(key) for key in base_vae.state_dict().keys())
    if keys and set(keys).issubset(base_keys):
        return dict(raw_state), "ldm_standalone_vae"

    return _convert_diffusers_state_for_ldm_runtime(base_vae, raw_state), "diffusers_standalone_vae"


def _load_normalized_vae_state(path: Path) -> tuple[dict[str, Any], str]:
    raw_state = _load_raw_state_dict(path)
    return normalize_vae_state_dict(raw_state)


def _clean_incompatible_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """Return only incompatibilities that represent actual VAE model state.

    A few common standalone VAE checkpoints carry EMA bookkeeping scalars
    alongside the network weights. They are not runtime parameters and should
    not make an otherwise complete VAE fail compatibility validation. Keep this
    allow-list exact so unknown extra model keys remain fatal.
    """

    return tuple(
        sorted(
            key_text
            for key in keys
            if (key_text := str(key))
            and not key_text.startswith("loss.")
            and key_text not in _VAE_AUXILIARY_STATE_KEYS
        )
    )


def _load_external_vae_for_ldm_runtime(
    base_vae: "Module",
    path: Path,
) -> tuple["Module", str]:
    from modules.ldm_vae_builder import LDMVAEBuilder

    raw_state = _load_raw_state_dict(path)
    runtime_state, source_format = _normalize_vae_state_for_ldm_runtime(base_vae, raw_state)
    target_device, target_dtype = _base_component_device_dtype(base_vae)
    result = LDMVAEBuilder(device=str(target_device), dtype=target_dtype).build_and_load(runtime_state)
    if not result.success:
        raise RuntimeError(
            "External VAE override could not be constructed for IMAGE_GEN's LDMVAE runtime: "
            f"{result.error}"
        )
    missing = _clean_incompatible_keys(result.missing_keys)
    unexpected = _clean_incompatible_keys(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "External VAE override is incompatible with IMAGE_GEN's LDMVAE runtime. "
            f"Missing keys: {missing[:12]}{' ...' if len(missing) > 12 else ''}; "
            f"Unexpected keys: {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}"
        )
    shell = result.model
    shell.eval()
    shell.requires_grad_(False)
    return shell, source_format


def _load_external_vae_for_diffusers_runtime(
    base_vae: "Module",
    path: Path,
) -> tuple["Module", str]:
    normalized_state, source_format = _load_normalized_vae_state(path)
    shell = _clone_diffusers_vae_shell(base_vae)
    incompatible = shell.load_state_dict(normalized_state, strict=False)
    missing = _clean_incompatible_keys(getattr(incompatible, "missing_keys", ()) or ())
    unexpected = _clean_incompatible_keys(getattr(incompatible, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise RuntimeError(
            "External VAE override is incompatible with the active AutoencoderKL configuration. "
            f"Missing keys: {missing[:12]}{' ...' if len(missing) > 12 else ''}; "
            f"Unexpected keys: {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}"
        )
    target_device, target_dtype = _base_component_device_dtype(base_vae)
    shell.to(device=target_device, dtype=target_dtype)
    shell.eval()
    shell.requires_grad_(False)
    return shell, source_format


def apply_external_vae_override(
    base_vae: "Module",
    vae_path: str | Path,
    *,
    project_context: Any | None = None,
) -> ExternalVAELoadResult:
    """Load an external VAE without changing the working embedded-VAE path.

    The external boundary adapts to the active runtime component type:
    IMAGE_GEN's LDMVAE receives LDM-shaped weights (with Diffusers files mapped
    back to that layout), while legacy/configured AutoencoderKL callers retain
    their historical Diffusers-targeted loading behavior.
    """

    path = Path(str(vae_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VAE override file not found: {path}")

    if _is_ldm_runtime_vae(base_vae):
        shell, source_format = _load_external_vae_for_ldm_runtime(base_vae, path)
    else:
        shell, source_format = _load_external_vae_for_diffusers_runtime(base_vae, path)

    source_sha256 = _sha256_file(path)
    registry_refresh = None
    if project_context is not None and path.suffix.lower() == ".safetensors":
        # Keep the shared asset registry synchronized with assets that actually
        # cross the runtime load boundary. The targeted refresher is cache-aware,
        # so a complete/current VAE does not re-hash on every generation.
        from modules.registry.component_refresh import ComponentRegistryRefresher

        refresher = ComponentRegistryRefresher(project_context)
        registry_refresh = refresher.ensure_path(
            path,
            explicit_kind="vae",
            source="external_vae_load",
        )
        registered = refresher.registry.get_asset_by_path(str(path))
        if registered is not None:
            refresher.registry.update_asset_sha256(registered.id, source_sha256)

    provenance = {
        "source_kind": "external_vae_override",
        "source_path": str(path),
        "sha256": source_sha256,
        "identity": f"external_vae_override:{path}",
        "display_name": path.name,
        "embedded_in_checkpoint": False,
        "override_applied": True,
        "source_format": source_format,
        "component_registry_refresh": registry_refresh,
    }
    attach_vae_provenance(shell, provenance)
    return ExternalVAELoadResult(
        vae=shell,
        provenance=provenance,
        source_format=source_format,
    )


def resolve_installed_vae(
    vae_dir: str | Path,
    requested: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve a user-supplied VAE reference without flattening library layout.

    Resolution order:
      1. existing absolute/direct path
      2. existing path relative to ``project_root`` when supplied
      3. deterministic recursive resolution under the configured VAE directory

    Recursive ambiguity is intentionally surfaced by ``resolve_nested_asset``;
    callers must provide a more-qualified relative path rather than silently
    selecting one of several matching VAE filenames.
    """

    text = str(requested or "").strip()
    if not text:
        raise ValueError("A VAE filename or path is required.")

    direct = Path(text).expanduser()
    if direct.is_file():
        return direct.resolve()

    if project_root is not None and not direct.is_absolute():
        project_candidate = (Path(project_root).expanduser().resolve() / direct).resolve()
        if project_candidate.is_file():
            return project_candidate

    resolved = resolve_nested_asset(
        Path(vae_dir).expanduser().resolve(),
        text,
        extensions=_SUPPORTED_VAE_EXTENSIONS,
        allow_stem_match=True,
    )
    if resolved is not None:
        return resolved

    raise FileNotFoundError(
        f"VAE {text!r} was not found as a direct path or anywhere under the configured VAE directory: "
        f"{Path(vae_dir).expanduser().resolve()}"
    )


def resolve_sd2_default_vae(vae_dir: str | Path) -> Path | None:
    root = Path(str(vae_dir)).expanduser()
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_VAE_EXTENSIONS:
            continue
        stem = path.stem.lower()
        score = 0
        if any(token in stem for token in _SD2_NEGATIVE_HINTS):
            score -= 100
        if "sd2" in stem or "sd_2" in stem or "stable-diffusion-2" in stem or "stable_diffusion_2" in stem:
            score += 80
        if "2.1" in stem or "2-1" in stem or "v2-1" in stem or "v2_1" in stem or "v21" in stem:
            score += 50
        if "default" in stem or "base" in stem:
            score += 10
        if score > 0:
            candidates.append((score, path.resolve()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1]).lower()))
    return candidates[0][1]
