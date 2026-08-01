# sampler_config_loader.py

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml

SAMPLER_CFG_DEFAULTS = {
    "prefer_config": False,

    "sampler": {
        "sampler_type": "euler",
        "eta": 0.0,
        "add_noise": True,
        "verbose": False,
    },

    "cfg_strategy": {
        "rescale_cfg": False,
        "rescale_cfg_factor": 1.0,
        "clamp_range": [-1.0, 1.0],

        "initial_noise_strength": 0.0,
        "eta_scale_factor": 1.0,
        "eta_schedule_mode": "none",
        "use_adaptive_eta": False,
        "noise_schedule_scaling": "none",

        "adaptive_time_mode": "none",
        "adaptive_delta_low_floor": 0.1,
        "adaptive_delta_high_floor": 1.0,
        "adaptive_low_adjustment_multiplier": 1.5,
        "adaptive_high_adjustment_multiplier": 0.5,
        "adaptive_denoised_floor": 0.05,
        "adaptive_denoised_adjustment_multiplier": 1.25,
        "adaptive_manual_low_adjustment": 1.0,
        "adaptive_manual_high_adjustment": 1.0,

        "cfg_guidance_mode": "legacy_flat",
        "cfg_curve_type": "smoothstep",
        "cfg_curve_strength": 1.0,
        "cfg_high_sigma_boost": 1.2,
        "cfg_low_sigma_taper": 0.3,
        "cfg_auto_low_cfg_threshold": 6.5,
        "cfg_early_floor_enabled": False,
        "cfg_early_floor_value": 6.2,
        "cfg_early_floor_until_fraction": 0.3,
    }
}

# ============================================================
# Path helpers
# ============================================================
def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)

    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result

def flatten_sampler_sections(config: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    reserved_sections = {"sampler", "cfg_strategy"}

    sampler_section = config.get("sampler", {})
    if isinstance(sampler_section, dict):
        flat.update(sampler_section)

    cfg_strategy_section = config.get("cfg_strategy", {})
    if isinstance(cfg_strategy_section, dict):
        flat.update(cfg_strategy_section)

    # Flat top-level values win so direct overrides survive the nested defaults.
    for key, value in (config or {}).items():
        if key not in reserved_sections:
            flat[key] = value

    return flat

def build_sampler_cfg_settings(
    settings: Optional[Dict[str, Any]],
    extra_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = deepcopy(SAMPLER_CFG_DEFAULTS)

    if extra_defaults:
        merged = deep_merge_dicts(merged, extra_defaults)

    if settings:
        merged = deep_merge_dicts(merged, settings)

   
    return flatten_sampler_sections(merged)
    
    

def get_sampler_config_root(base_path: Optional[str] = None) -> str:
    """
    Returns the shared sampler config root.

    Expected layout:
        modules/ss_registry/samplers/config/
            shared/
                presets/
            kes/
                default.yaml
                presets/
    """
    if base_path:
        return base_path
    return os.path.join(os.path.dirname(__file__), "config")


def get_sampler_default_config_path(
    sampler_name: str,
    *,
    base_path: Optional[str] = None,
    default_filename: str = "default.yaml",
) -> str:
    root = get_sampler_config_root(base_path)
    return os.path.join(root, sampler_name, default_filename)


def get_sampler_preset_path(
    sampler_name: str,
    preset_name: str,
    *,
    base_path: Optional[str] = None,
    presets_subdir: str = "presets",
    extension: str = ".yaml",
) -> str:
    root = get_sampler_config_root(base_path)
    filename = preset_name if preset_name.endswith(extension) else f"{preset_name}{extension}"
    return os.path.join(root, sampler_name, presets_subdir, filename)


def get_shared_preset_path(
    preset_name: str,
    *,
    base_path: Optional[str] = None,
    shared_dir: str = "shared",
    presets_subdir: str = "presets",
    extension: str = ".yaml",
) -> str:
    root = get_sampler_config_root(base_path)
    filename = preset_name if preset_name.endswith(extension) else f"{preset_name}{extension}"
    return os.path.join(root, shared_dir, presets_subdir, filename)


def list_sampler_presets(
    sampler_name: str,
    *,
    base_path: Optional[str] = None,
    include_shared: bool = True,
    presets_subdir: str = "presets",
    extension: str = ".yaml",
) -> Dict[str, List[str]]:
    """
    Returns available preset names.

    Example:
        {
            "sampler": ["smoothed_tail", "heun_detail"],
            "shared": ["balanced", "creative"]
        }
    """
    root = get_sampler_config_root(base_path)

    def _list_yaml_names(folder: str) -> List[str]:
        if not os.path.isdir(folder):
            return []
        names = []
        for entry in os.listdir(folder):
            if entry.endswith(extension):
                names.append(os.path.splitext(entry)[0])
        return sorted(names)

    sampler_folder = os.path.join(root, sampler_name, presets_subdir)
    result = {
        "sampler": _list_yaml_names(sampler_folder),
    }

    if include_shared:
        shared_folder = os.path.join(root, "shared", presets_subdir)
        result["shared"] = _list_yaml_names(shared_folder)

    return result


# ============================================================
# YAML loading
# ============================================================

def load_yaml_config_file(config_path: str) -> Dict[str, Any]:
    if not config_path:
        raise ValueError("config_path must be provided.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[SamplerConfigLoader] Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(
    *,
    sampler_name: str,
    config_path: Optional[str] = None,
    preset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    default_filename: str = "default.yaml",
) -> str:
    """
    Resolution order:
      1. explicit config_path
      2. sampler-specific preset
      3. shared preset
      4. sampler default
    """
    if config_path:
        return config_path

    if preset_name:
        sampler_preset = get_sampler_preset_path(
            sampler_name,
            preset_name,
            base_path=base_path,
        )
        if os.path.exists(sampler_preset):
            return sampler_preset

        shared_preset = get_shared_preset_path(
            preset_name,
            base_path=base_path,
        )
        if os.path.exists(shared_preset):
            return shared_preset

        raise FileNotFoundError(
            f"[SamplerConfigLoader] Preset '{preset_name}' not found for sampler "
            f"'{sampler_name}' in sampler-specific or shared preset folders."
        )
    

    default_path = get_sampler_default_config_path(
        sampler_name,
        base_path=base_path,
        default_filename=default_filename,
    )
    if not os.path.exists(default_path):
        raise FileNotFoundError(
            f"[SamplerConfigLoader] Default config not found for sampler "
            f"'{sampler_name}': {default_path}"
        )

    return default_path


# ============================================================
# Config normalization
# ============================================================

def flatten_sampler_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flatten nested sampler config into one effective runtime dict.

    Merge order:
      1. top-level keys
      2. postprocessing
      3. noise_schedule
      4. cfg_strategy

    Later sections override earlier sections.
    """
    config = deepcopy(config or {})
    flattened: Dict[str, Any] = {}

    reserved_sections = {"postprocessing", "noise_schedule", "cfg_strategy"}

    for key, value in config.items():
        if key not in reserved_sections:
            flattened[key] = value

    for section_name in ("postprocessing", "noise_schedule", "cfg_strategy"):
        section = config.get(section_name, {})
        if isinstance(section, dict):
            flattened.update(section)

    return flattened


def resolve_settings(
    config: Optional[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge config and overrides.

    If prefer_config is true, config wins.
    Otherwise, overrides win.
    """
    config = deepcopy(config or {})
    overrides = deepcopy(overrides or {})

    prefer_config = bool(config.get("prefer_config", False))

    if prefer_config:
        return {**overrides, **config}
    return {**config, **overrides}




def _normalize_sampler_aliases(
    sampler_name: str,
    settings: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = deepcopy(settings or {})
    if str(sampler_name or "").strip().lower() != "kes":
        return normalized

    if "rescale_cfg" in normalized:
        if (
            "legacy_clamp_guidance" not in normalized
            or (
                bool(normalized.get("legacy_clamp_guidance")) is False
                and bool(normalized.get("rescale_cfg")) is True
            )
        ):
            normalized["legacy_clamp_guidance"] = bool(normalized.get("rescale_cfg"))
    if "rescale_cfg_factor" in normalized:
        try:
            rescale_factor = float(normalized.get("rescale_cfg_factor"))
        except (TypeError, ValueError):
            rescale_factor = 1.0
        if (
            "legacy_guidance_multiplier" not in normalized
            or (
                float(normalized.get("legacy_guidance_multiplier") or 1.0) == 1.0
                and rescale_factor != 1.0
            )
        ):
            normalized["legacy_guidance_multiplier"] = rescale_factor

    if "legacy_clamp_guidance" in normalized and "rescale_cfg" not in normalized:
        normalized["rescale_cfg"] = bool(normalized.get("legacy_clamp_guidance"))
    if (
        "legacy_guidance_multiplier" in normalized
        and "rescale_cfg_factor" not in normalized
    ):
        normalized["rescale_cfg_factor"] = normalized.get("legacy_guidance_multiplier")
    return normalized

# ============================================================
# State application
# ============================================================

def apply_settings_to_object(
    target: Any,
    settings: Dict[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    if target is None:
        raise ValueError("[SamplerConfigLoader] target is not set.")

    for key, value in settings.items():
        if overwrite or not hasattr(target, key):
            setattr(target, key, value)


def apply_settings_to_sampler_state(
    sampler_state: Any,
    settings: Dict[str, Any],
    *,
    subsection: str = "cfg",
    overwrite: bool = True,
) -> None:
    if sampler_state is None:
        raise ValueError("[SamplerConfigLoader] sampler_state is not set.")

    target = getattr(sampler_state, subsection, None)
    if target is None:
        raise AttributeError(
            f"[SamplerConfigLoader] sampler_state has no attribute '{subsection}'"
        )

    cfg_settings = build_sampler_cfg_settings(settings)
    apply_settings_to_object(target, cfg_settings, overwrite=overwrite)

# ============================================================
# Main public helpers
# ============================================================

def build_sampler_settings(
    *,
    sampler_name: str,
    config_path: Optional[str] = None,
    preset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    default_filename: str = "default.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full config pipeline:
      resolve path -> load YAML -> flatten -> resolve overrides

    Returns:
        final effective flat settings dict
    """
    final_config_path = resolve_config_path(
        sampler_name=sampler_name,
        config_path=config_path,
        preset_name=preset_name,
        base_path=base_path,
        default_filename=default_filename,
    )

    raw_config = load_yaml_config_file(final_config_path)
    flattened = flatten_sampler_config(raw_config)
    resolved = resolve_settings(flattened, overrides)
    resolved = _normalize_sampler_aliases(sampler_name, resolved)

    # Keep a little provenance for metadata/debugging
    resolved["_config_path"] = final_config_path
    resolved["_config_source"] = (
        "file" if config_path else
        "preset" if preset_name else
        "default"
    )
    if preset_name:
        resolved["_preset_name"] = preset_name

    return resolved


def prepare_sampler_config(
    *,
    sampler_name: str,
    sampler_state: Optional[Any] = None,
    config_path: Optional[str] = None,
    preset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    default_filename: str = "default.yaml",
    overrides: Optional[Dict[str, Any]] = None,
    subsection: str = "cfg",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Convenience helper:
      build settings -> optionally apply into sampler_state.cfg

    Returns:
        final effective flat settings dict
    """
    settings = build_sampler_settings(
        sampler_name=sampler_name,
        config_path=config_path,
        preset_name=preset_name,
        base_path=base_path,
        default_filename=default_filename,
        overrides=overrides,
    )

    if sampler_state is not None:
        apply_settings_to_sampler_state(
            sampler_state,
            settings,
            subsection=subsection,
            overwrite=overwrite,
        )

    return settings