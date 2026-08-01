# sampler_config_loader.py

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml


# ============================================================
# Path helpers
# ============================================================

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

    apply_settings_to_object(target, settings, overwrite=overwrite)


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