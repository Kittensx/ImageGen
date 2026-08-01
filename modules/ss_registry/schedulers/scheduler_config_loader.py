from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml


SCHEDULER_CFG_DEFAULTS: Dict[str, Any] = {
    "prefer_config": False,
}


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dictionaries without mutating inputs.
    """
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


def get_scheduler_config_root(base_path: Optional[str] = None) -> str:
    """
    Returns the shared scheduler config root.

    Expected layout:
        modules/ss_registry/schedulers/config/
            shared/
                presets/
            simple_kes/
                default.yaml
                presets/
    """
    if base_path:
        return base_path
    return os.path.join(os.path.dirname(__file__), "config")


def get_scheduler_default_config_path(
    scheduler_name: str,
    *,
    base_path: Optional[str] = None,
    default_filename: str = "default.yaml",
) -> str:
    root = get_scheduler_config_root(base_path)
    return os.path.join(root, scheduler_name, default_filename)


def get_scheduler_preset_path(
    scheduler_name: str,
    preset_name: str,
    *,
    base_path: Optional[str] = None,
    presets_subdir: str = "presets",
    extension: str = ".yaml",
) -> str:
    root = get_scheduler_config_root(base_path)
    filename = preset_name if preset_name.endswith(extension) else f"{preset_name}{extension}"
    return os.path.join(root, scheduler_name, presets_subdir, filename)


def get_shared_scheduler_preset_path(
    preset_name: str,
    *,
    base_path: Optional[str] = None,
    shared_dir: str = "shared",
    presets_subdir: str = "presets",
    extension: str = ".yaml",
) -> str:
    root = get_scheduler_config_root(base_path)
    filename = preset_name if preset_name.endswith(extension) else f"{preset_name}{extension}"
    return os.path.join(root, shared_dir, presets_subdir, filename)


def load_yaml_config_file(config_path: str) -> Dict[str, Any]:
    if not config_path:
        raise ValueError("config_path must be provided.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[SchedulerConfigLoader] Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_scheduler_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flatten nested scheduler config into one effective runtime dict.

    Generic behavior:
    - preserve top-level keys
    - if a top-level 'scheduler' section exists, merge it over the top-level keys
    - preserve provenance keys when present

    Scheduler-specific reshaping belongs in scheduler-specific validator modules.
    """
    config = deepcopy(config or {})
    flattened: Dict[str, Any] = {}

    for key, value in config.items():
        if key == "scheduler" and isinstance(value, dict):
            continue
        flattened[key] = value

    scheduler_section = config.get("scheduler", {})
    if isinstance(scheduler_section, dict):
        flattened.update(scheduler_section)

    for key in ("_config_path", "_config_source", "_preset_name"):
        if key in config:
            flattened[key] = config[key]

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


def resolve_scheduler_config_path(
    *,
    scheduler_name: str,
    config_path: Optional[str] = None,
    preset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    default_filename: str = "default.yaml",
) -> str:
    """
    Resolution order:
      1. explicit config_path
      2. scheduler-specific preset
      3. shared preset
      4. scheduler default
    """
    if config_path:
        return config_path

    if preset_name:
        scheduler_preset = get_scheduler_preset_path(
            scheduler_name,
            preset_name,
            base_path=base_path,
        )
        if os.path.exists(scheduler_preset):
            return scheduler_preset

        shared_preset = get_shared_scheduler_preset_path(
            preset_name,
            base_path=base_path,
        )
        if os.path.exists(shared_preset):
            return shared_preset

        raise FileNotFoundError(
            f"[SchedulerConfigLoader] Preset '{preset_name}' not found for scheduler "
            f"'{scheduler_name}' in scheduler-specific or shared preset folders."
        )

    default_path = get_scheduler_default_config_path(
        scheduler_name,
        base_path=base_path,
        default_filename=default_filename,
    )
    if not os.path.exists(default_path):
        raise FileNotFoundError(
            f"[SchedulerConfigLoader] Default config not found for scheduler "
            f"'{scheduler_name}': {default_path}"
        )

    return default_path


def build_scheduler_settings(
    *,
    scheduler_name: str,
    config_path: Optional[str] = None,
    preset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    default_filename: str = "default.yaml",
    overrides: Optional[Dict[str, Any]] = None,
    extra_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full config pipeline:
      resolve path -> load YAML -> flatten -> merge defaults -> resolve overrides

    Returns:
        final effective flat settings dict
    """
    final_config_path = resolve_scheduler_config_path(
        scheduler_name=scheduler_name,
        config_path=config_path,
        preset_name=preset_name,
        base_path=base_path,
        default_filename=default_filename,
    )

    raw_config = load_yaml_config_file(final_config_path)
    flattened = flatten_scheduler_config(raw_config)

    merged_defaults = deepcopy(SCHEDULER_CFG_DEFAULTS)
    if extra_defaults:
        merged_defaults = deep_merge_dicts(merged_defaults, extra_defaults)

    with_defaults = deep_merge_dicts(merged_defaults, flattened)
    resolved = resolve_settings(with_defaults, overrides)

    resolved["_config_path"] = final_config_path
    resolved["_config_source"] = (
        "file" if config_path else
        "preset" if preset_name else
        "default"
    )
    if preset_name:
        resolved["_preset_name"] = preset_name

    return resolved


def apply_settings_to_object(
    target: Any,
    settings: Dict[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    if target is None:
        raise ValueError("[SchedulerConfigLoader] target is not set.")

    for key, value in settings.items():
        if overwrite or not hasattr(target, key):
            setattr(target, key, value)


def apply_settings_to_scheduler_state(
    shared_state: Any,
    settings: Dict[str, Any],
    *,
    section_name: str = "sched",
    settings_attr: str = "scheduler_settings",
    overwrite: bool = True,
) -> None:
    """
    Stores resolved settings on a shared state object, if available.

    By default this targets:
        shared_state.sched.scheduler_settings
    """
    if shared_state is None:
        raise ValueError("[SchedulerConfigLoader] shared_state is not set.")

    section = getattr(shared_state, section_name, None)
    if section is None:
        raise AttributeError(
            f"[SchedulerConfigLoader] shared_state has no attribute '{section_name}'"
        )

    current = getattr(section, settings_attr, None)
    if overwrite or current in (None, {}):
        setattr(section, settings_attr, deepcopy(settings))


def prepare_scheduler_config(
    *,
    scheduler_name: str,
    shared_state: Optional[Any] = None,
    config_path: Optional[str] = None,
    preset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    default_filename: str = "default.yaml",
    overrides: Optional[Dict[str, Any]] = None,
    extra_defaults: Optional[Dict[str, Any]] = None,
    apply_to_state: bool = True,
    section_name: str = "sched",
    settings_attr: str = "scheduler_settings",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Convenience helper:
      build settings -> optionally apply into shared_state.sched

    Returns:
        final effective flat settings dict
    """
    settings = build_scheduler_settings(
        scheduler_name=scheduler_name,
        config_path=config_path,
        preset_name=preset_name,
        base_path=base_path,
        default_filename=default_filename,
        overrides=overrides,
        extra_defaults=extra_defaults,
    )

    if shared_state is not None and apply_to_state:
        apply_settings_to_scheduler_state(
            shared_state,
            settings,
            section_name=section_name,
            settings_attr=settings_attr,
            overwrite=overwrite,
        )

    return settings
