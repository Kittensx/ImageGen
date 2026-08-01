import os
import yaml


def config_load(config_filename=None, base_path=None, subdir=None) -> dict:
    """
    Loads a YAML config file from a subdirectory if specified.
    """
    if not config_filename:
        raise ValueError("Config filename must be provided.")

    base_path = base_path or os.path.dirname(__file__)
    config_path = os.path.join(base_path, subdir or "", config_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[SSMixinConfig] Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}

def resolve_settings(config: dict, overrides: dict = None) -> dict:
    """
    Resolves final settings from config and overrides.
    """
    overrides = overrides or {}
    prefer_config = config.get("prefer_config", False)

    if prefer_config:
        return {**overrides, **config}
    else:
        return {**config, **overrides}
        
def apply_to_state_subsection(state, section: str, data: dict, overwrite: bool = True):
    """
    Applies a dictionary of key/values to a subsection of `state`.

    Args:
        section (str): The attribute of `state` to target (e.g., "p", "d", "aliases").
        data (dict): The settings to apply.
        overwrite (bool): Whether to overwrite existing values (default True).
    """
    if not state:
        raise ValueError("[SSMixinConfig] state is not set.")

    target = getattr(state, section, None)
    if target is None:
        raise AttributeError(f"[SSMixinConfig] state has no attribute '{section}'")

    for k, v in data.items():
        if overwrite or not hasattr(target, k):
            setattr(target, k, v)

    