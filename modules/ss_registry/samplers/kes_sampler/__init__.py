from __future__ import annotations

from copy import deepcopy
from typing import Any

from .simple_kes_sampler.kes_sampler import KESSampler, SAMPLER_CLASS
from .simple_kes_sampler.kes_sampler_adapter import KESSamplerAdapter, SAMPLER_ADAPTER_CLASS
from modules.ss_registry.samplers.sampler_config_loader import build_sampler_cfg_settings

required_args = {
    "raw_model_fn": "callable",
    "latents": "torch.Tensor",
    "schedule": "SchedulerOutput-like object with sigmas",
    "conditioning": "ConditioningOutput-like object with cond/uncond",
    "request": "request object with steps/cfg_scale",
}

optional_args = {
    "state": "shared pipeline state",
    "config_path": "optional sampler config path",
    "preset_name": "optional sampler preset name",
}

meta = {
    "name": "kes",
    "label": "KES Sampler",
    "description": "Pipeline-facing Euler/Heun sampler with config-driven CFG strategy support.",
    "config_key": "kes",
}

_KES_ALLOWED_OPTIONS: dict[str, list[str]] = {
    "sampler_type": ["euler", "heun"],
    "eta_schedule_mode": ["none", "linear", "cosine", "exp_decay", "ease_out", "auto"],
    "noise_schedule_scaling": ["none", "linear", "cosine", "exp_decay", "ease_out"],
    "adaptive_time_mode": ["none", "time_boost", "time_curve", "sigma", "manual"],
    "cfg_guidance_mode": ["legacy_flat", "auto_low_cfg", "sigma_shaped", "step_shaped"],
    "cfg_curve_type": ["linear", "cosine", "smoothstep", "exp_decay"],
}

_FIELD_DESCRIPTIONS: dict[str, str] = {
    "sampler_type": "Chooses Euler or Heun integration for the KES sampler runtime.",
    "eta": "Base stochastic noise amount. Zero keeps the sampler deterministic.",
    "add_noise": "Applies eta-driven noise injection when eta is above zero.",
    "verbose": "Enables verbose KES sampler logging.",
    "rescale_cfg": "Legacy compatibility alias for KES clamp guidance. New runs should use Legacy Clamp Guidance.",
    "rescale_cfg_factor": "Legacy compatibility alias for the KES guidance multiplier.",
    "legacy_clamp_guidance": "Retains the historical KES clamp path for exact replay of older outputs. This is different from canonical CFG rescale.",
    "legacy_guidance_multiplier": "Historical KES-only guidance multiplier retained for replay compatibility.",
    "cfg_guidance_mode": "Controls whether CFG stays flat or follows an effective per-step curve. The requested CFG remains recorded separately.",
    "cfg_curve_type": "Curve used to distribute early guidance support and late-step taper across the denoising schedule.",
    "cfg_curve_strength": "Overall strength applied to the selected effective-guidance curve.",
    "cfg_high_sigma_boost": "Additional effective CFG available during high-sigma early composition steps.",
    "cfg_low_sigma_taper": "Amount removed from effective CFG during late low-sigma detail steps.",
    "cfg_auto_low_cfg_threshold": "Auto low-CFG shaping activates only below this requested CFG value.",
    "cfg_early_floor_enabled": "Prevents effective CFG from dropping below the configured floor during the opening portion of the run.",
    "cfg_early_floor_value": "Minimum effective CFG used while the early guidance floor is active.",
    "cfg_early_floor_until_fraction": "Fraction of denoising steps covered by the early guidance floor, from 0.0 to 1.0.",
    "clamp_range": "Two-value clamp range used when CFG rescaling is enabled.",
    "initial_noise_strength": "Strength of initial latent noise added before denoising begins.",
    "eta_scale_factor": "Additional multiplier applied to per-step eta noise.",
    "eta_schedule_mode": "Shapes how eta noise is distributed across denoising steps.",
    "use_adaptive_eta": "Enables adaptive eta adjustments based on latent and denoised stability.",
    "noise_schedule_scaling": "Scales per-step stochastic noise across the denoising schedule.",
    "adaptive_time_mode": "Optional time-domain adjustment applied by the adaptive eta helper.",
    "adaptive_delta_low_floor": "If latent movement falls below this threshold, adaptive eta can be boosted.",
    "adaptive_delta_high_floor": "If latent movement exceeds this threshold, adaptive eta can be reduced.",
    "adaptive_low_adjustment_multiplier": "Multiplier applied when latent movement is below the low floor.",
    "adaptive_high_adjustment_multiplier": "Multiplier applied when latent movement is above the high floor.",
    "adaptive_denoised_floor": "Threshold used when comparing denoised output stability between steps.",
    "adaptive_denoised_adjustment_multiplier": "Multiplier applied when denoised output movement stays below the denoised floor.",
    "adaptive_manual_low_adjustment": "Manual adjustment factor used by adaptive_time_mode=manual.",
    "adaptive_manual_high_adjustment": "Second manual adjustment factor used by adaptive_time_mode=manual.",
}

_RANGE_OVERRIDES: dict[str, tuple[float, float, float]] = {
    "eta": (0.0, 5.0, 0.001),
    "rescale_cfg_factor": (0.0, 5.0, 0.001),
    "legacy_guidance_multiplier": (0.0, 5.0, 0.001),
    "cfg_curve_strength": (0.0, 3.0, 0.01),
    "cfg_high_sigma_boost": (0.0, 10.0, 0.05),
    "cfg_low_sigma_taper": (0.0, 10.0, 0.05),
    "cfg_auto_low_cfg_threshold": (0.0, 30.0, 0.1),
    "cfg_early_floor_value": (0.0, 30.0, 0.1),
    "cfg_early_floor_until_fraction": (0.0, 1.0, 0.01),
    "initial_noise_strength": (0.0, 5.0, 0.001),
    "eta_scale_factor": (0.0, 5.0, 0.001),
    "adaptive_delta_low_floor": (0.0, 5.0, 0.001),
    "adaptive_delta_high_floor": (0.0, 10.0, 0.001),
    "adaptive_low_adjustment_multiplier": (0.0, 5.0, 0.001),
    "adaptive_high_adjustment_multiplier": (0.0, 5.0, 0.001),
    "adaptive_denoised_floor": (0.0, 5.0, 0.001),
    "adaptive_denoised_adjustment_multiplier": (0.0, 5.0, 0.001),
    "adaptive_manual_low_adjustment": (0.0, 5.0, 0.001),
    "adaptive_manual_high_adjustment": (0.0, 5.0, 0.001),
}


def _title(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _group_for(name: str) -> str:
    if name in {"sampler_type", "eta", "add_noise", "verbose"}:
        return "Core Sampler"
    if name in {
        "cfg_guidance_mode", "cfg_curve_type", "cfg_curve_strength",
        "cfg_high_sigma_boost", "cfg_low_sigma_taper", "cfg_auto_low_cfg_threshold",
        "cfg_early_floor_enabled", "cfg_early_floor_value", "cfg_early_floor_until_fraction",
    }:
        return "CFG Lab"
    if name in {"legacy_clamp_guidance", "legacy_guidance_multiplier", "rescale_cfg", "rescale_cfg_factor", "clamp_range"}:
        return "Legacy Guidance Compatibility"
    if name in {
        "initial_noise_strength",
        "eta_scale_factor",
        "eta_schedule_mode",
        "noise_schedule_scaling",
        "use_adaptive_eta",
    }:
        return "Noise Injection"
    if name.startswith("adaptive_"):
        return "Adaptive Eta"
    return "KES Sampler"


def _numeric_range(name: str, default: Any) -> tuple[float, float, float] | None:
    if name in _RANGE_OVERRIDES:
        return _RANGE_OVERRIDES[name]
    if isinstance(default, bool):
        return None
    if isinstance(default, int):
        return (0.0, max(10.0, float(default) * 4.0), 1.0)
    if isinstance(default, float):
        magnitude = max(abs(float(default)), 1.0)
        return (0.0, max(5.0, magnitude * 4.0), 0.001)
    return None


def _primitive_schema(name: str, default: Any) -> dict[str, Any]:
    if isinstance(default, bool):
        field_type = "boolean"
    elif isinstance(default, int) and not isinstance(default, bool):
        field_type = "integer"
    elif isinstance(default, float):
        field_type = "number"
    else:
        field_type = "string"

    schema: dict[str, Any] = {
        "type": field_type,
        "default": deepcopy(default),
        "title": _title(name),
        "description": _FIELD_DESCRIPTIONS.get(name, ""),
        "x_group": _group_for(name),
    }
    enum = _KES_ALLOWED_OPTIONS.get(name)
    if enum and field_type == "string":
        schema["enum"] = list(enum)
    numeric_range = _numeric_range(name, default)
    if numeric_range:
        schema["minimum"], schema["maximum"], schema["x_slider_step"] = numeric_range
    if name in {
        "cfg_guidance_mode", "cfg_curve_type", "cfg_curve_strength",
        "cfg_high_sigma_boost", "cfg_low_sigma_taper", "cfg_auto_low_cfg_threshold",
        "cfg_early_floor_enabled", "cfg_early_floor_value", "cfg_early_floor_until_fraction",
    }:
        schema["x_hidden_in_advanced"] = True
        schema["x_surface"] = "cfg_lab"
    return schema


def _clamp_range_schema(default: list[float]) -> dict[str, Any]:
    return {
        "type": "array",
        "default": list(default),
        "title": "Clamp Range",
        "description": _FIELD_DESCRIPTIONS["clamp_range"],
        "x_group": _group_for("clamp_range"),
        "minItems": 2,
        "maxItems": 2,
        "x_item_titles": ["Minimum", "Maximum"],
        "items": {
            "type": "number",
            "minimum": -20.0,
            "maximum": 20.0,
            "x_slider_step": 0.01,
        },
    }


def _build_config_schema() -> dict[str, Any]:
    defaults = dict(build_sampler_cfg_settings(None))
    defaults.pop("prefer_config", None)
    for key in ("_config_path", "_config_source", "_preset_name"):
        defaults.pop(key, None)

    properties: dict[str, Any] = {}
    for name, default in defaults.items():
        if name == "clamp_range":
            properties[name] = _clamp_range_schema(list(default) if isinstance(default, (list, tuple)) else [-1.0, 1.0])
        else:
            properties[name] = _primitive_schema(name, default)

    return {
        "type": "object",
        "properties": properties,
        "required": [],
        "additionalProperties": True,
    }


PLUGIN_DESCRIPTOR = {
    "plugin_id": "sampler.kes",
    "kind": "sampler",
    "name": "kes",
    "label": "KES Sampler",
    "description": meta["description"],
    "version": "2",
    "module": __name__,
    "adapter_class": "KESSamplerAdapter",
    "aliases": ["kes_sampler", "simple_kes_sampler"],
    "capabilities": KESSamplerAdapter.SAMPLER_CAPABILITIES.to_serializable_dict(),
    "config_schema": _build_config_schema(),
}

__all__ = [
    "KESSampler",
    "KESSamplerAdapter",
    "SAMPLER_CLASS",
    "SAMPLER_ADAPTER_CLASS",
    "meta",
    "required_args",
    "optional_args",
    "PLUGIN_DESCRIPTOR",
]
