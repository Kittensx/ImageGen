from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

          
        
KES_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "blending_mode": "default",
    "blending_style": "softmax",
    "step_progress_mode": "linear",
    "allow_randomization_range_override": False,

    "sigma_auto_enabled": False,
    "sigma_auto_mode": "sigma_min",
    "sigma_scale_factor": 3.0,

    "sigma_min": 0.02,
    "sigma_max": 50.0,
    "rho": 7.0,
    "steps": 25,
    "device": "cpu",
    
    "exp_power": 2,
    "min_visual_sigma": 10,
    "sigma_save_subfolder": "saved_sigmas",

    "decay_pattern": "extrapolate",
    "decay_mode": "append",
    "tail_steps": 1,

    "allow_step_expansion": False,
    "apply_tail_steps": False,
    "apply_decay_tail": False,
    "apply_blended_tail": False,
    "apply_progressive_decay": False,

    "auto_tail_smoothing": True,
    "auto_tail_threshold": 0.05,
    "jaggedness_threshold": 0.01,
    "auto_stabilization_sequence": [
        "smooth_interpolation",
        "append_tail",
        "blend_tail",
        "apply_decay",
        "progressive_decay",
    ],

    "start_blend": 0.05,
    "end_blend": 0.4,
    "blend_midpoint": 0.5,
    "smooth_blend_factor": 5.0,

    "initial_step_size": 0.9,
    "final_step_size": 0.2,
    "step_size_factor": 0.80814932869181,

    "initial_noise_scale": 1.25,
    "final_noise_scale": 0.80,
    "noise_scale_factor": 0.8113992828873163,

    "early_stopping_threshold": 0.06,
    "early_stopping_method": "max",
    "sigma_variance_scale": 0.1,
    "safety_minimum_stop_step": 10,
    "recent_change_convergence_delta": 0.6,

    "sharpen_variance_threshold": 0.01,
    "sharpen_last_n_steps": 10,
    "sharpen_mode": "full",
    "sharpness": 0.85,

    "skip_prepass": True,
    "load_prepass_sigmas": False,
    "save_prepass_sigmas": False,
    "load_sigma_cache": False,
    "save_sigma_cache": False,

    "graph_save_enable": False,
    "graph_save_directory": "image_generation_data",
    "log_save_directory": "image_generation_data",

    "debug": False,
    "verbose": False,
    "global_randomize": False,

    "blend_methods": {
        "karras": {
            "weight": 1.0,
            "decay_pattern": "exponential",
            "decay_mode": "blend",
            "tail_steps": 1,
        },
        "exponential": {
            "weight": 1.0,
            "decay_pattern": "geometric",
            "decay_mode": "blend",
            "tail_steps": 1,
        },
    },

    "compatibility": {
        "pipeline_mode": "fixed_steps",
        "truncate_to_requested_steps": True,
        "warn_on_feature_downgrade": True,
    },
    "history_window": 10,
    "repair_steps": 4,
    "auto_history": False,
    
    # Randomization controls
    "rho_rand": False,
    "rho_rand_min": 3.0,
    "rho_rand_max": 8.0,
    "rho_enable_randomization_type": False,
    "rho_randomization_type": "log",
    "rho_randomization_percent": 0.1,

    "sigma_min_rand": False,
    "sigma_min_rand_min": 0.001,
    "sigma_min_rand_max": 0.02,
    "sigma_min_enable_randomization_type": False,
    "sigma_min_randomization_type": "asymmetric",
    "sigma_min_randomization_percent": 0.2,

    "sigma_max_rand": False,
    "sigma_max_rand_min": 25,
    "sigma_max_rand_max": 60,
    "sigma_max_enable_randomization_type": False,
    "sigma_max_randomization_type": "log",
    "sigma_max_randomization_percent": 0.25,

    "start_blend_rand": False,
    "start_blend_rand_min": 0.04,
    "start_blend_rand_max": 0.11,
    "start_blend_enable_randomization_type": False,
    "start_blend_randomization_type": "asymmetric",
    "start_blend_randomization_percent": 0.1,

    "end_blend_rand": False,
    "end_blend_rand_min": 0.4,
    "end_blend_rand_max": 0.6,
    "end_blend_enable_randomization_type": False,
    "end_blend_randomization_type": "asymmetric",
    "end_blend_randomization_percent": 0.2,

    "sharpness_rand": False,
    "sharpness_rand_min": 0.75,
    "sharpness_rand_max": 0.95,
    "sharpness_enable_randomization_type": False,
    "sharpness_randomization_type": "asymmetric",
    "sharpness_randomization_percent": 0.2,

    "smooth_blend_factor_rand": False,
    "smooth_blend_factor_rand_min": 6,
    "smooth_blend_factor_rand_max": 11,
    "smooth_blend_factor_enable_randomization_type": False,
    "smooth_blend_factor_randomization_type": "asymmetric",
    "smooth_blend_factor_randomization_percent": 0.2,

    "initial_step_size_rand": False,
    "initial_step_size_rand_min": 0.7,
    "initial_step_size_rand_max": 1.0,
    "initial_step_size_enable_randomization_type": False,
    "initial_step_size_randomization_type": "asymmetric",
    "initial_step_size_randomization_percent": 0.2,

    "final_step_size_rand": False,
    "final_step_size_rand_min": 0.1,
    "final_step_size_rand_max": 0.3,
    "final_step_size_enable_randomization_type": False,
    "final_step_size_randomization_type": "asymmetric",
    "final_step_size_randomization_percent": 0.2,

    "step_size_factor_rand": False,
    "step_size_factor_rand_min": 0.65,
    "step_size_factor_rand_max": 0.85,
    "step_size_factor_enable_randomization_type": False,
    "step_size_factor_randomization_type": "asymmetric",
    "step_size_factor_randomization_percent": 0.2,

    "initial_noise_scale_rand": False,
    "initial_noise_scale_rand_min": 1.0,
    "initial_noise_scale_rand_max": 1.5,
    "initial_noise_scale_enable_randomization_type": False,
    "initial_noise_scale_randomization_type": "asymmetric",
    "initial_noise_scale_randomization_percent": 0.2,

    "final_noise_scale_rand": False,
    "final_noise_scale_rand_min": 0.6,
    "final_noise_scale_rand_max": 1.0,
    "final_noise_scale_enable_randomization_type": False,
    "final_noise_scale_randomization_type": "asymmetric",
    "final_noise_scale_randomization_percent": 0.2,

    "noise_scale_factor_rand": False,
    "noise_scale_factor_rand_min": 0.75,
    "noise_scale_factor_rand_max": 0.95,
    "noise_scale_factor_enable_randomization_type": False,
    "noise_scale_factor_randomization_type": "asymmetric",
    "noise_scale_factor_randomization_percent": 0.2,

    "early_stopping_threshold_rand": False,
    "early_stopping_threshold_rand_min": 0.001,
    "early_stopping_threshold_rand_max": 0.02,
    "early_stopping_threshold_enable_randomization_type": False,
    "early_stopping_threshold_randomization_type": "asymmetric",
    "early_stopping_threshold_randomization_percent": 0.2,
}


KES_RANDOMIZATION_SAFE_BOUNDS: Dict[str, tuple[float, float]] = {
    key[:-9]: (
        float(value),
        float(KES_RUNTIME_DEFAULTS[f"{key[:-9]}_rand_max"]),
    )
    for key, value in KES_RUNTIME_DEFAULTS.items()
    if key.endswith("_rand_min") and f"{key[:-9]}_rand_max" in KES_RUNTIME_DEFAULTS
}


def enforce_randomization_safety(settings: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """Normalize configured randomization ranges against the supported defaults.

    The safety override only permits explicit user ranges to extend past the
    recommended defaults. Global randomization never invents wider limits; it
    always consumes the effective min/max values stored in the settings.
    """

    normalized = deepcopy(settings)
    warnings: list[str] = []
    override_enabled = _coerce_bool(
        normalized.get("allow_randomization_range_override"),
        KES_RUNTIME_DEFAULTS["allow_randomization_range_override"],
    )
    normalized["allow_randomization_range_override"] = override_enabled

    for base_key, (safe_min, safe_max) in KES_RANDOMIZATION_SAFE_BOUNDS.items():
        min_key = f"{base_key}_rand_min"
        max_key = f"{base_key}_rand_max"
        configured_min = _coerce_float(normalized.get(min_key), safe_min)
        configured_max = _coerce_float(normalized.get(max_key), safe_max)

        if configured_min > configured_max:
            configured_min, configured_max = configured_max, configured_min
            warnings.append(
                f"{min_key} and {max_key} were reversed and have been reordered."
            )

        if override_enabled:
            if configured_min < safe_min or configured_max > safe_max:
                warnings.append(
                    f"Randomization safety override is active for {base_key}: "
                    f"configured range {configured_min:g} to {configured_max:g} exceeds "
                    f"the recommended {safe_min:g} to {safe_max:g} range."
                )
        else:
            clamped_min = max(configured_min, safe_min)
            clamped_max = min(configured_max, safe_max)
            if clamped_min > clamped_max:
                clamped_min, clamped_max = safe_min, safe_max
            if clamped_min != configured_min or clamped_max != configured_max:
                warnings.append(
                    f"{base_key} randomization range was limited to the supported "
                    f"{clamped_min:g} to {clamped_max:g} range. Enable the safety "
                    "override to use an explicitly wider manual range."
                )
            configured_min, configured_max = clamped_min, clamped_max

        default_min = KES_RUNTIME_DEFAULTS[min_key]
        default_max = KES_RUNTIME_DEFAULTS[max_key]
        if isinstance(default_min, int) and not isinstance(default_min, bool):
            normalized[min_key] = int(round(configured_min))
        else:
            normalized[min_key] = float(configured_min)
        if isinstance(default_max, int) and not isinstance(default_max, bool):
            normalized[max_key] = int(round(configured_max))
        else:
            normalized[max_key] = float(configured_max)

    return normalized, warnings


KES_ALLOWEDS: Dict[str, set] = {
    "blending_mode": {"auto", "default", "smooth_blend", "weights"},
    "blending_style": {"softmax", "explicit"},
    "step_progress_mode": {"linear", "exponential", "logarithmic", "sigmoid"},
    "sigma_auto_mode": {"sigma_min", "sigma_max"},
    "decay_mode": {"append", "blend", "replace"},
    "early_stopping_method": {"mean", "max", "sum"},
    "sharpen_mode": {"last_n", "full", "both"},
}

KES_ALLOWED_DECAY_PATTERNS = {
    "zero",
    "soft_landing",
    "extrapolate",
    "fractional",
    "geometric",
    "harmonic",
    "logarithmic",
    "exponential",
    "linear",
}

KES_ALLOWED_STABILIZATION_METHODS = {
    "smooth_interpolation",
    "append_tail",
    "blend_tail",
    "apply_decay",
    "progressive_decay",
}

KES_ALLOWED_BLEND_METHOD_KEYS = {
    "weight",
    "decay_pattern",
    "decay_mode",
    "tail_steps",
}

KES_FEATURE_KEYS_THAT_CAN_EXPAND_STEPS = {
    "apply_tail_steps",
    "apply_decay_tail",
    "apply_blended_tail",
    "allow_step_expansion",
}

KES_COMPATIBILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "fixed_steps": {
        "allow_step_expansion": False,
        "allow_tail_append": False,
        "allow_decay_append": False,
        "truncate_to_requested_steps": True,
        "warn_on_feature_downgrade": True,
        "compatibility_mode": "fixed_steps",
    },
    "a1111": {
        "allow_step_expansion": False,
        "allow_tail_append": False,
        "allow_decay_append": False,
        "truncate_to_requested_steps": True,
        "warn_on_feature_downgrade": True,
        "compatibility_mode": "a1111",
    },
    "custom": {
        "allow_step_expansion": True,
        "allow_tail_append": True,
        "allow_decay_append": True,
        "truncate_to_requested_steps": False,
        "warn_on_feature_downgrade": False,
        "compatibility_mode": "custom",
    },
    "flexible": {
        "allow_step_expansion": True,
        "allow_tail_append": True,
        "allow_decay_append": True,
        "truncate_to_requested_steps": False,
        "warn_on_feature_downgrade": False,
        "compatibility_mode": "flexible",
    },
}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_choice(
    settings: Dict[str, Any],
    key: str,
    allowed: Iterable[str],
    default: str,
) -> None:
    value = settings.get(key, default)
    if isinstance(value, str):
        value = value.strip().lower()
    if value not in allowed:
        value = default
    settings[key] = value


def _clamp_min(value: float, floor: float) -> float:
    if value < floor:
        return floor
    return value


def _ensure_dict(value: Any, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return deepcopy(default or {})


def _select_legacy_shared_setting(values: list[tuple[str, Any]], weights: Dict[str, float]) -> Any:
    if not values:
        return None
    positive = [item for item in values if weights.get(item[0], 0.0) > 0]
    if positive:
        values = positive
    counts: Dict[str, tuple[int, Any]] = {}
    for _method, value in values:
        key = repr(value)
        current_count, _current_value = counts.get(key, (0, value))
        counts[key] = (current_count + 1, value)
    return max(counts.values(), key=lambda item: item[0])[1]


def _promote_legacy_blend_method_shared_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not settings:
        return {}
    blend_methods = _ensure_dict(settings.get("blend_methods"), {})
    if not blend_methods:
        return {}
    weights: Dict[str, float] = {}
    legacy_values: Dict[str, list[tuple[str, Any]]] = {
        "decay_pattern": [],
        "decay_mode": [],
        "tail_steps": [],
    }
    for method_name, method_cfg in blend_methods.items():
        if not isinstance(method_name, str) or not isinstance(method_cfg, dict):
            continue
        normalized_name = method_name.strip().lower()
        weights[normalized_name] = _coerce_float(method_cfg.get("weight"), 0.0)
        if "decay_pattern" in method_cfg:
            legacy_values["decay_pattern"].append((normalized_name, method_cfg.get("decay_pattern")))
        if "decay_mode" in method_cfg:
            legacy_values["decay_mode"].append((normalized_name, method_cfg.get("decay_mode")))
        if "tail_steps" in method_cfg:
            legacy_values["tail_steps"].append((normalized_name, _coerce_int(method_cfg.get("tail_steps"), 0)))
    promoted: Dict[str, Any] = {}
    for key, values in legacy_values.items():
        if key in settings:
            continue
        selected = _select_legacy_shared_setting(values, weights)
        if selected is not None:
            promoted[key] = selected
    return promoted


def validate_blend_methods(settings: Dict[str, Any]) -> Dict[str, Any]:
    settings = deepcopy(settings)
    default_methods = deepcopy(KES_RUNTIME_DEFAULTS["blend_methods"])
    blend_methods = _ensure_dict(settings.get("blend_methods"), default_methods)

    normalized_methods: Dict[str, Dict[str, Any]] = {}

    for method_name, method_cfg in blend_methods.items():
        if not isinstance(method_name, str):
            continue

        method_cfg = _ensure_dict(method_cfg, {})
        cleaned_cfg = {
            "weight": _coerce_float(
                method_cfg.get("weight", default_methods.get(method_name, {}).get("weight", 1.0)),
                default_methods.get(method_name, {}).get("weight", 1.0),
            ),
        }

        if cleaned_cfg["weight"] < 0:
            cleaned_cfg["weight"] = 0.0

        if cleaned_cfg["weight"] > 0:
            normalized_methods[method_name.strip().lower()] = cleaned_cfg

    if not normalized_methods:
        normalized_methods = {
            name: {"weight": _coerce_float(method_cfg.get("weight"), 0.0)}
            for name, method_cfg in default_methods.items()
            if isinstance(name, str) and _coerce_float(method_cfg.get("weight"), 0.0) > 0
        }

    settings["blend_methods"] = normalized_methods
    return settings

def _normalize_randomization_type(value: Any, default: str = "asymmetric") -> str:
    aliases = {
        "symmetric": "symmetric",
        "sym": "symmetric",
        "s": "symmetric",
        "asymmetric": "asymmetric",
        "assym": "asymmetric",
        "asym": "asymmetric",
        "a": "asymmetric",
        "logarithmic": "logarithmic",
        "log": "logarithmic",
        "l": "logarithmic",
        "exponential": "exponential",
        "exp": "exponential",
        "e": "exponential",
    }
    if not isinstance(value, str):
        return default
    return aliases.get(value.strip().lower(), default)
    
def normalize_simple_kes_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(KES_RUNTIME_DEFAULTS)
    legacy_shared_blend_settings = _promote_legacy_blend_method_shared_settings(settings)
    if settings:
        # shallow dict merge is fine at top-level; nested configs are normalized below
        for key, value in settings.items():
            if key == "compatibility" and isinstance(value, dict):
                merged["compatibility"] = {**merged["compatibility"], **value}
            else:
                merged[key] = value
    for key, value in legacy_shared_blend_settings.items():
        if settings is None or key not in settings:
            merged[key] = value

    for key, default in KES_RUNTIME_DEFAULTS.items():
        if key not in merged or merged[key] is None:
            merged[key] = deepcopy(default)

    for key in ("blending_mode", "blending_style", "step_progress_mode", "sigma_auto_mode",
                "decay_mode", "early_stopping_method", "sharpen_mode"):
        _normalize_choice(
            merged,
            key,
            KES_ALLOWEDS[key],
            KES_RUNTIME_DEFAULTS[key],
        )

    if merged.get("decay_pattern") not in KES_ALLOWED_DECAY_PATTERNS:
        merged["decay_pattern"] = KES_RUNTIME_DEFAULTS["decay_pattern"]

    merged["steps"] = max(1, _coerce_int(merged.get("steps"), KES_RUNTIME_DEFAULTS["steps"]))
    merged["tail_steps"] = max(0, _coerce_int(merged.get("tail_steps"), KES_RUNTIME_DEFAULTS["tail_steps"]))

    for key in (
        "sigma_min",
        "sigma_max",
        "rho",
        "sigma_scale_factor",
        "start_blend",
        "end_blend",
        "blend_midpoint",
        "smooth_blend_factor",
        "initial_step_size",
        "final_step_size",
        "step_size_factor",
        "initial_noise_scale",
        "final_noise_scale",
        "noise_scale_factor",
        "early_stopping_threshold",
        "sigma_variance_scale",
        "recent_change_convergence_delta",
        "sharpen_variance_threshold",
        "sharpness",
        "auto_tail_threshold",
        "jaggedness_threshold",
        "rho_rand_min",
        "rho_rand_max",
        "rho_randomization_percent",
        "sigma_min_rand_min",
        "sigma_min_rand_max",
        "sigma_min_randomization_percent",
        "sigma_max_rand_min",
        "sigma_max_rand_max",
        "sigma_max_randomization_percent",
        "start_blend_rand_min",
        "start_blend_rand_max",
        "start_blend_randomization_percent",
        "end_blend_rand_min",
        "end_blend_rand_max",
        "end_blend_randomization_percent",
        "sharpness_rand_min",
        "sharpness_rand_max",
        "sharpness_randomization_percent",
        "initial_step_size_rand_min",
        "initial_step_size_rand_max",
        "initial_step_size_randomization_percent",
        "final_step_size_rand_min",
        "final_step_size_rand_max",
        "final_step_size_randomization_percent",
        "step_size_factor_rand_min",
        "step_size_factor_rand_max",
        "step_size_factor_randomization_percent",
        "initial_noise_scale_rand_min",
        "initial_noise_scale_rand_max",
        "initial_noise_scale_randomization_percent",
        "final_noise_scale_rand_min",
        "final_noise_scale_rand_max",
        "final_noise_scale_randomization_percent",
        "noise_scale_factor_rand_min",
        "noise_scale_factor_rand_max",
        "noise_scale_factor_randomization_percent",
        "early_stopping_threshold_rand_min",
        "early_stopping_threshold_rand_max",
        "early_stopping_threshold_randomization_percent",
        "min_visual_sigma",
    ):
        merged[key] = _coerce_float(merged.get(key), KES_RUNTIME_DEFAULTS[key])

    for key in (
        "safety_minimum_stop_step",
        "sharpen_last_n_steps",
        "exp_power",
        "smooth_blend_factor_rand_min",
        "smooth_blend_factor_rand_max",
        "sigma_max_rand_min",
        "sigma_max_rand_max",
        "exp_power",
        "history_window",
        "repair_steps",
        
    ):
        merged[key] = max(0, _coerce_int(merged.get(key), KES_RUNTIME_DEFAULTS.get(key, 0)))

    for key in (
        "sigma_auto_enabled",
        "allow_step_expansion",
        "apply_tail_steps",
        "apply_decay_tail",
        "apply_blended_tail",
        "apply_progressive_decay",
        "auto_tail_smoothing",
        "skip_prepass",
        "load_prepass_sigmas",
        "save_prepass_sigmas",
        "load_sigma_cache",
        "save_sigma_cache",
        "graph_save_enable",
        "debug",
        "verbose",
        "global_randomize",
        "allow_randomization_range_override",
        "rho_rand",
        "rho_enable_randomization_type",
        "sigma_min_rand",
        "sigma_min_enable_randomization_type",
        "sigma_max_rand",
        "sigma_max_enable_randomization_type",
        "start_blend_rand",
        "start_blend_enable_randomization_type",
        "end_blend_rand",
        "end_blend_enable_randomization_type",
        "sharpness_rand",
        "sharpness_enable_randomization_type",
        "smooth_blend_factor_rand",
        "smooth_blend_factor_enable_randomization_type",
        "initial_step_size_rand",
        "initial_step_size_enable_randomization_type",
        "final_step_size_rand",
        "final_step_size_enable_randomization_type",
        "step_size_factor_rand",
        "step_size_factor_enable_randomization_type",
        "initial_noise_scale_rand",
        "initial_noise_scale_enable_randomization_type",
        "final_noise_scale_rand",
        "final_noise_scale_enable_randomization_type",
        "noise_scale_factor_rand",
        "noise_scale_factor_enable_randomization_type",
        "early_stopping_threshold_rand",
        "early_stopping_threshold_enable_randomization_type",
        "auto_history",
    ):
        merged[key] = _coerce_bool(merged.get(key), KES_RUNTIME_DEFAULTS[key])
    
    for key in (
        "rho_randomization_type",
        "sigma_min_randomization_type",
        "sigma_max_randomization_type",
        "start_blend_randomization_type",
        "end_blend_randomization_type",
        "sharpness_randomization_type",
        "smooth_blend_factor_randomization_type",
        "initial_step_size_randomization_type",
        "final_step_size_randomization_type",
        "step_size_factor_randomization_type",
        "initial_noise_scale_randomization_type",
        "final_noise_scale_randomization_type",
        "noise_scale_factor_randomization_type",
        "early_stopping_threshold_randomization_type",
    ):
        merged[key] = _normalize_randomization_type(merged.get(key), "asymmetric")
        
    merged["sigma_min"] = _clamp_min(merged["sigma_min"], 1e-5)
    merged["sigma_max"] = _clamp_min(merged["sigma_max"], merged["sigma_min"])
    if merged["sigma_max"] <= merged["sigma_min"]:
        merged["sigma_max"] = max(merged["sigma_min"] * 2.0, KES_RUNTIME_DEFAULTS["sigma_max"])

    if merged["start_blend"] < 0:
        merged["start_blend"] = 0.0
    if merged["end_blend"] < 0:
        merged["end_blend"] = 0.0

    auto_sequence = merged.get("auto_stabilization_sequence", KES_RUNTIME_DEFAULTS["auto_stabilization_sequence"])
    if not isinstance(auto_sequence, list):
        auto_sequence = deepcopy(KES_RUNTIME_DEFAULTS["auto_stabilization_sequence"])
    auto_sequence = [
        str(item).strip()
        for item in auto_sequence
        if str(item).strip() in KES_ALLOWED_STABILIZATION_METHODS
    ]
    if not auto_sequence:
        auto_sequence = deepcopy(KES_RUNTIME_DEFAULTS["auto_stabilization_sequence"])
    merged["auto_stabilization_sequence"] = auto_sequence

    compatibility = _ensure_dict(merged.get("compatibility"), KES_RUNTIME_DEFAULTS["compatibility"])
    if "pipeline_mode" not in compatibility or not compatibility["pipeline_mode"]:
        compatibility["pipeline_mode"] = KES_RUNTIME_DEFAULTS["compatibility"]["pipeline_mode"]
    compatibility["pipeline_mode"] = str(compatibility["pipeline_mode"]).strip().lower()
    compatibility["truncate_to_requested_steps"] = _coerce_bool(
        compatibility.get("truncate_to_requested_steps"),
        KES_RUNTIME_DEFAULTS["compatibility"]["truncate_to_requested_steps"],
    )
    compatibility["warn_on_feature_downgrade"] = _coerce_bool(
        compatibility.get("warn_on_feature_downgrade"),
        KES_RUNTIME_DEFAULTS["compatibility"]["warn_on_feature_downgrade"],
    )
    merged["compatibility"] = compatibility

    merged = validate_blend_methods(merged)
    merged, safety_warnings = enforce_randomization_safety(merged)
    merged["_validation_warnings"] = list(dict.fromkeys(safety_warnings))
    return merged


def resolve_simple_kes_pipeline_policy(
    settings: Dict[str, Any],
    pipeline_mode: Optional[str] = None,
) -> Dict[str, Any]:
    settings = deepcopy(settings)
    compatibility = _ensure_dict(settings.get("compatibility"), KES_RUNTIME_DEFAULTS["compatibility"])

    mode = pipeline_mode or compatibility.get("pipeline_mode") or "fixed_steps"
    mode = str(mode).strip().lower()
    if mode not in KES_COMPATIBILITY_PRESETS:
        mode = "fixed_steps"

    policy = deepcopy(KES_COMPATIBILITY_PRESETS[mode])

    policy["truncate_to_requested_steps"] = _coerce_bool(
        compatibility.get("truncate_to_requested_steps"),
        policy["truncate_to_requested_steps"],
    )
    policy["warn_on_feature_downgrade"] = _coerce_bool(
        compatibility.get("warn_on_feature_downgrade"),
        policy["warn_on_feature_downgrade"],
    )

    return policy


def validate_simple_kes_settings(
    settings: Optional[Dict[str, Any]],
    *,
    pipeline_mode: Optional[str] = None,
) -> Dict[str, Any]:
    validated = normalize_simple_kes_settings(settings)
    policy = resolve_simple_kes_pipeline_policy(validated, pipeline_mode=pipeline_mode)

    warnings: list[str] = list(validated.pop("_validation_warnings", []) or [])

    if not policy["allow_step_expansion"]:
        if validated["allow_step_expansion"]:
            warnings.append(
                "allow_step_expansion requested but disabled by compatibility policy."
            )
            validated["allow_step_expansion"] = False

    if not policy["allow_tail_append"]:
        for key in ("apply_tail_steps", "apply_blended_tail"):
            if validated.get(key):
                warnings.append(f"{key} requested but disabled by compatibility policy.")
                validated[key] = False

    if not policy["allow_decay_append"]:
        for key in ("apply_decay_tail", "apply_progressive_decay"):
            if validated.get(key):
                warnings.append(f"{key} requested but disabled by compatibility policy.")
                validated[key] = False

    active_methods = {
        name: config
        for name, config in validated.get("blend_methods", {}).items()
        if isinstance(config, dict) and _coerce_float(config.get("weight"), 0.0) > 0
    }
    if not active_methods:
        active_methods = {
            name: {"weight": _coerce_float(config.get("weight"), 0.0)}
            for name, config in KES_RUNTIME_DEFAULTS["blend_methods"].items()
            if isinstance(config, dict) and _coerce_float(config.get("weight"), 0.0) > 0
        }
    validated["blend_methods"] = active_methods

    if validated["blending_mode"] in {"default", "auto"}:
        # Blend weights are authoritative. Any method with a positive weight is
        # active; zero-weight methods are excluded. Weighted mode works for one,
        # two, or many active methods and preserves the user's relative weights.
        validated["blending_mode"] = "weights"
    elif validated["blending_mode"] == "smooth_blend" and len(active_methods) != 2:
        warnings.append(
            "smooth_blend requires exactly two active methods; weighted blending was used instead."
        )
        validated["blending_mode"] = "weights"

    compatibility = _ensure_dict(validated.get("compatibility"), {})
    compatibility["pipeline_mode"] = policy["compatibility_mode"]
    compatibility["truncate_to_requested_steps"] = policy["truncate_to_requested_steps"]
    compatibility["warn_on_feature_downgrade"] = policy["warn_on_feature_downgrade"]
    validated["compatibility"] = compatibility

    validated["_policy"] = policy
    validated["_validation_warnings"] = warnings

    return validated
