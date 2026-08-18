from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from modules.ss_registry.schedulers.simple_kes_sched.get_sigmas import scheduler_registry
from modules.ss_registry.schedulers.simple_kes_sched.simple_kes import SimpleKEScheduler, SCHEDULER_CLASS
from modules.ss_registry.schedulers.simple_kes_sched.kes_scheduler_adapter import SimpleKESSchedulerAdapter, SCHEDULER_ADAPTER_CLASS
from modules.ss_registry.schedulers.simple_kes_sched.simple_kes_config import (
    KES_ALLOWED_DECAY_PATTERNS,
    KES_ALLOWED_STABILIZATION_METHODS,
    KES_ALLOWEDS,
    KES_COMPATIBILITY_PRESETS,
    KES_RANDOMIZATION_SAFE_BOUNDS,
    KES_RUNTIME_DEFAULTS,
)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

config_root = os.path.join(PACKAGE_DIR, "kes_config")
default_config = os.path.join(config_root, "default_config.yaml")
user_config = os.path.join(config_root, "user_config.yaml")
presets_path = os.path.join(config_root, "presets")

inferred_settings_yaml = os.path.join(config_root, "inferred_scheduler_settings.yaml")
inferred_settings_json = os.path.join(config_root, "inferred_scheduler_settings.json")

settings_file = default_config

ARG_SCHEMA_ORDER = ["type", "default", "short_desc", "long_desc", "choices", "range", "required"]


_FIELD_DESCRIPTIONS = {
    "steps": "Uses the main generation Steps control so the scheduler cannot receive a conflicting step count.",
    "device": "Uses the generation runtime device.",
    "blending_mode": "Chooses how active sigma schedules are combined. Auto selects smooth blending for two methods and weighted blending for more than two.",
    "blending_style": "Softmax normalizes blend weights. Explicit keeps their relative numeric proportions.",
    "step_progress_mode": "Controls how step-size interpolation progresses from the initial value to the final value.",
    "sigma_auto_enabled": "Automatically derives one sigma boundary from the other using Sigma Scale Factor.",
    "sigma_auto_mode": "Selects whether automatic sigma scaling derives sigma_min or sigma_max.",
    "sigma_scale_factor": "Multiplier or divisor used by automatic sigma scaling.",
    "sigma_min": "Lowest positive sigma requested before the terminal zero transition.",
    "sigma_max": "Starting sigma. Values around 35 to 50 are the established IMAGE_GEN Simple KES working range.",
    "rho": "Curvature used by Karras-derived schedules.",
    "decay_pattern": "Shared decay pattern applied after blend weights are resolved. It affects every active blend method.",
    "decay_mode": "Shared decay mode applied to the blended methods after weights are resolved.",
    "tail_steps": "Shared tail-step count applied to every active blend method after weights are resolved.",
    "allow_step_expansion": "Allows scheduler features to increase effective steps when the selected compatibility policy permits it.",
    "apply_tail_steps": "Appends individual scheduler tails when compatibility policy permits it.",
    "apply_decay_tail": "Appends decay tails when compatibility policy permits it.",
    "apply_blended_tail": "Blends available tails and appends the result when compatibility policy permits it.",
    "apply_progressive_decay": "Applies blended decay progressively across the sigma sequence.",
    "auto_tail_smoothing": "Enables automatic tail stabilization when step expansion is allowed.",
    "auto_tail_threshold": "Maximum recent drop used by automatic instability detection.",
    "jaggedness_threshold": "Variance threshold used to detect jagged tail transitions.",
    "auto_stabilization_sequence": "Ordered stabilization methods. Select multiple entries; execution follows the displayed order.",
    "start_blend": "Initial blend factor at the beginning of the schedule.",
    "end_blend": "Final blend factor near the end of the schedule.",
    "blend_midpoint": "Midpoint used by the smooth blend sigmoid.",
    "smooth_blend_factor": "Controls how sharply the blend transitions between schedules.",
    "initial_step_size": "Step-size multiplier at the beginning of the sigma sequence.",
    "final_step_size": "Step-size multiplier near the end of the sigma sequence.",
    "step_size_factor": "Additional factor applied to interpolated step size.",
    "initial_noise_scale": "Noise-scale multiplier at the beginning of the sequence.",
    "final_noise_scale": "Noise-scale multiplier near the end of the sequence.",
    "noise_scale_factor": "Additional factor applied to interpolated noise scale.",
    "early_stopping_threshold": "Convergence threshold used by the optional prepass early-stop estimator.",
    "early_stopping_method": "Statistic used by the early-stop estimator.",
    "sigma_variance_scale": "Scales the minimum sigma threshold used by prepass convergence checks.",
    "safety_minimum_stop_step": "Prevents the early-stop estimator from considering termination before this step.",
    "recent_change_convergence_delta": "Maximum difference between recent sigma-change statistics for convergence.",
    "sharpen_variance_threshold": "Variance threshold used to choose the sharpening path.",
    "sharpen_last_n_steps": "Number of ending steps considered by last-N sharpening.",
    "sharpen_mode": "Selects last-N, full-sequence, or combined sharpening.",
    "sharpness": "Multiplier applied to sigma values selected by sharpening.",
    "skip_prepass": "Skips the optional prepass schedule evaluation.",
    "load_prepass_sigmas": "Loads saved prepass sigmas when available.",
    "save_prepass_sigmas": "Saves generated prepass sigmas.",
    "load_sigma_cache": "Loads the final sigma schedule cache when available.",
    "save_sigma_cache": "Saves the final sigma schedule cache.",
    "graph_save_enable": "Writes sigma-sequence graphs during debugging.",
    "graph_save_directory": "Directory used for saved sigma graphs.",
    "log_save_directory": "Directory used for scheduler debug logs.",
    "debug": "Enables detailed scheduler logging and validation output.",
    "verbose": "Enables verbose scheduler output.",
    "global_randomize": "Enables every supported randomization control, using the effective per-setting minimum and maximum values.",
    "allow_randomization_range_override": "Allows explicitly edited randomization minimum/maximum values to exceed the scheduler's recommended safety ranges. Global randomization still uses only the exact effective min/max values; it never expands them automatically.",
    "history_window": "Maximum history window used when repairing a truncated fixed-step tail.",
    "repair_steps": "Number of exposed tail values rebuilt after fixed-step truncation.",
    "auto_history": "Lets the tail-repair heuristic reduce the effective history window when the visible tail is already stable.",
    "blend_methods": "Per-scheduler blend weights. Set a higher number to increase that method's influence; set zero to disable it. Decay Pattern, Decay Mode, and Tail Steps are shared below and apply to the blended methods as a group.",
    "compatibility": "Pipeline compatibility policy applied before schedule construction.",
}


_RANGE_OVERRIDES: dict[str, tuple[float, float, float]] = {
    "steps": (1, 150, 1),
    "sigma_min": (0.00001, 10, 0.00001),
    "sigma_max": (0.01, 100, 0.01),
    "rho": (0.1, 50, 0.01),
    "tail_steps": (0, 150, 1),
    "sigma_scale_factor": (0.01, 1000, 0.01),
    "exp_power": (0, 20, 1),
    "min_visual_sigma": (0, 100, 0.1),
    "start_blend": (0, 1, 0.001),
    "end_blend": (0, 1, 0.001),
    "blend_midpoint": (0, 1, 0.001),
    "smooth_blend_factor": (0, 25, 0.01),
    "initial_step_size": (0, 5, 0.001),
    "final_step_size": (0, 5, 0.001),
    "step_size_factor": (0, 5, 0.001),
    "initial_noise_scale": (0, 5, 0.001),
    "final_noise_scale": (0, 5, 0.001),
    "noise_scale_factor": (0, 5, 0.001),
    "early_stopping_threshold": (0, 1, 0.0001),
    "sigma_variance_scale": (0, 5, 0.001),
    "safety_minimum_stop_step": (0, 150, 1),
    "recent_change_convergence_delta": (0, 5, 0.001),
    "sharpen_variance_threshold": (0, 5, 0.001),
    "sharpen_last_n_steps": (0, 150, 1),
    "sharpness": (0, 100, 0.01),
    "auto_tail_threshold": (0, 5, 0.001),
    "jaggedness_threshold": (0, 5, 0.001),
    "history_window": (3, 150, 1),
    "repair_steps": (2, 150, 1),
}


def _title(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _group_for(name: str) -> str:
    if name == "allow_randomization_range_override":
        return "Safety and Overrides"
    if name in {"steps", "device"}:
        return "Generation Links"
    if name in {"blend_methods", "decay_pattern", "decay_mode", "tail_steps"} or name.startswith("blend") or name in {"start_blend", "end_blend", "smooth_blend_factor"}:
        return "Schedule Blending"
    if name.startswith("sigma_") or name.startswith("rho") or name in {"sigma_min", "sigma_max", "rho", "min_visual_sigma"}:
        return "Sigma and Rho"
    if name.startswith("decay") or name.startswith("tail") or name.startswith("apply_") or name.startswith("auto_tail") or name in {
        "allow_step_expansion", "auto_stabilization_sequence", "jaggedness_threshold", "history_window", "repair_steps", "auto_history", "compatibility"
    }:
        return "Tail and Compatibility"
    if "step_size" in name or name in {"step_progress_mode", "exp_power"}:
        return "Step Shaping"
    if "noise_scale" in name:
        return "Noise Shaping"
    if name.startswith("early_") or name in {"sigma_variance_scale", "safety_minimum_stop_step", "recent_change_convergence_delta"}:
        return "Early Stop Estimator"
    if name.startswith("sharpen") or name.startswith("sharpness"):
        return "Sharpening"
    if name.endswith("_rand") or "randomization" in name or "_rand_" in name or name == "global_randomize":
        return "Randomization"
    if name.startswith("load_") or name.startswith("save_") or name.startswith("graph_") or name.startswith("log_") or name in {"skip_prepass", "debug", "verbose", "sigma_save_subfolder"}:
        return "Cache and Diagnostics"
    return "Core Scheduler"


def _enum_for(name: str) -> list[str] | None:
    if name in KES_ALLOWEDS:
        return sorted(KES_ALLOWEDS[name])
    if name == "decay_pattern":
        return sorted(KES_ALLOWED_DECAY_PATTERNS)
    if name.endswith("_randomization_type"):
        return ["asymmetric", "symmetric", "logarithmic", "exponential"]
    return None


def _numeric_range(name: str, default: Any) -> tuple[float, float, float] | None:
    if name in _RANGE_OVERRIDES:
        return _RANGE_OVERRIDES[name]
    if name.endswith("_randomization_percent"):
        return (0, 1, 0.001)
    if name.endswith("_rand_min") or name.endswith("_rand_max"):
        base_name = name.rsplit("_rand_", 1)[0]
        safe_bounds = KES_RANDOMIZATION_SAFE_BOUNDS.get(base_name)
        if safe_bounds:
            safe_min, safe_max = safe_bounds
            step = 1 if isinstance(default, int) and not isinstance(default, bool) else 0.001
            return (safe_min, safe_max, step)
        if name.startswith("sigma_"):
            return (0, 100, 0.001)
        if name.startswith("rho_"):
            return (0, 50, 0.01)
        if "sharpness" in name:
            return (0, 100, 0.01)
        return (0, 10, 0.001)
    if isinstance(default, int) and not isinstance(default, bool):
        return (0, 150, 1)
    if isinstance(default, float):
        return (0, max(5, abs(default) * 4), 0.001)
    return None


def _primitive_schema(name: str, default: Any, *, group: str | None = None) -> dict[str, Any]:
    if isinstance(default, bool):
        field_type = "boolean"
    elif isinstance(default, int):
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
        "x_group": group or _group_for(name),
    }
    enum = _enum_for(name) if isinstance(default, str) else None
    if enum:
        schema["enum"] = enum
    numeric_range = _numeric_range(name, default)
    if numeric_range:
        schema["minimum"], schema["maximum"], schema["x_slider_step"] = numeric_range
    if name.endswith("_rand_min") or name.endswith("_rand_max"):
        base_name = name.rsplit("_rand_", 1)[0]
        safe_bounds = KES_RANDOMIZATION_SAFE_BOUNDS.get(base_name)
        if safe_bounds:
            schema["x_recommended_minimum"] = safe_bounds[0]
            schema["x_recommended_maximum"] = safe_bounds[1]
            schema["x_safety_override_field"] = "allow_randomization_range_override"
    if name == "allow_randomization_range_override":
        schema["title"] = "Override Randomization Safety Limits"
        schema["x_warning"] = (
            "Use only when intentionally testing wider per-setting randomization ranges. "
            "Global randomization will still remain inside the min/max values you enter."
        )
    return schema


def _blend_methods_schema() -> dict[str, Any]:
    method_properties: dict[str, Any] = {}
    method_defaults: dict[str, Any] = {}
    runtime_methods = deepcopy(KES_RUNTIME_DEFAULTS["blend_methods"])
    for method_name in scheduler_registry:
        default_weight = (runtime_methods.get(method_name) or {}).get("weight", 0.0)
        method_default = {"weight": default_weight}
        method_defaults[method_name] = method_default
        method_properties[method_name] = {
            "type": "object",
            "title": _title(method_name),
            "default": method_default,
            "properties": {
                "weight": {
                    "type": "number",
                    "default": method_default["weight"],
                    "title": "Weight",
                    "description": "Relative contribution of this scheduler. Higher numbers increase its influence; set to zero to disable it.",
                    "minimum": 0,
                    "maximum": 10,
                    "x_slider_step": 0.01,
                },
            },
        }
    return {
        "type": "object",
        "title": "Blend Methods",
        "description": _FIELD_DESCRIPTIONS["blend_methods"],
        "default": method_defaults,
        "properties": method_properties,
        "x_group": "Schedule Blending",
        "x_editor": "blend_methods",
    }


def _compatibility_schema() -> dict[str, Any]:
    default = deepcopy(KES_RUNTIME_DEFAULTS["compatibility"])
    return {
        "type": "object",
        "title": "Compatibility Policy",
        "description": _FIELD_DESCRIPTIONS["compatibility"],
        "default": default,
        "x_group": "Tail and Compatibility",
        "properties": {
            "pipeline_mode": {
                "type": "string",
                "default": default["pipeline_mode"],
                "title": "Pipeline Mode",
                "description": "Selects the runtime compatibility preset.",
                "enum": sorted(KES_COMPATIBILITY_PRESETS),
            },
            "truncate_to_requested_steps": {
                "type": "boolean",
                "default": default["truncate_to_requested_steps"],
                "title": "Truncate To Requested Steps",
                "description": "Repairs and truncates the schedule to the main generation step count.",
            },
            "warn_on_feature_downgrade": {
                "type": "boolean",
                "default": default["warn_on_feature_downgrade"],
                "title": "Warn On Feature Downgrade",
                "description": "Records warnings when compatibility policy disables requested features.",
            },
        },
    }


def _build_config_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {}
    ordered_names = list(KES_RUNTIME_DEFAULTS)
    blended_tail_names = ["decay_pattern", "decay_mode", "tail_steps"]
    for name in reversed(blended_tail_names):
        if name in ordered_names:
            ordered_names.remove(name)
    if "blend_methods" in ordered_names:
        blend_index = ordered_names.index("blend_methods") + 1
        for offset, name in enumerate(blended_tail_names):
            ordered_names.insert(blend_index + offset, name)
    for name in ordered_names:
        default = KES_RUNTIME_DEFAULTS[name]
        if name == "blend_methods":
            properties[name] = _blend_methods_schema()
        elif name == "compatibility":
            properties[name] = _compatibility_schema()
        elif name == "auto_stabilization_sequence":
            properties[name] = {
                "type": "array",
                "default": deepcopy(default),
                "title": "Auto Stabilization Sequence",
                "description": _FIELD_DESCRIPTIONS[name],
                "items": {
                    "type": "string",
                    "enum": list(KES_RUNTIME_DEFAULTS["auto_stabilization_sequence"]),
                },
                "x_group": "Tail and Compatibility",
            }
        else:
            properties[name] = _primitive_schema(name, default)
    return {
        "type": "object",
        "properties": properties,
        "required": [],
        "additionalProperties": True,
    }


def _build_meta() -> dict:
    github_repo = "https://github.com/Kittensx/Simple_KES"
    author_link = {
        "author": "KittensX",
        "author links": {
            "civitai": "https://civitai.com/user/KittensX",
            "github": "https://github.com/Kittensx",
            "ko-fi": "https://ko-fi.com/kittensx",
        },
    }

    return {
        "name": "simple_kes",
        "label": "Simple Karras Exponential Scheduler",
        "summary_text": "Custom KES scheduler with scheduler blending.",
        "optional_links": {
            "github_repo": github_repo,
            "author": author_link,
        },
        "args": _build_config_schema()["properties"],
        "config_file": {
            "yaml": inferred_settings_yaml,
            "json": inferred_settings_json,
        },
        "supports_pipeline_modes": ["fixed_steps", "compatible"],
        "supports_step_expansion": True,
        "supports_tail_steps": True,
        "supports_decay_tail": True,
        "supports_blended_tail": True,
        "supports_progressive_decay": True,
        "scheduler_family": "kes",
        "schedule_domain": "vp_sigma",
        "config_root": config_root,
        "default_config": default_config,
        "user_config": user_config,
        "presets_path": presets_path,
        "inferred_settings_yaml": inferred_settings_yaml,
        "inferred_settings_json": inferred_settings_json,
    }


meta = _build_meta()

PLUGIN_DESCRIPTOR = {
    "plugin_id": "scheduler.simple_kes",
    "kind": "scheduler",
    "name": "simple_kes",
    "label": "Simple Karras Exponential Scheduler",
    "description": meta["summary_text"],
    "version": "2",
    "module": __name__,
    "adapter_class": "SimpleKESSchedulerAdapter",
    "aliases": ["simple kes", "karras exponential", "kes scheduler"],
    "capabilities": {
        "pipeline_modes": ["fixed_steps", "extended_steps", "compatible"],
        "supports_fixed_steps": True,
        "supports_step_expansion": True,
        "supports_tail_metadata": True,
        "supports_tail_steps": True,
        "supports_decay_tail": True,
        "supports_blended_tail": True,
        "supports_progressive_decay": True,
        "scheduler_family": "kes",
        "schedule_domain": "vp_sigma",
    },
    "config_schema": _build_config_schema(),
}

__all__ = [
    "SimpleKEScheduler",
    "SimpleKESSchedulerAdapter",
    "SCHEDULER_CLASS",
    "SCHEDULER_ADAPTER_CLASS",
    "meta",
    "config_root",
    "default_config",
    "user_config",
    "presets_path",
    "settings_file",
    "PLUGIN_DESCRIPTOR",
]
